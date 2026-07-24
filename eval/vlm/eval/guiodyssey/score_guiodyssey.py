# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACTION_BLOCK_RE = re.compile(r"```action\s*(\{.*?\})\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
COMMAND_COORD_RE = re.compile(
    r"\b(CLICK|LONG_PRESS)\s*:\s*[\(\[]\s*([-\d.]+)\s*,\s*([-\d.]+)\s*[\)\]]",
    re.IGNORECASE,
)
COMMAND_SCROLL_RE = re.compile(r"\bSCROLL\s*:\s*(UP|DOWN|LEFT|RIGHT)\b", re.IGNORECASE)
COMMAND_TEXT_RE = re.compile(r"\b(TYPE|TEXT)\s*:\s*(.+)", re.IGNORECASE)

CLICK_COORD_THRESHOLD = 0.14
TEXT_ANLS_THRESHOLD = 0.5

ACTION_ALIASES = {
    "click": "CLICK",
    "tap": "CLICK",
    "select": "CLICK",
    "long_press": "LONG_PRESS",
    "longpress": "LONG_PRESS",
    "scroll": "SCROLL",
    "input_text": "TYPE",
    "type": "TYPE",
    "text": "TYPE",
    "write": "TYPE",
    "complete": "COMPLETE",
    "done": "COMPLETE",
    "incomplete": "INCOMPLETE",
    "impossible": "INCOMPLETE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score GUI-Odyssey predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Raw prediction JSONL path.")
    parser.add_argument("--dataset-jsonl", type=str, required=True, help="Converted GUI-Odyssey JSONL path.")
    parser.add_argument(
        "--metric",
        choices=("auto", "macro", "micro"),
        default="auto",
        help="Metric aggregation mode. 'auto' infers from the dataset metadata.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store metrics.")
    return parser.parse_args()


def calculate_anls(predicted_str: str, ground_truth_str: str) -> float:
    predicted = (predicted_str or "").strip()
    ground_truth = (ground_truth_str or "").strip()
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    if predicted in ground_truth or ground_truth in predicted:
        return 1.0

    if len(predicted) > len(ground_truth):
        predicted, ground_truth = ground_truth, predicted

    distances = list(range(len(predicted) + 1))
    for idx2, char2 in enumerate(ground_truth):
        next_distances = [idx2 + 1]
        for idx1, char1 in enumerate(predicted):
            if char1 == char2:
                next_distances.append(distances[idx1])
            else:
                next_distances.append(1 + min(distances[idx1], distances[idx1 + 1], next_distances[-1]))
        distances = next_distances

    length = max(len(predicted), len(ground_truth))
    return 0.0 if length == 0 else 1.0 - float(distances[-1]) / float(length)


def _parse_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def extract_action_json(text: str) -> Dict[str, Any]:
    content = text or ""
    match = ACTION_BLOCK_RE.search(content)
    if match:
        parsed = _parse_json_object(match.group(1))
        if parsed:
            return parsed

    if "```action" in content:
        tail = content.rsplit("```action", 1)[-1]
        brace_index = tail.find("{")
        if brace_index != -1:
            parsed = _parse_json_object(tail[brace_index:])
            if parsed:
                return parsed

    if "## Action:" in content:
        tail = content.rsplit("## Action:", 1)[-1]
        brace_index = tail.find("{")
        if brace_index != -1:
            parsed = _parse_json_object(tail[brace_index:])
            if parsed:
                return parsed

    match = JSON_OBJECT_RE.search(content)
    if match:
        parsed = _parse_json_object(match.group(0))
        if parsed:
            return parsed
    return {}


def normalize_pred_coord(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        coord = float(value)
    except (TypeError, ValueError):
        return None
    if abs(coord) > 1.5:
        coord = coord / 1000.0
    return max(0.0, min(1.0, coord))


def normalize_gt_coord(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value) / 1000.0))
    except (TypeError, ValueError):
        return None


def extract_point(info: Any, *, pred: bool) -> Optional[Tuple[float, float]]:
    normalizer = normalize_pred_coord if pred else normalize_gt_coord
    if isinstance(info, list) and len(info) == 2 and all(isinstance(v, (int, float)) for v in info):
        x = normalizer(info[0])
        y = normalizer(info[1])
        return (x, y) if x is not None and y is not None else None
    if isinstance(info, list) and info and isinstance(info[0], list) and len(info[0]) >= 2:
        x = normalizer(info[0][0])
        y = normalizer(info[0][1])
        return (x, y) if x is not None and y is not None else None
    return None


def infer_scroll_direction(info: Any) -> Optional[str]:
    if not isinstance(info, list) or len(info) < 2:
        return None
    start, end = info[0], info[1]
    if not (
        isinstance(start, list)
        and isinstance(end, list)
        and len(start) >= 2
        and len(end) >= 2
        and all(isinstance(v, (int, float)) for v in start[:2] + end[:2])
    ):
        return None
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    if abs(delta_x) > abs(delta_y):
        return "left" if delta_x < 0 else "right"
    return "up" if delta_y < 0 else "down"


def canonicalize_action_type(action_type: Any) -> Optional[str]:
    if action_type is None:
        return None
    normalized = str(action_type).strip().lower()
    mapped = ACTION_ALIASES.get(normalized)
    if mapped is not None:
        return mapped
    return str(action_type).strip().upper()


def canonicalize_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_action = str(record.get("raw_action") or "").strip().upper()
    info = record.get("raw_info")

    if raw_action == "CLICK":
        if info == "KEY_HOME":
            return {"action_type": "CLICK", "key": "KEY_HOME"}
        if info == "KEY_BACK":
            return {"action_type": "CLICK", "key": "KEY_BACK"}
        if info == "KEY_APPSELECT":
            return {"action_type": "CLICK", "key": "KEY_APPSELECT"}
        coord = extract_point(info, pred=False)
        return {"action_type": "CLICK", "coord": coord}

    if raw_action == "LONG_PRESS":
        return {"action_type": "LONG_PRESS", "coord": extract_point(info, pred=False)}

    if raw_action in {"TEXT", "TYPE"}:
        return {"action_type": "TYPE", "text": str(info or "")}

    if raw_action == "SCROLL":
        direction = infer_scroll_direction(info)
        return {"action_type": "SCROLL", "direction": direction.upper() if direction else None}

    if raw_action == "COMPLETE":
        return {"action_type": "COMPLETE"}

    if raw_action == "INCOMPLETE":
        return {"action_type": "INCOMPLETE"}

    return {"action_type": None}


def canonicalize_json_prediction(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = canonicalize_action_type(action.get("action_type"))
    result: Dict[str, Any] = {"action_type": action_type}

    if action_type == "CLICK":
        key_text = action.get("text")
        if key_text is None:
            key_text = action.get("app_name")
        if isinstance(key_text, str):
            key_text = key_text.strip().upper()
            if key_text in {"KEY_HOME", "KEY_BACK", "KEY_APPSELECT"}:
                result["key"] = key_text
                return result
        x = normalize_pred_coord(action.get("x"))
        y = normalize_pred_coord(action.get("y"))
        result["coord"] = (x, y) if x is not None and y is not None else None
        return result

    if action_type == "LONG_PRESS":
        x = normalize_pred_coord(action.get("x"))
        y = normalize_pred_coord(action.get("y"))
        result["coord"] = (x, y) if x is not None and y is not None else None
        return result

    if action_type == "SCROLL":
        direction = action.get("direction")
        result["direction"] = str(direction).strip().upper() if direction is not None else None
        return result

    if action_type == "TYPE":
        text = action.get("text")
        if text is None:
            text = action.get("content")
        result["text"] = str(text or "")
        return result

    legacy_action = str(action.get("action_type") or "").strip().lower()
    if legacy_action in {"navigate_back", "press_back", "back"}:
        return {"action_type": "CLICK", "key": "KEY_BACK"}
    if legacy_action in {"press_home", "home", "navigate_home"}:
        return {"action_type": "CLICK", "key": "KEY_HOME"}
    if legacy_action in {"press_recent", "press_recents", "recent", "open_recents"}:
        return {"action_type": "CLICK", "key": "KEY_APPSELECT"}

    return result


def parse_command_action(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}

    for line in reversed([line.strip().strip("`") for line in content.splitlines() if line.strip()]):
        coord_match = COMMAND_COORD_RE.search(line)
        if coord_match:
            action_type = canonicalize_action_type(coord_match.group(1))
            x = normalize_pred_coord(coord_match.group(2))
            y = normalize_pred_coord(coord_match.group(3))
            return {
                "action_type": action_type,
                "coord": (x, y) if x is not None and y is not None else None,
            }

        scroll_match = COMMAND_SCROLL_RE.search(line)
        if scroll_match:
            return {
                "action_type": "SCROLL",
                "direction": scroll_match.group(1).strip().upper(),
            }

        text_match = COMMAND_TEXT_RE.search(line)
        if text_match:
            return {
                "action_type": "TYPE",
                "text": text_match.group(2).strip(),
            }

        upper_line = line.upper()
        if "PRESS_BACK" in upper_line:
            return {"action_type": "CLICK", "key": "KEY_BACK"}
        if "PRESS_HOME" in upper_line:
            return {"action_type": "CLICK", "key": "KEY_HOME"}
        if "PRESS_RECENT" in upper_line or "PRESS_RECENTS" in upper_line:
            return {"action_type": "CLICK", "key": "KEY_APPSELECT"}
        if re.search(r"\bCOMPLETE\b", upper_line):
            return {"action_type": "COMPLETE"}
        if re.search(r"\b(INCOMPLETE|IMPOSSIBLE)\b", upper_line):
            return {"action_type": "INCOMPLETE"}

    return {}


def canonicalize_prediction(text: str) -> Dict[str, Any]:
    action = extract_action_json(text)
    if action:
        return canonicalize_json_prediction(action)
    return parse_command_action(text)


def is_click_match(
    gt_coord: Optional[Tuple[float, float]],
    pred_coord: Optional[Tuple[float, float]],
    sam2_bbox: Any,
) -> bool:
    if gt_coord is None or pred_coord is None:
        return False

    if isinstance(sam2_bbox, list) and len(sam2_bbox) == 4:
        bbox = [normalize_gt_coord(value) for value in sam2_bbox]
        if all(value is not None for value in bbox):
            x1, y1, x2, y2 = bbox
            if x1 <= pred_coord[0] <= x2 and y1 <= pred_coord[1] <= y2:
                return True

    distance = math.dist(pred_coord, gt_coord)
    return distance <= CLICK_COORD_THRESHOLD


def evaluate_match(gt: Dict[str, Any], pred: Dict[str, Any], sam2_bbox: Any) -> Tuple[bool, str]:
    gt_type = gt.get("action_type")
    pred_type = pred.get("action_type")
    if gt_type != pred_type:
        return False, "action_fail"

    if gt_type == "TYPE":
        score = calculate_anls(str(pred.get("text") or ""), str(gt.get("text") or ""))
        return score >= TEXT_ANLS_THRESHOLD, "type_correct" if score >= TEXT_ANLS_THRESHOLD else "type_fail"

    if gt_type == "SCROLL":
        matched = str(pred.get("direction") or "").upper() == str(gt.get("direction") or "").upper()
        return matched, "scroll_correct" if matched else "scroll_fail"

    if gt_type == "CLICK" and gt.get("key") is not None:
        matched = str(pred.get("key") or "").upper() == str(gt.get("key") or "").upper()
        return matched, "click_correct" if matched else "click_fail"

    if gt_type in {"CLICK", "LONG_PRESS"}:
        matched = is_click_match(gt.get("coord"), pred.get("coord"), sam2_bbox)
        return matched, "click_correct" if matched else "click_fail"

    if gt_type in {"COMPLETE", "INCOMPLETE"}:
        return True, "action_correct"

    return False, "invalid"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_dataset_map(path: str) -> Dict[str, Dict[str, Any]]:
    dataset_map: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(path):
        sample_id = record.get("id")
        if sample_id is None:
            raise ValueError(f"Dataset entry missing 'id': {record}")
        dataset_map[str(sample_id)] = record
    return dataset_map


def infer_metric(records: List[Dict[str, Any]], override: str) -> str:
    if override != "auto":
        return override
    if records and isinstance(records[0].get("metric"), str):
        return records[0]["metric"]
    if records and "random_split" in str(records[0].get("split_name") or ""):
        return "micro"
    return "macro"


def compute_macro_accuracy(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for record in records if record["is_correct"])
    return correct / len(records) * 100.0


def compute_episode_success(records: List[Dict[str, Any]]) -> Tuple[float, int]:
    episodes: Dict[str, List[bool]] = defaultdict(list)
    step_lengths: Dict[str, int] = {}
    for record in records:
        episode_id = str(record["episode_id"])
        episodes[episode_id].append(bool(record["is_correct"]))
        step_lengths[episode_id] = int(record.get("step_length") or 0)
    if not episodes:
        return 0.0, 0

    success = 0
    total = 0
    for episode_id, results in episodes.items():
        if len(results) != step_lengths.get(episode_id, 0):
            continue
        total += 1
        if results and all(results):
            success += 1

    if total == 0:
        return 0.0, 0
    return success / total * 100.0, total


def compute_micro_metrics(records: List[Dict[str, Any]]) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    category_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        category_groups[str(record["category"])].append(record)

    per_category: Dict[str, Dict[str, float]] = {}
    if not category_groups:
        return 0.0, 0.0, per_category

    step_scores: List[float] = []
    episode_scores: List[float] = []
    for category, items in sorted(category_groups.items()):
        ams = compute_macro_accuracy(items)
        sr, total_episodes = compute_episode_success(items)
        per_category[category] = {
            "ams": ams,
            "sr": sr,
            "num_steps": len(items),
            "num_episodes": total_episodes,
        }
        step_scores.append(ams)
        episode_scores.append(sr)

    return sum(step_scores) / len(step_scores), sum(episode_scores) / len(episode_scores), per_category


def score_predictions(
    predictions_path: str,
    dataset_jsonl: str,
    output_dir: str,
    metric_override: str,
) -> None:
    dataset_map = load_dataset_map(dataset_jsonl)
    dataset_records = list(dataset_map.values())
    metric = infer_metric(dataset_records, metric_override)

    detailed: List[Dict[str, Any]] = []
    type_total = 0
    type_hit = 0
    text_total = 0
    text_hit = 0

    for record in load_jsonl(predictions_path):
        sample_id = str(record.get("id"))
        dataset_record = dataset_map.get(sample_id)
        if dataset_record is None:
            logger.warning("Prediction id %s not found in dataset JSONL. Skipping.", sample_id)
            continue

        gt = canonicalize_ground_truth(dataset_record)
        pred = canonicalize_prediction(record.get("prediction", ""))

        gt_type = gt.get("action_type")
        pred_type = pred.get("action_type")
        type_total += 1
        if gt_type == pred_type:
            type_hit += 1

        if gt_type == "TYPE":
            text_total += 1
            if calculate_anls(str(pred.get("text") or ""), str(gt.get("text") or "")) >= TEXT_ANLS_THRESHOLD:
                text_hit += 1

        is_correct, info = evaluate_match(gt, pred, dataset_record.get("sam2_bbox"))
        detailed.append(
            {
                "id": sample_id,
                "episode_id": dataset_record.get("episode_id"),
                "category": dataset_record.get("category"),
                "step_length": dataset_record.get("step_length"),
                "question": next(
                    (
                        conv.get("value")
                        for conv in reversed(dataset_record.get("conversations", []))
                        if conv.get("from") == "human"
                    ),
                    "",
                ),
                "target": record.get("target", ""),
                "prediction": record.get("prediction", ""),
                "gt_action": gt,
                "pred_action": pred,
                "is_correct": is_correct,
                "match_info": info,
            }
        )

    if metric == "micro":
        ams, sr, per_category = compute_micro_metrics(detailed)
        total_episodes = sum(info["num_episodes"] for info in per_category.values())
    else:
        ams = compute_macro_accuracy(detailed)
        sr, total_episodes = compute_episode_success(detailed)
        per_category = {}

    official_action_type_hit = sum(1 for item in detailed if item["match_info"] != "action_fail")
    official_text_correct = sum(1 for item in detailed if item["match_info"] == "type_correct")
    official_text_total = sum(1 for item in detailed if str(item["match_info"]).startswith("type_"))

    metrics = {
        "metric": metric,
        "ams": round(ams, 2),
        "sr": round(sr, 2),
        "action_type_accuracy": round(type_hit / type_total * 100.0, 2) if type_total else 0.0,
        "text_accuracy": round(text_hit / text_total * 100.0, 2) if text_total else 0.0,
        "total_steps": len(detailed),
        "total_episodes": total_episodes,
        "per_category": per_category,
        "official_metrics": {
            "AMS": round(ams, 2),
            "SR": round(sr, 2),
            "total": len(detailed),
            "action_type": (
                f"{official_action_type_hit} / {len(detailed)} = "
                f"{(official_action_type_hit / len(detailed) * 100.0) if detailed else 0.0:.2f}"
            ),
            "text": (
                f"{official_text_correct} / {official_text_total} = "
                f"{(official_text_correct / official_text_total * 100.0) if official_text_total else 0.0:.2f}"
            ),
        },
        "internal_metrics": {
            "exact_action_type_accuracy": round(type_hit / type_total * 100.0, 2) if type_total else 0.0,
            "exact_text_accuracy": round(text_hit / text_total * 100.0, 2) if text_total else 0.0,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    details_path = os.path.join(output_dir, "detailed_predictions.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved detailed predictions to %s", details_path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "score.log"), mode="a+")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    score_predictions(args.predictions, args.dataset_jsonl, args.output_dir, args.metric)


if __name__ == "__main__":
    main()
