# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple

from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACTION_BLOCK_RE = re.compile(r"```action\\s*(\\{.*?\\})\\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\\{.*\\}", re.DOTALL)

GROUNDING_ACTIONS = {"click", "long_press", "moveto", "doubleclick", "rightclick"}
TEXT_ACTIONS = {"open_app", "input_text", "scroll", "select"}
ACTION_ALIASES = {
    "press_back": "navigate_back",
    "type": "input_text",
    "longpress": "long_press",
    "double_click": "doubleclick",
    "right_click": "rightclick",
}


def calculate_f1_score(predicted_str: str, ground_truth_str: str) -> float:
    predicted_str = predicted_str.replace("[", "").replace("]", "")
    ground_truth_str = ground_truth_str.replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1.0

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    precision = len(common_tokens) / len(predicted_tokens) if predicted_tokens else 0.0
    recall = len(common_tokens) / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _parse_json_object(raw: str) -> Dict[str, object]:
    decoder = json.JSONDecoder()
    raw = raw.strip()
    if not raw:
        return {}
    try:
        obj, _ = decoder.raw_decode(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_action_json(text: str) -> Dict[str, object]:
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


def extract_text_field(action: Dict[str, object]) -> str:
    action_type = normalize_action_type(action.get("action_type"))
    if action_type == "open_app":
        return action.get("app_name") or ""
    if action_type == "input_text":
        return action.get("text") or ""
    if action_type == "scroll":
        return action.get("direction") or ""
    return ""


def extract_coord(action: Dict[str, object]) -> Optional[Tuple[float, float]]:
    x = action.get("x")
    y = action.get("y")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def normalize_action_type(action_type: Optional[str]) -> Optional[str]:
    if action_type is None:
        return None
    action_type = action_type.strip().lower()
    return ACTION_ALIASES.get(action_type, action_type)


def load_image_sizes(jsonl_path: str, image_root: str) -> Dict[str, Tuple[int, int]]:
    image_sizes: Dict[str, Tuple[int, int]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image = record.get("image")
            if isinstance(image, list):
                image = image[0] if image else None
            if not image:
                continue
            image_path = os.path.join(image_root, image)
            if record.get("id") in image_sizes:
                continue
            with Image.open(image_path) as img:
                image_sizes[record.get("id")] = img.size
    return image_sizes


def score_predictions(
    predictions_path: str,
    dataset_jsonl: str,
    image_root: str,
    output_dir: str,
) -> None:
    image_sizes = load_image_sizes(dataset_jsonl, image_root)

    score_dict = defaultdict(int)
    type_total = 0
    type_hit = 0
    step_total = 0
    step_hit = 0
    grounding_total = 0
    grounding_hit = 0

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if not record_id:
                continue

            pred_action = extract_action_json(record.get("prediction", ""))
            gt_action = extract_action_json(record.get("target", ""))

            gt_type = normalize_action_type(gt_action.get("action_type"))
            pred_type = normalize_action_type(pred_action.get("action_type"))

            category = f"android_control-{gt_type}"
            score_dict[f"{category}_full"] += 1
            type_total += 1
            if gt_type == pred_type:
                score_dict[category] += 1
                type_hit += 1

            if gt_type in GROUNDING_ACTIONS:
                grounding_total += 1
                score_dict[f"{category}-grounding_full"] += 1
                gt_coord = extract_coord(gt_action)
                pred_coord = extract_coord(pred_action)
                image_size = image_sizes.get(record_id)
                if gt_coord and pred_coord and image_size:
                    width, height = image_size
                    gt_x, gt_y = gt_coord[0] * width, gt_coord[1] * height
                    pred_x, pred_y = pred_coord[0] * width, pred_coord[1] * height
                    dist = ((gt_x - pred_x) / width) ** 2 + ((gt_y - pred_y) / height) ** 2
                    if dist < 0.14 ** 2:
                        score_dict[f"{category}-grounding"] += 1
                        grounding_hit += 1
                        step_hit += 1
                step_total += 1

            if gt_type in TEXT_ACTIONS:
                step_total += 1
                score_dict[f"{category}-text_full"] += 1
                gt_text = extract_text_field(gt_action)
                pred_text = extract_text_field(pred_action)
                if calculate_f1_score(gt_text, pred_text) >= 0.5:
                    score_dict[f"{category}-text"] += 1
                    step_hit += 1

    for key in [k for k in score_dict.keys() if not k.endswith("full")]:
        full_key = f"{key}_full"
        if score_dict[full_key] > 0:
            logger.info(f"{key} Length {score_dict[full_key]} : {score_dict[key] / score_dict[full_key]}")

    metrics = {
        "type_accuracy": type_hit / type_total if type_total else 0.0,
        "step_accuracy": step_hit / step_total if step_total else 0.0,
        "grounding_accuracy": grounding_hit / grounding_total if grounding_total else 0.0,
        "total": {
            "type": type_total,
            "step": step_total,
            "grounding": grounding_total,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Android Control predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Raw prediction JSONL path.")
    parser.add_argument("--dataset-jsonl", type=str, required=True, help="Dataset JSONL path.")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "score.log"), mode="a+")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    score_predictions(args.predictions, args.dataset_jsonl, args.image_root, args.output_dir)


if __name__ == "__main__":
    main()
