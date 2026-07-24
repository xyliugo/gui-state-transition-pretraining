from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AGENT_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of actions to complete the task.\n\n"
    "For each step, provide your response in this format:\n\n"
    "Action:\n"
    "  First provide the step instruction, then the action JSON object wrapped in a code block:\n"
    "  ```action\n"
    "  {action_str}\n"
    "  ```\n"
    "  The JSON object must include:\n"
    "  - action_type (string)\n"
    "  - app_name (string or null)\n"
    "  - direction (string or null)\n"
    "  - text (string or null)\n"
    "  - x (number in [0,1] or null)\n"
    "  - y (number in [0,1] or null)\n"
    "\nCoordinates x/y are normalized to [0,1] from GUI-Odyssey's original [0,1000] "
    "annotation space. Use null for fields that do not apply.\n"
    "\nUse the native GUI-Odyssey action space for action_type: CLICK, LONG_PRESS, "
    "SCROLL, TYPE, COMPLETE, INCOMPLETE.\n"
    "For CLICK on special system keys, keep action_type as CLICK and put one of "
    "KEY_HOME, KEY_BACK, KEY_APPSELECT in text. For normal CLICK or LONG_PRESS, "
    "put the normalized coordinate in x/y. For SCROLL, put the direction in "
    "direction using UP, DOWN, LEFT or RIGHT. For TYPE, put the typed string in text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GUI-Odyssey annotations into Bagel-style agent JSONL."
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Directory containing GUI-Odyssey episode annotation JSON files.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        required=True,
        help="Directory containing GUI-Odyssey screenshots.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split JSON file from GUI-Odyssey, e.g. random_split.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/gui_odyssey/bagel_format"),
        help="Directory to store the converted JSONL file.",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "test"),
        default="test",
        help="Which split partition to convert.",
    )
    parser.add_argument(
        "--instruction-level",
        choices=("high", "low"),
        default="high",
        help="Whether the task instruction uses high-level or low-level text.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=4,
        help="Maximum number of previous assistant turns to include. Use a negative value for full history.",
    )
    return parser.parse_args()


def load_split_items(split_file: Path, partition: str) -> List[str]:
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    items = split_data.get(partition)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Split file {split_file} does not contain a non-empty '{partition}' list.")
    return [str(item) for item in items]


def infer_metric(split_name: str) -> str:
    return "micro" if "random_split" in split_name else "macro"


def ensure_relative_screenshot(path_value: str, screenshots_dir: Path) -> str:
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return str(raw_path.relative_to(screenshots_dir))
    return str(raw_path)


def normalize_coord(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        coord = float(value) / 1000.0
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, coord))


def extract_point(info: Any) -> Optional[List[float]]:
    if isinstance(info, list) and len(info) == 2 and all(isinstance(v, (int, float)) for v in info):
        return [float(info[0]), float(info[1])]
    if isinstance(info, list) and info and isinstance(info[0], list) and len(info[0]) >= 2:
        point = info[0]
        if all(isinstance(v, (int, float)) for v in point[:2]):
            return [float(point[0]), float(point[1])]
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


def convert_action(action: str, info: Any) -> Dict[str, Any]:
    normalized = {
        "action_type": None,
        "app_name": None,
        "direction": None,
        "text": None,
        "x": None,
        "y": None,
    }
    action = str(action or "").strip().upper()

    if action == "CLICK":
        if info == "KEY_HOME":
            normalized["action_type"] = "CLICK"
            normalized["text"] = "KEY_HOME"
            return normalized
        if info == "KEY_BACK":
            normalized["action_type"] = "CLICK"
            normalized["text"] = "KEY_BACK"
            return normalized
        if info == "KEY_APPSELECT":
            normalized["action_type"] = "CLICK"
            normalized["text"] = "KEY_APPSELECT"
            return normalized
        point = extract_point(info)
        if point is None:
            raise ValueError(f"Unsupported CLICK info: {info}")
        normalized["action_type"] = "CLICK"
        normalized["x"] = normalize_coord(point[0])
        normalized["y"] = normalize_coord(point[1])
        return normalized

    if action == "LONG_PRESS":
        point = extract_point(info)
        if point is None:
            raise ValueError(f"Unsupported LONG_PRESS info: {info}")
        normalized["action_type"] = "LONG_PRESS"
        normalized["x"] = normalize_coord(point[0])
        normalized["y"] = normalize_coord(point[1])
        return normalized

    if action in {"TEXT", "TYPE"}:
        normalized["action_type"] = "TYPE"
        normalized["text"] = str(info or "")
        return normalized

    if action == "SCROLL":
        normalized["action_type"] = "SCROLL"
        direction = infer_scroll_direction(info)
        normalized["direction"] = direction.upper() if direction else None
        return normalized

    if action == "COMPLETE":
        normalized["action_type"] = "COMPLETE"
        return normalized

    if action == "INCOMPLETE":
        normalized["action_type"] = "INCOMPLETE"
        return normalized

    raise ValueError(f"Unsupported GUI-Odyssey action: {action}")


def format_coord(value: Optional[float]) -> str:
    if value is None:
        return "null"
    return f"{value:.3f}"


def format_action_json(action: Dict[str, Any]) -> str:
    return (
        "{"
        f"\"action_type\":{json.dumps(action['action_type'], ensure_ascii=False)},"
        f"\"app_name\":{json.dumps(action['app_name'], ensure_ascii=False)},"
        f"\"direction\":{json.dumps(action['direction'], ensure_ascii=False)},"
        f"\"text\":{json.dumps(action['text'], ensure_ascii=False)},"
        f"\"x\":{format_coord(action['x'])},"
        f"\"y\":{format_coord(action['y'])}"
        "}"
    )


def format_action_block(action_json: str) -> str:
    return f"```action\n{action_json}\n```"


def format_step_response(step_number: int, step_instruction: str, action_json: str) -> str:
    return (
        f"# Step {step_number}\n"
        "## Action:\n"
        f"{step_instruction}\n"
        f"{format_action_block(action_json)}"
    )


def build_human_instruction(
    task_instruction: str,
    instruction_level: str,
) -> str:
    if instruction_level == "high":
        header = "# Task Instruction:"
    else:
        header = "# Current Step Instruction:"
    return (
        "<image>\n"
        f"{header}\n{task_instruction}\n\n"
        "Please generate the next move according to the screenshot, "
        "task instruction and previous steps (if provided)."
    )


def trim_history(conversations: List[Dict[str, str]], history_steps: int) -> List[Dict[str, str]]:
    if history_steps < 0:
        return conversations
    if history_steps == 0:
        return []
    return conversations[-history_steps:]


def iter_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    split_items = load_split_items(args.split_file, args.partition)
    split_name = args.split_file.stem
    metric = infer_metric(split_name)

    for annotation_name in split_items:
        annotation_path = args.annotation_dir / annotation_name
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        episode = json.loads(annotation_path.read_text(encoding="utf-8"))
        episode_id = str(episode.get("episode_id") or annotation_path.stem)
        task_info = episode.get("task_info") or {}
        high_level_instruction = str(task_info.get("instruction") or "")
        category = str(task_info.get("category") or "")
        step_length = int(episode.get("step_length") or len(episode.get("steps") or []))
        steps = episode.get("steps") or []

        history_turns: List[Dict[str, str]] = []
        for step in steps:
            step_index = int(step.get("step", 0))
            screenshot_rel = ensure_relative_screenshot(str(step.get("screenshot") or ""), args.screenshots_dir)
            screenshot_path = args.screenshots_dir / screenshot_rel
            if not screenshot_path.is_file():
                raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

            low_level_instruction = str(step.get("low_level_instruction") or "")
            task_instruction = (
                high_level_instruction if args.instruction_level == "high" else low_level_instruction
            )
            action_struct = convert_action(step.get("action"), step.get("info"))
            action_json = format_action_json(action_struct)
            target_step_instruction = low_level_instruction or task_instruction
            prior_turns = trim_history(history_turns, args.history_steps)

            conversations: List[Dict[str, str]] = [{"from": "human", "value": AGENT_SYSTEM_PROMPT}]
            conversations.extend(prior_turns)
            conversations.append(
                {
                    "from": "human",
                    "value": build_human_instruction(task_instruction, args.instruction_level),
                }
            )
            conversations.append(
                {
                    "from": "gpt",
                    "value": format_step_response(step_index + 1, target_step_instruction, action_json),
                }
            )

            yield {
                "id": f"{episode_id}_{step_index + 1:03d}",
                "episode_id": episode_id,
                "step_index": step_index,
                "split_name": split_name,
                "partition": args.partition,
                "instruction_level": args.instruction_level,
                "metric": metric,
                "category": category,
                "step_length": step_length,
                "sam2_bbox": step.get("sam2_bbox") or [],
                "raw_action": step.get("action"),
                "raw_info": step.get("info"),
                "image": [screenshot_rel],
                "conversations": conversations,
            }

            history_turns.append(
                {
                    "from": "gpt",
                    "value": format_step_response(step_index + 1, target_step_instruction, action_json),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"agent_tasks__guiodyssey__{args.instruction_level}__{args.split_file.stem}__{args.partition}.jsonl"
    )

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in iter_records(args):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} GUI-Odyssey samples to {output_path}")


if __name__ == "__main__":
    main()
