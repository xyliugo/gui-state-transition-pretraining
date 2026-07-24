"""Convert GUI-Odyssey annotations into Bagel-format training data.

Produces three output formats:
  1. Agent  JSONL  – multi-turn conversations (matches test format exactly)
  2. Inverse JSONL – two consecutive screenshots → infer action
  3. Forward Parquet – two consecutive screenshot bytes → predict next screen
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0) or 0)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Agent prompt – identical to convert_guiodyssey_test.py
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

# Inverse prompt – adapted from android_control pattern for GUI-Odyssey action space
INVERSE_SYSTEM_PROMPT = (
    "You are a helpful assistant that infers actions from screen transitions.\n\n"
    "Previous screen:\n<image>\n\n"
    "Current screen:\n<image>\n\n"
    "Action format:\n"
    "<action_format> Action:\n"
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
    "direction using UP, DOWN, LEFT or RIGHT. For TYPE, put the typed string in text. "
    "</action_format>\n\n"
    "What action was performed?"
)


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GUI-Odyssey annotations into Bagel-style training data (agent + inverse + forward)."
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
        help="Directory to store the converted files.",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "test"),
        default="train",
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
        help="Maximum number of previous assistant turns to include. Negative for full history.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=1000,
        help="Rows per parquet chunk for the forward dataset.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100,
        help="Row group size for forward parquet chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to convert.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers shared with the test converter
# ---------------------------------------------------------------------------

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
    normalized: Dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent format builders (identical to test converter)
# ---------------------------------------------------------------------------

def build_human_instruction(task_instruction: str, instruction_level: str) -> str:
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


# ---------------------------------------------------------------------------
# Inverse format builder
# ---------------------------------------------------------------------------

def build_inverse_conversations(
    step_instruction: str,
    action_json: str,
) -> List[Dict[str, str]]:
    return [
        {"from": "human", "value": INVERSE_SYSTEM_PROMPT},
        {
            "from": "gpt",
            "value": f"## Action:\n{step_instruction}\n{format_action_block(action_json)}",
        },
    ]


# ---------------------------------------------------------------------------
# Forward format builder
# ---------------------------------------------------------------------------

def build_forward_instruction(step_instruction: str, action_json: str) -> str:
    return (
        "You are a helpful assistant that predicts the next screen state.\n"
        "Given the current screen, perform the move:\n"
        f"## Action:\n{step_instruction}\n{format_action_block(action_json)}\n\n"
        "What is the next screen?"
    )


# ---------------------------------------------------------------------------
# Parquet helpers (forward dataset)
# ---------------------------------------------------------------------------

def _rows_to_table(
    rows: List[Tuple[List[bytes], List[List[str]]]],
) -> pa.Table:
    image_list_type = pa.list_(pa.binary())
    instruction_list_type = pa.list_(pa.list_(pa.string()))
    image_list_arr = pa.array([r[0] for r in rows], type=image_list_type)
    instruction_list_arr = pa.array([r[1] for r in rows], type=instruction_list_type)
    return pa.Table.from_arrays(
        [image_list_arr, instruction_list_arr], names=["image_list", "instruction_list"]
    )


def write_parquet_chunk(
    rows: List[Tuple[List[bytes], List[List[str]]]],
    out_dir: Path,
    row_group_size: int,
    chunk_index: int,
) -> Tuple[str, Dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows)
    chunk_path = (out_dir / f"chunk_{chunk_index}.parquet").resolve()
    pq.write_table(table, chunk_path, row_group_size=row_group_size)
    num_rows = len(rows)
    num_row_groups = math.ceil(num_rows / row_group_size) if row_group_size > 0 else 1
    return str(chunk_path), {"num_row_groups": int(num_row_groups), "num_rows": int(num_rows)}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args: argparse.Namespace) -> None:
    split_items = load_split_items(args.split_file, args.partition)
    split_name = args.split_file.stem
    metric = infer_metric(split_name)

    if args.limit is not None:
        split_items = split_items[: args.limit]

    # Output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.instruction_level}__{split_name}__{args.partition}"
    agent_out = args.output_dir / f"agent_tasks__guiodyssey__{suffix}.jsonl"
    inverse_out = args.output_dir / f"inverse_tasks__guiodyssey__{suffix}.jsonl"
    forward_dir = args.output_dir / "forward_edit_guiodyssey"
    parquet_info_path = args.output_dir / "parquet_info" / "forward_edit_guiodyssey.json"
    parquet_info_path.parent.mkdir(parents=True, exist_ok=True)

    forward_rows: List[Tuple[List[bytes], List[List[str]]]] = []
    parquet_info: Dict[str, Dict[str, int]] = {}
    chunk_index = 0
    total_episodes = 0
    total_agent_steps = 0
    total_inverse_steps = 0
    total_forward_steps = 0

    def flush_forward(force: bool = False) -> None:
        nonlocal forward_rows, chunk_index, parquet_info
        if forward_rows and (force or len(forward_rows) >= args.rows_per_file):
            chunk_path, info = write_parquet_chunk(
                forward_rows, forward_dir, args.row_group_size, chunk_index
            )
            parquet_info[chunk_path] = info
            forward_rows = []
            chunk_index += 1

    with agent_out.open("w", encoding="utf-8") as agent_f, \
         inverse_out.open("w", encoding="utf-8") as inverse_f:

        for annotation_name in tqdm(split_items, desc="Episodes", unit="ep"):
            annotation_path = args.annotation_dir / annotation_name
            if not annotation_path.is_file():
                print(f"[Warning] Annotation file not found, skipping: {annotation_path}")
                continue

            episode = json.loads(annotation_path.read_text(encoding="utf-8"))
            episode_id = str(episode.get("episode_id") or annotation_path.stem)
            task_info = episode.get("task_info") or {}
            high_level_instruction = str(task_info.get("instruction") or "")
            category = str(task_info.get("category") or "")
            step_length = int(episode.get("step_length") or len(episode.get("steps") or []))
            steps = episode.get("steps") or []

            if not steps:
                total_episodes += 1
                continue

            # ----------------------------------------------------------
            # Pre-read all screenshot bytes for this episode once.
            # This avoids repeated reads on CephFS: without caching,
            # screenshot i is read as "next" in step i-1 and as "cur"
            # in step i – doubling the IO.  We also batch the is_file()
            # stat calls into the same up-front pass.
            # ----------------------------------------------------------
            screenshot_rels: List[str] = []
            screenshot_bytes_cache: Dict[str, Optional[bytes]] = {}
            screenshot_exists: Dict[str, bool] = {}

            for step in steps:
                rel = ensure_relative_screenshot(
                    str(step.get("screenshot") or ""), args.screenshots_dir
                )
                screenshot_rels.append(rel)
                if rel not in screenshot_exists:
                    p = args.screenshots_dir / rel
                    exists = p.is_file()
                    screenshot_exists[rel] = exists
                    # Only read bytes if the file exists; defer nothing.
                    if exists:
                        try:
                            screenshot_bytes_cache[rel] = p.read_bytes()
                        except Exception:
                            screenshot_bytes_cache[rel] = None
                    else:
                        screenshot_bytes_cache[rel] = None

            history_turns: List[Dict[str, str]] = []

            for i, step in enumerate(steps):
                step_index = int(step.get("step", 0))
                screenshot_rel = screenshot_rels[i]
                if not screenshot_exists.get(screenshot_rel, False):
                    print(f"[Warning] Screenshot not found, skipping step: "
                          f"{args.screenshots_dir / screenshot_rel}")
                    continue

                low_level_instruction = str(step.get("low_level_instruction") or "")
                task_instruction = (
                    high_level_instruction if args.instruction_level == "high" else low_level_instruction
                )

                try:
                    action_struct = convert_action(step.get("action"), step.get("info"))
                except ValueError as e:
                    print(f"[Warning] {e} in episode {episode_id} step {step_index}, skipping step.")
                    continue

                action_json = format_action_json(action_struct)
                target_step_instruction = low_level_instruction or task_instruction

                # ----------------------------------------------------------
                # 1. Agent record (same logic as test converter)
                # ----------------------------------------------------------
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

                agent_record = {
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
                agent_f.write(json.dumps(agent_record, ensure_ascii=False) + "\n")
                total_agent_steps += 1

                # Update history for future steps
                history_turns.append(
                    {
                        "from": "gpt",
                        "value": format_step_response(step_index + 1, target_step_instruction, action_json),
                    }
                )

                # ----------------------------------------------------------
                # 2. Inverse record (needs next screenshot)
                # ----------------------------------------------------------
                if i + 1 < len(steps):
                    next_screenshot_rel = screenshot_rels[i + 1]
                    if screenshot_exists.get(next_screenshot_rel, False):
                        inverse_record = {
                            "id": f"{episode_id}_{step_index + 1:03d}",
                            "image": [screenshot_rel, next_screenshot_rel],
                            "conversations": build_inverse_conversations(
                                target_step_instruction, action_json
                            ),
                        }
                        inverse_f.write(json.dumps(inverse_record, ensure_ascii=False) + "\n")
                        total_inverse_steps += 1

                        # ------------------------------------------------------
                        # 3. Forward record (use pre-cached bytes)
                        # ------------------------------------------------------
                        cur_bytes = screenshot_bytes_cache.get(screenshot_rel)
                        next_bytes = screenshot_bytes_cache.get(next_screenshot_rel)
                        if cur_bytes is not None and next_bytes is not None:
                            instruction = build_forward_instruction(
                                target_step_instruction, action_json
                            )
                            forward_rows.append(
                                ([cur_bytes, next_bytes], [[instruction]])
                            )
                            flush_forward()
                            total_forward_steps += 1

            # Free memory for this episode's screenshot cache
            del screenshot_bytes_cache
            total_episodes += 1

    # Flush remaining forward rows
    flush_forward(force=True)
    parquet_info_path.write_text(
        json.dumps(parquet_info, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    print(f"\nConversion complete for {total_episodes} episodes:")
    print(f"  Agent:   {total_agent_steps} steps -> {agent_out}")
    print(f"  Inverse: {total_inverse_steps} steps -> {inverse_out}")
    print(
        f"  Forward: {total_forward_steps} steps -> {forward_dir} "
        f"({sum(info['num_rows'] for info in parquet_info.values())} rows in "
        f"{len(parquet_info)} chunks)"
    )


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
