from __future__ import annotations

import argparse
import base64
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0) or 0)


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
    "\nCoordinates x/y are normalized by the original screenshot width/height. "
    "Use null for fields that do not apply."
)

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
    "\nCoordinates x/y are normalized by the original screenshot width/height. "
    "Use null for fields that do not apply. </action_format>\n\n"
    "What action was performed?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert android-control parquet into AgentNet-style agent/inverse/forward datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/android_control/data"),
        help="Directory containing android-control parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/android_control/bagel_format_v2"),
        help="Output base directory for converted data.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/android_control/images_v2"),
        help="Image root directory for decoded screenshots.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used to build image paths.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=1000,
        help="Rows per parquet chunk for forward dataset.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100,
        help="Row group size for parquet chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of trajectories to convert.",
    )
    parser.add_argument(
        "--skip-image-write",
        action="store_true",
        help="Do not write decoded screenshots to disk.",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Only produce agent JSONL; skip inverse JSONL and forward parquet.",
    )
    return parser.parse_args()


def iter_parquet_rows(
    parquet_files: List[Path],
    limit: Optional[int],
) -> Iterable[Dict[str, object]]:
    count = 0
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(
            batch_size=64,
            columns=["episode_id", "goal", "screenshots_b64", "actions", "step_instructions"],
        ):
            for row in batch.to_pylist():
                yield row
                count += 1
                if limit is not None and count >= limit:
                    return


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def decode_screenshots(
    screenshots_b64: List[str],
    episode_id: int,
    split: str,
    image_dir: Path,
    write_images: bool,
) -> Tuple[List[str], List[Tuple[int, int]], List[bytes]]:
    image_paths: List[str] = []
    sizes: List[Tuple[int, int]] = []
    bytes_list: List[bytes] = []
    for idx, b64 in enumerate(screenshots_b64):
        data = base64.b64decode(b64)
        bytes_list.append(data)
        with Image.open(BytesIO(data)) as img:
            w, h = img.size
        sizes.append((w, h))
        rel_path = f"{split}/{episode_id}_{idx:04d}.png"
        image_paths.append(rel_path)
        if write_images:
            abs_path = image_dir / rel_path
            ensure_dir(abs_path.parent)
            abs_path.write_bytes(data)
    return image_paths, sizes, bytes_list


def _normalize_coord(value: Optional[int], max_value: int) -> Optional[float]:
    if value is None:
        return None
    if max_value <= 0:
        return None
    return value / max_value


def _format_coord(value: Optional[float]) -> str:
    if value is None:
        return "null"
    return f"{value:.3f}"


def format_action_json(
    action: Dict[str, object],
    screen_size: Tuple[int, int],
) -> str:
    w, h = screen_size
    action_type = action.get("action_type")
    app_name = action.get("app_name")
    direction = action.get("direction")
    text = action.get("text")
    x_norm = _normalize_coord(action.get("x"), w)
    y_norm = _normalize_coord(action.get("y"), h)
    parts = [
        f"\"action_type\":{json.dumps(action_type, ensure_ascii=False)}",
        f"\"app_name\":{json.dumps(app_name, ensure_ascii=False)}",
        f"\"direction\":{json.dumps(direction, ensure_ascii=False)}",
        f"\"text\":{json.dumps(text, ensure_ascii=False)}",
        f"\"x\":{_format_coord(x_norm)}",
        f"\"y\":{_format_coord(y_norm)}",
    ]
    return "{" + ",".join(parts) + "}"


def format_action_block(action_json: str) -> str:
    return f"```action\n{action_json}\n```"


def format_step_response(step_number: int, step_instruction: str, action_json: str) -> str:
    return (
        f"# Step {step_number}\n"
        "## Action:\n"
        f"{step_instruction}\n"
        f"{format_action_block(action_json)}"
    )


def build_agent_conversations(
    goal: str,
    step_index: int,
    image_paths: List[str],
    actions: List[Dict[str, object]],
    sizes: List[Tuple[int, int]],
    step_instructions: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    conversations: List[Dict[str, str]] = [{"from": "human", "value": AGENT_SYSTEM_PROMPT}]
    images: List[str] = []

    if step_index > 0:
        for k in range(step_index):
            action_json = format_action_json(actions[k], sizes[k])
            step_instruction = step_instructions[k] if k < len(step_instructions) else ""
            conversations.append(
                {"from": "gpt", "value": format_step_response(k + 1, step_instruction, action_json)}
            )

    current_img = image_paths[step_index]
    images.append(current_img)
    human_value = (
        "<image>\n"
        f"# Task Instruction:\n{goal}\n\n"
        "Please generate the next move according to the screenshot, "
        "task instruction and previous steps (if provided)."
    )
    conversations.append({"from": "human", "value": human_value})

    current_action_json = format_action_json(actions[step_index], sizes[step_index])
    conversations.append(
        {
            "from": "gpt",
            "value": format_step_response(
                step_index + 1,
                step_instructions[step_index] if step_index < len(step_instructions) else "",
                current_action_json,
            ),
        }
    )

    return images, conversations


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


def build_forward_instruction(step_instruction: str, action_json: str) -> str:
    return (
        "You are a helpful assistant that predicts the next screen state.\n"
        "Given the current screen, perform the move:\n"
        f"## Action:\n{step_instruction}\n{format_action_block(action_json)}\n\n"
        "What is the next screen?"
    )


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
    ensure_dir(out_dir)
    table = _rows_to_table(rows)
    chunk_path = (out_dir / f"chunk_{chunk_index}.parquet").resolve()
    pq.write_table(table, chunk_path, row_group_size=row_group_size)
    num_rows = len(rows)
    num_row_groups = math.ceil(num_rows / row_group_size) if row_group_size > 0 else 1
    return str(chunk_path), {"num_row_groups": int(num_row_groups), "num_rows": int(num_rows)}


def convert(args: argparse.Namespace) -> None:
    parquet_files = sorted(args.input_dir.glob(f"{args.split}-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {args.input_dir}")
    total_rows = None
    try:
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
        if args.limit is not None:
            total_rows = min(total_rows, args.limit)
    except Exception:
        total_rows = args.limit

    vlm_dir = args.output_dir / "vlm_data"
    ensure_dir(vlm_dir)
    agent_out = vlm_dir / f"agent_tasks__android_control__l1.jsonl"

    forward_rows: List[Tuple[List[bytes], List[List[str]]]] = []
    parquet_info: Dict[str, Dict[str, int]] = {}
    chunk_index = 0
    total_traj = 0
    total_steps = 0
    pbar = tqdm(total=total_rows, desc="Trajectories", unit="traj")

    if not args.agent_only:
        inverse_out = vlm_dir / f"inverse_tasks__android_control__l1.jsonl"
        forward_dir = args.output_dir / "forward_edit_android_control"
        parquet_info_path = args.output_dir / "parquet_info" / "forward_edit_android_control.json"
        ensure_dir(parquet_info_path.parent)

    def flush_forward(force: bool = False) -> None:
        nonlocal forward_rows, chunk_index, parquet_info
        if forward_rows and (force or len(forward_rows) >= args.rows_per_file):
            chunk_path, info = write_parquet_chunk(
                forward_rows, forward_dir, args.row_group_size, chunk_index
            )
            parquet_info[chunk_path] = info
            forward_rows = []
            chunk_index += 1

    inverse_f_ctx = inverse_out.open("w", encoding="utf-8") if not args.agent_only else None
    try:
        with agent_out.open("w", encoding="utf-8") as agent_f:
            for row in iter_parquet_rows(parquet_files, args.limit):
                episode_id = row.get("episode_id")
                goal = row.get("goal", "")
                screenshots = row.get("screenshots_b64", [])
                actions = row.get("actions", [])
                step_instructions = row.get("step_instructions", [])

                if not screenshots or not actions:
                    total_traj += 1
                    pbar.update(1)
                    continue

                image_paths, sizes, bytes_list = decode_screenshots(
                    screenshots,
                    episode_id,
                    args.split,
                    args.image_dir,
                    write_images=not args.skip_image_write,
                )

                max_steps = min(len(actions), len(image_paths) - 1, len(sizes) - 1)
                if max_steps <= 0:
                    total_traj += 1
                    pbar.update(1)
                    continue

                for step_index in range(max_steps):
                    images, conversations = build_agent_conversations(
                        goal, step_index, image_paths, actions, sizes, step_instructions
                    )
                    record = {
                        "id": f"{episode_id}_{step_index + 1:03d}",
                        "image": images,
                        "conversations": conversations,
                    }
                    agent_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if not args.agent_only:
                        action_json = format_action_json(actions[step_index], sizes[step_index])
                        step_instruction = (
                            step_instructions[step_index] if step_index < len(step_instructions) else ""
                        )
                        inverse_record = {
                            "id": f"{episode_id}_{step_index + 1:03d}",
                            "image": [image_paths[step_index], image_paths[step_index + 1]],
                            "conversations": build_inverse_conversations(step_instruction, action_json),
                        }
                        inverse_f_ctx.write(json.dumps(inverse_record, ensure_ascii=False) + "\n")

                        instruction = build_forward_instruction(step_instruction, action_json)
                        forward_rows.append(
                            ([bytes_list[step_index], bytes_list[step_index + 1]], [[instruction]])
                        )
                        flush_forward()
                    total_steps += 1

                total_traj += 1
                pbar.update(1)
            pbar.close()
    finally:
        if inverse_f_ctx is not None:
            inverse_f_ctx.close()

    if not args.agent_only:
        flush_forward(force=True)
        parquet_info_path.write_text(
            json.dumps(parquet_info, ensure_ascii=False, indent=4), encoding="utf-8"
        )

    print(f"Converted {total_steps} step-level tasks from {total_traj} trajectories.")
    print(f"  Agent: {agent_out}")
    if not args.agent_only:
        print(f"  Inverse: {inverse_out}")
        print(
            f"  Forward: {sum(info['num_rows'] for info in parquet_info.values())} rows "
            f"written to {forward_dir}."
        )


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
