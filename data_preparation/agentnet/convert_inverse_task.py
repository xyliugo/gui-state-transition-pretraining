from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


action_description = [
    '<action_format> Action:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}} </action_format> '
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AgentNet trajectories into (s_t, s_{t+1}) -> a_t action inference JSONL (bagel-compatible)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/agentnet/agentnet_win_mac_18k.jsonl"),
        help="Path to the raw AgentNet trajectory JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/agentnet/bagel_format/vlm_data/inverse_tasks__winmac__l1.jsonl"),
        help="Destination JSONL file for converted action-inference samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of (s_t, s_{t+1}) -> a_t samples to write.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def format_action_label(action: Optional[str], code: Optional[str]) -> str:
    parts: List[str] = []
    if action:
        parts.append(f"## Action:\n{action}")
    if code:
        parts.append(f"## Code:\n```python\n{code}\n```")
    return "\n".join(parts) if parts else ""


def build_conversations_with_two_images(instruction_prefix: str) -> str:
    # Embed two <image> placeholders into the human prompt
    return (
        instruction_prefix
        + "\n\nPrevious screen:\n<image>\n\n"
        + "Current screen:\n<image>\n\n"
        + f"Action format:\n{action_description[0]}\n\n"
        + "What action was performed?"
    )


def convert(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    action_system = "You are a helpful assistant that infers actions from screen transitions."

    written = 0
    with args.output.open("w", encoding="utf-8") as out_file:
        for sample in iter_jsonl(args.input):
            task_id = sample.get("task_id")
            traj = sample.get("traj", [])
            if not isinstance(traj, list) or len(traj) < 2:
                continue

            # For each adjacent pair (t, t+1), build one sample with two images and label a_t
            for t in range(len(traj) - 1):
                step_t = traj[t] if isinstance(traj[t], dict) else {}
                step_tp1 = traj[t + 1] if isinstance(traj[t + 1], dict) else {}

                img_t = step_t.get("image")
                img_tp1 = step_tp1.get("image")
                value_t = step_t.get("value", {}) if isinstance(step_t, dict) else {}
                action = value_t.get("action")
                code = value_t.get("code")

                if not img_t or not img_tp1:
                    continue

                conversations = [
                    {
                        "from": "human",
                        "value": build_conversations_with_two_images(action_system),
                    },
                ]

                label = format_action_label(action, code)
                # Skip if we don't have a label at all
                if not label:
                    continue

                conversations.append({"from": "gpt", "value": label})

                record = {
                    "id": f"{task_id}_{t + 1:03d}",
                    "image": [img_t, img_tp1],
                    "conversations": conversations,
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if args.limit is not None and written >= args.limit:
                    break

            if args.limit is not None and written >= args.limit:
                break

    print(f"Wrote {written} action-inference samples.")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    convert(args)


if __name__ == "__main__":
    main()

