from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

L2_SHORT_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n\nAction:\n  Provide clear, concise, and actionable instructions.\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

L1_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    default_input = "data/agentnet/agentnet_win_mac_18k.jsonl"
    default_output = "data/agentnet/bagel_format/vlm_data/agent_tasks__winmac__l1.jsonl"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the raw AgentNet trajectory JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination JSONL file for converted samples.",
    )
    parser.add_argument(
        "--image-base",
        type=Path,
        default=base_dir,
        help="Base directory containing ubuntu_images and win_mac_images folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of converted step samples.",
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



def format_step_response(
    step_number: int,
    thought: Optional[str],
    action: Optional[str],
    code: Optional[str],
) -> str:
    segments: List[str] = [f"# Step {step_number}"]
    # if thought:
    #     segments.append(f"## Thought:\n{thought}")
    if action:
        segments.append(f"## Action:\n{action}")
    if code:
        segments.append(f"## Code:\n```python\n{code}\n```")
    return "\n".join(segments)


def build_bagel_conversations(
    traj: List[Dict[str, object]],
    step_index: int,
    instruction: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Build Bagel-compatible image list and conversations.

    - image: a list of image filenames (order matches <image> placeholders)
    - conversations: list of {from: 'human'|'gpt', value: str}
    """
    images_filenames: List[str] = []
    conversations: List[Dict[str, object]] = [{"from": "human", "value": L1_SYSTEM_PROMPTS[0]}]

    # 1-based step number for display
    S = step_index + 1

    # Long history actions only: steps 1 .. S-3
    if S - 1 >= 1:
        for k in range(1, S):
            hist_step = traj[k - 1]
            value = hist_step.get("value", {}) if isinstance(hist_step, dict) else {}
            action_text = value.get("action")
            response_text = format_step_response(k, None, action_text, None)
            conversations.append({"from": "gpt", "value": response_text})

    # Recent history with images: steps max(1, S-2) .. S-1
    recent_start = max(1, S)
    for k in range(recent_start, S):
        img_step = traj[k - 1]
        img_filename = img_step.get("image") if isinstance(img_step, dict) else None
        if img_filename:
            images_filenames.append(img_filename)
            conversations.append({"from": "human", "value": "<image>"})

        value = img_step.get("value", {}) if isinstance(img_step, dict) else {}
        action_text = value.get("action")
        response_text = format_step_response(k, None, action_text, None)
        conversations.append({"from": "gpt", "value": response_text})

    # Add current observation image (image_S) combined with instruction in a single human message
    if step_index < len(traj):
        cur_img_step = traj[step_index]
        img_filename = cur_img_step.get("image") if isinstance(cur_img_step, dict) else None
        if img_filename:
            images_filenames.append(img_filename)
            human_value = (
                "<image>\n"
                f"# Task Instruction:\n{instruction}\n\n"
                "Please generate the next move according to the screenshot, "
                "task instruction and previous steps (if provided)."
            )
        else:
            human_value = (
                f"# Task Instruction:\n{instruction}\n\n"
                "Please generate the next move according to the context above."
            )
        conversations.append({"from": "human", "value": human_value})

    # Target assistant response for step S
    current_step = traj[step_index]
    target_value = current_step.get("value", {}) if isinstance(current_step, dict) else {}
    target_response = format_step_response(
        S,
        target_value.get("thought"),
        target_value.get("action"),
        target_value.get("code"),
    )
    conversations.append({"from": "gpt", "value": target_response})

    return images_filenames, conversations


def convert_samples(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_trajectories = 0
    total_steps = 0

    # If running in sampled mode (limit set), save exactly one original trajectory for debugging
    orig_file = None
    if args.limit is not None:
        orig_output = args.output.with_name(args.output.stem + ".orig.jsonl")
        orig_file = orig_output.open("w", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as out_file:
        for sample in iter_jsonl(args.input):
            task_id = sample.get("task_id")
            instruction = sample.get("instruction", "")
            traj = sample.get("traj", [])

            if not traj:
                continue

            for step_index, step in enumerate(traj):
                # Bagel-compatible output
                image_list, conversations = build_bagel_conversations(traj, step_index, instruction)

                # 1-based step number in id for clarity
                converted_record = {
                    "id": f"{task_id}_{step_index + 1:03d}",
                    "image": image_list,
                    "conversations": conversations,
                }

                out_file.write(json.dumps(converted_record, ensure_ascii=False) + "\n")

                total_steps += 1

            # Save a single original trajectory for debugging when sampling
            if orig_file and total_trajectories == 0:
                orig_record = {
                    "task_id": task_id,
                    "instruction": instruction,
                    "natural_language_task": sample.get("natural_language_task"),
                    "actual_task": sample.get("actual_task"),
                    "traj": traj,
                }
                orig_file.write(json.dumps(orig_record, ensure_ascii=False) + "\n")

            total_trajectories += 1
            if args.limit and total_trajectories >= args.limit:
                break

    if orig_file:
        orig_file.close()

    summary = (
        f"Converted {total_steps} step-level tasks from {total_trajectories} trajectories."
    )
    print(summary)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    convert_samples(args)


if __name__ == "__main__":
    main()
