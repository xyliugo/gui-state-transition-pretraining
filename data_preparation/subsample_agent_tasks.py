"""Subsample an agent-task JSONL by trajectory.

All converters emit records with `id = "{trajectory_id}_{step:03d}"`. This
script groups records by trajectory id, samples N trajectories with a fixed
seed, and writes all their step records to a new JSONL.

Used to produce the fine-tuning subsets referenced by the paper's main tables
(Group 1a: 2K Win&Mac trajectories; Group 2: 3K AndroidControl; Group 3: 1K
GUIOdyssey).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Converted agent-task JSONL.")
    p.add_argument("--output", type=Path, required=True, help="Destination JSONL for the subsample.")
    p.add_argument("--num-trajectories", type=int, required=True, help="Number of trajectories to keep.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (fixed at 42 for release reproducibility).")
    return p.parse_args()


def trajectory_id(record_id: str) -> str:
    # id format from all converters: "{trajectory_id}_{step:03d}"
    return record_id.rsplit("_", 1)[0]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    groups: Dict[str, List[dict]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups[trajectory_id(record["id"])].append(record)

    all_trajs = sorted(groups.keys())
    if args.num_trajectories > len(all_trajs):
        print(
            f"Warning: requested {args.num_trajectories} trajectories but input has only "
            f"{len(all_trajs)}. Using all available.",
            file=sys.stderr,
        )
        chosen = all_trajs
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(all_trajs, args.num_trajectories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    with args.output.open("w", encoding="utf-8") as f:
        for tid in chosen:
            for record in groups[tid]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_steps += 1

    print(
        f"Wrote {total_steps} step records from {len(chosen)} trajectories "
        f"(seed={args.seed}) to {args.output}"
    )


if __name__ == "__main__":
    main()
