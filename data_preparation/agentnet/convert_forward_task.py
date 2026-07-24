from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    # tqdm provides nice progress bars with ETA and throughput
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    # Fallback no-op tqdm if the package is unavailable
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0) or 0)

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    base_dir = Path(os.environ.get("DATA_ROOT", "./datasets")) / "agentnet"
    parser = argparse.ArgumentParser(
        description=(
            "Convert AgentNet trajectories into forward editing data (s_t + a_t -> s_{t+1}) "
            "following BAGEL unified_edit parquet format."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "agentnet_win_mac_18k.jsonl",
        help="Path to the raw AgentNet trajectory JSONL file.",
    )
    parser.add_argument(
        "--image-base",
        type=Path,
        default=base_dir,
        help="Base directory containing ubuntu_images and win_mac_images folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "bagel_format" / "forward_edit",
        help="Output directory to store parquet chunks and parquet_info.json.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=1000,
        help="Number of rows per parquet chunk file.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100,
        help="Target row group size when writing parquet (controls rg count).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of pairs to convert.",
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


def resolve_image_path(base_dir: Path, filename: Optional[str]) -> Optional[Path]:
    if not filename:
        return None
    for sub in ("ubuntu_images", "win_mac_images"):
        p = base_dir / sub / "images" / filename
        if p.exists():
            return p
    return None


def load_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception:
        return None


def format_action_label(action: Optional[str], code: Optional[str]) -> str:
    parts: List[str] = []
    if action:
        parts.append(f"## Action:\n{action}")
    if code:
        parts.append(f"## Code:\n```python\n{code}\n```")
    return "\n".join(parts) if parts else ""


def build_instruction(action_str: Optional[str], code_str: Optional[str]) -> str:
    system = "You are a helpful assistant that predicts the next screen state."
    formatted = format_action_label(action_str or "", code_str or "")
    body = (
        "Given the current screen, perform the move:\n"
        f"{formatted}\n\n"
        "What is the next screen?"
    )
    return f"{system}\n{body}"


def _rows_to_table(
    rows: List[Tuple[List[bytes], List[List[str]]]],
) -> pa.Table:
    """Convert buffered rows into a PyArrow table.

    Keeping this as a small helper allows us to stream-write
    multiple chunks without holding all rows in memory.
    """
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
    """Write a single parquet file for the provided rows.

    Returns a tuple of (chunk_path_str, info_dict).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _rows_to_table(rows)
    chunk_path = (out_dir / f"chunk_{chunk_index}.parquet").resolve()
    pq.write_table(table, chunk_path, row_group_size=row_group_size)

    num_rows = len(rows)
    num_row_groups = math.ceil(num_rows / row_group_size) if row_group_size > 0 else 1
    return str(chunk_path), {"num_row_groups": int(num_row_groups), "num_rows": int(num_rows)}


def convert(args: argparse.Namespace) -> None:
    # Buffer rows and stream-write to avoid OOM.
    buffer_rows: List[Tuple[List[bytes], List[List[str]]]] = []
    written = 0
    chunk_idx = 0
    parquet_info: Dict[str, Dict[str, int]] = {}

    traj_pbar = tqdm(desc="Trajectories", unit="traj")
    pair_pbar = tqdm(desc="Pairs", unit="pair")

    def flush_if_needed(force: bool = False) -> None:
        nonlocal buffer_rows, chunk_idx, parquet_info
        if buffer_rows and (force or len(buffer_rows) >= args.rows_per_file):
            chunk_path, info = write_parquet_chunk(
                buffer_rows, args.out_dir, args.row_group_size, chunk_idx
            )
            parquet_info[chunk_path] = info
            buffer_rows = []  # free memory held by the buffer
            chunk_idx += 1

    for sample in iter_jsonl(args.input):
        traj = sample.get("traj", [])
        if not isinstance(traj, list) or len(traj) < 2:
            traj_pbar.update(1)
            continue
        for t in range(len(traj) - 1):
            step_t = traj[t] if isinstance(traj[t], dict) else {}
            step_tp1 = traj[t + 1] if isinstance(traj[t + 1], dict) else {}

            img_t = step_t.get("image")
            img_tp1 = step_tp1.get("image")
            value_t = step_t.get("value", {}) if isinstance(step_t, dict) else {}
            action = value_t.get("action")
            code = value_t.get("code")

            p_t = resolve_image_path(args.image_base, img_t)
            p_tp1 = resolve_image_path(args.image_base, img_tp1)
            if not p_t or not p_tp1:
                continue

            b_t = load_bytes(p_t)
            b_tp1 = load_bytes(p_tp1)
            if not b_t or not b_tp1:
                continue

            instruction = build_instruction(action, code)
            buffer_rows.append(([b_t, b_tp1], [[instruction]]))
            written += 1
            pair_pbar.update(1)
            flush_if_needed()

            if args.limit is not None and written >= args.limit:
                break
        traj_pbar.update(1)
        if args.limit is not None and written >= args.limit:
            break

    traj_pbar.close()
    pair_pbar.close()

    # Flush remaining rows in the last chunk
    flush_if_needed(force=True)

    # Save parquet_info.json next to chunks
    info_path = args.out_dir.parent / "parquet_info" / f"{args.out_dir.name}.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(parquet_info, ensure_ascii=False, indent=4), encoding="utf-8")
    print(
        f"Wrote {written} rows into {len(parquet_info)} parquet chunks. "
        f"Parquet info saved under {args.out_dir.parent / 'parquet_info' / (args.out_dir.name + '.json')}"
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    convert(args)


if __name__ == "__main__":
    main()
