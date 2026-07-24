# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Unified evaluator that loads the model **once** and runs multiple
agent benchmarks (Android Control, GUI-Odyssey) sequentially.

Usage (via torchrun):
    torchrun --nproc_per_node=8 -m eval.vlm.eval.evaluate_android_all \
        --model-path results/my_exp/checkpoints/0006000 \
        --benchmarks android_control guiodyssey \
        --android-control-jsonl /path/to/ac.jsonl \
        --android-control-image-root /path/to/ac/images \
        --guiodyssey-jsonl /path/to/go.jsonl \
        --guiodyssey-image-root /path/to/go/screenshots \
        --out-dir eval/vlm/outputs/my_exp/0006000
"""

import argparse
import os
import types
from typing import List

import torch

from data.transforms import ImageTransform
from eval.mobile.utils import load_model_and_tokenizer


def _wrap_model_prepares(model: torch.nn.Module) -> None:
    """Ensure model.prepare_* outputs land on the correct device.

    Identical to the helper used in the per-benchmark evaluators, duplicated
    here so that this module stays self-contained without importing from a
    specific benchmark package.
    """
    from functools import wraps

    device = next(model.parameters()).device

    def _move(data, dev):
        if data is None:
            return data
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(dev)
        return data

    def wrap_method(name: str):
        if not hasattr(model, name):
            return
        original = getattr(model, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, tuple):
                if result and isinstance(result[0], dict):
                    return (_move(result[0], device), *result[1:])
                return result
            if isinstance(result, dict):
                return _move(result, device)
            return result

        setattr(model, name, wrapped)

    for method_name in (
        "prepare_prompts",
        "prepare_vit_images",
        "prepare_vae_images",
        "prepare_start_tokens",
    ):
        wrap_method(method_name)


# ---------------------------------------------------------------------------
# Per-benchmark argument builder
# ---------------------------------------------------------------------------

def _build_android_control_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a namespace that looks like what evaluate_android_control.evaluate_dataset expects."""
    ns = argparse.Namespace(
        jsonl_path=args.android_control_jsonl,
        image_root=args.android_control_image_root,
        out_dir=os.path.join(args.out_dir, "android_control"),
        batch_size=1,
        num_workers=args.num_workers,
        temperature=args.temperature,
        max_length=args.max_length,
        do_sample=args.do_sample,
        max_samples=args.max_samples,
        enable_vae=args.enable_vae,
    )
    return ns


def _build_guiodyssey_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a namespace that looks like what evaluate_guiodyssey.evaluate_dataset expects."""
    ns = argparse.Namespace(
        jsonl_path=args.guiodyssey_jsonl,
        image_root=args.guiodyssey_image_root,
        out_dir=os.path.join(args.out_dir, "guiodyssey"),
        metric=args.guiodyssey_metric,
        batch_size=1,
        num_workers=args.num_workers,
        temperature=args.temperature,
        max_length=args.max_length,
        do_sample=args.do_sample,
        max_samples=args.max_samples,
        enable_vae=args.enable_vae,
    )
    return ns


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified multi-benchmark evaluator (single model load)."
    )
    # Model
    p.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint.")
    p.add_argument("--enable-vae", action="store_true", help="Enable VAE tokens.")

    # Benchmarks to run
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["android_control", "guiodyssey"],
        help="Benchmarks to evaluate (space-separated).",
    )

    # Common inference params
    p.add_argument("--out-dir", type=str, required=True, help="Base output directory.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # Data root for resolving relative dataset paths
    _data_root = os.environ.get("DATA_ROOT", "./datasets")

    # Android Control specific
    p.add_argument(
        "--android-control-jsonl",
        type=str,
        default=os.path.join(_data_root, "android_control/bagel_format_v2_test/vlm_data/agent_tasks__android_control__l1.jsonl"),
    )
    p.add_argument(
        "--android-control-image-root",
        type=str,
        default=os.path.join(_data_root, "android_control/images"),
    )

    # GUI Odyssey specific
    p.add_argument(
        "--guiodyssey-jsonl",
        type=str,
        default=os.path.join(_data_root, "gui_odyssey/bagel_format/agent_tasks__guiodyssey__high__random_split__test.jsonl"),
    )
    p.add_argument(
        "--guiodyssey-image-root",
        type=str,
        default=os.path.join(_data_root, "gui_odyssey/screenshots"),
    )
    p.add_argument(
        "--guiodyssey-metric",
        choices=("auto", "macro", "micro"),
        default="auto",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Distributed init
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    # 2. Load model ONCE
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    _wrap_model_prepares(model)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"[Info] Loaded model with {total_params:.2f}B parameters.")
        print(f"[Info] Benchmarks to evaluate: {args.benchmarks}")

    # 3. Run each benchmark sequentially, reusing the loaded model
    for bm_name in args.benchmarks:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Info] Starting benchmark: {bm_name}")
            print(f"{'='*60}")

        if bm_name == "android_control":
            from eval.vlm.eval.android_control.evaluate_android_control import (
                evaluate_dataset as eval_ac,
            )
            bm_args = _build_android_control_args(args)
            eval_ac(bm_args, model, tokenizer, new_token_ids, vit_transform, vae_transform)

        elif bm_name == "guiodyssey":
            from eval.vlm.eval.guiodyssey.evaluate_guiodyssey import (
                evaluate_dataset as eval_go,
            )
            bm_args = _build_guiodyssey_args(args)
            eval_go(bm_args, model, tokenizer, new_token_ids, vit_transform, vae_transform)

        else:
            if rank == 0:
                print(f"[Warning] Unknown benchmark '{bm_name}', skipping.")
            continue

        if rank == 0:
            print(f"[Info] Finished benchmark: {bm_name}")

    if rank == 0:
        print(f"\n[Info] All benchmarks done.")


if __name__ == "__main__":
    main()
