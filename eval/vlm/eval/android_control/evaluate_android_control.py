# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import itertools
import json
import os
import subprocess
import sys
from functools import wraps
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, Subset
from tqdm import tqdm

from data.transforms import ImageTransform
from eval.mobile.utils import load_model_and_tokenizer
from inferencer import DatasetInferencer


class AndroidControlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        self._records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]


def collate_fn(batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batches


class InferenceSampler(Sampler[int]):
    def __init__(self, size: int):
        self._size = int(size)
        if self._size <= 0:
            raise ValueError("Dataset size must be positive.")
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = list(self._get_local_indices(self._size, self._world_size, self._rank))

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int) -> Iterable[int]:
        shard_size = total_size // world_size
        remainder = total_size % world_size
        shard_sizes = [shard_size + int(r < remainder) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self) -> int:
        return len(self._local_indices)


def _move_tensor_dict_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if data is None:
        return data
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
    return data


def _wrap_model_prepares(model: torch.nn.Module) -> None:
    device = next(model.parameters()).device

    def wrap_method(name: str):
        if not hasattr(model, name):
            return
        original = getattr(model, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, tuple):
                if result and isinstance(result[0], dict):
                    moved = _move_tensor_dict_to_device(result[0], device)
                    return (moved, *result[1:])
                return result
            if isinstance(result, dict):
                return _move_tensor_dict_to_device(result, device)
            return result

        setattr(model, name, wrapped)

    for method_name in ("prepare_prompts", "prepare_vit_images", "prepare_vae_images", "prepare_start_tokens"):
        wrap_method(method_name)


def evaluate_dataset(args, model, tokenizer, new_token_ids, vit_transform, vae_transform):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    dataset = AndroidControlDataset(args.jsonl_path)
    if args.max_samples is not None:
        max_count = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_count)))

    if world_size > 1:
        sampler: Sampler[int] = InferenceSampler(len(dataset))
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    vae_model = getattr(model, "_vae_model", None)
    if args.enable_vae and vae_model is None:
        raise ValueError("VAE branch requested but autoencoder weights were not loaded.")

    inferencer = DatasetInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        image_root=args.image_root,
        use_vae=args.enable_vae,
    )

    local_outputs: List[Dict[str, Any]] = []
    progress = tqdm(dataloader, disable=(rank != 0), desc="Evaluating")

    with torch.no_grad():
        for batch in progress:
            if not batch:
                continue
            record = batch[0]
            result = inferencer.predict_next_step(
                record,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_length=args.max_length,
            )
            local_outputs.append(
                {
                    "id": record.get("id"),
                    "prediction": result["prediction"],
                    "target": result["target"],
                }
            )

    torch.distributed.barrier()

    serialized = json.dumps(local_outputs)
    gathered: List[str] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, serialized)

    if rank == 0:
        merged_outputs = [json.loads(buf) for buf in gathered]
        merged_outputs = list(itertools.chain.from_iterable(merged_outputs))

        os.makedirs(args.out_dir, exist_ok=True)
        output_path = os.path.join(args.out_dir, "raw_predictions.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in merged_outputs:
                f.write(json.dumps(item) + "\n")
        print(f"[Info] Saved {len(merged_outputs)} predictions to {output_path}.")

        score_cmd = [
            sys.executable,
            "eval/vlm/eval/android_control/score_android_control.py",
            "--predictions",
            output_path,
            "--dataset-jsonl",
            args.jsonl_path,
            "--image-root",
            args.image_root,
            "--output-dir",
            args.out_dir,
        ]
        print(f"[Info] Running scoring script: {' '.join(score_cmd)}")
        result = subprocess.run(score_cmd, check=False)
        if result.returncode != 0:
            print(f"[Warning] Scoring script exited with status code {result.returncode}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bagel on Android Control.")
    parser.add_argument("--model-path", type=str, default="hf/BAGEL-7B-MoT/", help="Path to the Bagel model directory.")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="eval/vlm/eval/android_control/bagel_format_v2/vlm_data/agent_tasks__android_control__l1_subset500.jsonl",
        help="Android Control JSONL file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="eval/vlm/eval/android_control/images",
        help="Root directory containing screenshot images.",
    )
    parser.add_argument("--out-dir", type=str, default="results/android_control", help="Directory to store outputs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size (must remain 1).")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of dataloader workers.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum decode length for predictions.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during decoding.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on number of samples to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--enable-vae", action="store_true", help="Enable VAE tokens during inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Only batch size 1 is supported for Android Control evaluation.")

    torch.manual_seed(args.seed)

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    _wrap_model_prepares(model)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if torch.distributed.get_rank() == 0:
        print(f"[Info] Loaded model with {total_params:.2f}B parameters.")
    evaluate_dataset(args, model, tokenizer, new_token_ids, vit_transform, vae_transform)


if __name__ == "__main__":
    main()
