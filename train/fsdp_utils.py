# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.checkpoint import (
    FileSystemReader,
    load_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=None, use_orig_params=False):
    if ignored_modules is None:
        ignored_modules = []
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=use_orig_params,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        logger, 
        fsdp_config,
        data_status=None,
        keep_total_ckpts=2,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
                del ema_state_dict
                gc.collect()

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
            del model_state_dict
            gc.collect()

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        FSDPCheckpoint._cleanup_old_optimizer_shards(
            ckpt_dir=ckpt_dir,
            keep_dir_name=f"{train_steps:07d}",
            fsdp_config=fsdp_config,
            logger=logger,
            keep_total_ckpts=keep_total_ckpts,
        )
        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            pos_embed_keys = ('latent_pos_embed.pos_embed', 'vit_pos_embed.pos_embed')
            legacy_model_state_path = os.path.join(
                resume_from,
                "ema.safetensors" if resume_from_ema else "model.safetensors",
            )
            sharded_model_dir = os.path.join(resume_from, "ema" if resume_from_ema else "model")

            if os.path.isfile(legacy_model_state_path):
                model_state_dict = load_file(legacy_model_state_path, device="cpu")
                model_state_dict.pop('latent_pos_embed.pos_embed', None)
                model_state_dict.pop('vit_pos_embed.pos_embed', None)
                msg = model.load_state_dict(model_state_dict, strict=False)
                logger.info(msg)
                del model_state_dict
                gc.collect()
            elif FSDPCheckpoint._has_sharded_state_dict(sharded_model_dir):
                msg = FSDPCheckpoint._load_sharded_state_dict_into_module(
                    model, sharded_model_dir, keys_to_pop=pos_embed_keys
                )
                logger.info(msg)

            if ema_model is not None:
                ema_sharded_dir = os.path.join(resume_from, "ema")
                model_sharded_dir = os.path.join(resume_from, "model")
                ema_safetensors_path = os.path.join(resume_from, "ema.safetensors")
                model_safetensors_path = os.path.join(resume_from, "model.safetensors")

                if os.path.isfile(ema_safetensors_path):
                    ema_load_path = ema_safetensors_path
                elif os.path.isfile(model_safetensors_path):
                    logger.info(f"replicaing ema model from {model_safetensors_path}.")
                    ema_load_path = model_safetensors_path
                else:
                    ema_load_path = None

                if ema_load_path is not None:
                    ema_state_dict = load_file(ema_load_path, device="cpu")
                    ema_state_dict.pop('latent_pos_embed.pos_embed', None)
                    ema_state_dict.pop('vit_pos_embed.pos_embed', None)
                    msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                    logger.info(msg)
                    del ema_state_dict
                    gc.collect()
                elif FSDPCheckpoint._has_sharded_state_dict(ema_sharded_dir):
                    msg = FSDPCheckpoint._load_sharded_state_dict_into_module(
                        ema_model, ema_sharded_dir, keys_to_pop=pos_embed_keys
                    )
                    logger.info(msg)
                elif FSDPCheckpoint._has_sharded_state_dict(model_sharded_dir):
                    logger.info(f"replicaing ema model from {model_sharded_dir}.")
                    msg = FSDPCheckpoint._load_sharded_state_dict_into_module(
                        ema_model, model_sharded_dir, keys_to_pop=pos_embed_keys
                    )
                    logger.info(msg)
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status

    @staticmethod
    def _should_handle_optimizer_shard(fsdp_config):
        if fsdp_config.sharding_strategy == "FULL_SHARD":
            shard_index = dist.get_rank()
            total_shards = dist.get_world_size()
            should_handle = True
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            shard_index = dist.get_rank() % fsdp_config.num_shard
            total_shards = fsdp_config.num_shard
            should_handle = dist.get_rank() < fsdp_config.num_shard
        else:
            raise NotImplementedError
        return should_handle, shard_index, total_shards

    @staticmethod
    def _cleanup_old_optimizer_shards(ckpt_dir, keep_dir_name, fsdp_config, logger, keep_total_ckpts=2):
        if not os.path.isdir(ckpt_dir):
            return

        keep_total_ckpts = max(int(keep_total_ckpts), 1)
        should_handle, shard_index, total_shards = FSDPCheckpoint._should_handle_optimizer_shard(fsdp_config)
        try:
            entries = [
                name
                for name in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, name))
            ]
        except FileNotFoundError:
            return

        numeric_entries = []
        for name in entries:
            try:
                int(name)
            except ValueError:
                continue
            numeric_entries.append(name)

        if not numeric_entries:
            return

        ordered_dirs = sorted(numeric_entries, key=int, reverse=True)

        # Ensure the latest checkpoint is always included first.
        dirs_to_keep = [keep_dir_name]
        for name in ordered_dirs:
            if name == keep_dir_name:
                continue
            if len(dirs_to_keep) >= keep_total_ckpts:
                break
            dirs_to_keep.append(name)

        dirs_to_remove_optimizer = [name for name in dirs_to_keep if name != keep_dir_name]
        dirs_to_delete = [name for name in ordered_dirs if name not in dirs_to_keep]

        if not dirs_to_remove_optimizer and not dirs_to_delete:
            return

        optimizer_cleanup_dirs = []

        if should_handle and dirs_to_remove_optimizer:
            optimizer_filename = f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            for dir_name in dirs_to_remove_optimizer:
                shard_path = os.path.join(ckpt_dir, dir_name, optimizer_filename)
                try:
                    os.remove(shard_path)
                    optimizer_cleanup_dirs.append(dir_name)
                except FileNotFoundError:
                    continue
                except IsADirectoryError:
                    continue

        dist.barrier()

        model_cleanup_dirs = []
        if dist.get_rank() == 0 and dirs_to_remove_optimizer:
            for name in dirs_to_remove_optimizer:
                ckpt_path = os.path.join(ckpt_dir, name)
                model_dir = os.path.join(ckpt_path, "model")
                legacy_model_file = os.path.join(ckpt_path, "model.safetensors")
                removed_any = False
                if os.path.isdir(model_dir):
                    try:
                        shutil.rmtree(model_dir)
                        removed_any = True
                    except (FileNotFoundError, OSError):
                        pass
                if os.path.isfile(legacy_model_file):
                    try:
                        os.remove(legacy_model_file)
                        removed_any = True
                    except OSError:
                        pass
                if removed_any:
                    model_cleanup_dirs.append(name)

        if dist.get_rank() == 0:
            fully_removed = []
            for name in dirs_to_delete:
                ckpt_path = os.path.join(ckpt_dir, name)
                if not os.path.isdir(ckpt_path):
                    continue
                try:
                    shutil.rmtree(ckpt_path)
                    fully_removed.append(name)
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

            if fully_removed and logger is not None:
                logger.info(
                    "Removed old checkpoints: %s",
                    ", ".join(sorted(fully_removed)),
                )

        if model_cleanup_dirs and logger is not None and dist.get_rank() == 0:
            logger.info(
                "Removed model shards from checkpoints: %s",
                ", ".join(sorted(model_cleanup_dirs)),
            )

        if optimizer_cleanup_dirs and logger is not None and dist.get_rank() == 0:
            logger.info(
                "Removed optimizer shards from checkpoints: %s",
                ", ".join(sorted(optimizer_cleanup_dirs)),
            )

    @staticmethod
    def _has_sharded_state_dict(ckpt_dir):
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            return False
        metadata_path = os.path.join(ckpt_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            return True
        try:
            for name in os.listdir(ckpt_dir):
                if name.endswith(".distcp"):
                    return True
        except FileNotFoundError:
            return False
        return False

    @staticmethod
    def _load_sharded_state_dict_into_module(module, ckpt_dir, keys_to_pop=()):
        state_dict = module.state_dict()
        reader = FileSystemReader(ckpt_dir)
        load_state_dict(state_dict, reader)
        for key in keys_to_pop or ():
            state_dict.pop(key, None)
        msg = module.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        return msg


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=None, use_orig_params=False):
    if ignored_modules is None:
        ignored_modules = []
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(
        ema_model,
        fsdp_config,
        ignored_modules=ignored_modules,
        use_orig_params=use_orig_params,
    )
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
