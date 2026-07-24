# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml
import torch
from torch.distributed.checkpoint import FileSystemReader, load_state_dict as load_sharded_state_dict

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, 
    Bagel, 
    Qwen2Config, 
    Qwen2ForCausalLM, 
    SiglipVisionConfig, 
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform
from modeling.autoencoder import load_ae


_POS_EMBED_KEYS = ("latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed")


def _has_sharded_state_dict(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.isfile(metadata_path):
        return True
    try:
        for name in os.listdir(path):
            if name.endswith(".distcp"):
                return True
    except FileNotFoundError:
        return False
    return False


def _pop_ignored_keys(state_dict):
    for key in _POS_EMBED_KEYS:
        state_dict.pop(key, None)


def _load_sharded_weights(module, ckpt_dir):
    state_dict = module.state_dict()
    reader = FileSystemReader(ckpt_dir)
    load_sharded_state_dict(state_dict, reader)
    _pop_ignored_keys(state_dict)
    msg = module.load_state_dict(state_dict, strict=False)
    del state_dict
    return msg


def _load_safetensor_weights(module, ckpt_path):
    weights = load_file(ckpt_path, device="cpu")
    _pop_ignored_keys(weights)
    msg = module.load_state_dict(weights, strict=False)
    del weights
    return msg


def _resolve_checkpoint_source(model_path: str):
    if not model_path:
        raise ValueError("Model path must be provided.")
    expanded = os.path.abspath(model_path)
    if os.path.isfile(expanded):
        return {"kind": "file", "path": expanded, "source": os.path.basename(expanded)}
    if _has_sharded_state_dict(expanded):
        return {"kind": "sharded", "path": expanded, "source": os.path.relpath(expanded)}
    if not os.path.isdir(expanded):
        raise FileNotFoundError(f"Model path does not exist: {expanded}")

    candidates = [
        ("ema (sharded)", "sharded", os.path.join(expanded, "ema")),
        ("ema.safetensors", "file", os.path.join(expanded, "ema.safetensors")),
        ("model (sharded)", "sharded", os.path.join(expanded, "model")),
        ("model.safetensors", "file", os.path.join(expanded, "model.safetensors")),
    ]

    for label, kind, path in candidates:
        if kind == "sharded" and _has_sharded_state_dict(path):
            return {"kind": kind, "path": path, "source": label}
        if kind == "file" and os.path.isfile(path):
            return {"kind": kind, "path": path, "source": label}

    raise FileNotFoundError(
        f"Could not locate checkpoint files under {expanded}. "
        "Expected sharded directories (ema/model) or *.safetensors."
    )


def load_model_and_tokenizer(args):
    enable_vae = getattr(args, "enable_vae", False)

    llm_config = Qwen2Config.from_json_file(os.path.join("checkpoints/BAGEL-7B-MoT", "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module ="Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join("checkpoints/BAGEL-7B-MoT", "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model = None
    vae_config = None
    if enable_vae:
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config_kwargs = dict(
        visual_gen=enable_vae,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    if enable_vae:
        config_kwargs.update(
            vae_config=vae_config,
            latent_patch_size=2,
            max_latent_size=64,
        )

    config = BagelConfig(**config_kwargs)
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained("checkpoints/BAGEL-7B-MoT")
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    ckpt = _resolve_checkpoint_source(args.model_path)
    print(f"[Info] Loading weights from {ckpt['path']} ({ckpt['source']}).")
    if ckpt["kind"] == "sharded":
        msg = _load_sharded_weights(model, ckpt["path"])
    else:
        msg = _load_safetensor_weights(model, ckpt["path"])
    print(msg)
    
    model = model.to(dtype=torch.bfloat16)
    model = model.to("cuda")
    model = model.eval()

    if vae_model is not None:
        vae_model = vae_model.to(dtype=torch.bfloat16).to("cuda").eval()
        setattr(model, "_vae_model", vae_model)

    return model, tokenizer, new_token_ids


def build_transform():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs",
        "ft_android_control_3k.yaml",
    )
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
    min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
    image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
    max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
