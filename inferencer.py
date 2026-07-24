# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(
        self,
        image,
        gen_context,
        vae=True,
        vit=True,
        vae_transform=None,
        vit_transform=None,
    ):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=vae_transform or self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=vit_transform or self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict



class _TransformWithImageCount:
    """Ensure ImageTransform receives a consistent img_num while preserving attributes."""

    def __init__(self, base_transform, img_num: int):
        self._base = base_transform
        self._img_num = max(int(img_num), 1)

    def __call__(self, image):
        return self._base(image, img_num=self._img_num)

    def __getattr__(self, name):
        return getattr(self._base, name)


class DatasetInferencer(InterleaveInferencer):
    """
    Utility inferencer tailored for dataset-style evaluation where the goal is to
    predict the final assistant response (the portion supervised during training).

    Given a json-like record with the same structure as the training data
    (containing ``image`` paths and ``conversations``), this inferencer will:
        1. Load and cache all context turns (human + previous assistant).
        2. Feed any visual frames through the ViT branch (optionally paired with VAE tokens).
        3. Autoregressively decode the next assistant turn.
    """

    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        image_root: Optional[str] = None,
        use_vae: bool = False,
    ):
        super().__init__(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
        self.image_root = image_root
        self.use_vae = use_vae
        if self.use_vae and self.vae_model is None:
            raise ValueError("use_vae=True requires a valid vae_model instance.")

    def _load_images(self, record: Dict[str, Any], image_root: Optional[str]) -> List[Image.Image]:
        root = image_root or self.image_root
        if record is None:
            return []
        if "video" in record:
            raise NotImplementedError("Video-based samples are not supported by DatasetInferencer.")
        image_entries = record.get("image")
        if image_entries is None:
            return []
        if root is None:
            raise ValueError("image_root must be provided to resolve image paths.")

        if isinstance(image_entries, list):
            image_names = image_entries
        else:
            image_names = [image_entries]

        images = []
        for name in image_names:
            image_path = os.path.join(root, name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as img:
                images.append(pil_img2rgb(img))
        return images

    def _build_context_elements(
        self,
        record: Dict[str, Any],
        images: List[Image.Image],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        conversations = record.get("conversations", [])
        last_gpt_idx = None
        for idx, conv in enumerate(conversations):
            if conv.get("from") == "gpt":
                last_gpt_idx = idx

        image_iter = iter(images)
        elements: List[Dict[str, Any]] = []
        target_text: Optional[str] = None

        def append_image_element(img: Image.Image):
            if self.use_vae:
                elements.append({"type": "vae_image", "value": img})
                elements.append({"type": "vit_image", "value": img})
            else:
                elements.append({"type": "image", "value": img})

        for idx, conv in enumerate(conversations):
            speaker = conv.get("from")
            value = conv.get("value", "")

            if speaker == "human":
                if "<image>" not in value:
                    if value.strip():
                        elements.append({"type": "text", "value": value})
                else:
                    parts = value.split("<image>")
                    for part_idx, part in enumerate(parts):
                        text_part = part.strip()
                        if text_part:
                            elements.append({"type": "text", "value": text_part})
                        if part_idx != len(parts) - 1:
                            try:
                                image = next(image_iter)
                            except StopIteration as exc:
                                raise ValueError("Number of <image> tokens exceeds provided images.") from exc
                            append_image_element(image)
            elif speaker == "gpt":
                if idx == last_gpt_idx:
                    target_text = value
                    # Skip adding the target into the context so we can predict it.
                    continue
                if value.strip():
                    elements.append({"type": "text", "value": value})
            else:
                raise ValueError(f"Unsupported speaker type: {speaker}")

        try:
            next(image_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("Number of provided images exceeds <image> placeholders in text.")

        return elements, target_text

    def predict_next_step(
        self,
        record: Dict[str, Any],
        *,
        image_root: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        max_length: int = 512,
    ) -> Dict[str, str]:
        """
        Predict the final assistant response for a dataset record.

        Args:
            record: Single json-like entry mirroring training data.
            image_root: Optional override for the image directory.
            do_sample: Whether to sample during decoding.
            temperature: Softmax temperature for sampling.
            max_length: Maximum number of generated tokens.

        Returns:
            Dict with the generated prediction and ground-truth target.
        """
        images = self._load_images(record, image_root)
        context_elements, target_text = self._build_context_elements(record, images)
        # print(context_elements, target_text)
        if target_text is None:
            raise ValueError("No assistant turn marked for prediction in the provided record.")

        gen_context = self.init_gen_context()
        total_images = len(images)
        vit_transform = self.vit_transform
        vae_transform = self.vae_transform
        if total_images > 0:
            vit_transform = _TransformWithImageCount(self.vit_transform, total_images)
            if self.use_vae:
                vae_transform = _TransformWithImageCount(self.vae_transform, total_images)

        use_autocast = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            for element in context_elements:
                if element["type"] == "text":
                    gen_context = self.update_context_text(element["value"], gen_context)
                elif element["type"] == "image":
                    gen_context = self.update_context_image(
                        element["value"],
                        gen_context,
                        vae=self.use_vae,
                        vit=True,
                        vae_transform=vae_transform if self.use_vae else None,
                        vit_transform=vit_transform,
                    )
                elif element["type"] == "vae_image":
                    gen_context = self.update_context_image(
                        element["value"],
                        gen_context,
                        vae=True,
                        vit=False,
                        vae_transform=vae_transform,
                    )
                elif element["type"] == "vit_image":
                    gen_context = self.update_context_image(
                        element["value"],
                        gen_context,
                        vae=False,
                        vit=True,
                        vit_transform=vit_transform,
                    )
                else:
                    raise ValueError(f"Unsupported element type: {element['type']}")

            prediction = self.gen_text(
                gen_context,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )
        return {"prediction": prediction, "target": target_text}
