# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from typing import List

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from .data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


action_description = [
    '<action_format> Action:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}} </action_format> '
]

action_description_ac = (
    "<action_format> Action:\n"
    "  Action as a JSON object wrapped in a code block:\n"
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
)

class ForwardEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """Iterates (s_t, a_t -> s_{t+1}) forward editing data.

    Each row is expected to contain two images (previous and next screen) and
    an instruction list describing the action that led to the transition.
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        num_used_data,
        parquet_info,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        self._action_suffix = (
            action_description_ac
            if any("android_control" in str(path) or "odyssey" in str(path) for path in data_dir_list)
            else action_description[0]
        )
        super().__init__(
            dataset_name=dataset_name,
            transform=transform,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            data_dir_list=data_dir_list,
            num_used_data=num_used_data,
            parquet_info=parquet_info,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
        )

    def parse_row(self, row):
        image_bytes_list: List[bytes] = row.get("image_list", [])
        instruction_list = row.get("instruction_list", [])

        if len(image_bytes_list) < 2:
            return {}

        # Previous screen (conditioning image)
        prev_image = pil_img2rgb(Image.open(io.BytesIO(image_bytes_list[0])))
        # Next screen (target image with loss)
        next_image = pil_img2rgb(Image.open(io.BytesIO(image_bytes_list[1])))

        data = self._init_data()

        data = self._add_image(
            data,
            prev_image,
            need_loss=False,
            need_vae=False,
            need_vit=True,
        )

        # Action instruction text (if available)
        instruction_candidates = []
        for bucket in instruction_list:
            instruction_candidates.extend(bucket)
        if instruction_candidates:
            instruction = random.choice(instruction_candidates)
            instruction = instruction.replace("You are a helpful assistant that predicts the next screen state.", "")
            instruction = instruction.replace("You are a helpful assistant that predicts future screen states after multi-step actions.", "")
            instruction = instruction + self._action_suffix
            data = self._add_text(data, instruction, need_loss=False)

        # Target image: supervise VAE decoder only
        data = self._add_image(
            data,
            next_image,
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )

        return data
