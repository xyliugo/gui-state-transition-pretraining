# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset registry for State Transition Pretraining (STP).

All paths are constructed relative to DATA_ROOT, which can be set via:
  1. Environment variable: export DATA_ROOT=/path/to/your/data
  2. Command-line override in the training script

Expected data layout under DATA_ROOT (see data_preparation/README.md for the full tree):
  DATA_ROOT/
  ├── agentnet/
  │   ├── ubuntu_images/images/
  │   ├── win_mac_images/images/
  │   └── bagel_format/
  │       ├── vlm_data/           (agent + inverse JSONL, per-platform)
  │       ├── forward_edit_ubuntu/
  │       ├── forward_edit_winmac/
  │       └── parquet_info/
  ├── android_control/
  │   ├── images/
  │   └── bagel_format_v2/
  │       ├── vlm_data/
  │       ├── forward_edit_android_control/
  │       └── parquet_info/
  └── gui_odyssey/
      ├── screenshots/
      └── bagel_format/
          ├── (agent + inverse JSONL, at group root)
          ├── forward_edit_guiodyssey/
          └── parquet_info/
"""

import os
from pathlib import Path

from .forward_dataset import ForwardEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


# ---------------------------------------------------------------------------
# Data root configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./datasets"))


def _p(relative_path: str) -> str:
    """Resolve a relative path against DATA_ROOT."""
    return str(DATA_ROOT / relative_path)


# ---------------------------------------------------------------------------
# Dataset class registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,           # Agent trajectory (FT) + Inverse dynamics (STP)
    'forward_edit': ForwardEditIterableDataset,    # Forward dynamics (STP image generation)
}


# ---------------------------------------------------------------------------
# Dataset path & metadata registry
# ---------------------------------------------------------------------------
DATASET_INFO = {
    # =========================================================================
    # Forward Dynamics Datasets (s_t, a_t -> s_{t+1})
    # Used in Stage 1: State Transition Pretraining
    # =========================================================================
    'forward_edit': {
        'agent_ubuntu_forward': {
            'data_dir': _p('agentnet/bagel_format/forward_edit_ubuntu'),
            'num_files': 78,
            'num_total_samples': 77000,
            'parquet_info_path': _p('agentnet/bagel_format/parquet_info/forward_edit_ubuntu.json'),
        },
        'agent_winmac_forward': {
            'data_dir': _p('agentnet/bagel_format/forward_edit_winmac'),
            'num_files': 322,
            'num_total_samples': 321000,
            'parquet_info_path': _p('agentnet/bagel_format/parquet_info/forward_edit_winmac.json'),
        },
        'android_control_forward': {
            'data_dir': _p('android_control/bagel_format_v2/forward_edit_android_control'),
            'num_files': 68,
            'num_total_samples': 67315,
            'parquet_info_path': _p('android_control/bagel_format_v2/parquet_info/forward_edit_android_control.json'),
        },
        'guiodyssey_forward': {
            'data_dir': _p('gui_odyssey/bagel_format/forward_edit_guiodyssey'),
            'num_files': 96,
            'num_total_samples': 95418,
            'parquet_info_path': _p('gui_odyssey/bagel_format/parquet_info/forward_edit_guiodyssey.json'),
        },
    },

    # =========================================================================
    # SFT JSONL Datasets
    #   *_inverse : Inverse dynamics (s_t, s_{t+1} -> a_t)     — used in STP
    #   *_agent   : Agent trajectory (instruction + screenshots -> actions) — used in FT
    # =========================================================================
    'vlm_sft': {
        # ---- Inverse (STP) ----
        'agentnet_ubuntu_inverse': {
            'data_dir': _p('agentnet/ubuntu_images/images'),
            'jsonl_path': _p('agentnet/bagel_format/vlm_data/inverse_tasks__ubuntu.jsonl'),
            'num_total_samples': 77000,
        },
        'agentnet_winmac_inverse': {
            'data_dir': _p('agentnet/win_mac_images/images'),
            'jsonl_path': _p('agentnet/bagel_format/vlm_data/inverse_tasks__winmac.jsonl'),
            'num_total_samples': 321000,
        },
        'android_control_inverse': {
            'data_dir': _p('android_control/images'),
            'jsonl_path': _p('android_control/bagel_format_v2/vlm_data/inverse_tasks__android_control__l1.jsonl'),
            'num_total_samples': 67315,
        },
        'guiodyssey_inverse': {
            'data_dir': _p('gui_odyssey/screenshots'),
            'jsonl_path': _p('gui_odyssey/bagel_format/inverse_tasks__guiodyssey__high__random_split__train.jsonl'),
            'num_total_samples': 95418,
        },
        # ---- Agent (FT) ----
        'agentnet_ubuntu_agent': {
            'data_dir': _p('agentnet/ubuntu_images/images'),
            'jsonl_path': _p('agentnet/bagel_format/vlm_data/agent_tasks__ubuntu__l1.jsonl'),
            'num_total_samples': 80000,
        },
        'agentnet_winmac_agent': {
            'data_dir': _p('agentnet/win_mac_images/images'),
            'jsonl_path': _p('agentnet/bagel_format/vlm_data/agent_tasks__winmac__l1.jsonl'),
            'num_total_samples': 339000,
        },
        # ---- Trajectory-level subsamples (produced by data_preparation/subsample_agent_tasks.py, seed=42) ----
        'agentnet_winmac_agent_2k': {
            'data_dir': _p('agentnet/win_mac_images/images'),
            'jsonl_path': _p('agentnet/bagel_format/vlm_data/agent_tasks__winmac__l1__2k.jsonl'),
            'num_total_samples': 32990,
        },
        'android_control_agent_3k': {
            'data_dir': _p('android_control/images'),
            'jsonl_path': _p('android_control/bagel_format_v2/vlm_data/agent_tasks__android_control__l1__3k.jsonl'),
            'num_total_samples': 16613,
        },
        'guiodyssey_agent_1k': {
            'data_dir': _p('gui_odyssey/screenshots'),
            'jsonl_path': _p('gui_odyssey/bagel_format/agent_tasks__guiodyssey__high__random_split__train__1k.jsonl'),
            'num_total_samples': 15259,
        },
    },
}
