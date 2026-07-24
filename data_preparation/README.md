# Data Preparation

This directory contains scripts to download raw datasets and convert them into the training format used by State Transition Pretraining (STP).

## Overview

STP uses three publicly available GUI datasets for training, plus AgentNetBench for desktop evaluation:

| Dataset | Platform | Source | Purpose | Transitions | Trajectories |
|---------|----------|--------|---------|-------------|--------------|
| [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) | Desktop (Win/Mac/Ubuntu) | HuggingFace | Training | 320K (Win/Mac) + 77K (Ubuntu) | 18K + 5K |
| [AndroidControl](https://huggingface.co/datasets/smolagents/android-control) | Mobile (Android) | HuggingFace | Training + Eval | 67K | 12K |
| [GUI-Odyssey](https://huggingface.co/datasets/hflqf88888/GUIOdyssey) | Mobile (Android) | HuggingFace | Training + Eval | 95K | 7K |
| [AgentNetBench](https://huggingface.co/datasets/xlangai/AgentNetBench) | Desktop | HuggingFace | Eval only | — | — |

## Quick Start

```bash
# Set your data directory
export DATA_ROOT=/path/to/your/data

# Step 1: Download all datasets
bash data_preparation/download_agentnet.sh
bash data_preparation/download_android_control.sh
bash data_preparation/download_guiodyssey.sh
bash data_preparation/download_agentnetbench.sh    # eval only

# Step 2: Convert to training format
bash data_preparation/convert_agentnet.sh winmac
bash data_preparation/convert_agentnet.sh ubuntu
bash data_preparation/convert_android_control.sh
bash data_preparation/convert_guiodyssey.sh

# Step 3 (optional): subsample fine-tuning trajectories for paper main-table Groups 1a / 2 / 3
#   Group 1a — 2K Win&Mac trajectories:
python data_preparation/subsample_agent_tasks.py \
  --input  ${DATA_ROOT:-./datasets}/agentnet/bagel_format/vlm_data/agent_tasks__winmac__l1.jsonl \
  --output ${DATA_ROOT:-./datasets}/agentnet/bagel_format/vlm_data/agent_tasks__winmac__l1__2k.jsonl \
  --num-trajectories 2000 --seed 42

#   Group 2 — 3K AndroidControl trajectories:
python data_preparation/subsample_agent_tasks.py \
  --input  ${DATA_ROOT:-./datasets}/android_control/bagel_format_v2/vlm_data/agent_tasks__android_control__l1.jsonl \
  --output ${DATA_ROOT:-./datasets}/android_control/bagel_format_v2/vlm_data/agent_tasks__android_control__l1__3k.jsonl \
  --num-trajectories 3000 --seed 42

#   Group 3 — 1K GUIOdyssey trajectories:
python data_preparation/subsample_agent_tasks.py \
  --input  ${DATA_ROOT:-./datasets}/gui_odyssey/bagel_format/agent_tasks__guiodyssey__high__random_split__train.jsonl \
  --output ${DATA_ROOT:-./datasets}/gui_odyssey/bagel_format/agent_tasks__guiodyssey__high__random_split__train__1k.jsonl \
  --num-trajectories 1000 --seed 42
```

## Data Format

Each dataset is converted into three formats:

### 1. Agent JSONL (Stage 2: Trajectory Fine-Tuning)
Multi-turn conversations with task instructions and step-by-step actions.
```json
{
  "conversations": [
    {"role": "system", "content": "You are a GUI agent..."},
    {"role": "user", "content": "<image>\nTask: Open Chrome and navigate to google.com"},
    {"role": "assistant", "content": "## Action:\nClick on the Chrome icon...\n## Code:\n```python\npyautogui.click(100, 200)\n```"}
  ],
  "images": ["path/to/screenshot.png"]
}
```

### 2. Inverse JSONL (Stage 1: STP - Inverse Dynamics)
Two consecutive screenshots → infer the action performed.
```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful assistant that infers actions from screen transitions."},
    {"role": "user", "content": "Previous screen:\n<image>\n\nCurrent screen:\n<image>\n\nWhat action was performed?"},
    {"role": "assistant", "content": "## Action:\nClicked on the search bar...\n## Code:\n```python\npyautogui.click(500, 300)\n```"}
  ],
  "images": ["path/to/s_t.png", "path/to/s_t+1.png"]
}
```

### 3. Forward Parquet (Stage 1: STP - Forward Dynamics)
Current screenshot + action text → predict next screenshot (stored as Parquet with image bytes).

Schema:
- `source_image`: bytes (PNG of s_t)
- `target_image`: bytes (PNG of s_{t+1})
- `edit_prompt`: str (action description)
- `source_vit_image`: bytes (optional, ViT-preprocessed source)

## Expected Directory Layout

After running all download and conversion scripts:

```
${DATA_ROOT}/
├── agentnet/
│   ├── agentnet_win_mac_18k.jsonl       (raw trajectories)
│   ├── agentnet_ubuntu_5k.jsonl         (raw trajectories)
│   ├── ubuntu_images/images/            (screenshots)
│   ├── win_mac_images/images/           (screenshots)
│   └── bagel_format/
│       ├── vlm_data/
│       │   ├── agent_tasks__winmac__l1.jsonl
│       │   ├── agent_tasks__ubuntu__l1.jsonl
│       │   ├── inverse_tasks__winmac.jsonl
│       │   └── inverse_tasks__ubuntu.jsonl
│       ├── forward_edit_winmac/          (Parquet chunks)
│       ├── forward_edit_ubuntu/          (Parquet chunks)
│       └── parquet_info/
├── android_control/
│   ├── train/data/                       (raw Parquet shards)
│   ├── test/data/
│   ├── images/                           (extracted screenshots)
│   ├── bagel_format_v2/                  (train split)
│   │   ├── vlm_data/
│   │   │   ├── agent_tasks__android_control__l1.jsonl
│   │   │   └── inverse_tasks__android_control__l1.jsonl
│   │   ├── forward_edit_android_control/
│   │   └── parquet_info/
│   └── bagel_format_v2_test/             (test split, agent-only, for eval)
│       └── vlm_data/agent_tasks__android_control__l1.jsonl
├── gui_odyssey/
│   ├── annotations/                      (episode JSON files)
│   ├── screenshots/                      (app UI screenshots)
│   ├── splits/                           (train/test splits)
│   └── bagel_format/
│       ├── agent_tasks__guiodyssey__high__random_split__train.jsonl
│       ├── agent_tasks__guiodyssey__high__random_split__test.jsonl
│       ├── inverse_tasks__guiodyssey__high__random_split__train.jsonl
│       ├── forward_edit_guiodyssey/
│       └── parquet_info/
└── agentnetbench/
    └── test_data/                        (AgentNetBench desktop eval)
        ├── mapping.json
        ├── s_*.json
        └── images/
```

## Requirements

```
huggingface_hub   (for downloading)
pyarrow           (for Parquet conversion)
Pillow            (for image processing)
tqdm              (optional, for progress bars)
```

## Notes

- Download scripts support resumption (`--resume-download`). Re-run if interrupted.
- GUI-Odyssey download includes retry logic for rate limiting.
- Conversion scripts can be run with `--limit N` for testing with a small subset.
- Set `HF_MAX_WORKERS` environment variable to control download parallelism (default: 16).
