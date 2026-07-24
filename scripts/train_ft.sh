#!/usr/bin/env bash
# ==============================================================================
# Stage 2: Trajectory Fine-Tuning
#
# Fine-tunes an STP-pretrained checkpoint on task trajectories with
# instruction supervision.
#
# Usage:
#   bash scripts/train_ft.sh [CONFIG_NAME] [STP_CHECKPOINT]
#
# Examples:
#   bash scripts/train_ft.sh ft_agentnet_winmac_2k results/stp_agentnet_winmac/checkpoints/0006000
#   bash scripts/train_ft.sh ft_android_control_3k results/stp_android_control/checkpoints/0004000
#   bash scripts/train_ft.sh ft_guiodyssey_1k       results/stp_guiodyssey/checkpoints/0004000
#
# If STP_CHECKPOINT is omitted the script fine-tunes from the base BAGEL
# checkpoint (this is the "FT w/o STP" baseline).
#
# Environment variables (override defaults):
#   PROJECT_ROOT   - Path to this project (default: script's parent dir)
#   DATA_ROOT      - Path to data directory (default: ./datasets)
#   MODEL_PATH     - Path to BAGEL-7B-MoT checkpoint (default: ./checkpoints/BAGEL-7B-MoT)
#   NUM_GPUS       - GPUs per node (default: 8)
#   NUM_NODES      - Number of nodes (default: 1)
#   LR             - Learning rate (default: 1e-5)
#   WARMUP_STEPS   - LR warmup steps (default: 500)
#   SAVE_EVERY     - Checkpoint save interval (default: 250)
#   KEEP_CKPTS     - Retain this many recent checkpoints and stop training after that many saves (default: 6)
#   MASTER_PORT    - torchrun master port (default: 29500)
#   OUTPUT_DIR     - Results output directory (default: results/<CONFIG_NAME>)
#   EXTRA_ARGS     - Additional flags forwarded to the trainer
# ==============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
CONFIG_NAME="${1:-ft_agentnet_winmac_2k}"
STP_CHECKPOINT="${2:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/BAGEL-7B-MoT}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_CKPTS="${KEEP_CKPTS:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/${CONFIG_NAME}}"

# STP checkpoint is used as the resume-from source if provided; otherwise
# resume from the base BAGEL checkpoint (i.e. FT-w/o-STP baseline).
RESUME_FROM="${STP_CHECKPOINT:-${MODEL_PATH}}"

# --- Environment setup --------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DATA_ROOT="${DATA_ROOT}"
export TOKENIZERS_PARALLELISM=false

# --- Logging ------------------------------------------------------------------
echo "============================================"
echo " Trajectory Fine-Tuning (Stage 2)"
echo "============================================"
echo " Config:       ${CONFIG_NAME}"
echo " STP ckpt:     ${STP_CHECKPOINT:-<none — FT w/o STP baseline>}"
echo " Resume from:  ${RESUME_FROM}"
echo " Model:        ${MODEL_PATH}"
echo " Data root:    ${DATA_ROOT}"
echo " GPUs:         ${NUM_GPUS} x ${NUM_NODES} node(s)"
echo " LR:           ${LR}"
echo " Warmup steps: ${WARMUP_STEPS}"
echo " Save every:   ${SAVE_EVERY} (keep ${KEEP_CKPTS})"
echo " Output:       ${OUTPUT_DIR}"
echo "============================================"

# --- Launch training ----------------------------------------------------------
torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT:-29500}" \
  "${PROJECT_ROOT}/train/pretrain_unified_navit.py" \
  --dataset_config_file "${PROJECT_ROOT}/configs/${CONFIG_NAME}.yaml" \
  --model_path "${MODEL_PATH}" \
  --vae_path "${MODEL_PATH}/ae.safetensors" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --num_shard 8 \
  --results_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
  --resume_from "${RESUME_FROM}" \
  --finetune_from_hf true \
  --auto_resume true \
  --resume_model_only true \
  --finetune_from_ema true \
  --wandb_resume allow \
  --wandb_project "stp" \
  --wandb_name "${CONFIG_NAME}" \
  --log_every 1 \
  --lr "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --expected_num_tokens 11520 \
  --max_num_tokens 12288 \
  --max_num_tokens_per_sample 11520 \
  --cpu_offload false \
  --save_every "${SAVE_EVERY}" \
  --keep_checkpoints "${KEEP_CKPTS}" \
  --stop_after_max_checkpoints true \
  --ema 0.995 \
  --visual_gen false \
  --visual_und true \
  ${EXTRA_ARGS:-}
