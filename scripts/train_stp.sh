#!/usr/bin/env bash
# ==============================================================================
# Stage 1: State Transition Pretraining (STP)
#
# Jointly trains inverse dynamics (s_t, s_{t+1} -> a_t) and forward dynamics
# (s_t, a_t -> s_{t+1}) on step-level state transitions.
#
# Usage:
#   bash scripts/train_stp.sh [CONFIG_NAME]
#
# Examples:
#   bash scripts/train_stp.sh stp_agentnet_winmac
#   bash scripts/train_stp.sh stp_agentnet_ubuntu
#   CE_WEIGHT=0.5 WARMUP_STEPS=1000 CPU_OFFLOAD=false \
#     bash scripts/train_stp.sh stp_android_control
#   WARMUP_STEPS=1000 CPU_OFFLOAD=false \
#     bash scripts/train_stp.sh stp_guiodyssey
#
# Environment variables (override defaults):
#   PROJECT_ROOT   - Path to this project (default: script's parent dir)
#   DATA_ROOT      - Path to data directory (default: ./datasets)
#   MODEL_PATH     - Path to BAGEL-7B-MoT checkpoint (default: ./checkpoints/BAGEL-7B-MoT)
#   NUM_GPUS       - GPUs per node (default: 8)
#   NUM_NODES      - Number of nodes (default: 1)
#   LR             - Learning rate (default: 1e-5)
#   WARMUP_STEPS   - LR warmup steps (default: 2500; use 1000 for mobile)
#   SAVE_EVERY     - Checkpoint save interval (default: 2000)
#   CE_WEIGHT      - Weight of the inverse-dynamics CE loss (default: 0.25; 0.5 for AndroidControl)
#   CPU_OFFLOAD    - FSDP CPU offload (default: true; safe to disable on mobile-sized runs)
#   MASTER_PORT    - torchrun master port (default: 29500)
#   OUTPUT_DIR     - Results output directory (default: results/<CONFIG_NAME>)
#   EXTRA_ARGS     - Additional flags forwarded to the trainer
# ==============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
CONFIG_NAME="${1:-stp_agentnet_winmac}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/BAGEL-7B-MoT}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2500}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
CE_WEIGHT="${CE_WEIGHT:-0.25}"
CPU_OFFLOAD="${CPU_OFFLOAD:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/${CONFIG_NAME}}"

# --- Environment setup --------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DATA_ROOT="${DATA_ROOT}"
export TOKENIZERS_PARALLELISM=false

# --- Logging ------------------------------------------------------------------
echo "============================================"
echo " State Transition Pretraining (STP)"
echo "============================================"
echo " Config:       ${CONFIG_NAME}"
echo " Model:        ${MODEL_PATH}"
echo " Data root:    ${DATA_ROOT}"
echo " GPUs:         ${NUM_GPUS} x ${NUM_NODES} node(s)"
echo " LR:           ${LR}"
echo " Warmup steps: ${WARMUP_STEPS}"
echo " CE weight:    ${CE_WEIGHT}"
echo " CPU offload:  ${CPU_OFFLOAD}"
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
  --resume_from "${MODEL_PATH}" \
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
  --ce_weight "${CE_WEIGHT}" \
  --expected_num_tokens 11520 \
  --max_num_tokens 12288 \
  --max_num_tokens_per_sample 11520 \
  --num_workers 1 \
  --vit_cond_dropout_prob 0 \
  --cpu_offload "${CPU_OFFLOAD}" \
  --save_every "${SAVE_EVERY}" \
  --ema 0.995 \
  --visual_gen true \
  --visual_und true \
  ${EXTRA_ARGS:-}
