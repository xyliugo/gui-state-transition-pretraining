#!/usr/bin/env bash
# ==============================================================================
# Evaluation Script
#
# Evaluates a trained checkpoint on GUI agent benchmarks.
#
# Usage:
#   bash scripts/eval.sh [CHECKPOINT_PATH] [BENCHMARK]
#
# Examples:
#   bash scripts/eval.sh results/ft_agentnet_winmac_2k/checkpoints/0006000 agentnetbench
#   bash scripts/eval.sh results/ft_android_control_3k/checkpoints/0003000 android_control
#   bash scripts/eval.sh results/ft_guiodyssey_1k/checkpoints/0003000     guiodyssey
#
# Supported benchmarks: agentnetbench, android_control, guiodyssey
# (Each checkpoint should be evaluated on the benchmark it was fine-tuned on.)
#
# Environment variables:
#   PROJECT_ROOT  - Path to this project
#   DATA_ROOT     - Path to data directory
#   NUM_GPUS      - Number of GPUs (default: 8)
#   TEMPERATURE   - Decoding temperature (default: 0.6)
# ==============================================================================

set -euo pipefail

CHECKPOINT_PATH="${1:?Usage: eval.sh CHECKPOINT_PATH BENCHMARK}"
BENCHMARK="${2:?Usage: eval.sh CHECKPOINT_PATH BENCHMARK (agentnetbench|android_control|guiodyssey)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
NUM_GPUS="${NUM_GPUS:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DATA_ROOT="${DATA_ROOT}"
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo " Evaluation"
echo "============================================"
echo " Checkpoint:  ${CHECKPOINT_PATH}"
echo " Benchmark:   ${BENCHMARK}"
echo " GPUs:        ${NUM_GPUS}"
echo " Temperature: ${TEMPERATURE}"
echo "============================================"

case "${BENCHMARK}" in
  agentnetbench)
    python "${PROJECT_ROOT}/eval/agentnetbench/run.py" \
      --data "${DATA_ROOT}/agentnetbench/test_data" \
      --image_dir "${DATA_ROOT}/agentnetbench/test_data/images" \
      --output "${PROJECT_ROOT}/results/eval_${BENCHMARK}" \
      --model_path "${CHECKPOINT_PATH}" \
      ${EXTRA_ARGS:-}
    ;;
  android_control|guiodyssey)
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      -m eval.mobile.evaluate_android_all \
      --model-path "${CHECKPOINT_PATH}" \
      --benchmarks "${BENCHMARK}" \
      --data-root "${DATA_ROOT}" \
      --temperature "${TEMPERATURE}" \
      ${EXTRA_ARGS:-}
    ;;
  *)
    echo "Error: Unknown benchmark '${BENCHMARK}'"
    echo "Supported: agentnetbench, android_control, guiodyssey"
    exit 1
    ;;
esac
