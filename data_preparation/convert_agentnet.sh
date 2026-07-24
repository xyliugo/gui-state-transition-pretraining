#!/usr/bin/env bash
# ==============================================================================
# Convert AgentNet raw data into STP training format
#
# Produces three outputs:
#   1. Agent JSONL  - multi-turn trajectory conversations (Stage 2)
#   2. Inverse JSONL - (s_t, s_{t+1}) -> a_t action inference (Stage 1)
#   3. Forward Parquet - (s_t, a_t) -> s_{t+1} image prediction (Stage 1)
#
# Usage:
#   bash data_preparation/convert_agentnet.sh [PLATFORM]
#
# PLATFORM: winmac (default) or ubuntu
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
PLATFORM="${1:-winmac}"

AGENTNET_DIR="${DATA_ROOT}/agentnet"

case "${PLATFORM}" in
  winmac)
    INPUT_FILE="${AGENTNET_DIR}/agentnet_win_mac_18k.jsonl"
    ;;
  ubuntu)
    INPUT_FILE="${AGENTNET_DIR}/agentnet_ubuntu_5k.jsonl"
    ;;
  *)
    echo "Error: Unknown platform '${PLATFORM}'. Use 'winmac' or 'ubuntu'."
    exit 1
    ;;
esac

if [ ! -f "${INPUT_FILE}" ]; then
  echo "Error: Input file not found: ${INPUT_FILE}"
  echo "Run download_agentnet.sh first."
  exit 1
fi

OUTPUT_BASE="${AGENTNET_DIR}/bagel_format"
mkdir -p "${OUTPUT_BASE}/vlm_data" "${OUTPUT_BASE}/forward_edit_${PLATFORM}" "${OUTPUT_BASE}/parquet_info"

echo "============================================"
echo " AgentNet Conversion: ${PLATFORM}"
echo "============================================"
echo " Input:  ${INPUT_FILE}"
echo " Output: ${OUTPUT_BASE}"
echo "============================================"

echo ""
echo "[1/3] Converting agent tasks (trajectory fine-tuning format)..."
python3 "${SCRIPT_DIR}/agentnet/convert_agent_task.py" \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_BASE}/vlm_data/agent_tasks__${PLATFORM}__l1.jsonl" \
  --image-base "${AGENTNET_DIR}"

echo ""
echo "[2/3] Converting inverse tasks (s_t, s_{t+1} -> a_t)..."
python3 "${SCRIPT_DIR}/agentnet/convert_inverse_task.py" \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_BASE}/vlm_data/inverse_tasks__${PLATFORM}.jsonl"

echo ""
echo "[3/3] Converting forward tasks (s_t, a_t -> s_{t+1})..."
python3 "${SCRIPT_DIR}/agentnet/convert_forward_task.py" \
  --input "${INPUT_FILE}" \
  --image-base "${AGENTNET_DIR}" \
  --out-dir "${OUTPUT_BASE}/forward_edit_${PLATFORM}"

echo ""
echo "============================================"
echo " Done. Outputs:"
echo "   ${OUTPUT_BASE}/vlm_data/agent_tasks__${PLATFORM}__l1.jsonl"
echo "   ${OUTPUT_BASE}/vlm_data/inverse_tasks__${PLATFORM}.jsonl"
echo "   ${OUTPUT_BASE}/forward_edit_${PLATFORM}/"
echo "============================================"
