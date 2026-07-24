#!/usr/bin/env bash
# ==============================================================================
# Download AgentNet dataset from HuggingFace
# Source: https://huggingface.co/datasets/xlangai/AgentNet
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
TARGET_DIR="${1:-${DATA_ROOT}/agentnet}"
MAX_WORKERS="${HF_MAX_WORKERS:-16}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli is required but not found in PATH." >&2
  echo "Install with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

echo "============================================"
echo " Downloading AgentNet from HuggingFace"
echo " Target: ${TARGET_DIR}"
echo "============================================"

huggingface-cli download xlangai/AgentNet \
  --repo-type dataset \
  --local-dir "${TARGET_DIR}" \
  --resume-download \
  --max-workers "${MAX_WORKERS}" \
  --local-dir-use-symlinks False

echo ""
echo "AgentNet download complete."
echo "Data root: ${TARGET_DIR}"
echo ""
echo "Expected layout:"
echo "  ${TARGET_DIR}/agentnet_win_mac_18k.jsonl"
echo "  ${TARGET_DIR}/agentnet_ubuntu_5k.jsonl"
echo "  ${TARGET_DIR}/ubuntu_images/"
echo "  ${TARGET_DIR}/win_mac_images/"
