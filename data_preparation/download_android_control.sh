#!/usr/bin/env bash
# ==============================================================================
# Download AndroidControl dataset from HuggingFace
# Source: https://huggingface.co/datasets/smolagents/android-control
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
TARGET_DIR="${1:-${DATA_ROOT}/android_control}"
MAX_WORKERS="${HF_MAX_WORKERS:-16}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli is required but not found in PATH." >&2
  echo "Install with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}/train" "${TARGET_DIR}/test"

echo "============================================"
echo " Downloading AndroidControl from HuggingFace"
echo " Target: ${TARGET_DIR}"
echo "============================================"

echo "Downloading train split..."
huggingface-cli download smolagents/android-control \
  --repo-type dataset \
  --include "data/train-*" \
  --local-dir "${TARGET_DIR}/train" \
  --resume-download \
  --max-workers "${MAX_WORKERS}" \
  --local-dir-use-symlinks False

echo "Downloading test split..."
huggingface-cli download smolagents/android-control \
  --repo-type dataset \
  --include "data/test-*" \
  --local-dir "${TARGET_DIR}/test" \
  --resume-download \
  --max-workers "${MAX_WORKERS}" \
  --local-dir-use-symlinks False

echo ""
echo "AndroidControl download complete."
echo "Data root: ${TARGET_DIR}"
echo ""
echo "Expected layout:"
echo "  ${TARGET_DIR}/train/data/train-*.parquet"
echo "  ${TARGET_DIR}/test/data/test-*.parquet"
