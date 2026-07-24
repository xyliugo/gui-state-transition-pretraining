#!/usr/bin/env bash
# ==============================================================================
# Convert GUI-Odyssey raw data into STP training format
#
# Produces agent JSONL + inverse JSONL + forward Parquet
#
# Usage:
#   bash data_preparation/convert_guiodyssey.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
GO_DIR="${DATA_ROOT}/gui_odyssey"

ANNO_DIR="${GO_DIR}/annotations"
SCREENSHOTS_DIR="${GO_DIR}/screenshots"
SPLIT_FILE="${GO_DIR}/splits/random_split.json"
OUTPUT_DIR="${GO_DIR}/bagel_format"

if [ ! -d "${ANNO_DIR}" ]; then
  echo "Error: Annotations directory not found: ${ANNO_DIR}"
  echo "Run download_guiodyssey.sh first."
  exit 1
fi

echo "============================================"
echo " GUI-Odyssey Conversion"
echo "============================================"

echo ""
echo "Converting train split (agent + inverse + forward)..."
python3 "${SCRIPT_DIR}/guiodyssey/convert_train.py" \
  --annotation-dir "${ANNO_DIR}" \
  --screenshots-dir "${SCREENSHOTS_DIR}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --partition train \
  --instruction-level high \
  --history-steps 4

echo ""
echo "Converting test split (agent only)..."
python3 "${SCRIPT_DIR}/guiodyssey/convert_test.py" \
  --annotation-dir "${ANNO_DIR}" \
  --screenshots-dir "${SCREENSHOTS_DIR}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --partition test \
  --instruction-level high \
  --history-steps 4

echo ""
echo "============================================"
echo " Done. Outputs in: ${OUTPUT_DIR}/"
echo "============================================"
ls -lh "${OUTPUT_DIR}"/*.jsonl 2>/dev/null || true
