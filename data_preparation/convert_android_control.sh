#!/usr/bin/env bash
# ==============================================================================
# Convert AndroidControl raw data into STP training format
#
# Produces agent JSONL + inverse JSONL + forward Parquet
#
# Usage:
#   bash data_preparation/convert_android_control.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
AC_DIR="${DATA_ROOT}/android_control"

echo "============================================"
echo " AndroidControl Conversion"
echo "============================================"

echo ""
echo "Converting train split (agent + inverse + forward)..."
python3 "${SCRIPT_DIR}/android_control/convert_android_control.py" \
  --input-dir "${AC_DIR}/train/data" \
  --output-dir "${AC_DIR}/bagel_format_v2" \
  --image-dir "${AC_DIR}/images" \
  --split train

echo ""
echo "Converting test split (agent only)..."
python3 "${SCRIPT_DIR}/android_control/convert_android_control.py" \
  --input-dir "${AC_DIR}/test/data" \
  --output-dir "${AC_DIR}/bagel_format_v2_test" \
  --image-dir "${AC_DIR}/images" \
  --split test \
  --agent-only

echo ""
echo "============================================"
echo " Done. Outputs in: ${AC_DIR}/bagel_format_v2/"
echo "============================================"
