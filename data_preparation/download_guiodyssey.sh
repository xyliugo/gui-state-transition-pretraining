#!/usr/bin/env bash
# ==============================================================================
# Download GUI-Odyssey dataset from HuggingFace
# Source: https://huggingface.co/datasets/hflqf88888/GUIOdyssey
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/datasets}"
TARGET_DIR="${1:-${DATA_ROOT}/gui_odyssey}"
MAX_WORKERS="${HF_MAX_WORKERS:-16}"
MAX_RETRIES="${HF_MAX_RETRIES:-20}"
RETRY_SLEEP_SECONDS="${HF_RETRY_SLEEP_SECONDS:-45}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli is required but not found in PATH." >&2
  echo "Install with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

download() {
  local pattern="$1"
  local attempt=1
  while true; do
    echo "Downloading ${pattern} from hflqf88888/GUIOdyssey into ${TARGET_DIR} (attempt ${attempt}/${MAX_RETRIES})"
    if huggingface-cli download "hflqf88888/GUIOdyssey" \
      --repo-type dataset \
      --include "${pattern}" \
      --local-dir "${TARGET_DIR}" \
      --resume-download \
      --max-workers "${MAX_WORKERS}" \
      --local-dir-use-symlinks False; then
      return 0
    fi

    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
      echo "Download failed for ${pattern} after ${MAX_RETRIES} attempts." >&2
      return 1
    fi

    echo "Download hit an error, likely rate limiting. Sleeping ${RETRY_SLEEP_SECONDS}s before retry."
    sleep "${RETRY_SLEEP_SECONDS}"
    attempt=$((attempt + 1))
  done
}

echo "============================================"
echo " Downloading GUI-Odyssey from HuggingFace"
echo " Target: ${TARGET_DIR}"
echo "============================================"

download "annotations/*"
download "screenshots/*"
download "splits/*.json"

echo ""
echo "GUI-Odyssey download complete."
echo "Data root: ${TARGET_DIR}"
echo ""
echo "Expected layout:"
echo "  ${TARGET_DIR}/annotations/"
echo "  ${TARGET_DIR}/screenshots/"
echo "  ${TARGET_DIR}/splits/"
