#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${BUNDLE_DIR}/input"
OUTPUT_DIR="${BUNDLE_DIR}/output"
export CELLCHAT_R_LIB="${BUNDLE_DIR}/.r_libs/cellchat"

Rscript "${BUNDLE_DIR}/install_cellchat_v2.R"

Rscript "${BUNDLE_DIR}/run_cellchat_spatial.R" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --min-dominant-fraction 0.0 \
  --min-spots-per-group 10 \
  --nboot 100 \
  --workers 4
