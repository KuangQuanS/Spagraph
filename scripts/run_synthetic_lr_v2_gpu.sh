#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
mkdir -p analysis results

output="${1:-results/synthetic_lr_v2_seed42}"
seed="${2:-42}"
epochs="${3:-30}"

echo "Started: $(date -Is)"
echo "Output: ${output}"
echo "Seed: ${seed}"
conda run --no-capture-output -n dl python scripts/run_synthetic_lr_v2_benchmark.py \
    --output "${output}" \
    --grid-side 22 \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --device cuda
echo "Finished: $(date -Is)"
