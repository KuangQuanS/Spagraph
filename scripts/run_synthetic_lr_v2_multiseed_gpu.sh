#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
root="${1:-results/synthetic_lr_v2_multiseed}"
epochs="${2:-30}"
mode="${3:-full}"
extra_args=()
if [[ "${mode}" == "no_lr_identity" ]]; then
    extra_args+=(--ablation-no-lr-identity)
fi
seeds=(11 23 42 67 101)

mkdir -p "${root}" analysis
echo "Started: $(date -Is)"
for seed in "${seeds[@]}"; do
    echo "Running seed ${seed}: $(date -Is)"
    conda run --no-capture-output -n dl python scripts/run_synthetic_lr_v2_benchmark.py \
        --output "${root}/seed_${seed}" \
        --grid-side 22 \
        --epochs "${epochs}" \
        --seed "${seed}" \
        --device cuda \
        "${extra_args[@]}"
done

conda run --no-capture-output -n dl python scripts/summarize_synthetic_lr_v2_multiseed.py \
    --root "${root}" \
    --seeds "${seeds[@]}"
echo "Finished: $(date -Is)"
