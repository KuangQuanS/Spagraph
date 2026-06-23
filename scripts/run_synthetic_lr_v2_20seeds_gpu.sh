#!/usr/bin/env bash
set -euo pipefail

cd /home/maweicheng/Spagraph

root="${1:-results/synthetic_lr_v2_context_20seeds}"
epochs="${2:-30}"
seeds=(11 17 23 29 37 42 53 61 67 73 79 83 89 97 101 107 113 127 131 139)

mkdir -p "${root}" analysis
echo "Started: $(date -Is)"
echo "Seeds: ${seeds[*]}"

for seed in "${seeds[@]}"; do
    if [[ -f "${root}/seed_${seed}/synthetic_v2_metrics.csv" ]]; then
        echo "Skipping completed seed ${seed}: $(date -Is)"
        continue
    fi
    echo "Running seed ${seed}: $(date -Is)"
    conda run --no-capture-output -n dl python scripts/run_synthetic_lr_v2_benchmark.py \
        --output "${root}/seed_${seed}" \
        --grid-side 22 \
        --epochs "${epochs}" \
        --seed "${seed}" \
        --device cuda
done

conda run --no-capture-output -n dl python scripts/summarize_synthetic_lr_v2_multiseed.py \
    --root "${root}" \
    --seeds "${seeds[@]}"

echo "Finished: $(date -Is)"
