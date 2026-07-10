#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
dataset="${1:-scc}"
seeds=(11 23 42 67 101)

case "$dataset" in
  scc)
    root="results/semisynthetic_lr_scc_multiseed"
    composition="evaluate/data/GSE144236/Spatial_composition.csv"
    expression="evaluate/data/GSE144236/Spatial_spot_cell_expr.csv"
    h5ad="spagraph_data/database/GSE144240/GSE144236_P2_ST.h5ad"
    ;;
  cid44971)
    root="results/semisynthetic_lr_cid44971_multiseed"
    composition="evaluate/data/CID44971/CID44971_ST_composition.csv"
    expression="evaluate/data/CID44971/CID44971_ST_spot_cell_expr.csv"
    h5ad="spagraph_data/database/Wu/CID44971/CID44971_ST.h5ad"
    ;;
  *)
    echo "Unknown dataset: $dataset"
    exit 2
    ;;
esac

mkdir -p analysis "$root"
exec >"analysis/semisynthetic_${dataset}_multiseed.log" 2>&1
echo "Started $dataset: $(date -Is)"

for seed in "${seeds[@]}"; do
  output="$root/seed_$seed"
  echo "Running seed $seed: $(date -Is)"
  conda run -n dl python scripts/run_semisynthetic_lr_benchmark.py \
    --output "$output" \
    --composition-csv "$composition" \
    --spot-cell-expr-csv "$expression" \
    --st-h5ad "$h5ad" \
    --epochs 30 \
    --seed "$seed" \
    --device cuda

  if [[ "$seed" != "42" ]]; then
    rm -f "$output/data/semisynthetic_spot_cell_expr.csv"
    rm -f "$output/cellcom/lr_communication.csv"
    rm -f "$output/cellcom/lr_scores.csv"
  fi
done

conda run -n dl python scripts/summarize_semisynthetic_multiseed.py \
  --root "$root" \
  --seeds "${seeds[@]}"

echo "Finished $dataset: $(date -Is)"
