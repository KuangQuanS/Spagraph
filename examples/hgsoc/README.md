# HGSOC Example (GSE211956 P3)

Runs the full three-stage Spagraph workflow on the high-grade serous ovarian
cancer (HGSOC) section **P3** from
[GSE211956](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211956).

## Usage

```bash
python examples/hgsoc/run.py \
  --sc-file /path/to/GSE211956_SC.h5ad \
  --st-file /path/to/GSE211956_ST_P3.h5ad \
  --output-dir results/hgsoc
```

Default settings match the manuscript record in [`README.md`](../../README.md#tumor-case-study-examples):
`resolution=4.0`, `top_n_per_type=100`, `k_celltype=[20, 30, 40]` (auto-k;
manuscript selected `30`), `k_cells_per_cluster=15`, `n_spot_neighbors=8`,
`ligand_expr_threshold=3.0`, `receptor_expr_threshold=3.0`, `epochs=200`,
`batch_size=128`, Stage 1–2 `seed=42`, Stage 3 `seeds=[11, 23, 42, 67, 101]`.
