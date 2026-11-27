# DCIS Example (GSE243275)

Runs the full three-stage Spagraph workflow on the breast ductal carcinoma in
situ (DCIS) case study from
[GSE243275](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275)
(scRNA-seq GSM7782698 + Visium ST GSM7782699).

## Usage

```bash
python examples/dcis/run.py \
  --sc-file /path/to/GSM7782698_SC.h5ad \
  --st-file /path/to/GSM7782699_ST.h5ad \
  --output-dir results/dcis
```

Default settings match [`docs/parameters.md`](../../docs/parameters.md):
`resolution=4.0`, `top_n_per_type=100`, `k_celltype=20` (fixed),
`k_cells_per_cluster=15`, `n_spot_neighbors=8`, `ligand_expr_threshold=3.0`,
`receptor_expr_threshold=3.0`, `epochs=200`, `batch_size=64`, Stage 1–2 `seed=42`,
Stage 3 `seeds=[11, 23, 42, 67, 101]`.
