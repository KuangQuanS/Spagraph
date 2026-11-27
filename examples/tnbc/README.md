# TNBC Example (CID44971)

Runs the full three-stage Spagraph workflow on the basal-like triple-negative
breast cancer (TNBC) section **CID44971** (Wu et al., *Nature Genetics* 2021;
GEO GSE176078 / Zenodo Visium release).

## Usage

```bash
python examples/tnbc/run.py \
  --sc-file /path/to/CID44971_SC.h5ad \
  --st-file /path/to/CID44971_ST.h5ad \
  --output-dir results/tnbc
```

Default settings match the manuscript record in [`README.md`](../../README.md#tumor-case-study-examples):
`resolution=4.0`, `top_n_per_type=100`, `k_celltype=[20, 30, 40]` (auto-k;
manuscript selected `40`), `k_cells_per_cluster=15`, `n_spot_neighbors=8`,
`ligand_expr_threshold=3.0`, `receptor_expr_threshold=3.0`, `epochs=200`,
`batch_size=128`, Stage 1–2 `seed=42`, Stage 3 `seeds=[11, 23, 42, 67, 101]`.
