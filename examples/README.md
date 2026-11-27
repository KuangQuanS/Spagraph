# Tumor Case-Study Examples

Each subdirectory provides a standalone Python script that runs the complete
three-stage Spagraph workflow (`spg.vae` -> `spg.deconv` -> `spg.cellcom`)
using only the public API and the explicit manuscript parameters recorded in
[`docs/parameters.md`](../docs/parameters.md).

| Case study | Directory | Dataset | Stage 2 `k_celltype` | Stage 3 `batch_size` |
|---|---|---|---|---:|
| Cutaneous squamous cell carcinoma (cSCC) | [`cscc/`](cscc/) | GSE144236 + GSE144239 (P2) | `40` (fixed) | `96` |
| Basal-like triple-negative breast cancer (TNBC) | [`tnbc/`](tnbc/) | CID44971 | `[20, 30, 40]` (selected: 40) | `128` |
| Ductal carcinoma in situ (DCIS) | [`dcis/`](dcis/) | GSE243275 | `20` (fixed) | `64` |
| High-grade serous ovarian cancer (HGSOC) | [`hgsoc/`](hgsoc/) | GSE211956 (P3) | `[20, 30, 40]` (selected: 30) | `128` |

## Common CLI options

Every case-study script accepts:

- `--sc-file`: path to the reference scRNA-seq `.h5ad` file
- `--st-file`: path to the spatial transcriptomics `.h5ad` file (`obsm["spatial"]` required)
- `--output-dir`: output directory (creates `deconv/` and `cellcom/` subdirectories)
- `--device`: compute device (`cuda`, `cuda:0`, `cpu`; defaults to automatic CUDA detection)
- `--seed`: random seed for Stage 1–2 (default: `42`)
- `--seeds`: explicit Stage 3 random seeds (default: `11 23 42 67 101` matching the manuscript)
- `--k-celltype`: one integer for fixed `k`, or multiple integers for auto-k selection
- `--n-repeats`: optional repeat count when generating Stage 3 seeds from `--seed`
- `--attention-weight`: Stage 3 consensus attention weight in `[0, 1]` (default: `1.0`; use `0.75` for w075 support-aware consensus)
- `--skip-cellcom`: stop after Stage 2 deconvolution
