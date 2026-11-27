# Manuscript reproduction entry points

Raw data and generated result directories are not versioned. The `main` branch
maintains the core `spagraph` library and the four clean tumor case-study
runners under [`examples/`](../examples/). All manuscript benchmark scripts,
baselines, ablations, statistical audit scripts, and historical figure recipes
are archived intact in the
[`paper-reproduction-20261127`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127)
and [`v1.0.0`](https://github.com/KuangQuanS/Spagraph/tree/v1.0.0) tags.

## Tumor case studies (Figures 3–6, on `main`)

Standalone Stage 1–3 runners using only the public Spagraph API and the exact
manuscript parameters (including Stage 3 seeds `11, 23, 42, 67, 101`):

- [`examples/cscc/`](../examples/cscc/) — GSE144236 + GSE144239 P2 (cSCC; includes `download.py` for compact `v1.0.0` release inputs)
- [`examples/tnbc/`](../examples/tnbc/) — CID44971 (basal-like TNBC)
- [`examples/dcis/`](../examples/dcis/) — GSE243275 (DCIS)
- [`examples/hgsoc/`](../examples/hgsoc/) — GSE211956 P3 (HGSOC)

To reproduce Stage 3 directly from the exact manuscript Stage 2 deconvolution
outputs (`*_composition.csv` and `*_spot_cell_expr.csv.gz`), download the
archived files from
[Hugging Face (`kmmuleelab/Spagraph-case-deconvolution`)](https://huggingface.co/datasets/kmmuleelab/Spagraph-case-deconvolution).

## Archived benchmark, baseline, and statistical scripts (`paper-reproduction-20261127`)

- **Deconvolution benchmarks (Figures 2 and S1):**
  - Per-dataset PCC, SSIM, RMSE, JSD evaluation and paired Spagraph/RCTD Wilcoxon test: [`evaluate/scripts/deconv/`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127/evaluate/scripts/deconv)
  - Simulated datasets (`Data1`–`Data32`) and STARmap/seqFISH+ notebooks: [`run_notebook/`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127/run_notebook)
- **SCC baselines, ablations, and Figure 3e statistics (Figure 3):**
  - Stage 3 ablations (`gse144240_ablation/`) and CellChat/COMMOT/Giotto spatial CCC baselines: [`run_notebook/`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127/run_notebook)
  - Figure 3e LR-pair Mann–Whitney U + Holm statistics: [`evaluate/scripts/cc_com/figure3e_lr_pair_statistics.py`](https://github.com/KuangQuanS/Spagraph/blob/paper-reproduction-20261127/evaluate/scripts/cc_com/figure3e_lr_pair_statistics.py) and [`scripts/summarize_communication_focality.py`](https://github.com/KuangQuanS/Spagraph/blob/paper-reproduction-20261127/scripts/summarize_communication_focality.py)
- **Semisynthetic and synthetic LR benchmarks (Figure S2):**
  - Per-seed runners and multi-seed summary scripts: [`scripts/`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127/scripts) (reusable logic remains in [`spagraph/analysis/`](../spagraph/analysis/))
- **Visium HD CRC check (Figure S3):**
  - LR coverage, 128-µm aggregation, and P1/P2/P5 Stage 3 runs: [`run_notebook/gse280315_visiumhd_crc/`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127/run_notebook/gse280315_visiumhd_crc)

## Configurable LR consensus ranking

The default attention weight is `1.0`, with the original aggregation and
filenames. The [README example](../README.md#optional-stage-2-ensemble--configurable-stage-3-consensus)
shows how to evaluate different weights on the same saved per-seed
`lr_pair_statistics.csv` files.

### Matched rankings from the same five runs

All weights used the same archived outputs, LR database and eligible pair set
within each section. Values are ranks within the Spagraph candidate set (rank 1
indicates highest priority).

| Section | Eligible pairs | LR pair | 1.0 (attention-only) | 0.75 | 0.5 |
|---|---:|---|---:|---:|---:|
| cSCC | 165 | TNC–SDC1 | 14 | 1 | 7 |
| HGSOC | 242 | THBS2–CD47 | 9 | 3 | 5 |
| HGSOC | 242 | CXCL12–CXCR4 | 41 | 6 | 3 |
| DCIS | 428 | CDH5–CDH5 | 46 | 7 | 8 |
| DCIS | 428 | CCL21–CCR7 | 24 | 10 | 23 |
| TNBC | 267 | CLEC2D–KLRB1 | 2 | 1 | 5 |
| TNBC | 267 | CXCL12–CXCR4 | 183 | 144 | 80 |

### Leave-one-seed-out stability

Range of Top-10 Jaccard overlaps between the five-run consensus and each
four-run omission (fixed five-run eligible candidate set, ties included):

| Section | Attention-only | w075 |
|---|---:|---:|
| cSCC | 0.750–1.000 | 1.000–1.000 |
| HGSOC | 0.750–1.000 | 0.818–1.000 |
| DCIS | 0.667–1.000 | 0.667–0.818 |
| TNBC | 0.429–1.000 | 0.538–1.000 |

### Common-set comparisons

For HGSOC, 18 of the 50 published stroma-to-tumor candidates from long-term
survivors were rankable in the 233-pair common set. At the top 10% cutoff,
Spagraph attention-only, Spagraph w075, CellChat and COMMOT recovered 3, 9, 6
and 2 of the 18 pairs, respectively. At the top 20% cutoff, they recovered
6, 10, 10 and 10 ([Ferri-Borgogno et al.](https://doi.org/10.1158/0008-5472.CAN-22-1821)).

Within the 257-pair TNBC common set, Spagraph w075 ranked CLEC2D–KLRB1 first,
while CXCL12–CXCR4 ranked 138th with Spagraph w075, 16th with CellChat and 7th
with COMMOT.
