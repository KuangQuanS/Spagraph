# Spagraph Benchmarks & Core Validation

This directory contains the reproducible runners, evaluation metrics, statistical
tests, ablations, and baseline method workflows supporting the Spagraph
manuscript.

## Deconvolution benchmarks (Figures 2 and S1)

| Script | Purpose |
|---|---|
| `run_deconvolution.py` | Run Stage 1–2 across simulated datasets `Data1`–`Data32`. |
| `evaluate_deconvolution.py` | Compute per-dataset PCC, SSIM, RMSE, and Jensen–Shannon divergence. |
| `compare_rctd_wilcoxon.py` | Paired Spagraph vs. RCTD Wilcoxon signed-rank tests across 32 datasets. |
| `run_starmap.py` | Run Stage 1–2 on the STARmap benchmark (`k_cells_per_cluster=10`). |
| `run_seqfish.py` | Run Stage 1–2 on the seqFISH+ benchmark (`top_n_per_type=50`, `k_celltype=20`). |
| `baselines/deconvolution/` | Reference scripts/notebooks for DWLS, RCTD, Seurat, SPOTlight, DestVI, Stereoscope, and Tangram. |

## Cell-communication benchmarks, statistics, ablations & baselines (Figures 3, S2, S3)

| Script / Folder | Purpose |
|---|---|
| `run_synthetic_lr.py` & `summarize_synthetic_lr.py` | Fully synthetic matched-control LR benchmark and multi-seed aggregation (Fig. S2). |
| `run_semisynthetic_lr.py` & `summarize_semisynthetic_lr.py` | CID44971 semi-synthetic LR prioritization benchmark and multi-seed aggregation (Fig. S2). |
| `figure3e_statistics.py` | LR-pair-level Mann–Whitney U tests with Holm adjustment (Fig. 3e). |
| `summarize_communication_focality.py` | Recompute 5-seed spatial focality metrics from complete LR support tables (Fig. 3e). |
| `run_visium_hd.py` & `visium_hd/` | Visium HD CRC (GSE280315 P1/P2/P5) 128-µm aggregation, LR coverage check, and Stage 3 run (Fig. S3). |
| `ablations/` | Four cSCC Stage 3 ablations (edge-masking only, node-masking only, no masking, no LR identity; Fig. 3f). |
| `baselines/communication/` | Spatial CCC baseline workflows for CellChat v2, COMMOT, and Giotto (Fig. 3). |

Archival plotting recipes and historical run configuration logs remain available
in the [`paper-reproduction-20261127`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127)
and [`v1.0.0`](https://github.com/KuangQuanS/Spagraph/tree/v1.0.0) tags.
