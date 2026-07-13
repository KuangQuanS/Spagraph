# Manuscript parameter record

This document is the corrected, public parameter record for the Spagraph
manuscript. Values were transcribed from the GPU run configurations and the
final scripts that produced the reported results. The supplementary spreadsheet
`parameter.xlsx` contains the compact table version of these settings.

Source priority is: (1) generated run configuration, (2) final result-producing
script, and (3) the manuscript-release code. The audited base revision is
`1265737c41ce6ead3ffc26d703c1b5af58d7fc50`; corrections and provenance notes
are part of the `codex/manuscript-release` branch.

## Shared model settings

| Stage | Parameter | Manuscript value | Notes | Source |
|---|---:|---:|---|---|
| Stage 1 | epochs | 300 | All recorded final runs | GPU `config_vae.txt` |
| Stage 1 | learning rate | 5e-4 | Adam | GPU `config_vae.txt` |
| Stage 1 | batch size | 512 | Real and recorded simulated runs | GPU `config_vae.txt` |
| Stage 1 | hidden dimensions | 512, 256 | Dual decoder | GPU `config_vae.txt` |
| Stage 1 | latent dimension | 256 | — | GPU `config_vae.txt` |
| Stage 1 | beta / lambda MMD | 0.1 / 0.03 | MSE reconstruction | GPU `config_vae.txt` |
| Stage 1 | seed | 42 | Unless explicitly overridden | GPU `config_vae.txt` |
| Stage 2 | epochs / learning rate | 300 / 5e-3 | — | GPU `config_deconv.txt` |
| Stage 2 | spatial k / weight threshold | 5 / 0.001 | `scale_basis=all` | GPU `config_deconv.txt` |
| Stage 2 | GAT | hidden 512; 4 layers; 4 heads; dropout 0.1 | — | GPU `config_deconv.txt` |
| Stage 2 | loss weights | spot Pearson 1; spot MSE 0; spot cosine 5; gene Pearson 1; gene cosine 5; regularization 0.1; sparsity 0; proportion 0.01 | Final manuscript runs | GPU `config_deconv.txt` |
| Stage 2 | `k_celltype` | Dataset-specific | Spot-to-cell-type graph sparsity; not `k_cells_per_cluster` | GPU `config_deconv.txt` |
| Stage 2 | `k_cells_per_cluster` | 15 (STARmap: 10) | Dynamic cluster representation | GPU `config_deconv.txt` |
| Stage 3 | graph/model | 8 spot neighbors; GAT 512,256,128; 8 heads; dropout 0.3; output 128 | Tumour case studies | Final run scripts |
| Stage 3 | training | 200 epochs; lr 1e-4; weight decay 1e-5; seed 42 | Tumour case studies | Final run scripts |
| Stage 3 | masking | edge 0.2; node 0.15; mask seed 1234 | Tumour case studies | Final run scripts and wrapper |

## Dataset-specific Stage 1 and Stage 2 settings

`k_celltype` controls spot-to-cell-type graph sparsity. It is distinct from
`k_cells_per_cluster`, which controls the number of nearest cells used for the
dynamic cluster representation.

| Dataset | Manuscript role | Leiden resolution | Markers per cluster | Recorded clusters | Shared genes | `k_celltype` candidates | Final `k_celltype` | `k_cells_per_cluster` | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| GSE211956 P3 | HGSOC | 4.0 | 100 | 62 | 1,195 | 20, 30, 40 | 30 | 15 | GPU `spagraph_data/evaluate/GSE211956/P3/config_{vae,deconv}.txt` |
| CID44971 | PDAC | 4.0 | 100 | 59 | 1,758 | 20, 30, 40 | 40 | 15 | GPU `spagraph_data/evaluate/CID44971/config_{vae,deconv}.txt` |
| GSE243275 | DCIS | 4.0 | 100 | 70 | 2,130 | fixed | 20 | 15 | GPU `spagraph_data/evaluate/GSE243275/config_{vae,deconv}.txt` |
| GSE144236 | cSCC | 4.0 | 100 | 63 | 1,807 | fixed | 40 | 15 | GPU `spagraph_data/evaluate/GSE144240/config_{vae,deconv}.txt` |
| STARmap | deconvolution benchmark | 4.0 | 100 | 89 | 817 | 20, 25, 30, 35, 40 | 40 | 10 | GPU `spagraph_data/evaluate/STARmap/config_{vae,deconv}.txt` |
| seqFISH+ | deconvolution benchmark | 4.0 | 50 | 68 | 1,301 | fixed | 20 | 15 | GPU `spagraph_data/evaluate/seqFISH/config_{vae,deconv}.txt` |

## Stage 3 case-study settings

| Dataset | Ligand threshold | Receptor threshold | LR-score threshold | Spot neighbors | Epochs | Batch size | Seed | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GSE211956 P3 (HGSOC) | 3 | 3 | 1 | 8 | 200 | 128 | 42 | `run_notebook/run_GSE211956.ipynb` |
| CID44971 (PDAC) | 3 | 3 | 1 | 8 | 200 | 128 | 42 | `run_notebook/run_CID44971.ipynb` |
| GSE243275 (DCIS) | 3 | 3 | 1 | 8 | 200 | 64 | 42 | `run_notebook/run_GSE243275.ipynb` |
| GSE144236 (cSCC), Figure 3 analysis | 3 | 3 | 1 | 8 | 200 | 96 | 42 | `run_notebook/rerun_GSE144236_lr_associated_411367e.py`; `n_repeats=5` for the final ensemble |
| GSE280315 Visium HD CRC, P1/P2/P5 | 3 | 1 | 1 | 8 | 10 | 4 | 42 | `run_notebook/gse280315_visiumhd_crc/run_p*_128um_cellcom_smoke.py` |

Expression thresholds are in CP10k space and the LR-score threshold is in the
pipeline's log1p score space. The tumour case-study runs use
`allow_same_celltype_comm=True`, `attention_threshold=1`, and
`ablation_no_lr_identity=True` in the final cSCC recheck.

## Figure 2 simulated deconvolution settings

The final Figure 2 and Supplementary Figure S1a results use the annotated
reference signature mode below. Ground-truth spot compositions are excluded
from fitting, gene selection, platform calibration, and parameter selection.

| Parameter | Final value |
|---|---|
| Reference grouping | Supplied single-cell `cell_type` annotations |
| Signature expression | Log-normalized |
| Gene selection | Cell-type-specific |
| Genes per cell type | 200 |
| `signature_only` | `True` |
| Platform calibration | Enabled |
| Calibration iterations | 5 |
| Ridge penalty | 0.0 |
| Composition power | 1.2 |
| Seed | 42 |
| Evaluation cell types | Intersection shared by truth and all eight methods within each dataset |

The composition power was selected on the predefined development split and
frozen before validation and blind-test evaluation.

## Figure 3e statistical specification

| Item | Specification |
|---|---|
| Statistical unit | One LR pair |
| Per-pair aggregation | All candidate communication edges for that LR pair |
| Ranking groups | Top 15 by attention; top 15 by frequency |
| Overlap handling | `TNC_SDC1` excluded from both groups |
| Final sample sizes | attention-only n=14; frequency-only n=14 |
| Tests | Two-sided Mann–Whitney U, two prespecified metrics |
| Multiplicity | Holm adjustment across the two tests |
| Edge spatial focality | median 0.851 vs 0.804; U=174; raw P=5.22e-4; Holm P=5.22e-4 |
| Cell-type-pair count | median 3.0 vs 28.5; U=11; raw P=6.80e-5; Holm P=1.36e-4 |

The calculation is implemented in
[`spagraph/analysis/figure3e_statistics.py`](../spagraph/analysis/figure3e_statistics.py)
and exercised by
[`tests/test_figure3e_statistics.py`](../tests/test_figure3e_statistics.py).
