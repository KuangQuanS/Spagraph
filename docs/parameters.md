# Manuscript parameter record

This document records the parameter settings used for the Spagraph manuscript.
Values were compiled from saved run configurations and the scripts used to
generate the reported results. The companion [`parameters.xlsx`](parameters.xlsx)
provides the same record in spreadsheet format.

## Shared model settings

| Stage | Parameter | Manuscript value | Notes | Source |
|---|---:|---:|---|---|
| Stage 1 | epochs | 300 | All recorded final runs | GPU `config_vae.txt` |
| Stage 1 | learning rate | 5e-4 | Adam | GPU `config_vae.txt` |
| Stage 1 | batch size | 512 | Real and recorded simulated runs | GPU `config_vae.txt` |
| Stage 1 | hidden dimensions | 512, 256 | Dual decoder | GPU `config_vae.txt` |
| Stage 1 | latent dimension | 256 | — | GPU `config_vae.txt` |
| Stage 1 | beta / lambda MMD | 0.1 / 0.03 | MSE reconstruction | GPU `config_vae.txt` |
| Stage 1 | pseudo-spot mixture loss / temperature | 0.01 / 0.15 | 4,096 training and 1,024 validation pseudo-spots | GPU mixture confirmation scripts, 20–21 August 2026 |
| Stage 1 | seed | 42 | Unless explicitly overridden | GPU `config_vae.txt` |
| Stage 2 | epochs / learning rate | 300 / 5e-3 | — | GPU `config_deconv.txt` |
| Stage 2 | spatial k / weight threshold | 5 / 0.001 | `scale_basis=all` | GPU `config_deconv.txt` |
| Stage 2 | GAT | hidden 512; 4 layers; 4 heads; dropout 0.1 | — | GPU `config_deconv.txt` |
| Stage 2 | attention temperature | 4/3 | Equivalent to row-normalized composition power gamma 0.75 | GPU mixture confirmation scripts, 20–21 August 2026 |
| Stage 2 | loss weights | spot Pearson 1; spot MSE 0; spot cosine 5; gene Pearson 1; gene cosine 5; regularization 0.1; sparsity 0; proportion 0.01; signature consistency 0 | Canonical no-signature model | GPU `config_deconv.txt` and mixture confirmation scripts |
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
| CID44971 | basal-like TNBC | 4.0 | 100 | 59 | 1,758 | 20, 30, 40 | 40 | 15 | GPU `spagraph_data/evaluate/CID44971/config_{vae,deconv}.txt` |
| GSE243275 | DCIS | 4.0 | 100 | 70 | 2,130 | fixed | 20 | 15 | GPU `spagraph_data/evaluate/GSE243275/config_{vae,deconv}.txt` |
| GSE144236 | cSCC | 4.0 | 100 | 63 | 1,807 | fixed | 40 | 15 | GPU `spagraph_data/evaluate/GSE144240/config_{vae,deconv}.txt` |
| STARmap | deconvolution benchmark | 4.0 | 100 | 89 | 817 | 20, 25, 30, 35, 40 | 40 | 10 | GPU `spagraph_data/evaluate/STARmap/config_{vae,deconv}.txt` |
| seqFISH+ | deconvolution benchmark | 4.0 | 50 | 68 | 1,301 | fixed | 20 | 15 | GPU `spagraph_data/evaluate/seqFISH/config_{vae,deconv}.txt` |

## Stage 3 case-study settings

| Dataset | Ligand threshold | Receptor threshold | LR-score threshold | Spot neighbors | Epochs | Batch size | Seed | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GSE211956 P3 (HGSOC) | 3 | 3 | 1 | 8 | 200 | 128 | 42 | `run_notebook/run_GSE211956.ipynb` |
| CID44971 (basal-like TNBC) | 3 | 3 | 1 | 8 | 200 | 128 | 42 | `run_notebook/run_CID44971.ipynb` |
| GSE243275 (DCIS) | 3 | 3 | 1 | 8 | 200 | 64 | 42 | `run_notebook/run_GSE243275.ipynb` |
| GSE144236 (cSCC), Figure 3 analysis | 3 | 3 | 1 | 8 | 200 | 96 | 42 | `run_notebook/rerun_GSE144236_lr_associated_411367e.py` |
| GSE280315 Visium HD CRC, P1/P2/P5 | 3 | 1 | 1 | 8 | 10 | 4 | 42 | `run_notebook/gse280315_visiumhd_crc/run_p*_128um_cellcom_smoke.py` |

Expression thresholds are in CP10k space and the LR-score threshold is in the
pipeline's log1p score space. The tumour case-study runs use
`allow_same_celltype_comm=True` and `attention_threshold=1`. Current Stage 3
releases score candidate LR pairs after aggregate graph message passing and
support repeated-seed ensembles.

## Simulated-data auto-k record

The manuscript evaluates Data1–Data32. Saved final configurations provide auto-k
records for 25 datasets, and contemporaneous `stage2_config.txt` files provide
the settings for Data21, Data22, Data24 and Data29. For Data27, Data28 and
Data32, the result files are available but the corresponding `k_celltype`
setting was not retained and is therefore listed as **not recorded**.

| Dataset | Recorded Leiden resolution | Final `k_celltype` | Source |
|---|---:|---:|---|
| Data1 | 2 | 35 | final GPU config |
| Data2 | 4.0 | 40 | final GPU config |
| Data3 | 4.0 | 40 | final GPU config |
| Data4 | 4.0 | 35 | final GPU config |
| Data5 | 4.0 | 40 | final GPU config |
| Data6 | 4.0 | 40 | final GPU config |
| Data7 | 4.0 | 40 | final GPU config |
| Data8 | 4.0 | 30 | final GPU config |
| Data9 | 4.0 | 40 | final GPU config |
| Data10 | 4.0 | 40 | final GPU config |
| Data11 | 4.0 | 40 | final GPU config |
| Data12 | 4.0 | 35 | final GPU config |
| Data13 | 4.0 | 35 | final GPU config |
| Data14 | 4.0 | 25 | final GPU config |
| Data15 | 4.0 | 30 | final GPU config |
| Data16 | 4.0 | 35 | final GPU config |
| Data17 | 2 | 40 | final GPU config |
| Data18 | 4.0 | 40 | final GPU config |
| Data19 | 4.0 | 35 | final GPU config |
| Data20 | 4.0 | 40 | final GPU config |
| Data21 | not recorded | 20 | legacy GPU config |
| Data22 | not recorded | 30 | legacy GPU config |
| Data23 | 4.0 | 30 | final GPU config |
| Data24 | not recorded | 20 | legacy GPU config |
| Data25 | 4.0 | 40 | final GPU config |
| Data26 | 4.0 | 30 | final GPU config |
| Data27 | not recorded | not recorded | result files only; config unavailable |
| Data28 | not recorded | not recorded | result files only; config unavailable |
| Data29 | not recorded | 20 | legacy GPU config |
| Data30 | 4.0 | 40 | final GPU config |
| Data31 | 4.0 | 20 | final GPU config |
| Data32 | not recorded | not recorded | result files only; config unavailable |

Recorded auto-k searches used candidates 20, 25, 30, 35 and 40 unless the run
configuration specified a smaller candidate set. No value is imputed when a
run configuration is unavailable.

## Current optional model settings

Current releases retain the older reference-affinity-guided Stage 2 arguments
for ablation reproduction, but the canonical model sets `signature_init=False`
and `lambda_signature_consistency=0`. For a final stability run,
`spg.deconv_ensemble(..., n_repeats=3)` reuses Stage 1 and averages independent
Stage 2 GAT residual predictions; `spg.deconv(...)` remains the single-run API.
Stage 3 supports `n_repeats=5` for final ensemble rankings while retaining
`n_repeats=1` as the exploratory default. These options are documented
separately from the recorded manuscript run configurations above.

## Figure 3e statistical specification

The corrected five-run consensus and frequency top-15 groups have no overlap
(15 pairs per group). Focality uses mean edge attention over the retained C0
seeds 11, 23, 42, 67 and 101; the unique edge sets and LR support agree across
all five runs. All 164 eligible LR support counts were checked against the
ranking input before calculating group summaries.
LR pairs can share supporting edges and are not independent biological
replicates.

| Item | Specification |
|---|---|
| Statistical unit | One LR pair |
| Per-pair aggregation | Unique directed spatial and cell-type edges supporting that LR pair |
| Ranking groups | Top 15 by mean within-run attention percentile; top 15 by unique supporting edge count |
| Overlap handling | No overlap between the two top-15 groups |
| Final sample sizes | attention n=15; frequency n=15 |
| Tests | Two-sided asymptotic Mann–Whitney U; descriptive nominal comparisons |
| Multiplicity | Holm adjustment across the two tests |
| Edge spatial focality | median 0.874156 vs 0.793058; U=175; raw P=0.00995523; Holm P=0.00995523 |
| Cell-type-pair count | median 2 vs 24; U=0; raw P=2.69542e-6; Holm P=5.39085e-6 |

The calculation is implemented in
[`scripts/summarize_communication_focality.py`](../scripts/summarize_communication_focality.py).
The resulting values and summary are stored under `results/communication/figures/`.

## Communication output compatibility

Current package runs export `communication_edge_statistics.csv` with complete
LR support and `lr_pair_statistics.csv` with single-run associated-edge ranks.
Repeated runs additionally produce `lr_pair_ensemble_statistics.csv` and a seed
manifest. The old representative-LR `lr_communication.csv` is no longer written,
including when deprecated export flags are supplied.

Use `ccc_minimal_analysis.py` and `ccc_paper_plots.py` in `evaluate/scripts/cc_com/`
for current single-run analysis. The SCC consensus replacement panels use
`scripts/plot_communication_ranks.py`, `plot_communication_abundance.py`,
`plot_communication_focality.py`, and `plot_scc_seed_stability.py`; compose the
replacement PDFs with `scripts/assemble_communication_figures.py`.

Older `lr_plot.py`, `panel_g_*.py`, `cellchat_scc_baseline_figures.py`, and
`fix_csv.py` scripts retain historical data contracts. They are archival figure
recipes, not current package entry points: do not rename a current output to
`lr_communication.csv` to make them run. Any reuse requires expansion through
`spagraph.cellcom.relation_ranker.read_associated_lr_events`, which checks
complete LR support and rejects representative-ID tables. Scripts in `trash/`
are also historical. This distinction does not revalidate historical figures.
