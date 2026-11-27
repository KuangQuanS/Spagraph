# Manuscript reproduction entry points

Raw data and generated result directories are not versioned. All public entry
points accept command-line inputs or resolve paths relative to the repository.

## Deconvolution benchmarks (Figures 2 and S1)

- `evaluate/scripts/deconv/evaluate_benchmark_metrics.py`: per-dataset PCC,
  SSIM, RMSE and Jensen-Shannon divergence evaluation.
- `evaluate/scripts/deconv/total_metrics_plot.py`: aggregate simulated-data
  benchmark panels.
- `evaluate/scripts/deconv/reproduce_spagraph_rctd_wilcoxon.py`: paired
  Spagraph/RCTD comparison across the 32 simulated datasets.
- `run_notebook/run_{CID44971,GSE144240,GSE211956,GSE243275}_stage1_only.py`:
  reproducible Stage 1 entry points for the four tumour datasets.
- `run_notebook/run_STARmap.ipynb` and `run_notebook/run_simulate.ipynb`:
  STARmap/seqFISH+ and simulated-data orchestration retained from the manuscript
  workflow.

## SCC baselines and ablations (Figure 3)

- `run_notebook/gse144240_ablation/`: edge masking, node masking, no masking
  and no-LR-identity Stage 3 runs.
- `run_notebook/gse144240_cellchat_baseline/`: CellChat input preparation and
  spatial baseline runner; generated matrices and R objects are excluded.
- `run_notebook/gse144240_commot_baseline/` and
  `run_notebook/gse144240_giotto_baseline/`: COMMOT and Giotto baseline
  entry points; generated input matrices are excluded.
- `evaluate/scripts/cc_com/ablation_rank_heatmap.py` and the SCC comparison
  scripts assemble the published rank and spatial panels.
- `evaluate/scripts/cc_com/figure3e_lr_pair_statistics.py` recalculates Figure
  3e from the selected LR-pair metrics and exports the panel, statistical
  summary, excluded overlap and per-pair audit table.

Example:

```bash
python evaluate/scripts/cc_com/figure3e_lr_pair_statistics.py \
  --pair-metrics path/to/selected_top_pairs.csv \
  --output-dir results/figure3e
```

## Semisynthetic and synthetic LR benchmarks (Figure S2)

- `spagraph/analysis/semisynthetic_lr_benchmark.py`
- `spagraph/analysis/synthetic_lr_v2_benchmark.py`
- `scripts/run_semisynthetic_lr_benchmark.py`
- `scripts/run_semisynthetic_multiseed_gpu.sh`
- `scripts/summarize_semisynthetic_multiseed.py`
- `scripts/run_synthetic_lr_v2_benchmark.py`
- `scripts/run_synthetic_lr_v2_gpu.sh`
- `scripts/run_synthetic_lr_v2_multiseed_gpu.sh`
- `scripts/summarize_synthetic_lr_v2_multiseed.py`
- `scripts/plot_synthetic_lr_v2_summary.py`

The local copies of the core benchmark modules, runners and multiseed
summarizers matched the final GPU workspace byte-for-byte before the release
scripts were made path-portable.

## Visium HD CRC check (Figure S3)

`run_notebook/gse280315_visiumhd_crc/` contains LR coverage checking, 128-µm
aggregation, three section-specific Stage 3 smoke runs and the summary plot.
These scripts reuse existing published deconvolution labels; they do not
redistribute raw matrices or result tables.

## Scope exclusions

This release deliberately excludes Figure 7/TCGA experiments, Xenium pilots,
the unrelated Visium HD colon pilot, partner-swap and archetype experiments,
historical outputs, checkpoints, server logs, private runbooks and manuscript
Word files. No full model retraining or additional CCC statistical expansion
is required to reproduce the documented Figure 3e correction.


## Optional w075 analysis

Attention-only is the default, with the original aggregation and filenames.
The [README example](../README.md#optional-support-aware-ranking-w075) shows
how to apply w075 to existing `lr_pair_statistics.csv` files.

### Matched rankings from the same five runs

Both modes used the same archived outputs, LR database and eligible pair set
within each section. The following are ranks within the Spagraph candidate
set, with rank 1 indicating highest priority.

| Section | Eligible pairs | LR pair | Attention-only | w075 |
|---|---:|---|---:|---:|
| cSCC | 165 | TNC–SDC1 | 14 | 1 |
| HGSOC | 242 | THBS2–CD47 | 9 | 3 |
| HGSOC | 242 | CXCL12–CXCR4 | 41 | 6 |
| DCIS | 428 | CDH5–CDH5 | 46 | 7 |
| DCIS | 428 | CCL21–CCR7 | 24 | 10 |
| TNBC | 267 | CLEC2D–KLRB1 | 2 | 1 |
| TNBC | 267 | CXCL12–CXCR4 | 183 | 144 |

The historical cSCC attention-only analysis placed TNC–SDC1 in the top 7%
of 164 eligible pairs. The matched comparison above uses a 165-pair analysis
that also includes MMP9–LRP1.

### Leave-one-seed-out stability

We compared the five-run top 10 with the top 10 from each four-run
aggregation, holding the five-run eligible candidate set fixed. The table
gives the range of Jaccard overlaps across the five omissions and includes
ties at the rank cutoff.

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
6, 10, 10 and 10. The reference is from the published spatial analysis of
[Ferri-Borgogno et al.](https://doi.org/10.1158/0008-5472.CAN-22-1821).

Within the 257-pair TNBC common set, Spagraph w075 ranked CLEC2D–KLRB1 first.
CXCL12–CXCR4 ranked 138th with Spagraph w075, 16th with CellChat and seventh
with COMMOT. The Spagraph-only ranks above use a different denominator.

These comparisons were checked against archived per-seed and frozen ranking
tables. The repository does not distribute those inputs. Figure S2 evaluates
attention against abundance under controlled conditions.
