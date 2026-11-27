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

The attention-only mode is the default and its existing output filenames and
aggregation are unchanged. The optional w075 mode is a separate post-processing
analysis of per-seed `lr_pair_statistics.csv` files. Use the README example
with explicit seeds 11, 23, 42, 67 and 101. Supply the exact same upstream inputs
and LR database when comparing weights.

The following are manuscript-reported case-study observations, not results
reproduced by this PR:

| Analysis | Reported observation |
|---|---|
| cSCC historical attention-only | TNC-SDC1 in top 7% of 164 eligible pairs |
| cSCC w075 | TNC-SDC1 first of 165 eligible pairs |
| HGSOC common-set reference recovery, w075 / CellChat / COMMOT | top 10%: 9/18, 6/18, 2/18; top 20%: 10/18 for all three |
| TNBC common-set comparison | CLEC2D-KLRB1 first by w075; CXCL12-CXCR4 ranked 138th by w075, 16th by CellChat and 7th by COMMOT |

The cSCC candidate universe changed (MMP9-LRP1 was added in the revised
analysis); the historical and revised ranks are not a controlled comparison
of weights alone. The HGSOC reference list is literature-derived, not
experimental ground truth. Existing synthetic benchmarks test attention
against abundance, not necessarily the w075 composite. Full case-study
reproduction requires the original per-seed outputs, exact database version,
baseline rankings, and HGSOC reference list; those frozen inputs are not
included in this change. No new biological or robustness claims are made.
