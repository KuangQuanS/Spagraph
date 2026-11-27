# Manuscript reproduction entry points

Raw data and generated result directories are not versioned. All public entry
points accept command-line inputs or resolve paths relative to the repository.
Archival figure-styling scripts, legacy notebooks, and historical intermediate
run logs remain intact in the
[`paper-reproduction-20261127`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127)
and [`v1.0.0`](https://github.com/KuangQuanS/Spagraph/tree/v1.0.0) tags.

## Tumor case studies (Figures 3–6)

Clean Stage 1–3 runners using only the public Spagraph API and explicit
manuscript parameters live under [`examples/`](../examples/):

- [`examples/cscc/`](../examples/cscc/) — GSE144236 + GSE144239 P2 (cSCC; includes `download.py` for compact `v1.0.0` release inputs)
- [`examples/tnbc/`](../examples/tnbc/) — CID44971 (basal-like TNBC)
- [`examples/dcis/`](../examples/dcis/) — GSE243275 (DCIS)
- [`examples/hgsoc/`](../examples/hgsoc/) — GSE211956 P3 (HGSOC)

## Deconvolution benchmarks (Figures 2 and S1)

- [`benchmarks/run_deconvolution.py`](../benchmarks/run_deconvolution.py):
  Stage 1–2 runner for simulated datasets `Data1`–`Data32`.
- [`benchmarks/evaluate_deconvolution.py`](../benchmarks/evaluate_deconvolution.py):
  per-dataset PCC, SSIM, RMSE and Jensen-Shannon divergence evaluation.
- [`benchmarks/compare_rctd_wilcoxon.py`](../benchmarks/compare_rctd_wilcoxon.py):
  paired Spagraph/RCTD Wilcoxon signed-rank comparison across the 32 simulated datasets.
- [`benchmarks/run_starmap.py`](../benchmarks/run_starmap.py) and
  [`benchmarks/run_seqfish.py`](../benchmarks/run_seqfish.py):
  STARmap and seqFISH+ Stage 1–2 deconvolution runners.
- [`benchmarks/baselines/deconvolution/`](../benchmarks/baselines/deconvolution/):
  reference scripts and notebooks for DWLS, RCTD, Seurat, SPOTlight, DestVI,
  Stereoscope, and Tangram.

## SCC baselines, ablations and Figure 3e statistics (Figure 3)

- [`benchmarks/ablations/`](../benchmarks/ablations/): edge masking, node
  masking, no masking, and no-LR-identity Stage 3 runs.
- [`benchmarks/baselines/communication/cellchat/`](../benchmarks/baselines/communication/cellchat/):
  CellChat v2 input preparation and spatial baseline runner.
- [`benchmarks/baselines/communication/commot/`](../benchmarks/baselines/communication/commot/) and
  [`benchmarks/baselines/communication/giotto/`](../benchmarks/baselines/communication/giotto/):
  COMMOT and Giotto spatial CCC baseline entry points.
- [`benchmarks/figure3e_statistics.py`](../benchmarks/figure3e_statistics.py):
  recalculates Figure 3e from LR-pair metrics (two-sided Mann–Whitney U tests
  with Holm adjustment) and exports the statistical summary and audit table.
- [`benchmarks/summarize_communication_focality.py`](../benchmarks/summarize_communication_focality.py):
  recomputes 5-seed edge spatial focality metrics from complete LR support tables.

Example:

```bash
python benchmarks/figure3e_statistics.py \
  --pair-metrics tests/fixtures/figure3e_selected_top_pairs.csv \
  --output-dir results/figure3e
```

## Semisynthetic and synthetic LR benchmarks (Figure S2)

| Analysis | Per-seed runner | Across-seed summary |
|---|---|---|
| CID44971 semisynthetic | [run_semisynthetic_lr.py](../benchmarks/run_semisynthetic_lr.py) | [summarize_semisynthetic_lr.py](../benchmarks/summarize_semisynthetic_lr.py) |
| Fully synthetic | [run_synthetic_lr.py](../benchmarks/run_synthetic_lr.py) | [summarize_synthetic_lr.py](../benchmarks/summarize_synthetic_lr.py) |

For each run, set `--output` to `<result-root>/seed_<seed>` and pass the
corresponding `--seed`. Then call the matching summary script with
`--root <result-root> --seeds <seed list>`. Both runners default to 30 epochs.

- **Semisynthetic.** Supply the CID44971 Stage 2 composition and reconstructed
  spot-cell expression through `--composition-csv` and `--spot-cell-expr-csv`,
  plus the matching Visium section through `--st-h5ad`. Use seeds 11, 23, 42,
  67 and 101.
- **Fully synthetic.** Use `--grid-side 22` and seeds 11, 17, 23, 29, 37, 42,
  53, 61, 67, 73, 79, 83, 89, 97, 101, 107, 113, 127, 131 and 139.

Historical launch and figure-plotting recipes remain preserved in
[`paper-reproduction-20261127`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127)
and [`v1.0.0`](https://github.com/KuangQuanS/Spagraph/tree/v1.0.0).

## Visium HD CRC check (Figure S3)

[`benchmarks/visium_hd/`](../benchmarks/visium_hd/) contains LR coverage
checking and 128-µm aggregation input preparation, and
[`benchmarks/run_visium_hd.py`](../benchmarks/run_visium_hd.py) runs Stage 3 on
sections P1, P2, or P5 (`--sample P1|P2|P5`). These scripts reuse published
deconvolution labels and do not redistribute raw matrices or result tables.

## Scope exclusions

This release deliberately excludes Figure 7/TCGA experiments, Xenium pilots,
the unrelated Visium HD colon pilot, partner-swap and archetype experiments,
historical outputs, checkpoints, server logs, private runbooks and manuscript
Word files. No full model retraining or additional CCC statistical expansion
is required to reproduce the documented Figure 3e correction.


## Configurable LR consensus ranking

The default attention weight is 1.0, with the original aggregation and
filenames. The [README example](../README.md#configurable-lr-consensus-ranking)
shows how to try different weights on the same `lr_pair_statistics.csv` files.

### Matched rankings from the same five runs

All weights used the same archived outputs, LR database and eligible pair set
within each section. The following are ranks within the Spagraph candidate
set, with rank 1 indicating highest priority.

| Section | Eligible pairs | LR pair | 1.0 (attention-only) | 0.75 | 0.5 |
|---|---:|---|---:|---:|---:|
| cSCC | 165 | TNC–SDC1 | 14 | 1 | 7 |
| HGSOC | 242 | THBS2–CD47 | 9 | 3 | 5 |
| HGSOC | 242 | CXCL12–CXCR4 | 41 | 6 | 3 |
| DCIS | 428 | CDH5–CDH5 | 46 | 7 | 8 |
| DCIS | 428 | CCL21–CCR7 | 24 | 10 | 23 |
| TNBC | 267 | CLEC2D–KLRB1 | 2 | 1 | 5 |
| TNBC | 267 | CXCL12–CXCR4 | 183 | 144 | 80 |

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
