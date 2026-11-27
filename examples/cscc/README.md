# cSCC Example (GSE144236 / GSE144239 P2)

This example runs the full three-stage Spagraph workflow on the P2 cutaneous
squamous cell carcinoma (cSCC) case study from
[GSE144236](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144236)
(scRNA-seq) and
[GSE144239](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144239)
(Visium ST), within SuperSeries GSE144240.

## 1. Download compact analysis-ready inputs

```bash
python examples/cscc/download.py
```

Files are saved under `examples/cscc/data/` and verified against their `v1.0.0`
release SHA256 checksums.

## 2. Run Stage 1–3

```bash
python examples/cscc/run.py
```

By default, outputs are written to `results/cscc/{deconv,cellcom}` using the
manuscript settings (`k_celltype=40`, `batch_size=96`, `epochs=200`,
`n_repeats=5`). To enable optional w075 support-aware consensus ranking or
override paths/device:

```bash
python examples/cscc/run.py --attention-weight 0.75 --device cuda:0
```
