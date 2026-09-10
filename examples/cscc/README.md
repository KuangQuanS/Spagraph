# cSCC example

This example contains the P2 cutaneous squamous cell carcinoma data used in
the Spagraph case study. The processed inputs are derived from the scRNA-seq
series [GSE144236](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144236)
and spatial transcriptomics series
[GSE144239](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144239), both
within SuperSeries GSE144240.

Download the compact, analysis-ready AnnData files from the `v1.0.0` release:

```bash
python examples/cscc/download.py
```

The files are written to `examples/cscc/data/` and checked against their
published SHA256 hashes. The spatial file retains the expression matrix and
spatial coordinates but omits the embedded high-resolution histology image,
which is not required by the Spagraph workflow.

Run the three stages:

```bash
python examples/cscc/run.py
```

Outputs are written to `results/cscc/`. Stage 3 uses five independent training
runs for the consensus ligand-receptor ranking and therefore takes longer than
the first two stages.
