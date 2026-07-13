# Spagraph

Spagraph is a three-stage framework for joint single-cell and spatial
transcriptomics analysis. It integrates the two modalities with a variational
autoencoder (Stage 1), estimates spot-level cell composition and reconstructed
expression with a graph attention network (Stage 2), and prioritizes spatial
ligand-receptor communication with a heterogeneous graph model (Stage 3).

![Spagraph workflow](docs/assets/spagraph_workflow.png)

## Installation

Spagraph was developed with Python 3.10 and a CUDA-enabled PyTorch environment.
For library use, install the package and its declared dependencies:

```bash
git clone https://github.com/KuangQuanS/Spagraph.git
cd Spagraph
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The exact software versions used for the manuscript are pinned in
[`requirements-paper.txt`](requirements-paper.txt). Install the CUDA build of
PyTorch appropriate for your system before installing that file. The looser
ranges in `pyproject.toml` are the supported library dependencies; the pinned
file is for manuscript reproduction.

## Three-stage quick start

```python
from pathlib import Path
import spagraph as spg

sc_file = "data/single_cell.h5ad"
st_file = "data/spatial.h5ad"
deconv_dir = Path("outputs/deconv")
cellcom_dir = Path("outputs/cellcom")

# Stage 1: joint scRNA-seq/ST representation learning.
artifacts = spg.vae(
    sc_file=sc_file,
    st_file=st_file,
    output_dir=str(deconv_dir),
    resolution=4.0,
    seed=42,
)

# Stage 2: spatial deconvolution. A list triggers auto-k selection.
deconv_result = spg.deconv(
    vae=artifacts,
    st_file=st_file,
    output_dir=str(deconv_dir),
    k_celltype=[20, 25, 30, 35, 40],
    k_cells_per_cluster=15,
    save_reconstructed_genes=True,
    seed=42,
)

# Stage 3: spatial cell-cell communication.
spg.cellcom(
    deconv_dir=str(deconv_dir),
    st_h5ad=st_file,
    output_dir=str(cellcom_dir),
    n_spot_neighbors=8,
    ligand_expr_threshold=3.0,
    receptor_expr_threshold=3.0,
    epochs=200,
    seed=42,
    n_repeats=5,
    export_unified_csv=False,
)
```

Stage 1 returns an in-memory `Stage1Artifacts` object. When `output_dir` is
provided it also writes the run configuration and modality-alignment plots.
Stage 2 writes `*_composition.csv`, configuration and training diagnostics;
`save_reconstructed_genes=True` additionally writes the
`*_spot_cell_expr.csv` file required by Stage 3. Stage 3 writes filtered or
unified LR communication tables and model diagnostics under its output
directory.

### Fast annotation-guided deconvolution

When the single-cell reference already has trusted `cell_type` or `celltype`
annotations, a deterministic signature-only route can skip VAE training,
Leiden clustering, and graph construction:

```python
result = spg.signature_deconv(
    sc_file=sc_file,
    st_file=st_file,
    output_dir=str(deconv_dir),
    celltype_key="cell_type",       # auto-detected when omitted
    genes_per_celltype=200,
    platform_calibration=True,
    composition_power=1.2,
)
```

This route builds balanced cell-type-specific signatures, corrects generic
scRNA/ST platform effects without using composition truth, and returns a
non-negative spot composition whose rows sum to one. It is useful for rapid
annotated-reference analysis and repeated benchmarking. It writes the same
`*_composition.csv` shape as Stage 2, but it does not reconstruct
`*_spot_cell_expr.csv`; use the full Stage 1/2 pipeline with
`save_reconstructed_genes=True` before Stage 3 communication analysis.

`composition_power=1.2` is a deterministic simplex calibration selected on the
fixed simulated development split. It reduces diffuse low-probability mass and
was frozen before validation and blind-test evaluation. Set it to `1.0` to
recover the uncalibrated signature proportions.

For the optimized Figure 2 configuration, store the reference labels in
`sc_adata.obs["cell_type"]` or `sc_adata.obs["celltype"]` and run:

```python
artifacts = spg.vae(sc_file=sc_file, st_file=st_file, seed=42)

result = spg.deconv(
    vae=artifacts,
    st_file=st_file,
    output_dir=str(deconv_dir),
    signature_init=True,
    signature_only=True,
    reference_grouping="celltype",
    reference_signature_mode="log_normalized",
    signature_gene_selection="celltype_specific",
    signature_genes_per_celltype=200,
    signature_platform_calibration=True,
    signature_calibration_iterations=5,
    signature_ridge=0.0,
    signature_composition_power=1.2,
    use_dynamic_cluster_repr=False,
    seed=42,
)
```

This uses only the supplied single-cell annotations and observed scRNA/ST
expression. Spot-level composition truth is not read by the model. Use the
default `reference_grouping="leiden"` when no reliable annotation is available.
For a custom annotation column name, pass `celltype_key="your_column"` to
`spg.signature_deconv()`; the full Stage 1 route currently auto-detects only
`cell_type` and `celltype`.

### Stable cell-communication ranking

Stage 3 is stochastic, so LR attention and rank can vary between training
runs. For a final analysis, `n_repeats=5` runs five independent models and
reports an ensemble ranking instead of relying on one seed. With `seed=42`,
the five repeat seeds are generated deterministically. For exact control and
reproduction, specify them explicitly:

```python
result = spg.cellcom_ensemble(
    deconv_dir=str(deconv_dir),
    st_h5ad=st_file,
    output_dir=str(cellcom_dir),
    seeds=[11, 23, 42, 67, 101],
    epochs=200,
)
```

Each run is written to `seed_<seed>/`. The combined
`lr_pair_ensemble_statistics.csv` contains the mean calibrated LR score,
`score_std`, ensemble `rank`, and `rank_std`; the accompanying
`cellcom_ensemble_manifest.json` records the seeds and calibration profile.
Five repeats cost approximately five times as much as one Stage 3 run, so
`n_repeats=1` remains the default for exploratory work. Keep the edge-level
exports disabled during an ensemble unless they are needed, because enabling
them writes a full communication table for every seed.

## Inputs and outputs

- Stage 1 inputs: an scRNA-seq `.h5ad` and an ST `.h5ad` with genes as
  variables; the spatial object must contain coordinates in `obsm["spatial"]`.
- Stage 2 input: the Stage 1 artifact plus the ST file. The principal output is
  a spot-by-cluster composition table.
- Stage 3 inputs: the ST file, Stage 2 composition table, reconstructed
  spot-cell expression, and the bundled `cellchat_human.csv` LR database.
- Generated data, checkpoints and manuscript result tables are intentionally
  not versioned. Dataset paths are supplied as command-line arguments in the
  reproduction entry points.

## License

Spagraph is released under the [MIT License](LICENSE).
