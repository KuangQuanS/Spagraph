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
    signature_init=True,
    signature_affinity_graph=True,
    signature_residual_scale=5.0,
    lambda_signature_consistency=3.0,
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

### Reference-affinity-guided residual GAT

With `signature_init=True`, Stage 2 derives a non-negative affinity matrix from
the annotated single-cell reference and observed spot expression. The matrix
initializes the graph and composition logits, while the GAT learns the final
residual correction. In logit form the default update is
`log(affinity) + 5 * tanh(GAT residual)`, with a soft consistency term rather
than a fixed output clamp. Spot-level composition truth is not read during
fitting or model selection. Setting `signature_init=False` retains the original
graph-mode workflow.

### Repeated cell-communication analysis

Stage 3 supports `n_repeats` for stochastic stability. `n_repeats=5` trains
five independent models and writes an ensemble LR ranking with score and rank
variation; `n_repeats=1` remains the faster exploratory default. Candidate
ranking combines neural attention with support, spatial specificity and
cross-run uncertainty, while retaining the raw neural score and rank in the
output. Explicit seeds can be supplied with `spg.cellcom_ensemble(...)`.

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
