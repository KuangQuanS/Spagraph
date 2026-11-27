<p align="center">
  <img src="assets/spagraph_logo.png" alt="Spagraph logo" width="260" />
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" alt="Python 3.10+" /></a>
  <a href="https://github.com/KuangQuanS/Spagraph/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/Release-v1.0.0-2ea44f.svg" alt="Release v1.0.0" /></a>
</p>

👋 **Welcome to Spagraph!**

**Spagraph** is a three-stage deep learning framework for joint single-cell RNA-seq (**scRNA-seq**) and spatial transcriptomics (**ST**) analysis. It aligns both modalities with a variational autoencoder (**Stage 1**), infers spot-level cell-type composition and cell-type-specific gene expression with a graph attention network (**Stage 2**), and prioritizes spatial ligand–receptor (LR) communication with a heterogeneous graph model (**Stage 3**).

If you find Spagraph helpful in your research, feel free to ⭐ the repository or open an issue with questions and feedback!

![Spagraph workflow](assets/spagraph_workflow.png)

## ✨ Key Features

- **Mixture-aware alignment (Stage 1):** Uses simulated pseudo-spots with known cell-type proportions to supervise joint scRNA-seq / ST latent representation learning.
- **Adaptive GAT deconvolution (Stage 2):** Combines auto-`k` reference prototype selection with graph attention over reference and spatial neighbors, reconstructing cell-type-specific expression for downstream communication inference.
- **Spatial cell–cell communication (Stage 3):** Prioritizes spatially localized LR signaling via masked edge/node reconstruction using the bundled human CellChat database (`spagraph/cellcom/cellchat_human.csv`).
- **Configurable multi-seed consensus ranking:** Supports adjustable attention vs. edge-support weighting (`attention_weight` in `[0, 1]`) directly on saved per-seed outputs without retraining.

## 📦 Installation

Spagraph targets **Python 3.10+** with a CUDA-enabled PyTorch environment:

```bash
git clone https://github.com/KuangQuanS/Spagraph.git
cd Spagraph
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 🚀 Tutorial: Three-Stage Workflow

### Input Data Requirements

> **Important:** Both `sc_file` and `st_file` must store **raw, unnormalized counts** in `adata.X`. Stage 1 estimates library-size ratios from raw counts and applies `normalize_total(target_sum=1e4)` and `log1p` internally — do **not** pass pre-normalized, log-transformed, or z-scored matrices.

- **scRNA-seq (`.h5ad`):**
  - `adata.X`: raw UMI/read counts (`cells × genes`).
  - `adata.var_names`: gene names using the same naming convention as the ST object (human gene symbols are required for the bundled Stage 3 `cellchat_human.csv` database).
  - `adata.obs`: reference cell-type labels in `cell_type` or `celltype` (auto-detected), or specify another column via `celltype_key="..."`.
- **Spatial transcriptomics (`.h5ad`):**
  - `adata.X`: raw counts (`spots × genes`).
  - `adata.var_names`: gene names matching `sc_file` (Stage 1 automatically takes the shared gene intersection).
  - `adata.obsm["spatial"]`: 2D spatial coordinates (`n_spots × 2`).

### Quick Start

```python
from pathlib import Path
import spagraph as spg

sc_file = "data/single_cell.h5ad"
st_file = "data/spatial.h5ad"
deconv_dir = Path("outputs/deconv")
cellcom_dir = Path("outputs/cellcom")

# Stage 1: Joint scRNA-seq / ST representation learning
artifacts = spg.vae(
    sc_file=sc_file,
    st_file=st_file,
    output_dir=str(deconv_dir),
    resolution=4.0,
    top_n_per_type=100,
    lambda_pseudospot_contrastive=0.01,
    mixture_temperature=0.15,
    seed=42,
)

# Stage 2: Spatial deconvolution (a list enables auto-k selection)
deconv_result = spg.deconv(
    vae=artifacts,
    st_file=st_file,
    output_dir=str(deconv_dir),
    k_celltype=[20, 25, 30, 35, 40],
    k_cells_per_cluster=15,
    attention_temperature=4 / 3,
    scale_basis="all",
    save_reconstructed_genes=True,  # Required by Stage 3
    seed=42,
)

# Stage 3: Spatial cell–cell communication (5-seed consensus)
spg.cellcom(
    deconv_dir=str(deconv_dir),
    st_h5ad=st_file,
    output_dir=str(cellcom_dir),
    n_spot_neighbors=8,
    ligand_expr_threshold=3.0,
    receptor_expr_threshold=3.0,
    lr_score_threshold=1.0,
    allow_same_celltype_comm=True,
    epochs=200,
    seeds=[11, 23, 42, 67, 101],
    export_unified_csv=False,
)
```

### Main Outputs

- **Stage 1:** Returns an in-memory `Stage1Artifacts` object and writes `config_vae.txt` plus modality-alignment plots when `output_dir` is set.
- **Stage 2:** Writes `*_composition.csv`, `config_deconv.txt`, training diagnostics, and (when `save_reconstructed_genes=True`) `*_spot_cell_expr.csv`.
- **Stage 3:** Writes `communication_edge_statistics.csv` (unique directed edges with complete LR support) and `lr_pair_statistics.csv` per run, plus `lr_pair_ensemble_statistics.csv` and `cellcom_ensemble_manifest.json` for multi-seed runs.

### Optional Stage 2 Ensemble & Configurable Stage 3 Consensus

Repeat only Stage 2 while reusing the same Stage 1 representation:

```python
deconv_result = spg.deconv_ensemble(
    vae=artifacts,
    st_file=st_file,
    output_dir="outputs/deconv_ensemble",
    n_repeats=3,
    seeds=[11, 23, 42],
)
```

For multi-seed Stage 3 runs, each run scores eligible LR pairs by
`α × attention percentile + (1 − α) × log1p(supporting edges) percentile`,
averaged across seeds:

| `attention_weight` (α) | Meaning | Output file |
|---:|---|---|
| `1.0` (default) | Attention percentile only | `lr_pair_ensemble_statistics.csv` |
| `0.75` (`ranking_mode="w075"`) | 75% attention + 25% edge support (≥10 edges/run) | `lr_pair_ensemble_w075_statistics.csv` |
| `0.5` | Equal attention and support weight (≥10 edges/run) | `lr_pair_ensemble_w0p5_statistics.csv` |

Pass `attention_weight=0.75` directly to `spg.cellcom(...)` / `spg.cellcom_ensemble(...)`,
or re-rank saved per-seed runs without retraining:

```python
from pathlib import Path
from spagraph.training.cellcom import aggregate_cellcom_seed_outputs

root = Path("outputs/cellcom")
seeds = [11, 23, 42, 67, 101]
dirs = [root / f"seed_{s}" for s in seeds]
aggregate_cellcom_seed_outputs(dirs, root, seeds, attention_weight=0.75)
```

## 🧫 Tumor Case-Study Examples

Four ready-to-run Stage 1–3 scripts under [`examples/`](examples/) implement
the tumor case studies with their exact manuscript parameters (Stage 1:
`resolution=4.0`, `top_n_per_type=100`, `lambda_pseudospot_contrastive=0.01`,
`mixture_temperature=0.15`, `seed=42`; Stage 2: `k_cells_per_cluster=15`,
`attention_temperature=4/3`, `scale_basis="all"`; Stage 3: `n_spot_neighbors=8`,
`ligand/receptor_expr_threshold=3.0`, `epochs=200`, `seeds=[11, 23, 42, 67, 101]`):

| Case study | Script | Dataset | Stage 2 `k_celltype` | Stage 3 `batch_size` |
|---|---|---|---:|---:|
| cSCC (P2) | [`examples/cscc/run.py`](examples/cscc/run.py) | GSE144236 + GSE144239 | `40` | `96` |
| TNBC | [`examples/tnbc/run.py`](examples/tnbc/run.py) | CID44971 | `[20, 30, 40]` | `128` |
| DCIS | [`examples/dcis/run.py`](examples/dcis/run.py) | GSE243275 | `20` | `64` |
| HGSOC (P3) | [`examples/hgsoc/run.py`](examples/hgsoc/run.py) | GSE211956 | `[20, 30, 40]` | `128` |

The cSCC example includes a one-command downloader for compact `v1.0.0` release inputs:

```bash
python examples/cscc/download.py
python examples/cscc/run.py
```

All four scripts accept `--sc-file`, `--st-file`, `--output-dir`, `--device`,
`--k-celltype`, `--seeds`, `--attention-weight`, and `--skip-cellcom`. See
[`examples/README.md`](examples/README.md) for full CLI options. To reproduce
Stage 3 directly from the exact manuscript Stage 2 deconvolution outputs without
re-running Stages 1–2, download the archived outputs from
[Hugging Face (`kmmuleelab/Spagraph-case-deconvolution`)](https://huggingface.co/datasets/kmmuleelab/Spagraph-case-deconvolution).

## 📚 Benchmark & Case-Study Datasets

All public datasets used for benchmarking and real-tissue case studies are
originally published in the following studies and repositories:

| Dataset | Usage in Spagraph | Data source / Accession | Original Publication |
|---|---|---|---|
| **32 Simulated ST datasets** | Deconvolution benchmark (`Data1`–`Data32`) | Spatial benchmarking suite of Li et al. | Li et al., *Nat. Methods* 2022 ([doi:10.1038/s41592-022-01480-9](https://doi.org/10.1038/s41592-022-01480-9)) |
| **STARmap** (mouse visual cortex) | Real-data deconvolution benchmark | Public STARmap release (via Li et al. benchmark) | Wang et al., *Science* 2018 ([doi:10.1126/science.aat5691](https://doi.org/10.1126/science.aat5691)) |
| **seqFISH+** (mouse cortex) | Real-data deconvolution & gene reconstruction benchmark | Public seqFISH+ release (via Li et al. benchmark) | Eng et al., *Nature* 2019 ([doi:10.1038/s41586-019-1049-y](https://doi.org/10.1038/s41586-019-1049-y)) |
| **cSCC (P2)** | Tumor case study, ablations & CCC baselines | GEO [GSE144236](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144236) / [GSE144239](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144239) ([GSE144240](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240)) | Ji et al., *Cell* 2020 ([doi:10.1016/j.cell.2020.05.039](https://doi.org/10.1016/j.cell.2020.05.039)) |
| **HGSOC (P3)** | Tumor case study & reference recovery check | GEO [GSE211956](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211956) | Denisenko et al., *Nat. Commun.* 2024 ([doi:10.1038/s41467-024-47271-y](https://doi.org/10.1038/s41467-024-47271-y)) |
| **DCIS** (breast cancer) | Tumor case study | GEO [GSE243275](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275) | Janesick et al., *Nat. Commun.* 2023 ([doi:10.1038/s41467-023-43458-x](https://doi.org/10.1038/s41467-023-43458-x)) |
| **TNBC (CID44971)** | Tumor case study & semi-synthetic LR benchmark | GEO [GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078) / Wu et al. Visium atlas | Wu et al., *Nat. Genet.* 2021 ([doi:10.1038/s41588-021-00911-1](https://doi.org/10.1038/s41588-021-00911-1)) |
| **Visium HD CRC (P1/P2/P5)** | High-resolution platform check (128 µm bins) | GEO [GSE280315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315) | Oliveira et al., *Nat. Genet.* 2025 ([doi:10.1038/s41588-025-02193-3](https://doi.org/10.1038/s41588-025-02193-3)) |

## 🗂️ Repository Layout

```text
Spagraph/
├── spagraph/               # Core Python package (Stage 1 VAE, Stage 2 GAT, Stage 3 CellCom)
├── examples/               # Four tumor case studies (cscc, tnbc, dcis, hgsoc)
├── docs/                   # Manuscript parameter record (parameters.md) and reproduction guide (reproduction.md)
├── assets/                 # Logo and workflow figure displayed in this README
├── README.md               # Welcome guide, installation, tutorial, and dataset sources
├── pyproject.toml          # Package metadata and dependencies
└── requirements.txt        # Runtime dependencies
```

## 🗄️ Paper Reproduction & Archive

- **Archived Stage 2 deconvolution outputs (4 tumor cases):** [Hugging Face (`kmmuleelab/Spagraph-case-deconvolution`)](https://huggingface.co/datasets/kmmuleelab/Spagraph-case-deconvolution)
- **Manuscript parameter record:** [`docs/parameters.md`](docs/parameters.md)
- **Reproduction guide & consensus ranking comparisons:** [`docs/reproduction.md`](docs/reproduction.md)
- **Full manuscript reproduction archive (benchmarks, baselines, ablations, figure scripts):** [`paper-reproduction-20261127`](https://github.com/KuangQuanS/Spagraph/tree/paper-reproduction-20261127)
- **Original v1.0.0 release:** [`v1.0.0`](https://github.com/KuangQuanS/Spagraph/tree/v1.0.0)

## 📜 License

Spagraph is released under the [MIT License](LICENSE).
