#!/usr/bin/env python3
"""Run the complete Spagraph workflow on the cSCC P2 example."""

from pathlib import Path

import spagraph as spg


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "cscc" / "data"
RESULTS = ROOT / "results" / "cscc"
SC_FILE = DATA / "GSE144236_P2_SC.h5ad"
ST_FILE = DATA / "GSE144239_P2_ST.h5ad"


def main() -> None:
    if not SC_FILE.exists() or not ST_FILE.exists():
        raise FileNotFoundError("Run `python examples/cscc/download.py` first.")

    deconv_dir = RESULTS / "deconv"
    cellcom_dir = RESULTS / "cellcom"

    artifacts = spg.vae(
        sc_file=str(SC_FILE),
        st_file=str(ST_FILE),
        output_dir=str(deconv_dir),
        resolution=4.0,
        lambda_pseudospot_contrastive=0.01,
        mixture_temperature=0.15,
        seed=42,
    )
    spg.deconv(
        vae=artifacts,
        st_file=str(ST_FILE),
        output_dir=str(deconv_dir),
        k_celltype=[20, 25, 30, 35, 40],
        k_cells_per_cluster=15,
        attention_temperature=4 / 3,
        save_reconstructed_genes=True,
        seed=42,
    )
    spg.cellcom(
        deconv_dir=str(deconv_dir),
        st_h5ad=str(ST_FILE),
        output_dir=str(cellcom_dir),
        n_spot_neighbors=8,
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=3.0,
        epochs=200,
        seed=42,
        n_repeats=5,
        export_unified_csv=False,
    )


if __name__ == "__main__":
    main()
