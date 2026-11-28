#!/usr/bin/env python3
"""Run the complete Spagraph Stage 1-3 workflow on the TNBC (CID44971) case study.

Default parameters match the manuscript settings:
- Stage 1: resolution=4.0, top_n_per_type=100, lambda_pseudospot_contrastive=0.01, mixture_temperature=0.15
- Stage 2: k_celltype=[20, 30, 40] (manuscript selected: 40), k_cells_per_cluster=15, attention_temperature=4/3, lambda_gene_pearson=1.0, lambda_gene_cosine=5.0, scale_basis='all'
- Stage 3: n_spot_neighbors=8, ligand/receptor thresholds=3.0, lr_score_threshold=1.0, epochs=200, batch_size=128
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import spagraph as spg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "tnbc"
MANUSCRIPT_STAGE3_SEEDS = [11, 23, 42, 67, 101]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spagraph Stage 1-3 on the basal-like TNBC case study (CID44971)."
    )
    parser.add_argument("--sc-file", type=Path, required=True, help="Reference scRNA-seq .h5ad path (CID44971).")
    parser.add_argument("--st-file", type=Path, required=True, help="Spatial transcriptomics .h5ad path (CID44971).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output root directory.")
    parser.add_argument(
        "--k-celltype",
        type=int,
        nargs="+",
        default=[20, 30, 40],
        help="Stage 2 k_celltype candidates or fixed value (manuscript: 20 30 40; selected: 40).",
    )
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Stage 1-2 (manuscript: 42).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit Stage 3 seeds (default: 11 23 42 67 101 matching the manuscript).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Optional number of Stage 3 repeats when generating seeds from --seed (default: uses --seeds).",
    )
    parser.add_argument(
        "--attention-weight",
        type=float,
        default=1.0,
        help="Stage 3 consensus attention weight in [0, 1] (1.0=attention-only, 0.75=w075).",
    )
    parser.add_argument("--stage3-epochs", type=int, default=200, help="Stage 3 epochs (manuscript: 200).")
    parser.add_argument("--stage3-batch-size", type=int, default=128, help="Stage 3 batch size (manuscript: 128).")
    parser.add_argument("--skip-cellcom", action="store_true", help="Stop after Stage 2 deconvolution.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.sc_file.is_file() or not args.st_file.is_file():
        raise FileNotFoundError(f"Input files not found: sc={args.sc_file}, st={args.st_file}")

    deconv_dir = args.output_dir / "deconv"
    cellcom_dir = args.output_dir / "cellcom"
    k_celltype = args.k_celltype[0] if len(args.k_celltype) == 1 else list(args.k_celltype)

    artifacts = spg.vae(
        sc_file=str(args.sc_file),
        st_file=str(args.st_file),
        output_dir=str(deconv_dir),
        resolution=4.0,
        top_n_per_type=100,
        lambda_pseudospot_contrastive=0.01,
        mixture_temperature=0.15,
        device=args.device,
        seed=args.seed,
    )

    spg.deconv(
        vae=artifacts,
        st_file=str(args.st_file),
        output_dir=str(deconv_dir),
        k_celltype=k_celltype,
        k_cells_per_cluster=15,
        attention_temperature=4.0 / 3.0,
        lambda_gene_pearson=1.0,
        lambda_gene_cosine=5.0,
        scale_basis="all",
        save_reconstructed_genes=True,
        device=args.device,
        seed=args.seed,
    )

    if args.skip_cellcom:
        return

    cellcom_kwargs = {
        "deconv_dir": str(deconv_dir),
        "st_h5ad": str(args.st_file),
        "output_dir": str(cellcom_dir),
        "n_spot_neighbors": 8,
        "ligand_expr_threshold": 3.0,
        "receptor_expr_threshold": 3.0,
        "lr_score_threshold": 1.0,
        "allow_same_celltype_comm": True,
        "epochs": args.stage3_epochs,
        "batch_size": args.stage3_batch_size,
        "seed": args.seed,
        "export_unified_csv": False,
        "attention_weight": args.attention_weight,
    }
    if args.seeds is not None:
        if args.n_repeats is not None and args.n_repeats != len(args.seeds):
            raise ValueError("--n-repeats must match the number of values in --seeds")
        cellcom_kwargs["seeds"] = list(args.seeds)
        cellcom_kwargs["n_repeats"] = len(args.seeds)
    elif args.n_repeats is not None:
        cellcom_kwargs["n_repeats"] = args.n_repeats
    else:
        cellcom_kwargs["seeds"] = list(MANUSCRIPT_STAGE3_SEEDS)
        cellcom_kwargs["n_repeats"] = len(MANUSCRIPT_STAGE3_SEEDS)
    if args.device is not None:
        cellcom_kwargs["device"] = args.device
    spg.cellcom(**cellcom_kwargs)


if __name__ == "__main__":
    main()
