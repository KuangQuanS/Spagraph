#!/usr/bin/env python3
"""Run Spagraph Stage 1-2 deconvolution on the STARmap benchmark dataset.

Default parameters match docs/parameters.md:
- Stage 1: resolution=4.0, top_n_per_type=100, lambda_pseudospot_contrastive=0.01, mixture_temperature=0.15
- Stage 2: k_celltype=[20, 25, 30, 35, 40] (manuscript selected: 40), k_cells_per_cluster=10,
  attention_temperature=4/3, scale_basis='all'
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import spagraph as spg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spagraph Stage 1-2 on the STARmap benchmark.")
    parser.add_argument("--sc-file", type=Path, required=True, help="STARmap reference scRNA-seq .h5ad path.")
    parser.add_argument("--st-file", type=Path, required=True, help="STARmap spatial .h5ad path.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/starmap"), help="Output directory.")
    parser.add_argument(
        "--k-celltype",
        type=int,
        nargs="+",
        default=[20, 25, 30, 35, 40],
        help="Stage 2 k_celltype candidates or single fixed value (manuscript selected: 40).",
    )
    parser.add_argument("--k-cells-per-cluster", type=int, default=10, help="Dynamic cluster k (manuscript: 10).")
    parser.add_argument("--save-reconstructed-genes", action="store_true", help="Export reconstructed spot-cell expression.")
    parser.add_argument("--save-all-trials", action="store_true", help="Save all auto-k trial outputs.")
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.sc_file.is_file() or not args.st_file.is_file():
        raise FileNotFoundError(f"Input files not found: sc={args.sc_file}, st={args.st_file}")

    k_celltype = args.k_celltype[0] if len(args.k_celltype) == 1 else list(args.k_celltype)

    artifacts = spg.vae(
        sc_file=str(args.sc_file),
        st_file=str(args.st_file),
        output_dir=str(args.output_dir),
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
        output_dir=str(args.output_dir),
        k_celltype=k_celltype,
        k_cells_per_cluster=args.k_cells_per_cluster,
        attention_temperature=4.0 / 3.0,
        scale_basis="all",
        save_reconstructed_genes=args.save_reconstructed_genes,
        save_all_trials=args.save_all_trials,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
