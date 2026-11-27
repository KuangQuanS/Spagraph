#!/usr/bin/env python3
"""Run Spagraph Stage 1-2 deconvolution across the 32 simulated datasets (Data1-Data32).

Default parameters match the manuscript record in docs/parameters.md:
- Stage 1: top_n_per_type=100, resolution=4.0 (2.0 for Data1 and Data17),
  lambda_pseudospot_contrastive=0.01, mixture_temperature=0.15
- Stage 2: k_celltype=[20, 25, 30, 35, 40] (auto-k), k_cells_per_cluster=15,
  attention_temperature=4/3, scale_basis='all'
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import spagraph as spg


MANUSCRIPT_RESOLUTION_OVERRIDES = {1: 2.0, 17: 2.0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spagraph Stage 1-2 on simulated spatial deconvolution datasets (Data1-Data32)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing dataset1/..dataset32/ (each with scRNA.h5ad and Spatial.h5ad).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/simulated_deconvolution"),
        help="Output root directory (creates Data<id>/ subdirectories).",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        nargs="+",
        default=list(range(1, 33)),
        help="Dataset indices to run (default: 1..32).",
    )
    parser.add_argument(
        "--k-celltype",
        type=int,
        nargs="+",
        default=[20, 25, 30, 35, 40],
        help="Stage 2 k_celltype candidates or single fixed value (default: 20 25 30 35 40).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Override Leiden resolution for all datasets (default: manuscript record, 4.0 except 2.0 for Data1/Data17).",
    )
    parser.add_argument("--k-cells-per-cluster", type=int, default=15, help="Dynamic cluster k (default: 15).")
    parser.add_argument("--save-all-trials", action="store_true", help="Save per-k composition files during auto-k.")
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    k_celltype = args.k_celltype[0] if len(args.k_celltype) == 1 else list(args.k_celltype)

    for dataset_id in args.datasets:
        dataset_dir = args.data_root / f"dataset{dataset_id}"
        sc_file = dataset_dir / "scRNA.h5ad"
        st_file = dataset_dir / "Spatial.h5ad"
        if not sc_file.is_file() or not st_file.is_file():
            raise FileNotFoundError(f"Missing inputs for dataset{dataset_id}: sc={sc_file}, st={st_file}")

        output_dir = args.output_root / f"Data{dataset_id}"
        resolution = (
            args.resolution
            if args.resolution is not None
            else MANUSCRIPT_RESOLUTION_OVERRIDES.get(dataset_id, 4.0)
        )

        artifacts = spg.vae(
            sc_file=str(sc_file),
            st_file=str(st_file),
            output_dir=str(output_dir),
            resolution=resolution,
            top_n_per_type=100,
            lambda_pseudospot_contrastive=0.01,
            mixture_temperature=0.15,
            device=args.device,
            seed=args.seed,
        )

        spg.deconv(
            vae=artifacts,
            st_file=str(st_file),
            output_dir=str(output_dir),
            k_celltype=k_celltype,
            k_cells_per_cluster=args.k_cells_per_cluster,
            attention_temperature=4.0 / 3.0,
            scale_basis="all",
            save_all_trials=args.save_all_trials,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
