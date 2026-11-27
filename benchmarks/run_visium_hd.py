#!/usr/bin/env python3
"""Run Spagraph Stage 3 on 128-um aggregated Visium HD CRC sections (GSE280315 P1/P2/P5).

Prepare inputs first with `benchmarks/visium_hd/prepare_stage3_inputs.py`.
Default parameters match docs/parameters.md:
- n_spot_neighbors=8, ligand_expr_threshold=3.0, receptor_expr_threshold=1.0,
  lr_score_threshold=1.0, epochs=10, batch_size=4, seed=42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import spagraph as spg


SAMPLE_PREFIXES = {
    "P1": "GSM8594567_P1CRC_128um",
    "P2": "GSM8594568_P2CRC_128um",
    "P5": "GSM8594569_P5CRC_128um",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spagraph Stage 3 on Visium HD CRC 128-um aggregated sections (GSE280315)."
    )
    parser.add_argument(
        "--sample",
        choices=sorted(SAMPLE_PREFIXES),
        default="P1",
        help="Default section preset when --deconv-dir / --st-h5ad / --output-dir are omitted.",
    )
    parser.add_argument("--deconv-dir", type=Path, default=None, help="Directory with *_composition.csv and *_spot_cell_expr.csv.")
    parser.add_argument("--st-h5ad", type=Path, default=None, help="Aggregated 128-um spatial .h5ad path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Stage 3 output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (manuscript smoke check: 10).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (manuscript: 4).")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of independent Stage 3 runs (default: 1).")
    parser.add_argument("--device", default="cuda", help="Compute device ('cuda', 'cuda:0', or 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slug = args.sample.lower()
    prefix = SAMPLE_PREFIXES[args.sample]

    deconv_dir = args.deconv_dir or Path("results") / f"gse280315_visiumhd_crc_{slug}_128um_inputs"
    st_h5ad = args.st_h5ad or (deconv_dir / f"{prefix}.h5ad")
    output_dir = args.output_dir or Path("results") / f"gse280315_visiumhd_crc_{slug}_128um_cellcom"

    if not deconv_dir.is_dir() or not st_h5ad.is_file():
        raise FileNotFoundError(
            f"Prepared Visium HD inputs not found: deconv_dir={deconv_dir}, st_h5ad={st_h5ad}\n"
            "Run `python benchmarks/visium_hd/prepare_stage3_inputs.py` first or pass explicit paths."
        )

    spg.cellcom(
        deconv_dir=str(deconv_dir),
        st_h5ad=str(st_h5ad),
        output_dir=str(output_dir),
        n_spot_neighbors=8,
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=1.0,
        lr_score_threshold=1.0,
        min_comm_edges=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_repeats=args.n_repeats,
        device=args.device,
        save_lr_scores_csv=False,
        export_unified_csv=False,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
