from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spagraph as spg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SCC no-LR-identity ablation for Spagraph Stage 3 on GSE144240/GSE144236."
    )
    parser.add_argument(
        "--deconv-dir",
        default=str(REPO_ROOT / "evaluate" / "data" / "GSE144236"),
        help="Directory containing Spatial_composition.csv and Spatial_spot_cell_expr.csv.",
    )
    parser.add_argument(
        "--st-h5ad",
        default=str(REPO_ROOT / "spagraph_data" / "database" / "GSE144240" / "GSE144236_P2_ST.h5ad"),
        help="Spatial h5ad input.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "evaluate" / "data" / "GSE144236" / "ablation_no_lr_identity"),
        help="Output directory for the no-LR-identity run.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    deconv_dir = Path(args.deconv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    composition_csv = deconv_dir / "Spatial_composition.csv"
    spot_cell_expr_csv = deconv_dir / "Spatial_spot_cell_expr.csv"

    if not composition_csv.exists():
        raise FileNotFoundError(f"Missing composition file: {composition_csv}")
    if not spot_cell_expr_csv.exists():
        raise FileNotFoundError(f"Missing spot-cell expression file: {spot_cell_expr_csv}")

    print("Running SCC no-LR-identity ablation with fixed Stage 3 settings:")
    print(f"  deconv_dir:         {deconv_dir}")
    print(f"  st_h5ad:            {args.st_h5ad}")
    print(f"  output_dir:         {output_dir}")
    print(f"  composition_csv:    {composition_csv}")
    print(f"  spot_cell_expr_csv: {spot_cell_expr_csv}")
    print(f"  epochs:             {args.epochs}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  seed:               {args.seed}")
    print("  ablation:           no LR identity embedding (zeroed out)")

    spg.cellcom(
        deconv_dir=str(deconv_dir),
        st_h5ad=args.st_h5ad,
        output_dir=str(output_dir),
        composition_csv=str(composition_csv),
        spot_cell_expr_csv=str(spot_cell_expr_csv),
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=3.0,
        lr_score_threshold=1.0,
        n_spot_neighbors=8,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        use_hvg_for_communication=False,
        allow_same_celltype_comm=True,
        save_lr_scores_csv=False,
        export_unified_csv=True,
        export_filtered_csv=True,
        ablation_no_lr_identity=True,
        early_stop_patience=5,
        early_stop_min_delta=1.0,
    )


if __name__ == "__main__":
    main()
