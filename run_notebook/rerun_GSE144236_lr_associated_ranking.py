"""Re-run GSE144236 Stage 3 with the legacy aggregation and paper parameters."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagraph.training.cellcom import run_cellcom


DECONV_DIR = REPO_ROOT / "evaluate" / "data" / "GSE144236"
OUTPUT_DIR = (
    REPO_ROOT
    / "evaluate"
    / "data"
    / "GSE144236_lr_associated_ranking_recheck"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_cellcom(
        deconv_dir=str(DECONV_DIR),
        st_h5ad=str(
            REPO_ROOT
            / "spagraph_data"
            / "database"
            / "GSE144240"
            / "GSE144236_P2_ST.h5ad"
        ),
        output_dir=str(OUTPUT_DIR),
        composition_csv=str(DECONV_DIR / "Spatial_composition.csv"),
        spot_cell_expr_csv=str(DECONV_DIR / "Spatial_spot_cell_expr.csv"),
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=3.0,
        n_spot_neighbors=8,
        allow_same_celltype_comm=True,
        epochs=200,
        batch_size=96,
        num_workers=4,
        seed=42,
        device="cuda:1",
        save_lr_scores_csv=False,
        export_unified_csv=True,
        export_filtered_csv=False,
        ablation_no_lr_identity=True,
    )


if __name__ == "__main__":
    main()
