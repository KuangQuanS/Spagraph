"""Run valid GSE144236 communication baselines and relation-ranker V2 seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagraph.training.cellcom import run_cellcom


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=[11, 23, 42, 67, 101])
    parser.add_argument("--variants", default="C0,C3")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    deconv_dir = REPO_ROOT / "evaluate" / "data" / "GSE144236"
    st_h5ad = REPO_ROOT / "spagraph_data" / "database" / "GSE144240" / "GSE144236_P2_ST.h5ad"
    variants = [item for item in args.variants.split(",") if item]
    unknown = set(variants).difference({"C0", "C3"})
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "valid_baseline": "multi_lr_attention_split",
        "invalid_old_attention_bug_excluded": True,
        "seeds": args.seeds,
        "variants": variants,
        "target_pair_used_as_validation_only": "TNC_SDC1",
        "hardcoded_pair_bonus": False,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for variant in variants:
        for seed in args.seeds:
            output = args.output_root / variant / f"seed_{seed}"
            if (output / "lr_pair_associated_edge_statistics.csv").exists():
                continue
            run_cellcom(
                deconv_dir=str(deconv_dir),
                st_h5ad=str(st_h5ad),
                output_dir=str(output),
                composition_csv=str(deconv_dir / "Spatial_composition.csv"),
                spot_cell_expr_csv=str(deconv_dir / "Spatial_spot_cell_expr.csv"),
                ligand_expr_threshold=3.0,
                receptor_expr_threshold=3.0,
                lr_score_threshold=1.0,
                n_spot_neighbors=8,
                use_hvg_for_communication=False,
                allow_same_celltype_comm=True,
                epochs=args.epochs,
                batch_size=96,
                num_workers=4,
                seed=seed,
                device=args.device,
                save_lr_scores_csv=False,
                # Ranking uses associated-edge statistics; the unified edge
                # table is ~50 MB/seed and is unnecessary for model selection.
                export_unified_csv=False,
                export_filtered_csv=False,
                model_variant="legacy" if variant == "C0" else "relation_ranker",
                ablation_no_lr_identity=(variant == "C0"),
                lambda_relation_rank=0.0 if variant == "C0" else 0.2,
            )


if __name__ == "__main__":
    main()
