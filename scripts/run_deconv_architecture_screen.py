#!/usr/bin/env python3
"""Run D0-D5 deconvolution ablations on the fixed benchmark split."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SPLITS = {
    "development": [3, 11, 26, 15, 1, 32, 8, 23, 9, 22, 13, 10, 29, 31, 5, 18, 6, 2],
    "validation": [19, 30, 21, 25, 28, 12, 7],
    "test": [20, 24, 14, 16, 4, 17, 27],
    "screen": [1, 2, 3, 11, 22, 31],
    "all": list(range(1, 33)),
}

VARIANTS = {
    "D0": dict(full_graph_training=False, restore_best_state=False),
    "D1": dict(full_graph_training=True, restore_best_state=True),
    "D2": dict(full_graph_training=True, restore_best_state=True, signature_init=True, signature_only=True),
    "D3": dict(full_graph_training=True, restore_best_state=True, signature_init=True, signature_prior_strength=1.0),
    "D4": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_prior_strength=1.0,
        lambda_poisson=0.05,
        heldout_gene_fraction=0.2,
    ),
    "D5": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_prior_strength=1.0,
        lambda_poisson=0.05,
        lambda_spatial=0.05,
        spatial_temperature=0.5,
        heldout_gene_fraction=0.2,
    ),
    "D6": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
    ),
    "D7": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_prior_strength=1.0,
        gat_hidden_dim=256,
        gat_layers=3,
        lambda_poisson=0.05,
        heldout_gene_fraction=0.2,
        reference_grouping="celltype",
    ),
    "D8": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
    ),
    "D9": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="cell_normalized",
    ),
    "D10": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D11": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        signature_prior_strength=0.25,
        gat_hidden_dim=256,
        gat_layers=3,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D12": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        signature_prior_strength=1.0,
        gat_hidden_dim=256,
        gat_layers=3,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D13": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        signature_prior_strength=4.0,
        gat_hidden_dim=256,
        gat_layers=3,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D14": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_platform_calibration=False,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D15": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D16": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.01,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D17": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=2,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D18": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=10,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
    ),
    "D19": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="celltype_specific",
        signature_genes_per_celltype=50,
    ),
    "D20": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="celltype_specific",
        signature_genes_per_celltype=100,
    ),
    "D21": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="celltype_specific",
        signature_genes_per_celltype=200,
        signature_composition_power=1.0,
    ),
    "D22": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="all_shared",
    ),
    "D23": dict(
        fast_signature_path=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="celltype_specific",
        signature_genes_per_celltype=200,
        signature_composition_power=1.0,
    ),
    "D25": dict(
        full_graph_training=True,
        restore_best_state=True,
        signature_init=True,
        signature_only=True,
        signature_ridge=0.0,
        signature_platform_calibration=True,
        signature_calibration_iterations=5,
        use_dynamic_cluster_repr=False,
        reference_grouping="celltype",
        reference_signature_mode="log_normalized",
        signature_gene_selection="celltype_specific",
        signature_genes_per_celltype=200,
        signature_composition_power=1.2,
    ),
}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_rctd_pcc(repo: Path, dataset: int) -> float:
    metrics = pd.read_csv(repo / "evaluate" / "data" / f"Data{dataset}" / "metrics_ARS.csv")
    row = metrics.loc[metrics["method_name"].eq("RCTD")]
    if len(row) != 1:
        raise ValueError(f"Expected one RCTD row for Data{dataset}")
    return float(row.iloc[0]["mean_pcc"])


def evaluate_prediction(
    repo: Path, code_root: Path, dataset: int, prediction: Path, output: Path
) -> dict:
    truth = repo / "evaluate" / "data" / f"Data{dataset}" / f"dataset{dataset}_density.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(code_root / "evaluate" / "scripts" / "deconv" / "evaluate_benchmark_metrics.py"),
        "--composition_pred_csv", str(prediction),
        "--composition_true_csv", str(truth),
        "--method1_name", "Spagraph",
        "--use_intersection",
        "--output_csv", str(output),
    ]
    subprocess.run(command, cwd=repo, check=True)
    metrics = pd.read_csv(output)
    return {
        "mean_pcc": float(np.nanmean(metrics["pcc"])),
        "mean_ssim": float(np.nanmean(metrics["ssim"])),
        "mean_rmse": float(np.nanmean(metrics["rmse"])),
        "mean_js": float(np.nanmean(metrics["js"])),
        "n_celltypes": int(len(metrics)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=SPLITS, default="screen")
    parser.add_argument("--datasets", type=parse_ints)
    parser.add_argument("--seeds", type=parse_ints, default=[11, 42])
    parser.add_argument("--variants", type=lambda x: [v for v in x.split(",") if v], default=list(VARIANTS))
    parser.add_argument("--vae-epochs", type=int, default=300)
    parser.add_argument("--deconv-epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k-celltype", type=int, default=20)
    args = parser.parse_args()

    repo = args.repo.resolve()
    code_root = (args.code_root or repo).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    import spagraph as spg

    module_path = Path(spg.__file__).resolve()
    if not module_path.is_relative_to(code_root):
        raise RuntimeError(
            f"Imported spagraph from {module_path}, expected it under code root {code_root}"
        )

    datasets = args.datasets or SPLITS[args.phase]
    unknown = set(args.variants).difference(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    manifest = {
        "phase": args.phase,
        "datasets": datasets,
        "seeds": args.seeds,
        "variants": args.variants,
        "vae_epochs": args.vae_epochs,
        "deconv_epochs": args.deconv_epochs,
        "device": args.device,
        "data_repo": str(repo),
        "code_root": str(code_root),
        "spagraph_module": str(module_path),
        "split_seed": 20260711,
        "ground_truth_used_for_training": False,
        "graph_policy": "spatial coordinates when present; otherwise VAE-embedding KNN",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    summary_path = output / "deconv_vs_rctd_runs.csv"
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
    completed = {(str(r["variant"]), int(r["dataset"]), int(r["seed"])) for r in rows if r.get("status") == "ok"}

    for dataset in datasets:
        sc_file = repo / "spagraph_data" / "database" / "SimualtedSpatalData" / f"dataset{dataset}" / "scRNA.h5ad"
        st_file = repo / "spagraph_data" / "database" / "SimualtedSpatalData" / f"dataset{dataset}" / "Spatial.h5ad"
        for seed in args.seeds:
            needed = [variant for variant in args.variants if (variant, dataset, seed) not in completed]
            if not needed:
                continue
            fast_variants = {
                variant for variant in needed
                if VARIANTS[variant].get("fast_signature_path", False)
                and VARIANTS[variant].get("reference_grouping") == "celltype"
                and VARIANTS[variant].get("signature_gene_selection") in {
                    "celltype_specific", "all_shared"
                }
            }
            artifacts = None
            stage1_seconds = 0.0
            if len(fast_variants) != len(needed):
                stage1_started = time.time()
                artifacts = spg.vae(
                    sc_file=str(sc_file),
                    st_file=str(st_file),
                    n_epochs=args.vae_epochs,
                    seed=seed,
                    device=args.device,
                )
                stage1_seconds = time.time() - stage1_started
            for variant in needed:
                run_dir = output / variant / f"Data{dataset}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                row = {"variant": variant, "dataset": dataset, "seed": seed, "status": "failed"}
                try:
                    config = VARIANTS[variant]
                    if variant in fast_variants:
                        result = spg.signature_deconv(
                            sc_file=str(sc_file),
                            st_file=str(st_file),
                            output_dir=str(run_dir),
                            gene_selection=config["signature_gene_selection"],
                            genes_per_celltype=config.get("signature_genes_per_celltype", 200),
                            reference_scale=config.get("reference_signature_mode", "log_normalized"),
                            platform_calibration=config.get("signature_platform_calibration", False),
                            calibration_iterations=config.get("signature_calibration_iterations", 5),
                            ridge=config.get("signature_ridge", 1e-4),
                            composition_power=config.get("signature_composition_power", 1.2),
                        )
                    else:
                        config = {key: value for key, value in config.items() if key != "fast_signature_path"}
                        result = spg.deconv(
                            vae=artifacts,
                            st_file=str(st_file),
                            output_dir=str(run_dir),
                            n_epochs=args.deconv_epochs,
                            k_celltype=args.k_celltype,
                            seed=seed,
                            device=args.device,
                            **config,
                        )
                    prediction = Path(result["deconv_path"])
                    metrics = evaluate_prediction(
                        repo, code_root, dataset, prediction, run_dir / "composition_metrics.csv"
                    )
                    rctd_pcc = load_rctd_pcc(repo, dataset)
                    row.update(
                        status="ok",
                        runtime_seconds=time.time() - started + stage1_seconds,
                        stage1_seconds=stage1_seconds,
                        fast_signature_path=variant in fast_variants,
                        peak_gpu_memory_mb=(
                            torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
                            if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0
                        ),
                        rctd_pcc=rctd_pcc,
                        pcc_delta=metrics["mean_pcc"] - rctd_pcc,
                        best_epoch=result.get("best_epoch"),
                        graph_source=result.get("graph_source"),
                        reference_grouping=result.get("reference_grouping"),
                        **metrics,
                    )
                except Exception as exc:
                    row.update(
                        runtime_seconds=time.time() - started + stage1_seconds,
                        stage1_seconds=stage1_seconds,
                        fast_signature_path=variant in fast_variants,
                        peak_gpu_memory_mb=(
                            torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
                            if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                rows.append(row)
                pd.DataFrame(rows).to_csv(summary_path, index=False)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
