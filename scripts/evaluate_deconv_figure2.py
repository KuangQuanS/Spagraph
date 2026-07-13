#!/usr/bin/env python3
"""Recompute Figure 2a/2b with optimized Spagraph on a shared intersection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluate.scripts.deconv.evaluate_benchmark_metrics import (  # noqa: E402
    clean_column_names,
    compute_ars,
    compute_metrics,
)


METHOD_FILES = {
    "Seurat": "Seurat.csv",
    "DestVI": "DestVI.csv",
    "SPOTlight": "SPOTlight.csv",
    "SpatialDWLS": "SpatialDWLS_result.csv",
    "Stereoscope": "Stereoscope.csv",
    "RCTD": "RCTD_results.csv",
    "Tangram": "tangram.csv",
}
METHODS = [*METHOD_FILES, "Spagraph"]
COLORS = {
    "Spagraph": "#0072B2", "Tangram": "#D55E00", "RCTD": "#009E73",
    "Seurat": "#CC79A7", "SpatialDWLS": "#F0E442", "SPOTlight": "#56B4E9",
    "DestVI": "#E69F00", "Stereoscope": "#999999",
}


def read_composition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    return clean_column_names(frame).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    totals = frame.sum(axis=1).mask(lambda values: values == 0, 1.0)
    return frame.div(totals, axis=0)


def find_prediction(run_roots: list[Path], dataset: int, seed: int) -> Path:
    matches = []
    for root in run_roots:
        matches.extend(root.glob(f"D21/Data{dataset}/seed_{seed}/*_composition.csv"))
    matches = sorted(set(path.resolve() for path in matches))
    if len(matches) != 1:
        raise RuntimeError(f"Data{dataset}: expected one D21 prediction, found {matches}")
    return matches[0]


def plot_figure2(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [
        ("mean_pcc", "PCC", "higher is better"),
        ("mean_ssim", "SSIM", "higher is better"),
        ("mean_rmse", "RMSE", "lower is better"),
        ("mean_js", "JS", "lower is better"),
    ]
    for ax, (column, label, direction) in zip(axes.flat, specs):
        values = [metrics.loc[metrics.method_name.eq(method), column].to_numpy() for method in METHODS]
        boxes = ax.boxplot(
            values, tick_labels=METHODS, patch_artist=True, showmeans=True, showfliers=False
        )
        for box, method in zip(boxes["boxes"], METHODS):
            box.set_facecolor(COLORS[method]); box.set_alpha(0.8)
        ax.set_title(f"{label} ({direction})")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "figure2a_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    ars = metrics.groupby("method_name", as_index=False)["ARS"].mean().set_index("method_name").loc[METHODS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(METHODS, ars["ARS"], color=[COLORS[method] for method in METHODS])
    ax.set_xlabel(f"Mean ARS across {metrics['dataset'].nunique()} datasets (higher is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output / "figure2b_ars.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", default=",".join(str(i) for i in range(1, 33)))
    args = parser.parse_args()
    repo = args.repo.resolve()
    roots = [root.resolve() for root in args.run_roots]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    result_rows, coverage_rows, celltype_rows = [], [], []
    datasets = [int(value) for value in args.datasets.split(",") if value.strip()]
    for dataset in datasets:
        data_dir = repo / "evaluate" / "data" / f"Data{dataset}"
        truth = read_composition(data_dir / f"dataset{dataset}_density.csv")
        predictions = {
            method: read_composition(data_dir / filename)
            for method, filename in METHOD_FILES.items()
        }
        spagraph_path = find_prediction(roots, dataset, args.seed)
        predictions["Spagraph"] = read_composition(spagraph_path)

        shared_spots = truth.index
        for frame in predictions.values():
            shared_spots = shared_spots.intersection(frame.index)
        common_celltypes = set(truth.columns)
        for frame in predictions.values():
            common_celltypes &= set(frame.columns)
        common = sorted(common_celltypes)
        if not len(shared_spots) or not common:
            raise RuntimeError(f"Data{dataset}: empty shared spots/cell types")

        truth = normalize_rows(truth.loc[shared_spots])
        predictions = {
            method: normalize_rows(frame.loc[shared_spots]) for method, frame in predictions.items()
        }
        metrics_list = []
        for method in METHODS:
            metrics = compute_metrics(
                truth[common].to_numpy(), predictions[method][common].to_numpy(), common
            )
            metrics_list.append(metrics)
            annotated = metrics.copy()
            annotated.insert(0, "method_name", method)
            annotated.insert(0, "dataset", dataset)
            celltype_rows.extend(annotated.to_dict("records"))
        ars = compute_ars(metrics_list)
        ars["method_name"] = [METHODS[int(index)] for index in ars["method_id"]]
        ars.insert(0, "dataset", dataset)
        result_rows.extend(ars.drop(columns="method_id").to_dict("records"))
        coverage_rows.append({
            "dataset": dataset,
            "spots": len(shared_spots),
            "paired_celltypes": len(common),
            "truth_celltypes": len(truth.columns),
            "spagraph_path": str(spagraph_path),
        })

    results = pd.DataFrame(result_rows)
    results.to_csv(output / "figure2_dataset_metrics_ars.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "figure2_intersection_coverage.csv", index=False)
    pd.DataFrame(celltype_rows).to_csv(output / "figure2_celltype_metrics.csv", index=False)
    method_summary = results.groupby("method_name", as_index=False).agg(
        mean_pcc=("mean_pcc", "mean"), mean_ssim=("mean_ssim", "mean"),
        mean_rmse=("mean_rmse", "mean"), mean_js=("mean_js", "mean"), mean_ars=("ARS", "mean"),
    )
    method_summary.to_csv(output / "figure2_method_summary.csv", index=False)
    indexed = method_summary.set_index("method_name")
    spagraph, rctd = indexed.loc["Spagraph"], indexed.loc["RCTD"]
    checks = pd.DataFrame([
        {"metric": "PCC", "spagraph": spagraph.mean_pcc, "rctd": rctd.mean_pcc, "pass": spagraph.mean_pcc > rctd.mean_pcc},
        {"metric": "SSIM", "spagraph": spagraph.mean_ssim, "rctd": rctd.mean_ssim, "pass": spagraph.mean_ssim > rctd.mean_ssim},
        {"metric": "RMSE", "spagraph": spagraph.mean_rmse, "rctd": rctd.mean_rmse, "pass": spagraph.mean_rmse < rctd.mean_rmse},
        {"metric": "JS", "spagraph": spagraph.mean_js, "rctd": rctd.mean_js, "pass": spagraph.mean_js < rctd.mean_js},
        {"metric": "ARS", "spagraph": spagraph.mean_ars, "rctd": rctd.mean_ars, "pass": spagraph.mean_ars > rctd.mean_ars},
    ])
    checks.to_csv(output / "figure2_spagraph_vs_rctd_checks.csv", index=False)
    plot_figure2(results, output)
    print(method_summary.to_string(index=False))
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
