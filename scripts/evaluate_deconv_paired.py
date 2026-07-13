#!/usr/bin/env python3
"""Evaluate Spagraph and RCTD on exactly the same spots and cell types."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluate.scripts.deconv.evaluate_benchmark_metrics import (
    clean_column_names,
    compute_metrics,
)


DEVELOPMENT = {3, 11, 26, 15, 1, 32, 8, 23, 9, 22, 13, 10, 29, 31, 5, 18, 6, 2}
VALIDATION = {19, 30, 21, 25, 28, 12, 7}
TEST = {20, 24, 14, 16, 4, 17, 27}


def split_name(dataset: int) -> str:
    if dataset in DEVELOPMENT:
        return "development"
    if dataset in VALIDATION:
        return "validation"
    if dataset in TEST:
        return "test"
    raise ValueError(dataset)


def read_composition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    return clean_column_names(frame).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    totals = frame.sum(axis=1)
    totals = totals.mask(totals == 0, 1.0)
    return frame.div(totals, axis=0)


def find_prediction(run_roots: list[Path], dataset: int, seed: int) -> Path:
    matches = []
    for root in run_roots:
        matches.extend(root.glob(f"D21/Data{dataset}/seed_{seed}/*_composition.csv"))
    matches = sorted(set(path.resolve() for path in matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one D21 prediction for Data{dataset} seed {seed}, found {len(matches)}: {matches}"
        )
    return matches[0]


def mean_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "pcc": float(np.nanmean(metrics["pcc"])),
        "ssim": float(np.nanmean(metrics["ssim"])),
        "rmse": float(np.nanmean(metrics["rmse"])),
        "js": float(np.nanmean(metrics["js"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = args.repo.resolve()
    run_roots = [root.resolve() for root in args.run_roots]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, celltype_rows = [], []
    for dataset in range(1, 33):
        data_dir = repo / "evaluate" / "data" / f"Data{dataset}"
        truth = read_composition(data_dir / f"dataset{dataset}_density.csv")
        rctd = read_composition(data_dir / "RCTD_results.csv")
        prediction_path = find_prediction(run_roots, dataset, args.seed)
        spagraph = read_composition(prediction_path)

        shared_spots = truth.index.intersection(spagraph.index).intersection(rctd.index)
        if not len(shared_spots):
            raise RuntimeError(f"Data{dataset} has no jointly shared spots")
        truth = normalize_rows(truth.loc[shared_spots])
        spagraph = normalize_rows(spagraph.loc[shared_spots])
        rctd = normalize_rows(rctd.loc[shared_spots])
        common = sorted(set(truth.columns) & set(spagraph.columns) & set(rctd.columns))
        if not common:
            raise RuntimeError(f"Data{dataset} has no jointly shared cell types")

        spagraph_metrics = compute_metrics(
            truth[common].to_numpy(), spagraph[common].to_numpy(), common
        )
        rctd_metrics = compute_metrics(
            truth[common].to_numpy(), rctd[common].to_numpy(), common
        )
        spagraph_mean, rctd_mean = mean_metrics(spagraph_metrics), mean_metrics(rctd_metrics)
        row = {
            "dataset": dataset,
            "split": split_name(dataset),
            "seed": args.seed,
            "prediction_path": str(prediction_path),
            "spots": len(shared_spots),
            "truth_celltypes": len(truth.columns),
            "spagraph_celltypes": len(set(truth.columns) & set(spagraph.columns)),
            "rctd_celltypes": len(set(truth.columns) & set(rctd.columns)),
            "paired_celltypes": len(common),
        }
        for metric in ("pcc", "ssim", "rmse", "js"):
            row[f"spagraph_{metric}"] = spagraph_mean[metric]
            row[f"rctd_{metric}"] = rctd_mean[metric]
        row["pcc_delta"] = row["spagraph_pcc"] - row["rctd_pcc"]
        row["ssim_delta"] = row["spagraph_ssim"] - row["rctd_ssim"]
        row["rmse_improvement"] = row["rctd_rmse"] - row["spagraph_rmse"]
        row["js_improvement"] = row["rctd_js"] - row["spagraph_js"]
        rows.append(row)
        for method, metrics in (("Spagraph", spagraph_metrics), ("RCTD", rctd_metrics)):
            method_rows = metrics.copy()
            method_rows.insert(0, "method", method)
            method_rows.insert(0, "dataset", dataset)
            celltype_rows.extend(method_rows.to_dict("records"))

    dataset_results = pd.DataFrame(rows)
    dataset_results.to_csv(output / "deconv_paired_vs_rctd_32datasets.csv", index=False)
    pd.DataFrame(celltype_rows).to_csv(output / "deconv_paired_celltype_metrics.csv", index=False)

    summary_rows = []
    for split, group in list(dataset_results.groupby("split")) + [("all", dataset_results)]:
        summary_rows.append({
            "split": split,
            "datasets": len(group),
            "pcc_wins": int((group["pcc_delta"] > 0).sum()),
            "ssim_wins": int((group["ssim_delta"] > 0).sum()),
            "rmse_wins": int((group["rmse_improvement"] > 0).sum()),
            "js_wins": int((group["js_improvement"] > 0).sum()),
            "median_pcc_delta": float(group["pcc_delta"].median()),
            "mean_pcc_delta": float(group["pcc_delta"].mean()),
            "worst_pcc_delta": float(group["pcc_delta"].min()),
            "mean_ssim_delta": float(group["ssim_delta"].mean()),
            "mean_rmse_improvement": float(group["rmse_improvement"].mean()),
            "mean_js_improvement": float(group["js_improvement"].mean()),
            "mean_paired_celltypes": float(group["paired_celltypes"].mean()),
        })
    pd.DataFrame(summary_rows).to_csv(output / "deconv_paired_summary.csv", index=False)


if __name__ == "__main__":
    main()
