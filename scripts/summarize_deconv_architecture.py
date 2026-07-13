#!/usr/bin/env python3
"""Combine historical RCTD references with D0-D5 experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def historical_rows(repo: Path) -> list[dict]:
    rows = []
    for dataset in range(1, 33):
        metrics = pd.read_csv(repo / "evaluate" / "data" / f"Data{dataset}" / "metrics_ARS.csv")
        spagraph = metrics.loc[metrics["method_name"].eq("Spagraph")].iloc[0]
        rctd = metrics.loc[metrics["method_name"].eq("RCTD")].iloc[0]
        rows.append(
            {
                "variant": "historical",
                "dataset": dataset,
                "split": split_name(dataset),
                "seed": np.nan,
                "mean_pcc": float(spagraph["mean_pcc"]),
                "mean_ssim": float(spagraph["mean_ssim"]),
                "mean_rmse": float(spagraph["mean_rmse"]),
                "mean_js": float(spagraph["mean_js"]),
                "rctd_pcc": float(rctd["mean_pcc"]),
                "rctd_ssim": float(rctd["mean_ssim"]),
                "rctd_rmse": float(rctd["mean_rmse"]),
                "rctd_js": float(rctd["mean_js"]),
                "pcc_delta": float(spagraph["mean_pcc"] - rctd["mean_pcc"]),
                "ssim_delta": float(spagraph["mean_ssim"] - rctd["mean_ssim"]),
                "rmse_improvement": float(rctd["mean_rmse"] - spagraph["mean_rmse"]),
                "js_improvement": float(rctd["mean_js"] - spagraph["mean_js"]),
                "beats_rctd": bool(spagraph["mean_pcc"] > rctd["mean_pcc"]),
                "status": "historical_reference",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = historical_rows(args.repo.resolve())
    for run_path in args.runs or []:
        if not run_path.exists():
            continue
        runs = pd.read_csv(run_path)
        if "status" in runs.columns:
            runs = runs.loc[runs["status"].isin({"ok", "reevaluated"})].copy()
        else:
            # Re-evaluated prediction tables contain only successfully scored runs.
            runs = runs.copy()
            runs["status"] = "reevaluated"
        runs["split"] = runs["dataset"].astype(int).map(split_name)
        rctd_metrics = {}
        for dataset in runs["dataset"].astype(int).unique():
            metrics = pd.read_csv(args.repo.resolve() / "evaluate" / "data" / f"Data{dataset}" / "metrics_ARS.csv")
            rctd = metrics.loc[metrics["method_name"].eq("RCTD")].iloc[0]
            rctd_metrics[dataset] = rctd
        for metric in ("pcc", "ssim", "rmse", "js"):
            runs[f"rctd_{metric}"] = runs["dataset"].astype(int).map(
                lambda dataset: float(rctd_metrics[dataset][f"mean_{metric}"])
            )
        runs["pcc_delta"] = runs["mean_pcc"] - runs["rctd_pcc"]
        runs["ssim_delta"] = runs["mean_ssim"] - runs["rctd_ssim"]
        runs["rmse_improvement"] = runs["rctd_rmse"] - runs["mean_rmse"]
        runs["js_improvement"] = runs["rctd_js"] - runs["mean_js"]
        runs["beats_rctd"] = runs["mean_pcc"] > runs["rctd_pcc"]
        rows.extend(runs.to_dict("records"))

    output = pd.DataFrame(rows)
    for column in ("runtime_seconds", "stage1_seconds", "peak_gpu_memory_mb"):
        if column not in output.columns:
            output[column] = np.nan
    if "fast_signature_path" not in output.columns:
        output["fast_signature_path"] = False
    output["full_runtime_seconds"] = output["runtime_seconds"].where(
        output["stage1_seconds"].notna() | output["fast_signature_path"].eq(True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    per_dataset = output.groupby(["variant", "dataset", "split"], as_index=False).agg(
        n_seeds=("seed", lambda values: int(values.notna().sum()) or 1),
        median_pcc=("mean_pcc", "median"),
        mean_pcc=("mean_pcc", "mean"),
        pcc_std=("mean_pcc", "std"),
        mean_ssim=("mean_ssim", "mean"),
        ssim_std=("mean_ssim", "std"),
        mean_rmse=("mean_rmse", "mean"),
        rmse_std=("mean_rmse", "std"),
        mean_js=("mean_js", "mean"),
        js_std=("mean_js", "std"),
        rctd_pcc=("rctd_pcc", "first"),
        rctd_ssim=("rctd_ssim", "first"),
        rctd_rmse=("rctd_rmse", "first"),
        rctd_js=("rctd_js", "first"),
        median_pcc_delta=("pcc_delta", "median"),
        mean_pcc_delta=("pcc_delta", "mean"),
        worst_seed_pcc_delta=("pcc_delta", "min"),
        mean_ssim_delta=("ssim_delta", "mean"),
        mean_rmse_improvement=("rmse_improvement", "mean"),
        mean_js_improvement=("js_improvement", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_full_runtime_seconds=("full_runtime_seconds", "mean"),
        mean_stage1_seconds=("stage1_seconds", "mean"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
    )
    per_dataset["beats_rctd"] = per_dataset["median_pcc"] > per_dataset["rctd_pcc"]
    per_dataset["beats_rctd_ssim"] = per_dataset["mean_ssim"] > per_dataset["rctd_ssim"]
    per_dataset["beats_rctd_rmse"] = per_dataset["mean_rmse"] < per_dataset["rctd_rmse"]
    per_dataset["beats_rctd_js"] = per_dataset["mean_js"] < per_dataset["rctd_js"]
    per_dataset.to_csv(args.output.with_name("deconv_vs_rctd_dataset_summary.csv"), index=False)

    aggregation = dict(
        datasets=("dataset", "nunique"),
        wins=("beats_rctd", "sum"),
        ssim_wins=("beats_rctd_ssim", "sum"),
        rmse_wins=("beats_rctd_rmse", "sum"),
        js_wins=("beats_rctd_js", "sum"),
        median_pcc_delta=("median_pcc_delta", "median"),
        mean_pcc_delta=("mean_pcc_delta", "mean"),
        worst_pcc_delta=("worst_seed_pcc_delta", "min"),
        mean_ssim_delta=("mean_ssim_delta", "mean"),
        mean_rmse_improvement=("mean_rmse_improvement", "mean"),
        mean_js_improvement=("mean_js_improvement", "mean"),
        mean_runtime_seconds=("mean_runtime_seconds", "mean"),
        mean_full_runtime_seconds=("mean_full_runtime_seconds", "mean"),
        mean_stage1_seconds=("mean_stage1_seconds", "mean"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
    )
    summary = per_dataset.groupby(["variant", "split"], as_index=False).agg(**aggregation)
    overall = per_dataset.groupby("variant", as_index=False).agg(**aggregation)
    overall.insert(1, "split", "all")
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(args.output.with_name("deconv_vs_rctd_summary.csv"), index=False)


if __name__ == "__main__":
    main()
