from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_DIRECTION = {
    "mean_pcc": "greater",
    "mean_ssim": "greater",
    "mean_rmse": "less",
    "mean_js": "less",
    "ARS": "greater",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir.parent.parent / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce paired Wilcoxon signed-rank tests across benchmark datasets "
            "using the per-dataset metrics_ARS.csv files."
        )
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--start", type=int, default=1, help="First dataset index.")
    parser.add_argument("--end", type=int, default=32, help="Last dataset index.")
    parser.add_argument(
        "--dataset-template",
        type=str,
        default="Data{idx}",
        help="Dataset directory template. Example: Data{idx}",
    )
    parser.add_argument("--metrics-file", type=str, default="metrics_ARS.csv")
    parser.add_argument("--method-a", type=str, default="Spagraph")
    parser.add_argument("--method-b", type=str, default="RCTD")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_ssim", "mean_js", "mean_pcc", "mean_rmse"],
        help="Metric columns to compare from metrics_ARS.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for a summary CSV.",
    )
    return parser.parse_args()


def load_metric_pairs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    used_datasets = []

    for idx in range(args.start, args.end + 1):
        dataset_name = args.dataset_template.format(idx=idx)
        metrics_path = args.data_root / dataset_name / args.metrics_file
        if not metrics_path.exists():
            continue

        df = pd.read_csv(metrics_path)
        row_a = df.loc[df["method_name"] == args.method_a]
        row_b = df.loc[df["method_name"] == args.method_b]
        if row_a.empty or row_b.empty:
            continue

        row = {"dataset": dataset_name}
        for metric in args.metrics:
            if metric not in df.columns:
                raise ValueError(f"Column '{metric}' not found in {metrics_path}")
            row[f"{metric}_a"] = float(row_a.iloc[0][metric])
            row[f"{metric}_b"] = float(row_b.iloc[0][metric])
        rows.append(row)
        used_datasets.append(idx)

    if not rows:
        raise ValueError("No usable datasets were found.")

    return pd.DataFrame(rows), used_datasets


def summarize_metric(pairs_df: pd.DataFrame, metric: str) -> dict[str, object]:
    series_a = pairs_df[f"{metric}_a"]
    series_b = pairs_df[f"{metric}_b"]
    diff = series_a - series_b
    preferred_alt = DEFAULT_DIRECTION.get(metric, "two-sided")

    result = {
        "metric": metric,
        "method_a_mean": series_a.mean(),
        "method_b_mean": series_b.mean(),
        "median_diff_a_minus_b": diff.median(),
        "n_positive_diff": int((diff > 0).sum()),
        "n_negative_diff": int((diff < 0).sum()),
        "n_zero_diff": int((diff == 0).sum()),
        "preferred_alternative": preferred_alt,
    }

    for alternative in ("two-sided", "greater", "less"):
        test = wilcoxon(series_a, series_b, alternative=alternative, zero_method="wilcox", method="auto")
        result[f"{alternative}_statistic"] = float(test.statistic)
        result[f"{alternative}_pvalue"] = float(test.pvalue)

    return result


def print_summary(summary_df: pd.DataFrame, args: argparse.Namespace, used_datasets: list[int]) -> None:
    print(f"Data root: {args.data_root}")
    print(f"Datasets used ({len(used_datasets)}): {used_datasets}")
    print(f"Method A: {args.method_a}")
    print(f"Method B: {args.method_b}")
    print("")

    for row in summary_df.to_dict(orient="records"):
        preferred_alt = row["preferred_alternative"]
        print(f"{row['metric']}:")
        print(f"  mean({args.method_a}) = {row['method_a_mean']:.10f}")
        print(f"  mean({args.method_b}) = {row['method_b_mean']:.10f}")
        print(f"  median_diff(a-b) = {row['median_diff_a_minus_b']:.10f}")
        print(
            "  sign_counts(a-b) = "
            f"+{row['n_positive_diff']} / -{row['n_negative_diff']} / 0={row['n_zero_diff']}"
        )
        print(
            f"  manuscript-direction ({preferred_alt}) "
            f"p = {row[f'{preferred_alt}_pvalue']:.10f}"
        )
        print(f"  two-sided p = {row['two-sided_pvalue']:.10f}")
        print(f"  greater p = {row['greater_pvalue']:.10f}")
        print(f"  less p = {row['less_pvalue']:.10f}")
        print("")


def main() -> None:
    args = parse_args()
    pairs_df, used_datasets = load_metric_pairs(args)
    summary_rows = [summarize_metric(pairs_df, metric) for metric in args.metrics]
    summary_df = pd.DataFrame(summary_rows)

    print_summary(summary_df, args, used_datasets)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output_csv, index=False)
        print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
