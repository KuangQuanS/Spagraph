#!/usr/bin/env python3
"""Aggregate semi-synthetic LR benchmark results across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCORES = ["avg_attention_score", "q90_edge_attention", "mean_edge_attention"]


def _ci95(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / np.sqrt(len(clean)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)

    metric_rows = []
    ranking_rows = []
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        with open(seed_dir / "semisynthetic_lr_metrics.json", encoding="utf-8") as handle:
            metrics = json.load(handle)
        flat_metrics = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        flat_metrics["seed"] = seed
        metric_rows.append(flat_metrics)
        ranking = pd.read_csv(seed_dir / "semisynthetic_lr_ranking.csv")
        ranking["seed"] = seed
        ranking_rows.append(ranking)

    metrics_df = pd.DataFrame(metric_rows).sort_values("seed")
    metrics_df.to_csv(root / "multiseed_metrics_by_seed.csv", index=False)
    summary_rows = []
    for column in metrics_df.columns:
        if column in {"seed", "n_candidates", "n_positives"}:
            continue
        summary_rows.append({
            "metric": column,
            "mean": float(metrics_df[column].mean()),
            "std": float(metrics_df[column].std(ddof=1)),
            "ci95_half_width": _ci95(metrics_df[column]),
            "min": float(metrics_df[column].min()),
            "max": float(metrics_df[column].max()),
        })
    pd.DataFrame(summary_rows).to_csv(root / "multiseed_metrics_summary.csv", index=False)

    rankings = pd.concat(ranking_rows, ignore_index=True)
    rankings.to_csv(root / "multiseed_rankings.csv", index=False)
    matched_rows = []
    for (seed, group), frame in rankings.groupby(["seed", "matched_group"]):
        positive = frame.loc[frame["is_positive"].eq(1)]
        if len(positive) != 1:
            continue
        positive = positive.iloc[0]
        for decoy_pattern in ["diffuse_edge_count_matched", "spatially_separated_matched"]:
            decoy = frame.loc[frame["pattern"].eq(decoy_pattern)]
            if len(decoy) != 1:
                continue
            decoy = decoy.iloc[0]
            for score in SCORES:
                if score not in frame.columns:
                    continue
                matched_rows.append({
                    "seed": seed,
                    "matched_group": group,
                    "scenario": positive["pattern"],
                    "template": positive["template"],
                    "decoy_pattern": decoy_pattern,
                    "score": score,
                    "positive_score": positive[score],
                    "decoy_score": decoy[score],
                    "difference": positive[score] - decoy[score],
                    "positive_wins": int(positive[score] > decoy[score]),
                })

    matched = pd.DataFrame(matched_rows)
    matched.to_csv(root / "matched_pair_comparisons.csv", index=False)
    paired_summary = []
    for keys, frame in matched.groupby(["score", "decoy_pattern"]):
        differences = frame["difference"].to_numpy(dtype=float)
        try:
            test = stats.wilcoxon(differences, alternative="greater")
            p_value = float(test.pvalue)
        except ValueError:
            p_value = 1.0
        paired_summary.append({
            "score": keys[0],
            "decoy_pattern": keys[1],
            "n_pairs": len(frame),
            "win_rate": float(frame["positive_wins"].mean()),
            "median_difference": float(np.median(differences)),
            "mean_difference": float(np.mean(differences)),
            "wilcoxon_greater_p": p_value,
        })
    pd.DataFrame(paired_summary).to_csv(root / "matched_pair_summary.csv", index=False)


if __name__ == "__main__":
    main()
