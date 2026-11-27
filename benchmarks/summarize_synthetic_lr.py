#!/usr/bin/env python3
"""Aggregate fully synthetic LR benchmark results across random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)

    metric_frames = []
    ranking_frames = []
    paired_frames = []
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        metrics = pd.read_csv(seed_dir / "synthetic_v2_metrics.csv")
        metrics.insert(0, "seed", seed)
        metric_frames.append(metrics)
        ranking = pd.read_csv(seed_dir / "synthetic_v2_ranking.csv")
        ranking.insert(0, "seed", seed)
        ranking_frames.append(ranking)
        paired = pd.read_csv(seed_dir / "synthetic_v2_paired_results.csv")
        paired.insert(0, "seed", seed)
        paired_frames.append(paired)

    metrics = pd.concat(metric_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    metrics.to_csv(root / "synthetic_v2_multiseed_metrics.csv", index=False)
    rankings.to_csv(root / "synthetic_v2_multiseed_rankings.csv", index=False)
    paired.to_csv(root / "synthetic_v2_multiseed_paired.csv", index=False)

    numeric = metrics.drop(columns="seed")
    summary = pd.DataFrame({
        "metric": numeric.columns,
        "mean": numeric.mean().to_numpy(),
        "median": numeric.median().to_numpy(),
        "std": numeric.std(ddof=1).to_numpy(),
        "min": numeric.min().to_numpy(),
        "max": numeric.max().to_numpy(),
    })
    summary.to_csv(root / "synthetic_v2_multiseed_summary.csv", index=False)

    local_diffuse = paired["local_minus_diffuse"]
    local_separated = paired["local_minus_separated"]
    pooled = pd.DataFrame([{
        "n_seeds": len(args.seeds),
        "n_matched_families": len(paired),
        "local_beats_diffuse_rate": float((local_diffuse > 0).mean()),
        "local_vs_diffuse_median_difference": float(local_diffuse.median()),
        "local_vs_diffuse_wilcoxon_p": float(
            stats.wilcoxon(local_diffuse, alternative="greater").pvalue
        ),
        "local_beats_separated_rate": float((local_separated > 0).mean()),
        "local_vs_separated_wilcoxon_p": float(
            stats.wilcoxon(local_separated, alternative="greater").pvalue
        ),
        "positive_median_attention_rank": float(
            rankings.loc[
                rankings["pattern"].eq("local_positive"), "attention_rank"
            ].median()
        ),
        "diffuse_median_attention_rank": float(
            rankings.loc[
                rankings["pattern"].eq("matched_diffuse"), "attention_rank"
            ].median()
        ),
        "global_median_attention_rank": float(
            rankings.loc[
                rankings["pattern"].eq("global_high_coverage"), "attention_rank"
            ].median()
        ),
        "global_median_abundance_rank": float(
            rankings.loc[
                rankings["pattern"].eq("global_high_coverage"), "abundance_rank"
            ].median()
        ),
    }])
    pooled.to_csv(root / "synthetic_v2_multiseed_pooled.csv", index=False)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
