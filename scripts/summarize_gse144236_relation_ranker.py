#!/usr/bin/env python3
"""Summarize valid GSE144236 multi-LR rankings and seed uncertainty."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagraph.cellcom.relation_ranker import calibrate_lr_statistics, ensemble_lr_rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="TNC_SDC1")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    by_variant: dict[str, list[pd.DataFrame]] = {}
    paths = sorted(args.input_root.glob("C*/seed_*/lr_pair_associated_edge_statistics.csv"))
    for path in paths:
        variant = path.parents[1].name
        match = re.search(r"seed_(\d+)", path.parent.name)
        seed = int(match.group(1)) if match else -1
        source = pd.read_csv(path)
        calibrated = calibrate_lr_statistics(source)
        calibrated.to_csv(path.parent / "lr_pair_calibrated_ranking.csv", index=False)
        by_variant.setdefault(variant, []).append(calibrated)
        target = calibrated.loc[calibrated["lr_pair"].eq(args.target)]
        if target.empty:
            rows.append({"variant": variant, "seed": seed, "target": args.target, "status": "missing"})
            continue
        item = target.iloc[0]
        raw_rank = source["associated_edge_attention_mean"].rank(method="min", ascending=False)
        raw_target_rank = int(raw_rank[source["lr_pair"].eq(args.target)].iloc[0])
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "target": args.target,
                "status": "valid_multi_lr",
                "raw_attention_rank": raw_target_rank,
                "calibrated_rank": int(item["rank"]),
                "calibrated_score": float(item["calibrated_score"]),
                "invalid_old_first_lr_bug": False,
                "source": str(path),
            }
        )

    history = pd.DataFrame(rows)
    history.to_csv(args.output / "gse144236_tnc_sdc1_rank_history.csv", index=False)
    for variant, frames in by_variant.items():
        ensemble = ensemble_lr_rankings(frames)
        ensemble.to_csv(args.output / f"{variant}_ensemble_lr_ranking.csv", index=False)

    if not history.empty:
        summary = history.loc[history["status"].eq("valid_multi_lr")].groupby("variant", as_index=False).agg(
            n_seeds=("seed", "nunique"),
            median_raw_rank=("raw_attention_rank", "median"),
            median_calibrated_rank=("calibrated_rank", "median"),
            best_calibrated_rank=("calibrated_rank", "min"),
            worst_calibrated_rank=("calibrated_rank", "max"),
        )
        summary.to_csv(args.output / "gse144236_variant_summary.csv", index=False)


if __name__ == "__main__":
    main()
