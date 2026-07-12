#!/usr/bin/env python3
"""Freeze generic LR calibration on synthetic-v2 labels, then apply to GSE."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spagraph.cellcom.relation_ranker import (  # noqa: E402
    CalibrationWeights,
    calibrate_lr_statistics,
    ensemble_lr_rankings,
)


def candidates():
    for attention, support, confidence, penalty in itertools.product(
        [0.30, 0.40, 0.50, 0.60],
        [0.05, 0.15, 0.25, 0.35],
        [0.10, 0.20, 0.30],
        [0.05, 0.15, 0.25],
    ):
        spatial = 0.05
        total = attention + support + confidence + spatial
        yield CalibrationWeights(
            attention=attention / total,
            support=support / total,
            confidence=confidence / total,
            spatial_specificity=spatial / total,
            uncertainty_penalty=penalty,
        )


def score(frame: pd.DataFrame, weights: CalibrationWeights) -> float:
    ranked = calibrate_lr_statistics(frame, weights)
    return float(average_precision_score(ranked["is_positive"].astype(int), ranked["calibrated_score"]))


def select(frames: list[pd.DataFrame]) -> tuple[CalibrationWeights, dict]:
    best = None
    for weights in candidates():
        values = [score(frame, weights) for frame in frames]
        key = (-float(np.mean(values)), -float(np.min(values)), float(np.std(values)))
        if best is None or key < best[0]:
            best = (key, weights, values)
    _, weights, values = best
    return weights, {
        "mean_auprc": float(np.mean(values)),
        "worst_auprc": float(np.min(values)),
        "std_auprc": float(np.std(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True)
    parser.add_argument("--gse", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    synthetic = [pd.read_csv(path) for path in args.synthetic]
    loo_rows = []
    for heldout in range(len(synthetic)):
        train = [frame for index, frame in enumerate(synthetic) if index != heldout]
        weights, train_metrics = select(train)
        heldout_auprc = score(synthetic[heldout], weights)
        raw_auprc = float(
            average_precision_score(
                synthetic[heldout]["is_positive"].astype(int),
                synthetic[heldout]["avg_attention_score"],
            )
        )
        loo_rows.append(
            {
                "heldout": str(args.synthetic[heldout]),
                "raw_attention_auprc": raw_auprc,
                "calibrated_auprc": heldout_auprc,
                **train_metrics,
                **weights.__dict__,
            }
        )
    pd.DataFrame(loo_rows).to_csv(args.output / "synthetic_v2_leave_one_seed_out.csv", index=False)

    weights, training_metrics = select(synthetic)
    (args.output / "synthetic_frozen_calibration.json").write_text(
        json.dumps(
            {
                **weights.__dict__,
                **training_metrics,
                "selection_data": "synthetic_lr_v2 only",
                "gse_target_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gse_rankings = []
    target_rows = []
    for path in args.gse:
        source = pd.read_csv(path)
        ranking = calibrate_lr_statistics(source, weights)
        match = re.search(r"seed_(\d+)", path.name)
        seed = int(match.group(1)) if match else -1
        ranking.to_csv(args.output / f"gse_seed_{seed}_independent_ranking.csv", index=False)
        gse_rankings.append(ranking)
        target = ranking.loc[ranking["lr_pair"].eq("TNC_SDC1")].iloc[0]
        target_rows.append(
            {
                "seed": seed,
                "rank": int(target["rank"]),
                "score": float(target["calibrated_score"]),
            }
        )
    pd.DataFrame(target_rows).sort_values("seed").to_csv(
        args.output / "gse_tnc_sdc1_independent_seed_ranks.csv", index=False
    )
    ensemble = ensemble_lr_rankings(gse_rankings)
    ensemble.to_csv(args.output / "gse_independent_ensemble_ranking.csv", index=False)


if __name__ == "__main__":
    main()
