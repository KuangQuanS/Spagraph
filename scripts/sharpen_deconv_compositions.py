#!/usr/bin/env python3
"""Deterministically calibrate composition concentration with a simplex power transform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_deconv_figure2 import find_prediction  # noqa: E402
from spagraph.models.deconv_initialization import power_calibrate_composition  # noqa: E402


def power_calibrate(composition: np.ndarray, power: float) -> np.ndarray:
    return power_calibrate_composition(composition, power=power)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power", type=float, required=True)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.run_roots]
    output = args.output.resolve()
    for dataset in [int(value) for value in args.datasets.split(",") if value.strip()]:
        source = find_prediction(roots, dataset, args.seed)
        composition = pd.read_csv(source, index_col=0)
        calibrated = power_calibrate(composition.to_numpy(dtype=np.float64), args.power)
        destination = output / "D21" / f"Data{dataset}" / f"seed_{args.seed}"
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(calibrated, index=composition.index, columns=composition.columns).to_csv(
            destination / "Spatial_composition.csv"
        )


if __name__ == "__main__":
    main()
