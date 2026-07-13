"""Calibrated, relation-aware ranking for ligand-receptor candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CalibrationWeights:
    attention: float = 0.45
    support: float = 0.25
    confidence: float = 0.20
    spatial_specificity: float = 0.10
    uncertainty_penalty: float = 0.15


# Selected on five synthetic-v2 seeds by mean AUPRC, then worst-seed AUPRC
# and cross-seed variance; case-study datasets and pair names were not used.
SYNTHETIC_V2_FROZEN_WEIGHTS = CalibrationWeights(
    attention=0.60,
    support=0.05,
    confidence=0.30,
    spatial_specificity=0.05,
    uncertainty_penalty=0.25,
)
DEFAULT_CALIBRATION_PROFILE = "synthetic_v2_frozen"


class LRCalibrationHead(nn.Module):
    """Frozen output head for robust LR candidate scores.

    Inputs are percentile-normalized neural attention, support, confidence,
    spatial specificity, and uncertainty. Coefficients are model buffers and
    therefore appear in ``state_dict``; they are not refit on a case study.
    """

    feature_names = (
        "neural_attention",
        "support",
        "confidence",
        "spatial_specificity",
        "uncertainty",
    )

    def __init__(
        self, weights: CalibrationWeights = SYNTHETIC_V2_FROZEN_WEIGHTS
    ) -> None:
        super().__init__()
        self.register_buffer(
            "coefficients",
            torch.tensor(
                [
                    weights.attention,
                    weights.support,
                    weights.confidence,
                    weights.spatial_specificity,
                    -weights.uncertainty_penalty,
                ],
                dtype=torch.float64,
            ),
        )

    def forward(self, normalized_features: torch.Tensor) -> torch.Tensor:
        if normalized_features.ndim != 2 or normalized_features.shape[1] != 5:
            raise ValueError("normalized_features must have shape [n_candidates, 5]")
        return normalized_features.to(self.coefficients) @ self.coefficients


def _percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return values.rank(method="average", pct=True).astype(float)


def calibrate_lr_statistics(
    statistics: pd.DataFrame,
    weights: Optional[CalibrationWeights] = None,
    calibration_profile: Optional[str] = None,
) -> pd.DataFrame:
    """Add robust, support-aware ranking columns without pair-name priors.

    Works with both legacy ``lr_pair_statistics.csv`` and the preferred
    ``lr_pair_associated_edge_statistics.csv`` schema.
    """
    frame = statistics.copy()
    if weights is None:
        weights = SYNTHETIC_V2_FROZEN_WEIGHTS
        calibration_profile = calibration_profile or DEFAULT_CALIBRATION_PROFILE
    else:
        calibration_profile = calibration_profile or "custom"
    if "associated_edge_attention_mean" in frame:
        mean_col = "associated_edge_attention_mean"
        std_col = "associated_edge_attention_std"
        count_col = "supporting_unique_edges"
    else:
        mean_col = "avg_attention_score"
        std_col = "std_attention_score"
        count_col = "occurrence_count"

    required = {"lr_pair", mean_col, std_col, count_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing LR ranking columns: {sorted(missing)}")

    mean = pd.to_numeric(frame[mean_col], errors="coerce").fillna(0.0)
    std = pd.to_numeric(frame[std_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    count = pd.to_numeric(frame[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    standard_error = std / np.sqrt(count.clip(lower=1.0))
    robust_attention = mean - standard_error

    if {"n_source_spots", "n_target_spots"}.issubset(frame.columns):
        source = pd.to_numeric(frame["n_source_spots"], errors="coerce").fillna(1.0).clip(lower=1.0)
        target = pd.to_numeric(frame["n_target_spots"], errors="coerce").fillna(1.0).clip(lower=1.0)
        density = count / (source * target).clip(lower=1.0)
        spatial_specificity = _percentile(density)
    else:
        spatial_specificity = pd.Series(0.5, index=frame.index, dtype=float)

    attention_feature = _percentile(mean)
    support_feature = _percentile(np.log1p(count))
    confidence_feature = _percentile(robust_attention)
    uncertainty_feature = _percentile(standard_error)
    normalized_features = np.column_stack(
        [
            attention_feature.to_numpy(),
            support_feature.to_numpy(),
            confidence_feature.to_numpy(),
            spatial_specificity.to_numpy(),
            uncertainty_feature.to_numpy(),
        ]
    )
    calibration_head = LRCalibrationHead(weights)
    with torch.no_grad():
        calibrated = calibration_head(
            torch.as_tensor(normalized_features, dtype=torch.float64)
        ).cpu().numpy()

    frame["neural_attention_score"] = mean.astype(float)
    frame["attention_percentile"] = attention_feature.astype(float)
    frame["support_percentile"] = support_feature.astype(float)
    frame["confidence_percentile"] = confidence_feature.astype(float)
    frame["uncertainty_percentile"] = uncertainty_feature.astype(float)
    frame["calibrated_score"] = calibrated.astype(float)
    frame["raw_attention_rank"] = mean.rank(method="min", ascending=False).astype(int)
    frame["rank"] = frame["calibrated_score"].rank(method="min", ascending=False).astype(int)
    frame["score_std"] = std.astype(float)
    frame["rank_std"] = np.nan
    frame["spatial_specificity"] = spatial_specificity.astype(float)
    frame["null_pvalue"] = np.nan
    frame["calibration_profile"] = calibration_profile
    return frame.sort_values(["rank", "lr_pair"]).reset_index(drop=True)


def ensemble_lr_rankings(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate calibrated and raw neural LR rankings across seeds."""
    if not frames:
        raise ValueError("at least one ranking frame is required")
    normalized = []
    for seed_index, frame in enumerate(frames):
        required = {
            "calibrated_score",
            "rank",
            "neural_attention_score",
            "raw_attention_rank",
        }
        calibrated = (
            frame if required.issubset(frame.columns)
            else calibrate_lr_statistics(frame)
        )
        normalized.append(
            calibrated[
                [
                    "lr_pair",
                    "calibrated_score",
                    "rank",
                    "neural_attention_score",
                    "raw_attention_rank",
                ]
            ].assign(seed_index=seed_index)
        )
    merged = pd.concat(normalized, ignore_index=True)
    result = merged.groupby("lr_pair", as_index=False).agg(
        calibrated_score=("calibrated_score", "mean"),
        score_std=("calibrated_score", "std"),
        rank=("rank", "mean"),
        rank_std=("rank", "std"),
        neural_attention_score=("neural_attention_score", "mean"),
        neural_attention_score_std=("neural_attention_score", "std"),
        raw_attention_rank=("raw_attention_rank", "mean"),
        raw_attention_rank_std=("raw_attention_rank", "std"),
        n_seeds=("seed_index", "nunique"),
    )
    result["rank"] = result["calibrated_score"].rank(method="min", ascending=False).astype(int)
    result["raw_attention_rank"] = result["neural_attention_score"].rank(
        method="min", ascending=False
    ).astype(int)
    return result.sort_values(["rank", "lr_pair"]).reset_index(drop=True)
