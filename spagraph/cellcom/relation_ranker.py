"""Calibrated, relation-aware ranking for ligand-receptor candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CalibrationWeights:
    attention: float = 0.45
    support: float = 0.25
    confidence: float = 0.20
    spatial_specificity: float = 0.10
    uncertainty_penalty: float = 0.15


# Frozen without using GSE144236 or TNC-SDC1. Selected on five synthetic-v2
# seeds by mean AUPRC, then worst-seed AUPRC and cross-seed variance.
SYNTHETIC_V2_FROZEN_WEIGHTS = CalibrationWeights(
    attention=0.60,
    support=0.05,
    confidence=0.30,
    spatial_specificity=0.05,
    uncertainty_penalty=0.25,
)
DEFAULT_CALIBRATION_PROFILE = "synthetic_v2_frozen"


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

    calibrated = (
        weights.attention * _percentile(mean)
        + weights.support * _percentile(np.log1p(count))
        + weights.confidence * _percentile(robust_attention)
        + weights.spatial_specificity * spatial_specificity
        - weights.uncertainty_penalty * _percentile(standard_error)
    )

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
    """Aggregate calibrated LR rankings across seeds with uncertainty."""
    if not frames:
        raise ValueError("at least one ranking frame is required")
    normalized = []
    for seed_index, frame in enumerate(frames):
        calibrated = frame if {"calibrated_score", "rank"}.issubset(frame.columns) else calibrate_lr_statistics(frame)
        normalized.append(calibrated[["lr_pair", "calibrated_score", "rank"]].assign(seed_index=seed_index))
    merged = pd.concat(normalized, ignore_index=True)
    result = merged.groupby("lr_pair", as_index=False).agg(
        calibrated_score=("calibrated_score", "mean"),
        score_std=("calibrated_score", "std"),
        rank=("rank", "mean"),
        rank_std=("rank", "std"),
        n_seeds=("seed_index", "nunique"),
    )
    result["rank"] = result["calibrated_score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["rank", "lr_pair"]).reset_index(drop=True)


class LRRelationRanker(nn.Module):
    """Candidate-level scorer with explicit LR identity and direction."""

    def __init__(
        self,
        numeric_dim: int,
        n_lr_pairs: int,
        n_celltypes: int,
        identity_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lr_embedding = nn.Embedding(n_lr_pairs + 1, identity_dim)
        self.sender_embedding = nn.Embedding(n_celltypes + 1, identity_dim)
        self.receiver_embedding = nn.Embedding(n_celltypes + 1, identity_dim)
        input_dim = numeric_dim + 3 * identity_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(
        self,
        numeric_features: torch.Tensor,
        lr_id: torch.Tensor,
        sender_id: torch.Tensor,
        receiver_id: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                numeric_features,
                self.lr_embedding(lr_id),
                self.sender_embedding(sender_id),
                self.receiver_embedding(receiver_id),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


def hard_negative_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Pairwise margin loss for shared-ligand/receptor and shuffled negatives."""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive_scores and negative_scores must have matching shapes")
    return F.softplus(margin - positive_scores + negative_scores).mean()


def within_context_ranking_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    edge_index: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Rank LR candidates that compete on the same directed cell-cell edge."""
    if predicted_scores.numel() < 2:
        return predicted_scores.new_zeros(())
    src, dst = edge_index
    context = src.to(torch.int64) * (int(dst.max().item()) + 1) + dst.to(torch.int64)
    positive = []
    negative = []
    for key in torch.unique(context):
        indices = torch.nonzero(context == key, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        ordered = indices[torch.argsort(target_scores[indices])]
        if target_scores[ordered[-1]] > target_scores[ordered[0]]:
            positive.append(predicted_scores[ordered[-1]])
            negative.append(predicted_scores[ordered[0]])
    if not positive:
        return predicted_scores.new_zeros(())
    return hard_negative_ranking_loss(torch.stack(positive), torch.stack(negative), margin=margin)
