"""Statistics used by manuscript Figure 3e.

The independent statistical unit is one ligand-receptor (LR) pair.  Pairs
selected by both ranking procedures are excluded from both groups before the
two-sided Mann-Whitney U tests are run.  The two prespecified Figure 3e tests
are adjusted together with Holm's method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RANKING_GROUPS = ("attention", "frequency")


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Return Holm-adjusted P values while preserving input order."""

    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted

    finite_values = values[finite_indices]
    order = np.argsort(finite_values, kind="stable")
    ranked = finite_values[order]
    multipliers = np.arange(len(ranked), 0, -1, dtype=float)
    ranked_adjusted = np.maximum.accumulate(ranked * multipliers)
    ranked_adjusted = np.minimum(ranked_adjusted, 1.0)

    restored = np.empty_like(ranked_adjusted)
    restored[order] = ranked_adjusted
    adjusted[finite_indices] = restored
    return adjusted


def make_disjoint_lr_pair_groups(
    metrics_df: pd.DataFrame,
    *,
    ranking_col: str = "ranking_type",
    pair_col: str = "lr_pair",
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude LR pairs appearing in both prespecified ranking groups.

    A duplicated LR pair within one ranking group would violate the declared
    statistical unit and therefore raises an error instead of silently
    inflating the sample size.
    """

    required = {ranking_col, pair_col}
    missing = required.difference(metrics_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = metrics_df.loc[metrics_df[ranking_col].isin(RANKING_GROUPS)].copy()
    duplicated = selected.duplicated([ranking_col, pair_col], keep=False)
    if duplicated.any():
        examples = selected.loc[duplicated, [ranking_col, pair_col]].drop_duplicates()
        raise ValueError(
            "Each LR pair must occur once per ranking group; duplicates found: "
            + examples.to_dict(orient="records").__repr__()
        )

    attention_pairs = set(
        selected.loc[selected[ranking_col] == "attention", pair_col].astype(str)
    )
    frequency_pairs = set(
        selected.loc[selected[ranking_col] == "frequency", pair_col].astype(str)
    )
    overlap = sorted(attention_pairs.intersection(frequency_pairs))
    disjoint = selected.loc[~selected[pair_col].astype(str).isin(overlap)].copy()
    disjoint["statistical_unit"] = "LR pair"
    disjoint["overlap_excluded"] = ";".join(overlap) if overlap else "none"
    return disjoint.reset_index(drop=True), overlap


def compare_lr_pair_groups(
    metrics_df: pd.DataFrame,
    metrics: Mapping[str, str],
    *,
    ranking_col: str = "ranking_type",
    pair_col: str = "lr_pair",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compare disjoint attention- and frequency-ranked LR-pair groups.

    Parameters
    ----------
    metrics:
        Mapping from metric column name to its publication-facing label.  Holm
        adjustment is applied across exactly these prespecified tests.
    """

    missing_metrics = set(metrics).difference(metrics_df.columns)
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {sorted(missing_metrics)}")

    disjoint, overlap = make_disjoint_lr_pair_groups(
        metrics_df, ranking_col=ranking_col, pair_col=pair_col
    )
    rows: list[dict[str, object]] = []
    for metric, label in metrics.items():
        attention = disjoint.loc[
            disjoint[ranking_col] == "attention", metric
        ].dropna().astype(float)
        frequency = disjoint.loc[
            disjoint[ranking_col] == "frequency", metric
        ].dropna().astype(float)

        statistic = float("nan")
        raw_p = float("nan")
        if len(attention) and len(frequency):
            result = mannwhitneyu(attention, frequency, alternative="two-sided")
            statistic = float(result.statistic)
            raw_p = float(result.pvalue)

        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "statistical_unit": "LR pair",
                "attention_n": int(len(attention)),
                "frequency_n": int(len(frequency)),
                "overlap_count": int(len(overlap)),
                "overlap_excluded": ";".join(overlap) if overlap else "none",
                "attention_median": float(attention.median()) if len(attention) else np.nan,
                "frequency_median": float(frequency.median()) if len(frequency) else np.nan,
                "mannwhitney_u": statistic,
                "mannwhitney_raw_p": raw_p,
            }
        )

    summary = pd.DataFrame(rows)
    summary["holm_p"] = holm_adjust(summary["mannwhitney_raw_p"].to_numpy())
    return summary, disjoint, overlap
