"""User segmentation and dimension-specific personalization patterns."""

import numpy as np
import pandas as pd

from felt_mer.config import PATTERN_IMPROVEMENT_THRESHOLD

PATTERN_ORDER = ("both", "arousal_only", "valence_only", "neither")


def segment_users_by_personalization(res_df: pd.DataFrame) -> pd.DataFrame:
    """Segment users into high_personalizer, baseline_equivalent, low_personalizer.

    Returns DataFrame with user_id, segment, mean_improvement, pct_dims_beat_baseline.
    """
    user_summary = res_df.groupby("user_id").agg({
        "acc_improvement": ["mean", "min", "max", "std"],
        "beat_baseline_acc": "sum",
        "dimension": "count",
    }).reset_index()
    user_summary.columns = [
        "user_id", "mean_improvement", "min_improvement",
        "max_improvement", "std_improvement",
        "n_dims_beat_baseline", "n_dims_total",
    ]
    user_summary["pct_dims_beat_baseline"] = (
        user_summary["n_dims_beat_baseline"] / user_summary["n_dims_total"]
    )

    def _classify(row):
        if row["pct_dims_beat_baseline"] >= 0.5 and row["mean_improvement"] > 0.05:
            return "high_personalizer"
        if row["pct_dims_beat_baseline"] <= 0.5 and row["mean_improvement"] < -0.05:
            return "low_personalizer"
        return "baseline_equivalent"

    user_summary["segment"] = user_summary.apply(_classify, axis=1)
    return user_summary


def classify_pattern(
    arousal_improvement: float | None,
    valence_improvement: float | None,
    threshold: float = PATTERN_IMPROVEMENT_THRESHOLD,
) -> str:
    """Classify one user as both / arousal_only / valence_only / neither.

    A dimension counts as helped when its accuracy gain exceeds `threshold`.
    A missing improvement (the user has no CV result for that dimension) is
    treated as absent rather than as zero, so a user evaluated on only one
    dimension is classified on that dimension alone.
    """
    helped_arousal = arousal_improvement is not None and arousal_improvement > threshold
    helped_valence = valence_improvement is not None and valence_improvement > threshold

    if arousal_improvement is None:
        return "valence_only" if helped_valence else "neither"
    if valence_improvement is None:
        return "arousal_only" if helped_arousal else "neither"
    if helped_arousal and helped_valence:
        return "both"
    if helped_arousal:
        return "arousal_only"
    if helped_valence:
        return "valence_only"
    return "neither"


def dimension_specific_patterns(
    res_df: pd.DataFrame,
    threshold: float = PATTERN_IMPROVEMENT_THRESHOLD,
) -> pd.DataFrame:
    """Classify each user by dimension-specific personalization pattern.

    Patterns: both, arousal_only, valence_only, neither.
    Returns DataFrame with user_id, arousal_improvement, valence_improvement, pattern.
    """
    pivot = res_df.pivot_table(
        index="user_id",
        columns="dimension",
        values="acc_improvement",
        aggfunc="first",
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={
        "arousal": "arousal_improvement",
        "valence": "valence_improvement",
    })
    for col in ("arousal_improvement", "valence_improvement"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot.reset_index()

    pivot["pattern"] = [
        classify_pattern(
            row["arousal_improvement"] if pd.notna(row["arousal_improvement"]) else None,
            row["valence_improvement"] if pd.notna(row["valence_improvement"]) else None,
            threshold=threshold,
        )
        for _, row in pivot.iterrows()
    ]
    return pivot[["user_id", "arousal_improvement", "valence_improvement", "pattern"]]
