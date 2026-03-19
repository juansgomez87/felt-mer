"""User segmentation and dimension-specific personalization patterns."""

import pandas as pd


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


def dimension_specific_patterns(res_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each user by dimension-specific personalization pattern.

    Patterns: both, arousal_only, valence_only, neither, mixed.
    Returns DataFrame with user_id, arousal_improvement, valence_improvement, pattern.
    """
    pivot = res_df.pivot_table(
        index="user_id",
        columns="dimension",
        values="acc_improvement",
        aggfunc="first",
    )
    patterns = []
    for uid, row in pivot.iterrows():
        arousal_imp = row.get("arousal", 0)
        valence_imp = row.get("valence", 0)

        if arousal_imp > 0.1 and valence_imp < -0.05:
            pattern = "arousal_only"
        elif valence_imp > 0.1 and arousal_imp < -0.05:
            pattern = "valence_only"
        elif arousal_imp > 0.05 and valence_imp > 0.05:
            pattern = "both"
        elif arousal_imp < -0.05 and valence_imp < -0.05:
            pattern = "neither"
        else:
            pattern = "mixed"

        patterns.append({
            "user_id": uid,
            "arousal_improvement": arousal_imp,
            "valence_improvement": valence_imp,
            "pattern": pattern,
        })
    return pd.DataFrame(patterns)
