"""Feature engineering: feature matrices and pairwise feature diffs."""

import numpy as np
import pandas as pd

from tac_personalization.config import AUDIO_FEATURE_COLS


def get_feature_matrix(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return feature matrix keyed by track_id and list of column names.

    Parameters
    ----------
    features_df : DataFrame with track_id column (or track_id as index).
    feature_cols : Columns to use. Defaults to AUDIO_FEATURE_COLS.

    Returns
    -------
    (feats, cols) where feats is indexed by track_id (str).
    """
    if feature_cols is None:
        feature_cols = AUDIO_FEATURE_COLS

    if "track_id" in features_df.columns:
        feats = features_df.set_index("track_id")
    else:
        feats = features_df.copy()

    feats.index = feats.index.astype(str)
    available = [c for c in feature_cols if c in feats.columns]
    feats = feats[available].copy()

    for col in feats.columns:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")
    feats = feats.fillna(0).astype(float)
    return feats, list(feats.columns)


def create_pairwise_features(
    comparisons: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X (A - B feature diff) and y (1 = A won).

    Only includes pairs where both tracks have features.
    """
    feats, _ = get_feature_matrix(features_df, feature_cols)
    x_list, y_list = [], []
    for _, row in comparisons.iterrows():
        a, b = str(row["song_a_id"]), str(row["song_b_id"])
        if a not in feats.index or b not in feats.index:
            continue
        x_list.append(feats.loc[a].values - feats.loc[b].values)
        y_list.append(1 if row["winner"] == "A" else 0)
    return np.array(x_list, dtype=float), np.array(y_list, dtype=int)
