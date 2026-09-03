"""Acoustic baseline: predict using Spotify energy (arousal) or valence."""

import numpy as np
import pandas as pd
from scipy.special import expit

from felt_mer.evaluation.metrics import compute_binary_metrics


class AcousticBaseline:
    """Non-parametric baseline using Spotify energy/valence features.

    For arousal: P(A wins) = sigmoid(energy_A - energy_B).
    For valence: P(A wins) = sigmoid(valence_A - valence_B).

    Uses expit(diff) instead of hard 0/1 predictions so that Brier and
    log-loss are not artificially capped.
    """

    @property
    def name(self) -> str:
        return "acoustic"

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        dimension: str,
    ) -> np.ndarray:
        """Predict P(A wins) based on acoustic feature ordering.

        Returns array of probabilities (nan where tracks are missing).
        """
        feature_col = "energy" if dimension == "arousal" else "valence"
        feats = audio_df.set_index("track_id") if "track_id" in audio_df.columns else audio_df
        feats.index = feats.index.astype(str)

        if feature_col not in feats.columns:
            return np.full(len(test_df), np.nan)

        values = feats[feature_col]
        probas = np.full(len(test_df), np.nan)
        for i, (_, row) in enumerate(test_df.iterrows()):
            a, b = str(row["song_a_id"]), str(row["song_b_id"])
            if a not in values.index or b not in values.index:
                continue
            diff = float(values[a]) - float(values[b])
            probas[i] = float(expit(diff))
        return probas

    def evaluate(
        self,
        test_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        dimension: str,
    ) -> dict | None:
        """Compute metrics on test set. Returns metrics dict or None."""
        probas = self.predict_proba(test_df, audio_df, dimension)
        valid = ~np.isnan(probas)
        if valid.sum() == 0:
            return None
        y_true = np.array([1 if row["winner"] == "A" else 0 for _, row in test_df.iterrows()])
        return compute_binary_metrics(y_true[valid], probas[valid])
