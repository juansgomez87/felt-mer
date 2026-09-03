"""General model baseline: predict using precomputed general_probability from test JSON files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from felt_mer.config import DIMENSION_TEST_FILE, USER_DATA_DIR
from felt_mer.data.loading import get_user_dir
from felt_mer.evaluation.metrics import compute_binary_metrics


class GeneralModelBaseline:
    """Baseline using precomputed general_probability from user test files.

    Each user's test_arousal.json / test_valence.json contains
    track_a.general_probability and track_b.general_probability.
    P(A wins) = sigmoid(prob_a - prob_b).
    """

    def __init__(self, user_data_dir: Path = USER_DATA_DIR):
        self._user_data_dir = user_data_dir

    @property
    def name(self) -> str:
        return "general_model"

    def _load_lookup(self, user_id: int, dimension: str) -> dict | None:
        """Load (song_a, song_b) -> (prob_a, prob_b) lookup from test JSON."""
        user_dir = get_user_dir(self._user_data_dir, user_id)
        filename = DIMENSION_TEST_FILE.get(dimension)
        if not filename:
            return None
        path = user_dir / filename
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        evaluations = data.get("evaluations") or []
        lookup = {}
        for ev in evaluations:
            ta = ev.get("track_a") or {}
            tb = ev.get("track_b") or {}
            id_a = str(ta.get("track_id", ""))
            id_b = str(tb.get("track_id", ""))
            prob_a = ta.get("general_probability")
            prob_b = tb.get("general_probability")
            if not id_a or not id_b or prob_a is None or prob_b is None:
                continue
            lookup[(id_a, id_b)] = (float(prob_a), float(prob_b))
            lookup[(id_b, id_a)] = (float(prob_b), float(prob_a))
        return lookup if lookup else None

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        dimension: str,
        user_id: int | None = None,
    ) -> np.ndarray:
        """Predict P(A wins) using general_probability lookup.

        Requires user_id to load the correct test file.
        """
        if user_id is None:
            return np.full(len(test_df), np.nan)

        lookup = self._load_lookup(user_id, dimension)
        if not lookup:
            return np.full(len(test_df), np.nan)

        probas = np.full(len(test_df), np.nan)
        for i, (_, row) in enumerate(test_df.iterrows()):
            a, b = str(row["song_a_id"]), str(row["song_b_id"])
            key = (a, b)
            if key not in lookup:
                continue
            prob_a, prob_b = lookup[key]
            diff = prob_a - prob_b
            p_a = float(expit(np.clip(diff, -500, 500)))
            probas[i] = np.clip(p_a, 1e-15, 1 - 1e-15)
        return probas

    def evaluate(
        self,
        test_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        dimension: str,
        user_id: int | None = None,
    ) -> dict | None:
        """Compute metrics on test set. Returns metrics dict or None."""
        probas = self.predict_proba(test_df, audio_df, dimension, user_id=user_id)
        valid = ~np.isnan(probas)
        if valid.sum() == 0:
            return None
        y_true = np.array([1 if row["winner"] == "A" else 0 for _, row in test_df.iterrows()])
        return compute_binary_metrics(y_true[valid], probas[valid])
