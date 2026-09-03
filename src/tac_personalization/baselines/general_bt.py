"""General (pooled) BT baseline: trained on all other users' comparisons."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from tac_personalization.config import BT_MAX_ITER, CV_SEED
from tac_personalization.data.features import create_pairwise_features


class GeneralBTBaseline:
    """BT model trained on all users' comparisons except the test user.

    This baseline answers: does personalization (per-user model on ~60 pairs)
    outperform a general preference model (pooled model on ~2400 pairs)?

    Usage within the CV harness:
      1. Before the CV loop, call `fit_pooled(all_comparisons, audio_df, exclude_user_id)`
         to train on all other users.
      2. During each fold, call `predict_proba(test_df, audio_df)` — no re-fitting needed
         since the general model doesn't use the test user's data at all.
    """

    def __init__(self):
        self._model = None
        self._scaler = None

    @property
    def name(self) -> str:
        return "general_bt"

    def fit_pooled(
        self,
        comparisons_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        exclude_user_id: int,
    ) -> None:
        """Train on all comparisons except those from exclude_user_id."""
        other_users = comparisons_df[comparisons_df["user_id"] != exclude_user_id]
        X, y = create_pairwise_features(other_users, audio_df)
        if len(y) == 0:
            self._model = None
            return
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._model = LogisticRegressionCV(
            cv=3,
            scoring="accuracy",
            max_iter=BT_MAX_ITER,
            random_state=CV_SEED,
            l1_ratios=(0.0,),
            use_legacy_attributes=False,
        )
        self._model.fit(X, y)

    def predict_proba(
        self,
        test_df: pd.DataFrame,
        audio_df: pd.DataFrame,
        dimension: str | None = None,
    ) -> np.ndarray:
        """Predict P(A wins) using the pooled general model.

        The dimension parameter is accepted for interface compatibility
        but not used (the general model is dimension-specific by training data).
        """
        if self._model is None:
            return np.full(len(test_df), np.nan)
        X, _ = create_pairwise_features(test_df, audio_df)
        if len(X) == 0:
            return np.full(len(test_df), np.nan)
        X = self._scaler.transform(X)
        return self._model.predict_proba(X)[:, 1]
