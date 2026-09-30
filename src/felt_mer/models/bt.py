"""Bradley-Terry model via logistic regression on pairwise feature diffs."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from felt_mer.config import BT_MAX_ITER, CV_SEED
from felt_mer.data.features import create_pairwise_features
from felt_mer.models.base import PairwisePreferenceModel


class BradleyTerryModel(PairwisePreferenceModel):
    """Feature-based Bradley-Terry via logistic regression."""

    def __init__(self, feature_cols: list[str] | None = None):
        self._feature_cols = feature_cols
        self._model = None
        self._scaler = None

    @property
    def name(self) -> str:
        return "bt"

    def fit(self, train_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
        X, y = create_pairwise_features(train_df, audio_df, self._feature_cols)
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
            # solver="saga", penalty="elasticnet", l1_ratios=(0.0, 0.5, 1.0),
            use_legacy_attributes=False,
        )
        self._model.fit(X, y)

    def predict_proba(self, test_df: pd.DataFrame, audio_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.full(len(test_df), np.nan)
        X, _ = create_pairwise_features(test_df, audio_df, self._feature_cols)
        if len(X) == 0:
            return np.full(len(test_df), np.nan)
        X = self._scaler.transform(X)
        return self._model.predict_proba(X)[:, 1]

    @property
    def coefficients(self) -> np.ndarray | None:
        """Return logistic regression coefficients (for feature importance)."""
        if self._model is None:
            return None
        return self._model.coef_[0].copy()
