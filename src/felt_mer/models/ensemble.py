"""Ensemble model: late fusion of BT, LambdaMART, and DPO."""

import numpy as np
import pandas as pd

from felt_mer.models.base import PairwisePreferenceModel
from felt_mer.models.bt import BradleyTerryModel
from felt_mer.models.dpo import DPOModel
from felt_mer.models.lambdamart import LambdaMARTModel


def _fuse_probs(probs: list[float | None]) -> float | None:
    """Equal-weight mean of non-None probabilities."""
    valid = [p for p in probs if p is not None and np.isfinite(p)]
    if not valid:
        return None
    return float(np.mean(valid))


def _majority_vote(probs: list[float | None]) -> float | None:
    """Majority vote: fraction of models predicting A wins."""
    valid = [p for p in probs if p is not None and np.isfinite(p)]
    if not valid:
        return None
    votes = np.array(valid) >= 0.5
    return float(np.mean(votes))


class EnsembleModel(PairwisePreferenceModel):
    """Late fusion of sub-models (default: BT + LambdaMART + DPO).

    Supported modes:
      - 'fuse': equal-weight probability average
      - 'majority': majority vote (fraction predicting A)
    """

    def __init__(
        self,
        models: list[PairwisePreferenceModel] | None = None,
        mode: str = "fuse",
        device: str = "cpu",
    ):
        if models is None:
            models = [
                BradleyTerryModel(),
                LambdaMARTModel(),
                DPOModel(device=device),
            ]
        self._models = models
        self._mode = mode

    @property
    def name(self) -> str:
        return f"ensemble_{self._mode}"

    def fit(self, train_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
        for m in self._models:
            m.fit(train_df, audio_df)

    def predict_proba(self, test_df: pd.DataFrame, audio_df: pd.DataFrame) -> np.ndarray:
        all_probas = [m.predict_proba(test_df, audio_df) for m in self._models]
        n = len(test_df)
        result = np.full(n, np.nan)

        for i in range(n):
            probs = [float(p[i]) if not np.isnan(p[i]) else None for p in all_probas]
            if self._mode == "majority":
                p = _majority_vote(probs)
            else:
                p = _fuse_probs(probs)
            if p is not None:
                result[i] = np.clip(p, 1e-15, 1 - 1e-15)

        return result
