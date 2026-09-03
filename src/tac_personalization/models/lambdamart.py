"""LambdaMART model via XGBoost ranking on audio features."""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from tac_personalization.config import (
    LAMBDAMART_NUM_BOOST_ROUND,
    LAMBDAMART_PARAMS,
)
from tac_personalization.data.features import get_feature_matrix
from tac_personalization.models.base import PairwisePreferenceModel


def _prepare_ranking_data(
    comparisons: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pairwise comparisons to XGBoost ranking format.

    Returns (X, y, qid) where each comparison has 2 items with the same qid.
    """
    feats, _ = get_feature_matrix(features_df)
    x_list, y_list, qid_list = [], [], []

    qid = 0
    for _, row in comparisons.iterrows():
        a, b = str(row["song_a_id"]), str(row["song_b_id"])
        if a not in feats.index or b not in feats.index:
            continue
        x_list.append(feats.loc[a].values)
        x_list.append(feats.loc[b].values)
        if row["winner"] == "A":
            y_list.extend([1, 0])
        else:
            y_list.extend([0, 1])
        qid_list.extend([qid, qid])
        qid += 1

    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(qid_list, dtype=np.int32),
    )


class LambdaMARTModel(PairwisePreferenceModel):
    """LambdaMART via XGBoost rank:pairwise objective."""

    def __init__(self, params: dict | None = None, num_boost_round: int = LAMBDAMART_NUM_BOOST_ROUND):
        self._model = None
        self._scaler = None
        self._params = params or LAMBDAMART_PARAMS
        self._num_boost_round = num_boost_round

    @property
    def name(self) -> str:
        return "lambdamart"

    def fit(self, train_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
        X, y, qid = _prepare_ranking_data(train_df, audio_df)
        if len(X) == 0:
            self._model = None
            return

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)

        unique_qids = np.unique(qid)
        groups = [int(np.sum(qid == q)) for q in unique_qids]

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)

        self._model = xgb.train(
            self._params,
            dtrain,
            num_boost_round=self._num_boost_round,
            verbose_eval=False,
        )

    def predict_proba(self, test_df: pd.DataFrame, audio_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.full(len(test_df), np.nan)

        feats, _ = get_feature_matrix(audio_df)
        probas = np.full(len(test_df), np.nan)

        for i, (_, row) in enumerate(test_df.iterrows()):
            a, b = str(row["song_a_id"]), str(row["song_b_id"])
            if a not in feats.index or b not in feats.index:
                continue
            X_pair = np.vstack([feats.loc[a].values, feats.loc[b].values])
            if self._scaler is not None:
                X_pair = self._scaler.transform(X_pair)
            scores = self._model.predict(xgb.DMatrix(X_pair))
            score_a, score_b = float(scores[0]), float(scores[1])
            probas[i] = float(np.clip(expit(score_a - score_b), 1e-15, 1 - 1e-15))

        return probas
