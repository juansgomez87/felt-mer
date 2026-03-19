"""Abstract base class for pairwise preference models."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PairwisePreferenceModel(ABC):
    """Interface for personalized pairwise preference models.

    All models learn from AB comparisons and predict P(A wins).
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
        """Train on pairwise comparisons.

        Parameters
        ----------
        train_df : Comparisons with columns song_a_id, song_b_id, winner.
        audio_df : Audio features with track_id column.
        """

    @abstractmethod
    def predict_proba(self, test_df: pd.DataFrame, audio_df: pd.DataFrame) -> np.ndarray:
        """Predict P(A wins) for each comparison.

        Parameters
        ----------
        test_df : Comparisons with columns song_a_id, song_b_id.
        audio_df : Audio features with track_id column.

        Returns
        -------
        Array of probabilities, one per row in test_df (nan for missing tracks).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model identifier (e.g. 'bt', 'lambdamart', 'dpo')."""
