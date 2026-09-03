"""Tests for model implementations."""

import numpy as np
import pandas as pd
import pytest

from felt_mer.baselines.acoustic import AcousticBaseline
from felt_mer.data.loading import get_dimension_comparisons
from felt_mer.models.bt import BradleyTerryModel
from felt_mer.models.dpo import DPOModel
from felt_mer.models.ensemble import EnsembleModel
from felt_mer.models.lambdamart import LambdaMARTModel


class TestBradleyTerryModel:
    def test_fit_predict(self, sample_comparisons, sample_audio_df):
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        user_comp = comp[comp["user_id"] == 0]
        model = BradleyTerryModel()
        model.fit(user_comp, sample_audio_df)
        probas = model.predict_proba(user_comp, sample_audio_df)
        assert len(probas) == len(user_comp)
        valid = ~np.isnan(probas)
        assert valid.sum() > 0
        assert np.all(probas[valid] >= 0)
        assert np.all(probas[valid] <= 1)

    def test_name(self):
        assert BradleyTerryModel().name == "bt"

    def test_empty_data(self, sample_audio_df):
        empty = pd.DataFrame(columns=["user_id", "song_a_id", "song_b_id", "winner"])
        model = BradleyTerryModel()
        model.fit(empty, sample_audio_df)
        probas = model.predict_proba(empty, sample_audio_df)
        assert len(probas) == 0


class TestLambdaMARTModel:
    def test_fit_predict(self, sample_comparisons, sample_audio_df):
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        user_comp = comp[comp["user_id"] == 0]
        model = LambdaMARTModel()
        model.fit(user_comp, sample_audio_df)
        probas = model.predict_proba(user_comp, sample_audio_df)
        assert len(probas) == len(user_comp)
        valid = ~np.isnan(probas)
        assert valid.sum() > 0

    def test_name(self):
        assert LambdaMARTModel().name == "lambdamart"


@pytest.mark.skipif(
    "darwin" in __import__("sys").platform,
    reason="torch + pytest fork issue on macOS; verified via direct invocation",
)
class TestDPOModel:
    def test_fit_predict(self, sample_comparisons, sample_audio_df):
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        user_comp = comp[comp["user_id"] == 0]
        model = DPOModel(epochs=5, device="cpu")
        model.fit(user_comp, sample_audio_df)
        probas = model.predict_proba(user_comp, sample_audio_df)
        assert len(probas) == len(user_comp)
        valid = ~np.isnan(probas)
        assert valid.sum() > 0

    def test_name(self):
        assert DPOModel().name == "dpo"


class TestEnsembleModel:
    def test_fit_predict_no_dpo(self, sample_comparisons, sample_audio_df):
        """Ensemble with BT + LambdaMART only (no torch dependency)."""
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        user_comp = comp[comp["user_id"] == 0]
        model = EnsembleModel(
            models=[BradleyTerryModel(), LambdaMARTModel()],
            mode="fuse",
        )
        model.fit(user_comp, sample_audio_df)
        probas = model.predict_proba(user_comp, sample_audio_df)
        assert len(probas) == len(user_comp)

    def test_name(self):
        assert EnsembleModel(mode="fuse").name == "ensemble_fuse"


class TestAcousticBaseline:
    def test_predict_proba_arousal(self, sample_comparisons, sample_audio_df):
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        baseline = AcousticBaseline()
        probas = baseline.predict_proba(comp, sample_audio_df, "arousal")
        assert len(probas) == len(comp)
        valid = ~np.isnan(probas)
        assert valid.sum() > 0
        # Probabilities should be between 0 and 1
        assert np.all(probas[valid] >= 0)
        assert np.all(probas[valid] <= 1)

    def test_evaluate(self, sample_comparisons, sample_audio_df):
        comp = get_dimension_comparisons(sample_comparisons, "arousal")
        baseline = AcousticBaseline()
        metrics = baseline.evaluate(comp, sample_audio_df, "arousal")
        assert metrics is not None
        assert "acc" in metrics
        assert "brier" in metrics
