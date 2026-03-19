"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_audio_df():
    """Audio features for 10 synthetic tracks."""
    rng = np.random.default_rng(42)
    n_tracks = 10
    tracks = {
        "track_id": [f"track_{i}" for i in range(n_tracks)],
        "danceability": rng.uniform(0, 1, n_tracks),
        "energy": rng.uniform(0, 1, n_tracks),
        "valence": rng.uniform(0, 1, n_tracks),
        "acousticness": rng.uniform(0, 1, n_tracks),
        "instrumentalness": rng.uniform(0, 1, n_tracks),
        "speechiness": rng.uniform(0, 1, n_tracks),
        "liveness": rng.uniform(0, 1, n_tracks),
        "tempo": rng.uniform(60, 200, n_tracks),
        "key": rng.integers(0, 12, n_tracks).astype(float),
        "loudness": rng.uniform(-30, 0, n_tracks),
        "mode": rng.integers(0, 2, n_tracks).astype(float),
    }
    return pd.DataFrame(tracks)


@pytest.fixture
def sample_comparisons(sample_audio_df):
    """50 synthetic pairwise comparisons for 2 users x 2 dimensions."""
    rng = np.random.default_rng(123)
    track_ids = sample_audio_df["track_id"].tolist()
    rows = []
    for user_id in [0, 1]:
        for dim in ["arousal", "valence"]:
            for _ in range(25):
                a, b = rng.choice(track_ids, size=2, replace=False)
                winner = rng.choice(["A", "B"])
                rows.append({
                    "user_id": user_id,
                    "song_a_id": a,
                    "song_b_id": b,
                    "preference_winner": None,
                    "arousal_winner": winner if dim == "arousal" else None,
                    "valence_winner": winner if dim == "valence" else None,
                })
    return pd.DataFrame(rows)
