"""Data loading utilities: comparisons, mapping, user directories."""

import json
from pathlib import Path

import pandas as pd

from tac_personalization.config import (
    COMPARISONS_PARQUET,
    COMPARISONS_CSV,
    MAPPING_FILE,
    USER_DATA_DIR,
)


def load_comparisons(
    parquet_path: Path = COMPARISONS_PARQUET,
    csv_path: Path = COMPARISONS_CSV,
) -> pd.DataFrame:
    """Load comparison table (parquet preferred, csv fallback)."""
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Neither {parquet_path} nor {csv_path} found. Run scripts/build_dataset.py first."
    )


def load_mapping(mapping_path: Path = MAPPING_FILE) -> dict:
    """Load unified user mapping JSON."""
    if not mapping_path.exists():
        return {}
    with open(mapping_path) as f:
        return json.load(f)


def get_real_user_ids(mapping: dict | None = None) -> list[int]:
    """Return sorted list of real (non-simulated) user IDs."""
    if mapping is None:
        mapping = load_mapping()
    id_to_info = mapping.get("id_to_info", {})
    return sorted(int(k) for k, v in id_to_info.items() if v.get("is_real", True))


def get_dimension_comparisons(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Filter to rows where the dimension winner is not null.

    Returns columns: user_id, song_a_id, song_b_id, winner (A or B).
    """
    col = f"{dimension}_winner"
    sub = df[df[col].notna()][["user_id", "song_a_id", "song_b_id", col]].copy()
    return sub.rename(columns={col: "winner"})


def get_user_dir(user_data_dir: Path = USER_DATA_DIR, user_id: int | str = 0) -> Path:
    """Return directory for a user. Prefers real/ subdirectory when it exists."""
    real_dir = user_data_dir / "real"
    if real_dir.exists():
        return real_dir / str(user_id)
    return user_data_dir / str(user_id)


def load_audio_features(path: Path) -> pd.DataFrame:
    """Load audio features parquet."""
    return pd.read_parquet(path)
