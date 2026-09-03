#!/usr/bin/env python3
"""One-time data prep: build unified user mapping and comparison dataset.

Merges logic from the original build_bt_dataset.py and export_features.py.

Usage:
  python scripts/build_dataset.py --source-dir /path/to/spotimeta/user_data
  python scripts/build_dataset.py --source-dir /path/to/spotimeta/user_data --audio-parquet /path/to/track_audio_features.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from felt_mer.config import (
    ALL_AUDIO_COLS,
    AUDIO_FEATURES_PATH,
    COMPARISONS_CSV,
    COMPARISONS_PARQUET,
    DATA_DIR,
    MAPPING_FILE,
)


def find_final_results_dirs(root: Path) -> list[Path]:
    """Return directories containing final_results.json."""
    return sorted(
        (p.parent for p in root.rglob("final_results.json")),
        key=lambda p: str(p),
    )


def is_simulated(profile: dict | None) -> bool:
    if profile is None:
        return False
    return profile.get("profile_type") == "simulated"


def load_profile(session_dir: Path) -> dict | None:
    p = session_dir / "user_profile.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_unified_mapping(root: Path) -> tuple[dict, dict]:
    """Scan for sessions, classify real vs simulated, assign IDs."""
    session_dirs = find_final_results_dirs(root)
    path_to_id = {}
    id_to_info = {}
    for new_id, session_dir in enumerate(session_dirs):
        try:
            rel = str(session_dir.relative_to(root))
        except ValueError:
            rel = str(session_dir)
        path_to_id[rel] = new_id
        profile = load_profile(session_dir)
        id_to_info[new_id] = {"path": rel, "is_real": not is_simulated(profile)}
    return path_to_id, id_to_info


def build_comparison_rows(session_dir: Path, user_id: int) -> list[dict]:
    """Load test_*.json and build comparison rows."""
    rows = []
    for dim, filename in [
        ("preference", "test_preference.json"),
        ("arousal", "test_arousal.json"),
        ("valence", "test_valence.json"),
    ]:
        p = session_dir / filename
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for e in data.get("evaluations", []):
            ta = e.get("track_a") or {}
            tb = e.get("track_b") or {}
            song_a = ta.get("track_id") if isinstance(ta, dict) else None
            song_b = tb.get("track_id") if isinstance(tb, dict) else None
            if not song_a or not song_b:
                continue
            selected = e.get("selected_track")
            if selected == "track_a":
                winner = "A"
            elif selected == "track_b":
                winner = "B"
            else:
                winner = None
            row = {
                "user_id": user_id,
                "song_a_id": song_a,
                "song_b_id": song_b,
                "preference_winner": None,
                "arousal_winner": None,
                "valence_winner": None,
            }
            row[f"{dim}_winner"] = winner
            rows.append(row)
    return rows


def build_comparisons(source_dir: Path) -> pd.DataFrame:
    """Build full comparison DataFrame from source user_data."""
    path_to_id, id_to_info = build_unified_mapping(source_dir)

    # Save mapping
    mapping = {
        "path_to_id": {k: v for k, v in path_to_id.items()},
        "id_to_info": {str(k): v for k, v in id_to_info.items()},
    }
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping: {len(path_to_id)} sessions -> {MAPPING_FILE}")

    # Build comparisons
    real_dir = source_dir / "real"
    all_rows = []
    for new_id, info in id_to_info.items():
        if not info["is_real"]:
            continue
        candidate = real_dir / str(new_id)
        session_dir = candidate if candidate.exists() else (source_dir / info["path"])
        if not session_dir.exists():
            continue
        all_rows.extend(build_comparison_rows(session_dir, new_id))

    if not all_rows:
        raise ValueError("No comparison rows from real users.")

    df = pd.DataFrame(all_rows)
    df = df[["user_id", "song_a_id", "song_b_id", "preference_winner", "arousal_winner", "valence_winner"]]

    COMPARISONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(COMPARISONS_PARQUET, index=False)
        print(f"Comparisons: {len(df)} rows -> {COMPARISONS_PARQUET}")
    except Exception:
        df.to_csv(COMPARISONS_CSV, index=False)
        print(f"Comparisons: {len(df)} rows -> {COMPARISONS_CSV}")
    return df


def export_audio_features(comparisons: pd.DataFrame, audio_parquet: Path) -> pd.DataFrame:
    """Filter audio features to comparison tracks and save."""
    import duckdb

    track_ids = list(
        set(comparisons["song_a_id"].astype(str)) | set(comparisons["song_b_id"].astype(str))
    )

    con = duckdb.connect(database=":memory:")
    path_str = str(audio_parquet.resolve())
    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path_str]).fetchall()
    col_names = [row[0] for row in schema]
    id_col = "track_id" if "track_id" in col_names else "id"

    available = [c for c in ALL_AUDIO_COLS if c in col_names]
    select_expr = f"CAST({id_col} AS VARCHAR) AS track_id, " + ", ".join(
        f"CAST({c} AS DOUBLE) AS {c}" for c in available
    )
    query = f"""
        SELECT {select_expr}
        FROM read_parquet(?)
        WHERE CAST({id_col} AS VARCHAR) IN (SELECT unnest(?))
    """
    out = con.execute(query, [path_str, track_ids]).fetchdf()
    con.close()

    out = out.dropna().reset_index(drop=True)
    AUDIO_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AUDIO_FEATURES_PATH, index=False)
    print(f"Audio features: {len(out)} tracks -> {AUDIO_FEATURES_PATH}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build dataset from source user data.")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Path to original user_data directory.")
    parser.add_argument("--audio-parquet", type=Path, default=None,
                        help="Path to track_audio_features.parquet (for re-exporting).")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = build_comparisons(args.source_dir)

    if args.audio_parquet and args.audio_parquet.exists():
        export_audio_features(comparisons, args.audio_parquet)
    else:
        print("Skipping audio feature export (no --audio-parquet provided).")
        print("Copy audio_features.parquet to data/raw/ manually if needed.")


if __name__ == "__main__":
    main()
