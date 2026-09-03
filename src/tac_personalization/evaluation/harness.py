"""Unified evaluation harness: per-user cross-validation for any model + baseline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tac_personalization.config import (
    AUDIO_FEATURES_PATH,
    CV_SEED,
    DIMENSIONS,
    MIN_PAIRS_CV,
    N_FOLDS,
)
from tac_personalization.baselines.acoustic import AcousticBaseline
from tac_personalization.baselines.general_bt import GeneralBTBaseline
from tac_personalization.data.features import get_feature_matrix
from tac_personalization.data.loading import (
    get_dimension_comparisons,
    get_real_user_ids,
    load_audio_features,
    load_comparisons,
    load_mapping,
)
from tac_personalization.evaluation.metrics import compute_binary_metrics
from tac_personalization.evaluation.results import build_results_row
from tac_personalization.models.base import PairwisePreferenceModel


def _nanmean(x: list[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else np.nan


def _nanstd(x: list[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanstd(a)) if np.sum(np.isfinite(a)) > 1 else np.nan


def evaluate_user_cv(
    model: PairwisePreferenceModel,
    comparisons_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    user_id: int,
    dimension: str,
    baseline: AcousticBaseline | None = None,
    general_baseline: GeneralBTBaseline | None = None,
    n_folds: int = N_FOLDS,
) -> dict | None:
    """Run n-fold stratified CV for one user x dimension.

    Trains model per fold; evaluates model, acoustic baseline, and optionally
    general BT baseline on the same test fold.

    The general_baseline should already be fit (via fit_pooled) before calling
    this function — it is not re-fit per fold since it excludes the test user
    entirely.

    Returns dict with mean/std metrics for personalized and baseline, or None
    if insufficient data.
    """
    user_comp = comparisons_df[comparisons_df["user_id"] == user_id].reset_index(drop=True)
    if len(user_comp) == 0:
        return None

    # Pre-filter to pairs where both tracks have features
    feats, _ = get_feature_matrix(audio_df)
    valid_mask = user_comp.apply(
        lambda row: str(row["song_a_id"]) in feats.index and str(row["song_b_id"]) in feats.index,
        axis=1,
    )
    user_comp = user_comp[valid_mask].reset_index(drop=True)
    if len(user_comp) < MIN_PAIRS_CV:
        return None

    y_all = np.array([1 if row["winner"] == "A" else 0 for _, row in user_comp.iterrows()])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED)

    acc_pers, acc_bl = [], []
    auc_pers, auc_bl = [], []
    brier_pers, brier_bl = [], []
    ll_pers, ll_bl = [], []
    acc_gbl, auc_gbl, brier_gbl, ll_gbl = [], [], [], []

    if baseline is None:
        baseline = AcousticBaseline()

    for train_idx, test_idx in skf.split(np.arange(len(y_all)), y_all):
        train_df = user_comp.iloc[train_idx].reset_index(drop=True)
        test_df = user_comp.iloc[test_idx].reset_index(drop=True)
        y_test = y_all[test_idx]

        # Train and predict with personalized model
        model.fit(train_df, audio_df)
        p_pers = model.predict_proba(test_df, audio_df)

        # Acoustic baseline on same test fold
        p_bl = baseline.predict_proba(test_df, audio_df, dimension)

        # Only evaluate where both have valid predictions
        valid = ~np.isnan(p_pers) & ~np.isnan(p_bl)
        if valid.sum() == 0:
            continue

        m_pers = compute_binary_metrics(y_test[valid], np.clip(p_pers[valid], 1e-15, 1 - 1e-15))
        m_bl = compute_binary_metrics(y_test[valid], np.clip(p_bl[valid], 1e-15, 1 - 1e-15))

        acc_pers.append(m_pers["acc"])
        acc_bl.append(m_bl["acc"])
        auc_pers.append(m_pers["auc_roc"])
        auc_bl.append(m_bl["auc_roc"])
        brier_pers.append(m_pers["brier"])
        brier_bl.append(m_bl["brier"])
        ll_pers.append(m_pers["ll"])
        ll_bl.append(m_bl["ll"])

        # General BT baseline (already fit on all other users)
        if general_baseline is not None:
            p_gbl = general_baseline.predict_proba(test_df, audio_df, dimension)
            valid_gbl = valid & ~np.isnan(p_gbl)
            if valid_gbl.sum() > 0:
                m_gbl = compute_binary_metrics(y_test[valid_gbl], np.clip(p_gbl[valid_gbl], 1e-15, 1 - 1e-15))
                acc_gbl.append(m_gbl["acc"])
                auc_gbl.append(m_gbl["auc_roc"])
                brier_gbl.append(m_gbl["brier"])
                ll_gbl.append(m_gbl["ll"])

    if not acc_pers:
        return None

    result = {
        "acc_personalized_mean": _nanmean(acc_pers),
        "acc_personalized_std": _nanstd(acc_pers),
        "acc_baseline_mean": _nanmean(acc_bl),
        "acc_baseline_std": _nanstd(acc_bl),
        "auc_personalized_mean": _nanmean(auc_pers),
        "auc_personalized_std": _nanstd(auc_pers),
        "auc_baseline_mean": _nanmean(auc_bl),
        "auc_baseline_std": _nanstd(auc_bl),
        "brier_personalized_mean": _nanmean(brier_pers),
        "brier_personalized_std": _nanstd(brier_pers),
        "brier_baseline_mean": _nanmean(brier_bl),
        "brier_baseline_std": _nanstd(brier_bl),
        "ll_personalized_mean": _nanmean(ll_pers),
        "ll_personalized_std": _nanstd(ll_pers),
        "ll_baseline_mean": _nanmean(ll_bl),
        "ll_baseline_std": _nanstd(ll_bl),
        "n_folds": n_folds,
        "n_pairs_cv": len(user_comp),
    }

    if acc_gbl:
        result["acc_general_bt_mean"] = _nanmean(acc_gbl)
        result["acc_general_bt_std"] = _nanstd(acc_gbl)
        result["auc_general_bt_mean"] = _nanmean(auc_gbl)
        result["brier_general_bt_mean"] = _nanmean(brier_gbl)
        result["ll_general_bt_mean"] = _nanmean(ll_gbl)

    return result


def run_experiment(
    model: PairwisePreferenceModel,
    baseline: AcousticBaseline | None = None,
    audio_features_path=AUDIO_FEATURES_PATH,
    output_path=None,
    dimensions: tuple[str, ...] = DIMENSIONS,
    n_folds: int = N_FOLDS,
    include_general_bt: bool = True,
) -> pd.DataFrame | None:
    """Run full experiment: loop user x dimension, CV, collect results.

    When include_general_bt=True, also evaluates a pooled BT baseline
    trained on all other users' comparisons (leave-one-user-out).
    """
    df = load_comparisons()
    mapping = load_mapping()
    real_user_ids = get_real_user_ids(mapping)
    audio_df = load_audio_features(audio_features_path)

    if baseline is None:
        baseline = AcousticBaseline()

    print(f"=== {model.name.upper()} vs {baseline.name} baseline ({n_folds}-fold CV) ===")
    print(f"Users: {len(real_user_ids)}, Audio features: {len(audio_df)} tracks\n")

    results = []
    for dim in dimensions:
        comp_all = get_dimension_comparisons(df, dim)
        if len(comp_all) == 0:
            print(f"[{dim}] No comparisons, skipping.")
            continue
        print(f"--- {dim.upper()} ---")

        for uid in real_user_ids:
            # Fit general BT baseline on all other users (leave-one-user-out)
            general_baseline = None
            if include_general_bt:
                general_baseline = GeneralBTBaseline()
                general_baseline.fit_pooled(comp_all, audio_df, exclude_user_id=uid)

            cv_res = evaluate_user_cv(
                model, comp_all, audio_df, uid, dim,
                baseline=baseline,
                general_baseline=general_baseline,
                n_folds=n_folds,
            )
            if cv_res is None:
                continue

            row = build_results_row(uid, dim, cv_res)
            results.append(row)

    if not results:
        print("\nNo results (insufficient data or audio coverage).")
        return None

    res_df = pd.DataFrame(results)

    if output_path is not None:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    return res_df
