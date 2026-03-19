"""Result building and summary printing utilities."""

import numpy as np
import pandas as pd


def build_results_row(user_id: int, dimension: str, cv_res: dict) -> dict:
    """Build a standardized results row from CV output.

    Computes improvement and beat_baseline flags for all metrics.
    """
    acc_pers = cv_res["acc_personalized_mean"]
    acc_bl = cv_res["acc_baseline_mean"]
    auc_pers = cv_res["auc_personalized_mean"]
    auc_bl = cv_res["auc_baseline_mean"]
    brier_pers = cv_res["brier_personalized_mean"]
    brier_bl = cv_res["brier_baseline_mean"]
    ll_pers = cv_res["ll_personalized_mean"]
    ll_bl = cv_res["ll_baseline_mean"]

    def _safe_diff(a, b):
        if np.isfinite(a) and np.isfinite(b):
            return a - b
        return np.nan

    return {
        "user_id": user_id,
        "dimension": dimension,
        "n_pairs_cv": cv_res["n_pairs_cv"],
        "n_folds": cv_res["n_folds"],
        "acc_personalized_mean": acc_pers,
        "acc_personalized_std": cv_res["acc_personalized_std"],
        "acc_baseline_mean": acc_bl,
        "acc_baseline_std": cv_res["acc_baseline_std"],
        "acc_improvement": _safe_diff(acc_pers, acc_bl),
        "auc_personalized_mean": auc_pers,
        "auc_personalized_std": cv_res["auc_personalized_std"],
        "auc_baseline_mean": auc_bl,
        "auc_baseline_std": cv_res["auc_baseline_std"],
        "auc_improvement": _safe_diff(auc_pers, auc_bl),
        "brier_personalized_mean": brier_pers,
        "brier_personalized_std": cv_res["brier_personalized_std"],
        "brier_baseline_mean": brier_bl,
        "brier_baseline_std": cv_res["brier_baseline_std"],
        # Lower brier is better, so improvement = baseline - personalized
        "brier_improvement": _safe_diff(brier_bl, brier_pers),
        "ll_personalized_mean": ll_pers,
        "ll_personalized_std": cv_res["ll_personalized_std"],
        "ll_baseline_mean": ll_bl,
        "ll_baseline_std": cv_res["ll_baseline_std"],
        "ll_improvement": _safe_diff(ll_pers, ll_bl),
        "beat_baseline_acc": acc_pers > acc_bl,
        "beat_baseline_auc": _safe_diff(auc_pers, auc_bl) > 0 if np.isfinite(auc_pers) and np.isfinite(auc_bl) else np.nan,
        "beat_baseline_brier": brier_pers < brier_bl if np.isfinite(brier_pers) and np.isfinite(brier_bl) else np.nan,
        "beat_baseline_ll": ll_pers > ll_bl if np.isfinite(ll_pers) and np.isfinite(ll_bl) else np.nan,
    }


def print_summary(res_df: pd.DataFrame, baseline_label: str = "Baseline") -> None:
    """Print aggregate summary from results DataFrame."""
    if res_df is None or len(res_df) == 0:
        print("No results.")
        return

    print(f"\nTotal evaluations: {len(res_df)} (user x dimension)")
    print(f"Unique users: {res_df['user_id'].nunique()}")

    for metric, label in [
        ("acc", "Accuracy"),
        ("auc", "AUC-ROC"),
        ("brier", "Brier"),
        ("ll", "Log-Likelihood"),
    ]:
        mean_p = f"{metric}_personalized_mean"
        mean_b = f"{metric}_baseline_mean"
        imp = f"{metric}_improvement"
        beat = f"beat_baseline_{metric}"
        if mean_p not in res_df.columns:
            continue
        print(f"\n--- {label} ---")
        print(f"  Personalized: {res_df[mean_p].mean():.4f}")
        print(f"  {baseline_label}: {res_df[mean_b].mean():.4f}")
        print(f"  Improvement: {res_df[imp].mean():.4f}")
        if beat in res_df.columns:
            print(f"  Beat baseline: {100 * res_df[beat].mean():.1f}%")

    if "acc_improvement" in res_df.columns:
        try:
            from scipy import stats
            vals = res_df["acc_improvement"].dropna().values
            if len(vals) >= 2:
                _, p = stats.wilcoxon(vals, alternative="greater")
                print(f"\nWilcoxon (acc improvement > 0): p={p:.4f}")
        except Exception:
            pass

    if "dimension" in res_df.columns:
        print("\n--- Per Dimension ---")
        for dim in res_df["dimension"].unique():
            sub = res_df[res_df["dimension"] == dim]
            acc_imp = sub["acc_improvement"].mean()
            beat_pct = 100 * sub["beat_baseline_acc"].mean()
            print(f"  {dim}: N={len(sub)}, acc improvement={acc_imp:.3f}, beat baseline={beat_pct:.1f}%")
