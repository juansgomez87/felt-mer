"""Publication figures for personalization results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tac_personalization.analysis.segmentation import PATTERN_ORDER
from tac_personalization.config import FIGURE_RCPARAMS, USER_DATA_DIR
from tac_personalization.data.loading import get_user_dir

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _load_surveys(user_ids, user_data_dir=USER_DATA_DIR):
    """Load survey.json for each user ID."""
    out = {}
    for uid in user_ids:
        path = get_user_dir(user_data_dir, uid) / "survey.json"
        if path.exists():
            try:
                with open(path) as f:
                    out[int(uid)] = json.load(f)
            except Exception:
                out[int(uid)] = None
        else:
            out[int(uid)] = None
    return out


def plot_agreement_with_spotify(res_df, segment_df, output_path):
    """Agreement with Spotify: mean by dimension + violin by segment."""
    if not HAS_MPL or "agreement_with_audio_all_pct" not in res_df.columns:
        return None
    # Use agreement_with_audio_all_pct as the agreement column
    res_df = res_df.copy()
    if "agreement_with_audio_pct" not in res_df.columns:
        res_df["agreement_with_audio_pct"] = res_df["agreement_with_audio_all_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dims = ["arousal", "valence"]
    labels = ["Arousal\n(energy)", "Valence"]
    colors = ["#e74c3c", "#3498db"]

    means = [res_df[res_df["dimension"] == d]["agreement_with_audio_pct"].mean() for d in dims]
    stds = [res_df[res_df["dimension"] == d]["agreement_with_audio_pct"].std() for d in dims]
    ns = [res_df[res_df["dimension"] == d]["agreement_with_audio_pct"].count() for d in dims]

    bars = axes[0].bar(labels, means, color=colors, edgecolor="black", linewidth=1.5, alpha=0.7)
    axes[0].errorbar(range(len(labels)), means, yerr=stds, fmt="none", ecolor="black", capsize=5)
    axes[0].set_ylabel("Agreement with Spotify (%)")
    axes[0].set_title("A. Mean Agreement with Spotify")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(50, color="gray", linestyle="--", alpha=0.7, label="Chance (50%)")
    axes[0].legend(loc="upper right")

    for i, (b, m, s, n) in enumerate(zip(bars, means, stds, ns)):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + (s or 0) + 2,
                     f"{m:.1f}%\n(n={int(n)})", ha="center", fontsize=10)

    # Panel B: Distribution
    data = [res_df[res_df["dimension"] == d]["agreement_with_audio_pct"].dropna().values for d in dims]
    if all(len(d) > 0 for d in data):
        parts = axes[1].violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        for idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.6)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Agreement with Spotify (%)")
    axes[1].set_title("B. Distribution of Agreement")
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return fig


def plot_improvement_heatmap(res_df, output_path):
    """Outcome counts and improvement heatmap by user x dimension."""
    if not HAS_MPL:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    dims = ["arousal", "valence"]
    labels_short = ["Arousal", "Valence"]

    def _outcome_counts(series):
        return (series > 0.05).sum(), ((series >= -0.05) & (series <= 0.05)).sum(), (series < -0.05).sum()

    x = np.arange(len(dims))
    h_counts = [_outcome_counts(res_df[res_df["dimension"] == d]["acc_improvement"])[0] for d in dims]
    n_counts = [_outcome_counts(res_df[res_df["dimension"] == d]["acc_improvement"])[1] for d in dims]
    hurt_counts = [_outcome_counts(res_df[res_df["dimension"] == d]["acc_improvement"])[2] for d in dims]

    axes[0].bar(x, h_counts, 0.6, label="Helped", color="#2ecc71", edgecolor="black")
    axes[0].bar(x, n_counts, 0.6, bottom=h_counts, label="Neutral", color="#95a5a6", edgecolor="black")
    axes[0].bar(x, hurt_counts, 0.6, bottom=np.array(h_counts) + np.array(n_counts),
                label="Hurt", color="#e74c3c", alpha=0.8, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_short)
    axes[0].set_ylabel("Number of models")
    axes[0].set_title("A. Outcome: Helped | Neutral | Hurt")
    axes[0].legend(loc="upper right")

    pivot_imp = res_df.pivot_table(index="user_id", columns="dimension", values="acc_improvement", aggfunc="first")
    pivot_imp = pivot_imp.dropna(how="all")
    pivot_imp["_mean"] = pivot_imp.mean(axis=1)
    pivot_imp = pivot_imp.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    im = axes[1].imshow(pivot_imp.values * 100, aspect="auto", cmap="RdYlGn", vmin=-25, vmax=25)
    axes[1].set_xticks(range(len(pivot_imp.columns)))
    axes[1].set_xticklabels(pivot_imp.columns.tolist())
    axes[1].set_yticks(range(len(pivot_imp)))
    axes[1].set_yticklabels(pivot_imp.index.tolist(), fontsize=7)
    axes[1].set_ylabel("User ID")
    axes[1].set_title("B. Improvement by User and Dimension (% points)")
    plt.colorbar(im, ax=axes[1], label="Accuracy change (%)")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return fig


def plot_user_patterns(res_df, pattern_df, output_path):
    """Pattern distribution donut + per-user delta space (manuscript Fig. 3)."""
    if not HAS_MPL or pattern_df is None:
        return None

    labels = ["Both helped", "Arousal only", "Valence only", "Neither"]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]
    counts = pattern_df["pattern"].value_counts()
    sizes = [int(counts.get(p, 0)) for p in PATTERN_ORDER]

    with plt.rc_context(FIGURE_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].pie(sizes, labels=[f"{lbl}\n(n={n})" for lbl, n in zip(labels, sizes)],
                    colors=colors, startangle=90, pctdistance=0.75,
                    autopct=lambda pct: f"{pct:.0f}%" if pct else "",
                    textprops={"fontsize": 10})
        axes[0].add_artist(plt.Circle((0, 0), 0.5, fc="white"))
        axes[0].set_title("A. Pattern distribution across users")

        for pattern, color in zip(PATTERN_ORDER, colors):
            sub = pattern_df[pattern_df["pattern"] == pattern]
            if sub.empty:
                continue
            axes[1].scatter(sub["arousal_improvement"] * 100, sub["valence_improvement"] * 100,
                            s=55, alpha=0.85, color=color, edgecolors="black", linewidth=0.8,
                            label=f"{pattern.replace('_', ' ')} (n={len(sub)})")
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[1].axvline(0, color="gray", linewidth=1)
        axes[1].set_xlabel("Arousal accuracy delta (pp)")
        axes[1].set_ylabel("Valence accuracy delta (pp)")
        axes[1].set_title("B. Per-user delta space")
        axes[1].legend(loc="best", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig


def plot_survey_correlates(pattern_df, surveys, output_path):
    """BMRQ, music experience, age, genre differences by benefited vs not."""
    if not HAS_MPL:
        return None

    success_patterns = {"both", "arousal_only", "valence_only"}
    rows = []
    for uid, s in (surveys or {}).items():
        if s is None:
            continue
        uid_int = int(uid)
        pat_row = pattern_df[pattern_df["user_id"].astype(int) == uid_int] if pattern_df is not None else pd.DataFrame()
        pat = pat_row["pattern"].iloc[0] if len(pat_row) > 0 else "neither"

        bmrq = 0
        for k, v in (s.get("bmrq_responses") or {}).items():
            try:
                score = int(v)
                if k in ("question_2", "question_5"):
                    score = 6 - score
                bmrq += score
            except (ValueError, TypeError):
                pass

        rows.append({
            "user_id": uid_int,
            "pattern": pat,
            "benefited": pat in success_patterns,
            "bmrq_total": bmrq if bmrq else np.nan,
            "age": s.get("demographics", {}).get("age"),
            "music_experience": s.get("demographics", {}).get("music_experience"),
            "survey": s,
        })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if not rows:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No survey data", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        return fig

    survey_df = pd.DataFrame(rows)

    # A: BMRQ by benefited
    ax = axes[0, 0]
    b_vals = survey_df[survey_df["benefited"]]["bmrq_total"].dropna()
    o_vals = survey_df[~survey_df["benefited"]]["bmrq_total"].dropna()
    to_plot, labels = [], []
    if len(b_vals):
        to_plot.append(b_vals.values)
        labels.append(f"Benefited (n={len(b_vals)})")
    if len(o_vals):
        to_plot.append(o_vals.values)
        labels.append(f"Mixed/Neither (n={len(o_vals)})")
    if to_plot:
        ax.violinplot(to_plot, positions=np.arange(len(to_plot)), showmeans=True, showmedians=True)
        ax.set_xticks(np.arange(len(to_plot)))
        ax.set_xticklabels(labels)
    ax.set_ylabel("BMRQ Score")
    ax.set_title("A. BMRQ by Personalization Outcome")

    # B: Music experience
    ax = axes[0, 1]
    if "music_experience" in survey_df.columns and survey_df["music_experience"].notna().any():
        exp_means = survey_df.groupby("music_experience").agg(
            benefited=("benefited", "mean"), n=("user_id", "count")
        ).reset_index()
        x = np.arange(len(exp_means))
        ax.bar(x, exp_means["benefited"] * 100, 0.4, color="#2ecc71", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_means["music_experience"], rotation=20, ha="right")
        for i, (_, row) in enumerate(exp_means.iterrows()):
            ax.text(i, row["benefited"] * 100 + 2, f"n={row['n']}", ha="center", fontsize=9)
    ax.set_ylabel("% Benefited")
    ax.set_title("B. Music Experience")

    # C: Age
    ax = axes[1, 0]
    survey_df["age_num"] = pd.to_numeric(survey_df["age"], errors="coerce")
    b_age = survey_df[survey_df["benefited"]]["age_num"].dropna()
    o_age = survey_df[~survey_df["benefited"]]["age_num"].dropna()
    if len(b_age):
        ax.hist(b_age, bins=10, alpha=0.6, label=f"Benefited (med={b_age.median():.0f})", color="#2ecc71", edgecolor="black")
    if len(o_age):
        ax.hist(o_age, bins=10, alpha=0.6, label=f"Mixed/Neither (med={o_age.median():.0f})", color="#e74c3c", edgecolor="black")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("C. Age Distribution")
    ax.legend()

    # D: Genre differences
    ax = axes[1, 1]
    stomp_b, stomp_o = {}, {}
    for _, row in survey_df.iterrows():
        s = row.get("survey") or {}
        for g, v in (s.get("stomp_responses") or {}).items():
            try:
                val = int(v)
                target = stomp_b if row["benefited"] else stomp_o
                target[g] = target.get(g, []) + [val]
            except (ValueError, TypeError):
                pass
    if stomp_b and stomp_o:
        diff = {}
        for g in set(stomp_b) | set(stomp_o):
            mb = np.mean(stomp_b[g]) if g in stomp_b else 0
            mo = np.mean(stomp_o[g]) if g in stomp_o else 0
            diff[g] = mb - mo
        top = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)[:18]
        genres = [t[0] for t in top]
        diffs = [t[1] for t in top]
        colors_d = ["#2ecc71" if d > 0 else "#e74c3c" for d in diffs]
        y_pos = np.arange(len(genres))
        ax.barh(y_pos, diffs, color=colors_d, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genres, fontsize=9)
        ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean preference diff")
    ax.set_title("D. Top Genre Differences")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return fig


def plot_statistical_summary(res_df, pattern_df, output_path):
    """Effect size table and 2x2 outcome grid."""
    if not HAS_MPL:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Effect size table
    axes[0].axis("off")
    dims = ["arousal", "valence"]
    lines = ["Dimension | Mean delta Acc (%) | 95% CI | Wilcoxon p | n"]
    lines.append("-" * 55)
    for d in dims:
        sub = res_df[res_df["dimension"] == d]["acc_improvement"] * 100
        n = len(sub)
        mean = sub.mean()
        se = sub.std() / np.sqrt(n) if n > 0 else 0
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        pstr = "--"
        if scipy_stats and n >= 2:
            try:
                sub_vals = np.asarray(sub, dtype=float)
                if np.any(sub_vals != 0):
                    _, p = scipy_stats.wilcoxon(sub_vals, alternative="two-sided")
                    pstr = f"{p:.4f}"
            except Exception:
                pass
        lines.append(f"{d:10} | {mean:+.1f}           | [{ci_lo:.1f}, {ci_hi:.1f}] | {pstr} | {n}")
    axes[0].text(0.1, 0.9, "\n".join(lines), transform=axes[0].transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    axes[0].set_title("A. Effect Size Summary")

    # Panel B: 2x2 outcome grid
    if pattern_df is not None:
        both = (pattern_df["pattern"] == "both").sum()
        a_only = (pattern_df["pattern"] == "arousal_only").sum()
        v_only = (pattern_df["pattern"] == "valence_only").sum()
        neither = (pattern_df["pattern"] == "neither").sum()
        grid = np.array([[both, v_only], [a_only, neither]])
        im = axes[1].imshow(grid, cmap="Blues")
        for i in range(2):
            for j in range(2):
                val = grid[i, j]
                lbl = ["Both", "Valence only", "Arousal only", "Neither"][i * 2 + j]
                pct = 100 * val / grid.sum() if grid.sum() > 0 else 0
                axes[1].text(j, i, f"{lbl}\n{int(val)} ({pct:.0f}%)", ha="center", va="center",
                             fontsize=9, fontweight="bold",
                             color="white" if val > grid.max() / 2 else "black")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Arousal helped", "Arousal no"])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Valence helped", "Valence no"])
        axes[1].set_title("B. Outcome Grid (user counts)")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return fig


def plot_feature_importance(raw_csv_path, output_path, metric="mean_abs_coef"):
    """Bar chart with error bars for BT logistic regression feature importance."""
    if not HAS_MPL:
        return None

    df = pd.read_csv(raw_csv_path)
    if df.empty:
        return None

    # Aggregate raw format to summary
    if "coef" in df.columns and "fold" in df.columns:
        agg = df.groupby(["dimension", "feature"])["coef"].agg(
            mean_coef="mean",
            std_coef="std",
            mean_abs_coef=lambda x: np.abs(x).mean(),
            std_abs_coef=lambda x: np.abs(x).std(),
        ).reset_index()
        df = agg

    err_col = "std_abs_coef" if metric == "mean_abs_coef" else "std_coef"
    dimensions = df["dimension"].unique().tolist()
    n_dims = len(dimensions)

    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 4 + 0.3 * df["feature"].nunique()))
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):
        sub = df[df["dimension"] == dim].sort_values(metric, ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(sub))
        vals = sub[metric].values
        err = sub[err_col].values if err_col in sub.columns else None
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sub)))
        ax.barh(y_pos, vals, color=colors, edgecolor="gray", linewidth=0.5)
        if err is not None:
            ax.errorbar(vals, y_pos, xerr=err, fmt="none", ecolor="gray", capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["feature"].tolist(), fontsize=10)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(dim.capitalize())
        if metric == "mean_coef":
            ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
