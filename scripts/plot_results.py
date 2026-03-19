#!/usr/bin/env python3
"""Generate publication figures from results CSV."""

import argparse
from pathlib import Path

import pandas as pd

from tac_personalization.analysis.plotting import (
    _load_surveys,
    plot_agreement_with_spotify,
    plot_feature_importance,
    plot_improvement_heatmap,
    plot_statistical_summary,
    plot_survey_correlates,
    plot_user_patterns,
)
from tac_personalization.analysis.segmentation import (
    dimension_specific_patterns,
    segment_users_by_personalization,
)
from tac_personalization.config import RESULTS_DIR, USER_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "bt_results.csv",
                        help="Results CSV to plot.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--feature-importance", type=Path, default=None,
                        help="Path to bt_feature_importance_raw.csv")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return

    res_df = pd.read_csv(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pattern_df = dimension_specific_patterns(res_df)
    segment_df = segment_users_by_personalization(res_df)
    user_ids = res_df["user_id"].unique().tolist()
    surveys = _load_surveys(user_ids, USER_DATA_DIR)

    print(f"Loaded {len(res_df)} rows, {len(user_ids)} users")

    plot_agreement_with_spotify(res_df, segment_df, args.output_dir / "agreement_with_spotify.png")
    plot_improvement_heatmap(res_df, args.output_dir / "improvement_heatmap.png")
    plot_user_patterns(res_df, pattern_df, args.output_dir / "user_patterns.png")
    plot_survey_correlates(pattern_df, surveys, args.output_dir / "survey_correlates.png")
    plot_statistical_summary(res_df, pattern_df, args.output_dir / "statistical_summary.png")

    if args.feature_importance and args.feature_importance.exists():
        plot_feature_importance(args.feature_importance, args.output_dir / "feature_importance.png")

    print(f"\nFigures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
