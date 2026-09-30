#!/usr/bin/env python3
"""Run a single model experiment: --model bt|lambdamart|dpo|ensemble_fuse|ensemble_majority."""

import argparse
from pathlib import Path

import torch

from felt_mer.baselines.acoustic import AcousticBaseline
from felt_mer.config import (
    AROUSAL_VALENCE_COLS,
    AUDIO_FEATURE_COLS,
    AUDIO_FEATURES_PATH,
    RESULTS_DIR,
)
from felt_mer.evaluation.harness import run_experiment
from felt_mer.evaluation.results import print_summary


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(model_name: str, device: str, feature_cols: list[str] | None = None):
    if model_name == "bt":
        from felt_mer.models.bt import BradleyTerryModel
        return BradleyTerryModel(feature_cols=feature_cols)
    if model_name == "lambdamart":
        from felt_mer.models.lambdamart import LambdaMARTModel
        return LambdaMARTModel()
    if model_name == "dpo":
        from felt_mer.models.dpo import DPOModel
        return DPOModel(device=device)
    if model_name.startswith("ensemble"):
        from felt_mer.models.ensemble import EnsembleModel
        mode = model_name.split("_", 1)[1] if "_" in model_name else "fuse"
        return EnsembleModel(mode=mode, device=device)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run personalization experiment.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bt", "lambdamart", "dpo", "ensemble_fuse", "ensemble_majority"],
        help="Model to evaluate.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--audio-features", type=Path, default=AUDIO_FEATURES_PATH)
    parser.add_argument(
        "--drop-av",
        action="store_true",
        help="Ablation: drop Spotify energy/valence from both BT models (bt only).",
    )
    args = parser.parse_args()
    if args.drop_av and args.model != "bt":
        parser.error("--drop-av is only supported with --model bt")

    feature_cols = None
    if args.drop_av:
        feature_cols = [c for c in AUDIO_FEATURE_COLS if c not in AROUSAL_VALENCE_COLS]

    device = _get_device()
    model = _build_model(args.model, device, feature_cols)
    baseline = AcousticBaseline()

    suffix = "_no_av" if args.drop_av else ""
    output_path = args.output or RESULTS_DIR / f"{model.name}{suffix}_results.csv"

    res_df = run_experiment(
        model=model,
        baseline=baseline,
        audio_features_path=args.audio_features,
        output_path=output_path,
        general_bt_feature_cols=feature_cols,
    )

    if res_df is not None:
        print_summary(res_df, baseline_label="Acoustic")


if __name__ == "__main__":
    main()
