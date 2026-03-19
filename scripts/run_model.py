#!/usr/bin/env python3
"""Run a single model experiment: --model bt|lambdamart|dpo|ensemble_fuse|ensemble_majority."""

import argparse
from pathlib import Path

import torch

from tac_personalization.baselines.acoustic import AcousticBaseline
from tac_personalization.config import AUDIO_FEATURES_PATH, RESULTS_DIR
from tac_personalization.evaluation.harness import run_experiment
from tac_personalization.evaluation.results import print_summary


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(model_name: str, device: str):
    if model_name == "bt":
        from tac_personalization.models.bt import BradleyTerryModel
        return BradleyTerryModel()
    if model_name == "lambdamart":
        from tac_personalization.models.lambdamart import LambdaMARTModel
        return LambdaMARTModel()
    if model_name == "dpo":
        from tac_personalization.models.dpo import DPOModel
        return DPOModel(device=device)
    if model_name.startswith("ensemble"):
        from tac_personalization.models.ensemble import EnsembleModel
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
    args = parser.parse_args()

    device = _get_device()
    model = _build_model(args.model, device)
    baseline = AcousticBaseline()

    output_path = args.output or RESULTS_DIR / f"{model.name}_results.csv"

    res_df = run_experiment(
        model=model,
        baseline=baseline,
        audio_features_path=args.audio_features,
        output_path=output_path,
    )

    if res_df is not None:
        print_summary(res_df, baseline_label="Acoustic")


if __name__ == "__main__":
    main()
