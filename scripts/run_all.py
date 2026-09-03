#!/usr/bin/env python3
"""Run all models and produce combined results CSV."""


import pandas as pd
import torch

from felt_mer.baselines.acoustic import AcousticBaseline
from felt_mer.config import AUDIO_FEATURES_PATH, RESULTS_DIR
from felt_mer.evaluation.harness import run_experiment
from felt_mer.evaluation.results import print_summary
from felt_mer.models.bt import BradleyTerryModel
from felt_mer.models.dpo import DPOModel
from felt_mer.models.lambdamart import LambdaMARTModel


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = _get_device()
    baseline = AcousticBaseline()

    models = [
        BradleyTerryModel(),
        LambdaMARTModel(),
        DPOModel(device=device),
        # EnsembleModel(mode="fuse", device=device),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model in models:
        output_path = RESULTS_DIR / f"{model.name}_results.csv"
        print(f"\n{'=' * 60}")
        res_df = run_experiment(
            model=model,
            baseline=baseline,
            audio_features_path=AUDIO_FEATURES_PATH,
            output_path=output_path,
        )
        if res_df is not None:
            res_df["model"] = model.name
            all_results.append(res_df)
            print_summary(res_df, baseline_label="Acoustic")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = RESULTS_DIR / "all_results.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
