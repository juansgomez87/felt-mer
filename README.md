# tac-personalization

Personalized music emotion annotation via pairwise preference learning. Companion code for an IEEE Transactions on Affective Computing submission.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot,data]"
```

## Data preparation

Build the comparison dataset and audio features from the original user data:

```bash
python scripts/build_dataset.py \
  --source-dir /path/to/spotimeta/user_data \
  --audio-parquet /path/to/track_audio_features.parquet
```

This writes `data/raw/bt_comparisons.parquet`, `data/raw/audio_features.parquet`, and `data/raw/user_mapping.json`.

## Running experiments

Single model:

```bash
python scripts/run_model.py --model bt
python scripts/run_model.py --model lambdamart
python scripts/run_model.py --model dpo
python scripts/run_model.py --model ensemble_fuse
```

All models (writes `results/all_results.csv`):

```bash
python scripts/run_all.py
```

## Generating figures

```bash
python scripts/plot_results.py --results results/bt_results.csv --output-dir results/figures
```

## Tests

```bash
pytest
```

## Project structure

```
src/tac_personalization/
├── config.py              # paths, feature cols, hyperparams
├── data/                  # loading + feature engineering
├── models/                # bt, lambdamart, dpo, ensemble (shared ABC)
├── baselines/             # acoustic (energy/valence), general model
├── evaluation/            # metrics, CV harness, result formatting
└── analysis/              # user segmentation, publication plots
```

Models implement `PairwisePreferenceModel` (fit/predict_proba) and plug into a single evaluation harness (`evaluation/harness.py`) that runs per-user stratified k-fold CV with the baseline evaluated on the same test folds.
