# felt-mer

**From Relevance Feedback to Preference Learning: Two Approaches to Personalizing Induced Music Emotion Recognition**

Juan Sebastián Gómez-Cañón¹, Perfecto Herrera-Boyer², Dmitry Bogdanov², Beatriz Barroso-Gstrein³, Daniel L. Bowling¹

¹ Department of Psychiatry and Behavioral Sciences, Stanford University, Stanford, CA, USA
² Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain
³ Universität Innsbruck, Innsbruck, Austria

Submitted to *IEEE Transactions on Affective Computing*.

## Abstract

We present a study of personalised Music Emotion Recognition (MER) for induced (felt) emotion, where the target of prediction is each listener's own affective response rather than an aggregated group label. Forty-one participants completed an online protocol in two phases. In the first, a relevance-feedback pipeline collected arousal, valence, and preference ratings on excerpts from the MusAV dataset, restricted to the genres each participant reported listening to. Personalised Gaussian Mixture Models fitted on these ratings were combined with a general MER model, and the two were compared in a subjective AB testing. The second phase used the pairwise choices from the AB stage as a preference-learning problem, training Bradley-Terry (BT) models on Spotify audio features and comparing them against an existing industrial baseline and a general BT model validated leave-one-user-out. Relevance feedback improved how closely each model matched its listener's ratings, yet listeners could not distinguish its predictions from the general model's. Preference learning instead produced per-listener models that predicted those listeners' own choices better than baselines, from comparisons that are easier to collect than absolute ratings—though the improvement varied widely between people.

## Scope of this repository

This repository contains the **phase-two** experiments: preference learning from the pairwise choices collected during AB testing. The phase-one relevance-feedback pipeline (personalised GMMs, excerpt selection, AB test administration) is not included; its output enters here only through the precomputed `general_probability` scores read by `baselines/general_model.py`.

## Setup

Requires Python 3.11 (pinned in `.python-version`).

```bash
uv sync --extra dev --extra plot --extra data
```

This creates `.venv/` and installs the project in editable mode. Run commands through `uv run`:

```bash
uv run python scripts/run_all.py
uv run pytest
```

If Python 3.11 is not on the system, `uv` will fetch it:

```bash
uv python install 3.11
```

## Data preparation

Build the comparison dataset and audio features from the original user data:

```bash
uv run python scripts/build_dataset.py \
  --source-dir /path/to/spotimeta/user_data \
  --audio-parquet /path/to/track_audio_features.parquet
```

This writes `data/raw/bt_comparisons.parquet`, `data/raw/audio_features.parquet`, and `data/raw/user_mapping.json`.

## Running experiments

Single model:

```bash
uv run python scripts/run_model.py --model bt
uv run python scripts/run_model.py --model lambdamart
uv run python scripts/run_model.py --model dpo
uv run python scripts/run_model.py --model ensemble_fuse
```

All models (writes `results/all_results.csv`):

```bash
uv run python scripts/run_all.py
```

## Generating figures

```bash
uv run python scripts/plot_results.py --results results/bt_results.csv --output-dir results/figures
```

## Tests

```bash
uv run pytest
```

## Project structure

```
src/felt_mer/
├── config.py              # paths, feature cols, hyperparams
├── data/                  # loading + feature engineering
├── models/                # bt, lambdamart, dpo, ensemble (shared ABC)
├── baselines/             # acoustic (energy/valence), general model
├── evaluation/            # metrics, CV harness, result formatting
└── analysis/              # user segmentation, publication plots
```

Models implement `PairwisePreferenceModel` (fit/predict_proba) and plug into a single evaluation harness (`evaluation/harness.py`) that runs per-user stratified k-fold CV with the baseline evaluated on the same test folds.

## Citation

If you use this code, please cite:

```bibtex
@article{GomezCanon2026TAC,
author = {Gómez-Cañón, Juan Sebastián and
          Herrera-Boyer, Perfecto and
          Bogdanov, Dmitry and
          Barroso-Gstrein, Beatriz and
          Bowling, Daniel L.},
title = {{From Relevance Feedback to Preference Learning: Two Approaches to Personalizing Induced Music Emotion Recognition}},
journal = {IEEE Transactions on Affective Computing},
volume = {},
issue = {},
year = {2026},
pages = {},
doi = {}
}
```
