# felt-mer

**Personalizing Induced Music Emotion Recognition: from relevance feedback to preference learning**

Juan Sebastián Gómez-Cañón¹, Perfecto Herrera-Boyer², Dmitry Bogdanov², Beatriz Barroso-Gstrein³, Daniel L. Bowling¹

¹ Department of Psychiatry and Behavioral Sciences, Stanford University, Stanford, CA, USA
² Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain
³ Universität Innsbruck, Innsbruck, Austria

Submitted to *IEEE Transactions on Affective Computing*.

## Abstract

Two listeners can agree on the emotion conveyed by a piece of music and yet feel very different in response. This implies that modeling individual differences in induced emotion is essential for emotionally meaningful music recommendations. Yet prior approaches to music emotion recognition (MER) have mainly focused on perceived emotion, averaging judgments across listeners into shared "ground truths." This approach treats disagreement as noise. 
By contrast, predicting induced (felt) emotion requires treating disagreement as the signal. 

We present a study of audio-based MER in which the target of prediction is how music makes an individual listener feel, based on induced arousal, valence, and preference ratings from 41 participants who listened online to excerpts from the MusAV dataset in their preferred genres. We examine how these ratings can support personalization in two ways. First, we use relevance feedback on excerpts represented with MAEST embeddings, fitting personalized Gaussian Mixture Models that are blended with a general MER model. The predictions of the blended and general models were evaluated to determine if the personalization layer improved correspondence with self-report using AB testing (e.g., which excerpt makes you feel more energized?). Second, we used the pairwise choices from the AB testing stage as a preference learning problem, training Bradley--Terry (BT) models on Spotify audio features and comparing them against a Spotify baseline and a general BT model trained to predict the ratings of all other participants. 
Relevance feedback improved how closely each model matched its listener's ratings, yet listener comparisons did not differentiate the performance of the two models. 

The preference learning approach instead produced personalized models that predicted individual listeners choices more accurately than both baselines in average (60.7% vs. 56.6% for Spotify and 56.1% for general BT; Wilcoxon's p = 0.007 and p = 0.004), with the added advantage of lower cognitive burden through AB comparison. Inspection of individual-level data revealed wide variation in model performance. Where personalization changed accuracy by more than 5 percentage points, listeners gained 14.6 (arousal) and 12.0 (valence) points on average when helped, and lost 12.8 and 8.6 when hurt. Disagreement between listeners is the signal, and pairwise comparisons appear to be an efficient way to capture it. 

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
