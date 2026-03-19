# Plan: Improving Personalization Results (v2)

## What the data tells us

After including energy/valence features and regularizing LambdaMART/DPO:

| Model | Acc Improvement | p-value | Brier vs BL |
|---|---|---|---|
| BT | +0.041 | 0.0036 | -0.008 |
| LambdaMART | -0.006 | 0.77 | +0.000 |
| DPO | -0.022 | 0.98 | -0.003 |

**BT improved** (0.034 → 0.041, p: 0.023 → 0.0036). Including energy/valence helped.

**LambdaMART/DPO calibration is fixed** (DPO Brier went from 0.334 to 0.246) but they still don't beat the baseline on accuracy. With heavy regularization they converge *to* the baseline — they learn to reproduce it, not improve on it.

## Root cause

~52 pairwise comparisons per user-dimension is only enough data for a model with ~11 parameters. BT (logistic regression on 11 feature diffs) is right-sized. LambdaMART and DPO have the wrong inductive bias for this data regime:

- **LambdaMART** learns per-item scores from absolute features. With ~10 unique tracks per user, it can't generalize item-level scores. BT directly models *pairwise differences*, which is what the task actually is.
- **DPO** has the same problem (item-level scoring) plus neural network overhead. Even a [32,16] net has ~600 parameters — 10x more than BT, on 52 training examples.

The models that work need to: (a) operate on pairwise diffs, and (b) have ≤ O(features) parameters.

## Proposed improvements

### 1. Data augmentation via pair symmetry

Each (A, B, winner=A) implies (B, A, winner=B). Doubling training data from ~52 to ~104 pairs. This is free information the models aren't using.

- Modify `create_pairwise_features` to optionally include both orderings
- Modify `_build_pairs` (DPO) and `_prepare_ranking_data` (LambdaMART) similarly
- Expected: most benefit for BT (more data for the same model); some for LambdaMART/DPO

### 2. Pairwise-diff LambdaMART

Current LambdaMART uses absolute features with rank:pairwise objective. Switch to the same representation BT uses: train XGBoost binary classifier on feature diffs (A-B) with label 1=A won. This gives LambdaMART the right inductive bias.

- New class `PairwiseLambdaMARTModel` using `xgb.XGBClassifier` on diff features
- Same regularization (shallow trees, few rounds)
- Expected: should match or exceed BT since it can learn nonlinear interactions between feature diffs

### 3. Pairwise-diff DPO

Same idea: instead of scoring items individually and comparing, feed the model the feature *difference* vector and predict P(A wins) directly. This halves the input dimension and matches BT's inductive bias.

- Modify `DPOMusicModel` to accept diff vectors instead of two separate item inputs
- Use standard binary cross-entropy loss instead of DPO loss (DPO's KL-to-reference adds nothing when the reference is random)
- This effectively becomes a nonlinear logistic regression — right-sized for this problem

### 4. Adaptive ensemble (BT-only fallback)

The ensemble should not average models that underperform the baseline. Strategy:

- On each CV train fold, evaluate each sub-model's accuracy on a held-out inner fold
- Only include sub-models that beat 50% accuracy on the inner fold
- If none beat 50%, fall back to BT alone

### 5. Elastic net BT (squeeze more from the linear model)

Current BT uses `LogisticRegressionCV` with `l1_ratios=(0.0,)` — pure L2. Try elastic net to enable feature selection:

- `l1_ratios=(0.0, 0.5, 1.0)`, `solver="saga"`
- May help by zeroing out noise features for users where only 2-3 features matter

## Implementation order

1. **Pair symmetry augmentation** (#1) — apply to all models
2. **Pairwise-diff LambdaMART** (#2) — new model class
3. **Pairwise-diff DPO** (#3) — restructure DPO
4. **Elastic net BT** (#5) — config change
5. Re-run and compare
6. **Adaptive ensemble** (#4) — if individual models improve

## What NOT to do

- Add more capacity — the constraint is data, not model expressiveness
- Pool across users for training — user-specific preferences are the whole point
- Hyperparameter search — 41 users is not enough to validate tuning decisions
