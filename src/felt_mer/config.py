"""Central configuration: paths, feature columns, hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
USER_DATA_DIR = DATA_DIR / "users"
COMPARISONS_PARQUET = DATA_DIR / "bt_comparisons.parquet"
COMPARISONS_CSV = DATA_DIR / "bt_comparisons.csv"
AUDIO_FEATURES_PATH = DATA_DIR / "audio_features.parquet"
MAPPING_FILE = DATA_DIR / "user_mapping.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
DIMENSIONS = ("arousal", "valence")

# ---------------------------------------------------------------------------
# Feature columns (energy/valence included — no leakage since the baseline
# IS energy/valence; giving the model access lets it learn deviations)
# ---------------------------------------------------------------------------
AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "key",
    "loudness",
    "mode",
]

# Spotify's own arousal/valence proxies, dropped in the feature ablation
AROUSAL_VALENCE_COLS = ("energy", "valence")

ALL_AUDIO_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "key",
    "loudness",
    "mode",
]

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_FOLDS = 5
CV_SEED = 42

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
# Accuracy gain (in absolute points) above which a dimension counts as helped
# by personalization. Used to classify per-user patterns for Fig. 3.
PATTERN_IMPROVEMENT_THRESHOLD = 0.05

# Matplotlib settings for manuscript figures (fonttype 42 embeds TrueType so
# the PDF text stays selectable and editable).
FIGURE_RCPARAMS = {
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}

# ---------------------------------------------------------------------------
# Minimum data thresholds
# ---------------------------------------------------------------------------
MIN_PAIRS_CV = N_FOLDS * 2

# ---------------------------------------------------------------------------
# BT hyperparameters
# ---------------------------------------------------------------------------
BT_MAX_ITER = 2000

# ---------------------------------------------------------------------------
# LambdaMART hyperparameters
# ---------------------------------------------------------------------------
LAMBDAMART_PARAMS = {
    "objective": "rank:pairwise",
    "eta": 0.05,
    "max_depth": 2,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 5.0,
    "seed": 42,
}
LAMBDAMART_NUM_BOOST_ROUND = 15

# ---------------------------------------------------------------------------
# DPO hyperparameters
# ---------------------------------------------------------------------------
DPO_HIDDEN_DIMS = [32, 16]
DPO_DROPOUT = 0.5
DPO_EPOCHS = 15
DPO_BATCH_SIZE = 8
DPO_LR = 1e-3
DPO_BETA = 0.5

# ---------------------------------------------------------------------------
# Dimension → test file mapping (for general model baseline)
# ---------------------------------------------------------------------------
DIMENSION_TEST_FILE = {
    "arousal": "test_arousal.json",
    "valence": "test_valence.json",
}
