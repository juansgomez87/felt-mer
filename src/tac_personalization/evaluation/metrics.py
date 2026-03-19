"""Binary classification metrics for pairwise preference evaluation."""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


def compute_binary_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute accuracy, log-likelihood, AUC-ROC, and Brier score.

    Parameters
    ----------
    y_true : Binary labels (0 or 1).
    y_pred_proba : Predicted P(positive class).

    Returns
    -------
    Dict with keys: acc, ll, auc_roc, brier. auc_roc is nan if only one class.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n = len(y_true)
    if n == 0:
        return {"acc": np.nan, "ll": np.nan, "auc_roc": np.nan, "brier": np.nan}

    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))

    p = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    ll = float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    auc_roc = np.nan
    if len(np.unique(y_true)) >= 2:
        try:
            auc_roc = float(roc_auc_score(y_true, y_pred_proba))
        except Exception:
            pass

    brier = float(brier_score_loss(y_true, y_pred_proba))
    return {"acc": acc, "ll": ll, "auc_roc": auc_roc, "brier": brier}
