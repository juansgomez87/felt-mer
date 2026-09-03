"""DPO (Direct Preference Optimization) model for music preference learning."""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tac_personalization.config import (
    AUDIO_FEATURE_COLS,
    DPO_BATCH_SIZE,
    DPO_BETA,
    DPO_DROPOUT,
    DPO_EPOCHS,
    DPO_HIDDEN_DIMS,
    DPO_LR,
)
from tac_personalization.data.features import get_feature_matrix
from tac_personalization.models.base import PairwisePreferenceModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class _PairwiseDataset(Dataset):
    """Dataset of (preferred_features, dispreferred_features) pairs."""

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], scaler=None):
        self.pairs = pairs
        self.scaler = scaler

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        preferred, dispreferred = self.pairs[idx]
        if self.scaler is not None:
            preferred = self.scaler.transform(preferred.reshape(1, -1)).astype(np.float32).squeeze()
            dispreferred = self.scaler.transform(dispreferred.reshape(1, -1)).astype(np.float32).squeeze()
        return torch.FloatTensor(preferred), torch.FloatTensor(dispreferred)


def _build_pairs(comparisons_df, audio_df):
    """Build list of (preferred_features, dispreferred_features) from comparisons."""
    feature_cols = AUDIO_FEATURE_COLS
    feats = audio_df.set_index("track_id") if "track_id" in audio_df.columns else audio_df
    feats.index = feats.index.astype(str)
    available = [c for c in feature_cols if c in feats.columns]
    feats = feats[available].astype(np.float32)
    features_dict = {tid: feats.loc[tid].values for tid in feats.index}

    pairs = []
    for _, row in comparisons_df.iterrows():
        a, b = str(row["song_a_id"]), str(row["song_b_id"])
        if a not in features_dict or b not in features_dict:
            continue
        if row["winner"] == "A":
            pairs.append((features_dict[a].copy(), features_dict[b].copy()))
        else:
            pairs.append((features_dict[b].copy(), features_dict[a].copy()))
    return pairs


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class DPOMusicModel(nn.Module):
    """Maps audio features to a scalar preference strength score."""

    def __init__(
        self,
        input_dim: int = len(AUDIO_FEATURE_COLS),
        hidden_dims: list[int] | None = None,
        dropout: float = DPO_DROPOUT,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DPO_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # Reference model: frozen deep copy (not shared params)
        self.ref_network = copy.deepcopy(self.network)
        for param in self.ref_network.parameters():
            param.requires_grad = False

    def forward(self, features):
        return self.network(features).squeeze(-1)

    def get_reference_score(self, features):
        with torch.no_grad():
            return self.ref_network(features).squeeze(-1)


def _dpo_loss(model, preferred, dispreferred, beta=DPO_BETA):
    """DPO loss: -log sigmoid(beta * (policy_logits - ref_logits))."""
    policy_logits = model(preferred) - model(dispreferred)
    ref_logits = model.get_reference_score(preferred) - model.get_reference_score(dispreferred)
    loss = -F.logsigmoid(beta * (policy_logits - ref_logits)).mean()
    with torch.no_grad():
        accuracy = (policy_logits > 0).float().mean()
    return loss, accuracy


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class DPOModel(PairwisePreferenceModel):
    """DPO-based preference model wrapping DPOMusicModel."""

    def __init__(
        self,
        epochs: int = DPO_EPOCHS,
        batch_size: int = DPO_BATCH_SIZE,
        lr: float = DPO_LR,
        beta: float = DPO_BETA,
        device: str = "cpu",
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._beta = beta
        self._device = device
        self._model = None
        self._scaler = None

    @property
    def name(self) -> str:
        return "dpo"

    def fit(self, train_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
        pairs = _build_pairs(train_df, audio_df)
        if not pairs:
            self._model = None
            return

        # Fit scaler on training features
        all_feats = []
        for p, d in pairs:
            all_feats.append(p)
            all_feats.append(d)
        self._scaler = StandardScaler()
        self._scaler.fit(np.array(all_feats))

        dataset = _PairwiseDataset(pairs, scaler=self._scaler)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        model = DPOMusicModel(input_dim=len(pairs[0][0])).to(self._device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._epochs)

        model.train()
        for _ in range(self._epochs):
            for preferred, dispreferred in loader:
                preferred = preferred.to(self._device)
                dispreferred = dispreferred.to(self._device)
                optimizer.zero_grad()
                loss, _ = _dpo_loss(model, preferred, dispreferred, beta=self._beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        self._model = model

    def predict_proba(self, test_df: pd.DataFrame, audio_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.full(len(test_df), np.nan)

        feats, _ = get_feature_matrix(audio_df)
        probas = np.full(len(test_df), np.nan)

        self._model.eval()
        with torch.no_grad():
            for i, (_, row) in enumerate(test_df.iterrows()):
                a, b = str(row["song_a_id"]), str(row["song_b_id"])
                if a not in feats.index or b not in feats.index:
                    continue
                va = feats.loc[a].values.astype(np.float32)
                vb = feats.loc[b].values.astype(np.float32)
                if self._scaler is not None:
                    va = self._scaler.transform(va.reshape(1, -1)).squeeze()
                    vb = self._scaler.transform(vb.reshape(1, -1)).squeeze()
                va_t = torch.tensor(va, dtype=torch.float32, device=self._device).unsqueeze(0)
                vb_t = torch.tensor(vb, dtype=torch.float32, device=self._device).unsqueeze(0)
                score_a = self._model(va_t).item()
                score_b = self._model(vb_t).item()
                diff = score_a - score_b
                p_a = 1.0 / (1.0 + np.exp(-diff))
                probas[i] = float(np.clip(p_a, 1e-15, 1 - 1e-15))

        return probas
