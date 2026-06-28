from __future__ import annotations

import math

import torch
from torch import nn


class CNN1DRegressor(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.lstm(z)
        return self.head(z[:, -1]).squeeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return context, weights


class FeatureAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.sigmoid(self.conv(h.transpose(1, 2))).transpose(1, 2)
        z = h * weights
        return z.mean(dim=1), weights.mean(dim=1)


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
        attention: str = "none",
    ):
        super().__init__()
        self.attention = attention
        self.input_norm = nn.LayerNorm(n_features)
        self.lstm1 = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        dim = hidden_size * 2
        if attention in {"temporal", "dual"}:
            self.temporal_attn = TemporalAttention(dim)
        if attention in {"feature", "dual"}:
            self.feature_attn = FeatureAttention(dim)
        head_dim = dim
        if attention == "dual":
            self.gate = nn.Linear(dim * 2, dim * 2)
            self.fuse = nn.Linear(dim * 2, dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h, _ = self.lstm1(h)
        h = self.norm1(h)
        h, _ = self.lstm2(h)
        h = self.norm2(h)

        if self.attention == "temporal":
            context, _ = self.temporal_attn(h)
        elif self.attention == "feature":
            context, _ = self.feature_attn(h)
        elif self.attention == "dual":
            temporal, _ = self.temporal_attn(h)
            feature, _ = self.feature_attn(h)
            both = torch.cat([temporal, feature], dim=-1)
            gated = torch.sigmoid(self.gate(both)) * both
            context = self.fuse(gated) + h[:, -1]
        else:
            context = h[:, -1]
        return self.head(context).squeeze(-1)


class BiLSTMAttnRegressor(nn.Module):
    """Paper-inspired dual-attention BiLSTM with LayerNorm, TCN, InstanceNorm, and gated fusion."""

    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.tcn = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.instance_norm = nn.InstanceNorm1d(32)
        self.pre_lstm_norm = nn.LayerNorm(32)
        self.lstm1 = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        dim = hidden_size * 2
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.feature_attn = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Linear(dim * 2, dim * 2)
        self.fuse = nn.Linear(dim * 2, dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_norm(x)
        z = torch.relu(self.tcn(z.transpose(1, 2)))
        z = self.instance_norm(z).transpose(1, 2)
        z = self.pre_lstm_norm(z)
        h, _ = self.lstm1(z)
        h = self.norm1(h)
        h, _ = self.lstm2(h)
        h = self.norm2(h)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(h.shape[-1]), dim=-1)
        z_temporal = torch.matmul(attn, v).mean(dim=1)

        feature_weights = torch.sigmoid(self.feature_attn(h.transpose(1, 2))).transpose(1, 2)
        z_feature = (feature_weights * h).mean(dim=1)

        both = torch.cat([z_temporal, z_feature], dim=-1)
        gated = torch.sigmoid(self.gate(both)) * both
        fused = self.fuse(gated) + h[:, -1]
        return self.head(fused).squeeze(-1)


def build_torch_model(name: str, n_features: int, hidden_size: int = 64, dropout: float = 0.3) -> nn.Module:
    if name == "cnn":
        return CNN1DRegressor(n_features=n_features, dropout=dropout)
    if name == "cnn_lstm":
        return CNNLSTMRegressor(n_features=n_features, hidden_size=hidden_size, dropout=dropout)
    if name == "bilstm":
        return BiLSTMRegressor(n_features=n_features, hidden_size=hidden_size, dropout=dropout)
    if name == "bilstm_temporal_attn":
        return BiLSTMRegressor(n_features=n_features, hidden_size=hidden_size, dropout=dropout, attention="temporal")
    if name == "bilstm_feature_attn":
        return BiLSTMRegressor(n_features=n_features, hidden_size=hidden_size, dropout=dropout, attention="feature")
    if name == "bilstm_attn":
        return BiLSTMAttnRegressor(n_features=n_features, hidden_size=hidden_size, dropout=dropout)
    raise ValueError(f"Unknown torch model: {name}")
