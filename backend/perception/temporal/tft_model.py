# backend/perception/temporal/tft_model.py
"""Temporal Fusion Transformer — compact reference implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config.constants import DIM_PRICE, OHLCV_WINDOW_MINUTES


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = torch.sigmoid(self.gate(h))
        return self.norm(self.skip(x) + g * h)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.per_feature_grn = nn.ModuleList(
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout) for _ in range(num_features)
        )
        self.selection_grn = GatedResidualNetwork(num_features, hidden_dim, num_features, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _b, _t, f = x.shape
        per_feat = torch.stack(
            [self.per_feature_grn[i](x[..., i : i + 1]) for i in range(f)], dim=-2
        )
        weights = F.softmax(self.selection_grn(x), dim=-1).unsqueeze(-1)
        selected = (per_feat * weights).sum(dim=-2)
        return selected, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.15,
        output_dim: int = DIM_PRICE,
        window_length: int = OHLCV_WINDOW_MINUTES,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(num_features)]
        )
        if len(self.feature_names) != num_features:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != num_features {num_features}"
            )
        self.vsn = VariableSelectionNetwork(num_features, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.mc_dropout = nn.Dropout(0.2)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mc_sample: bool = False,
        return_vsn_weights: bool = False,
    ):
        """Forward pass.

        Default return is ``(embedding, attn_weights)`` for backward compatibility
        with all existing callers. When ``return_vsn_weights=True`` is passed, the
        second tuple element is replaced by a dict mapping feature name to its
        per-timestep VSN weight tensor (shape ``(B, T)``). This is the form the
        Spartan Arena's attribution extractor expects (Phase X.3 of the Arena
        roadmap): no extra model call, just a different view of weights that
        were already computed in the VSN.
        """
        selected, vsn_weights = self.vsn(x)  # vsn_weights: (B, T, F)
        lstm_out, _ = self.lstm(selected)
        gated = self.post_lstm_grn(lstm_out)
        t = gated.size(1)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        attn_out, attn_w = self.attention(gated, gated, gated, attn_mask=mask)
        attn_out = self.attention_grn(attn_out + gated)
        pooled = 0.5 * (attn_out[:, -1] + attn_out.mean(dim=1))
        if mc_sample or self.training:
            pooled = self.mc_dropout(pooled)
        embedding = self.output_head(pooled)
        if return_vsn_weights:
            vsn_dict = {
                name: vsn_weights[..., i].detach() for i, name in enumerate(self.feature_names)
            }
            return embedding, vsn_dict
        return embedding, attn_w

    @torch.no_grad()
    def encode_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()
        samples = torch.stack([self.forward(x, mc_sample=True)[0] for _ in range(n_samples)])
        self.eval()
        return samples.mean(0), samples.std(0)
