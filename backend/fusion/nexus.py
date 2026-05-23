# backend/fusion/nexus.py
"""Deep Fusion Nexus — the "Thalamus" of Lumina V3 (§4 of arch spec).

The Nexus orchestrates:

    Encoders → raw concat (224) → Cross-Modal Attention → re-weighted (224)
        → gating (sigmoid) → MLP head → 256-d latent state

The 256-d output is the *Latent State Representation* fed to the cognitive
agent. Its dimensionality is independent of the 224-d super-vector: 256 is
chosen to be a round power of two that comfortably contains the 224-d
information without bottlenecking it, and it is a familiar size for
PPO/SAC trunks.

MC-Dropout uncertainty
----------------------
The Nexus exposes ``encode_with_uncertainty(...)`` which runs N stochastic
forward passes and returns ``(mean, std)`` over the latent state. This is
the *fusion-level* epistemic uncertainty (different from the action-level
uncertainty harvested by the agent). The downstream `state_assembler`
service writes both to Redis under separate keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backend.config.constants import (
    DIM_FUSED,
    MC_DROPOUT_SAMPLES,
    NEXUS_OUTPUT_DIM,
)
from backend.fusion.cross_attention import CrossModalAttention


class DeepFusionNexus(nn.Module):
    """Combines Cross-Modal Attention + gating + MLP head."""

    def __init__(
        self,
        num_heads: int = 4,
        d_head: int = 32,
        num_layers: int = 2,
        dropout: float = 0.15,
        mc_dropout_p: float = 0.2,
        output_dim: int = NEXUS_OUTPUT_DIM,
    ):
        super().__init__()
        self.cross_attention = CrossModalAttention(
            num_heads=num_heads,
            d_head=d_head,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Sigmoid "gate" that lets the model up- or down-weight any region
        # of the 224-d super-vector. This is a learned soft mask; it is
        # *additional* to the attention re-weighting because attention can
        # only redistribute, not zero-out.
        self.gate = nn.Sequential(
            nn.Linear(DIM_FUSED, DIM_FUSED),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(DIM_FUSED, DIM_FUSED),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(DIM_FUSED, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.mc_dropout = nn.Dropout(mc_dropout_p)

    def forward(
        self,
        price_emb: torch.Tensor,
        semantic_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_attention: bool = False,
        mc_sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns a dict with keys:
            "market_state"      — (B, 256) latent state
            "gate_values"       — (B, 224) post-sigmoid gate values
            "attention_weights" — present iff return_attention=True
        """
        fused, attn = self.cross_attention(
            price_emb,
            semantic_emb,
            graph_emb,
            return_attention=return_attention,
        )
        gate = self.gate(fused)
        gated = fused * gate
        if mc_sample or self.training:
            gated = self.mc_dropout(gated)
        state = self.head(gated)
        out = {"market_state": state, "gate_values": gate}
        if return_attention and attn is not None:
            out["attention_weights"] = attn["cross_modal"]
        return out

    @torch.no_grad()
    def encode_with_uncertainty(
        self,
        price_emb: torch.Tensor,
        semantic_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        n_samples: int = MC_DROPOUT_SAMPLES,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate epistemic uncertainty via MC-Dropout.

        Returns
        -------
        mean : (B, 256) — averaged latent state
        std  : (B, 256) — sample std-dev across the N forward passes
        """
        was_training = self.training
        self.train()
        samples = torch.stack(
            [
                self.forward(price_emb, semantic_emb, graph_emb, mc_sample=True)["market_state"]
                for _ in range(n_samples)
            ]
        )
        self.train(was_training)
        return samples.mean(0), samples.std(0)
