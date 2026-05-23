# backend/fusion/cross_attention.py
"""Cross-Modal Attention block — section 4 of the architecture spec.

Architectural commitments
-------------------------
The spec is unambiguous: the three encoder outputs are RAW-CONCATENATED
into a 224-d "Super-Vector" *before* attention, NOT projected to a shared
dimension first. Each modality's information capacity is therefore
preserved, and the attention block learns to *re-weight* the streams.

    [Price (128) | Semantic (64) | Graph (32)] → 224
        ↓
    Cross-Modal Attention (multi-head)
        ↓
    Latent fused state (224)

The block is implemented as a multi-head self-attention layer where the
"sequence" has length 3 — one token per modality — but each token has a
*different* dimension. We solve the dim mismatch by introducing per-modality
*read* and *write* projections of small width (the attention head dimension
``d_head``) that operate INSIDE the attention block, never replacing the
concatenated 224-d state itself. The output is a re-weighted 224-d vector,
the same shape as the input, so it can be passed forward directly.

Why three tokens, not one?
--------------------------
A single 224-d token cannot attend cross-modally — there is nothing to
attend *to*. We need three "queries" so the model can decide, for example,
"the Semantic stream should dominate this step, the Price stream is noise".
This is exactly the "Earnings Call vs Flash Crash" example in §4.

References
----------
* Vaswani et al., 2017 — "Attention Is All You Need"
* Tsai et al., 2019 — "Multimodal Transformer for Unaligned Multimodal
  Language Sequences", ACL — the formal name for this block.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from backend.config.constants import DIM_FUSED, DIM_GRAPH, DIM_PRICE, DIM_SEMANTIC


class CrossModalAttention(nn.Module):
    """Three-modality cross-attention block returning a 224-d fused vector.

    Parameters
    ----------
    num_heads : int
        Number of attention heads. Each head operates on ``d_head = 32``
        per modality (so the price stream has 4 heads-worth of capacity,
        the semantic stream 2, the graph stream 1).
    d_head : int
        Size of each attention head's working dimension.
    num_layers : int
        Stack depth. Two layers is enough by spec (§4 says "block",
        singular, but we follow Transformer convention with two).
    dropout : float
        Dropout in attention + FFN.

    Returns
    -------
    fused : (B, 224)
        Re-weighted concatenation, same dim as the raw concat input.
    attn_weights : dict | None
        Per-modality attention weights, only when ``return_attention=True``.
    """

    def __init__(
        self, num_heads: int = 4, d_head: int = 32, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_attn = num_heads * d_head  # working dim inside attn
        self.num_layers = num_layers

        # Per-modality "read" projections: native dim → d_attn
        self.read_price = nn.Linear(DIM_PRICE, self.d_attn)
        self.read_semantic = nn.Linear(DIM_SEMANTIC, self.d_attn)
        self.read_graph = nn.Linear(DIM_GRAPH, self.d_attn)

        # Per-modality "write" projections: d_attn → native dim
        self.write_price = nn.Linear(self.d_attn, DIM_PRICE)
        self.write_semantic = nn.Linear(self.d_attn, DIM_SEMANTIC)
        self.write_graph = nn.Linear(self.d_attn, DIM_GRAPH)

        # Stacked Transformer encoder layers operating on 3-token sequences
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_attn,
                    nhead=num_heads,
                    dim_feedforward=self.d_attn * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Modality-type embeddings: tells the model "this token is price"
        # vs "this token is semantic", separately from the values.
        self.modality_embed = nn.Parameter(torch.randn(3, self.d_attn) * 0.02)

        # Layer-norms on the native-dim residuals
        self.norm_price = nn.LayerNorm(DIM_PRICE)
        self.norm_semantic = nn.LayerNorm(DIM_SEMANTIC)
        self.norm_graph = nn.LayerNorm(DIM_GRAPH)

    # ------------------------------------------------------------------
    def forward(
        self,
        price_emb: torch.Tensor,  # (B, 128)
        semantic_emb: torch.Tensor,  # (B, 64)
        graph_emb: torch.Tensor,  # (B, 32)
        return_attention: bool = False,
        return_modality_weights: bool = False,
    ):
        """Forward pass.

        Default returns ``(fused, None)`` for backward compatibility. The two
        attention flags are mutually orthogonal; when both are requested,
        ``return_modality_weights`` wins and the second tuple element is a
        ``(B, 3)`` softmaxed tensor over (price, news, graph) — the form the
        Spartan Arena's attribution extractor expects.
        """
        # --- 1. Read-project each modality into the shared attn space ---
        p_tok = self.read_price(price_emb).unsqueeze(1)  # (B, 1, d_attn)
        s_tok = self.read_semantic(semantic_emb).unsqueeze(1)
        g_tok = self.read_graph(graph_emb).unsqueeze(1)

        # --- 2. Stack into a 3-token "sequence" + add modality embedding ---
        seq = torch.cat([p_tok, s_tok, g_tok], dim=1)  # (B, 3, d_attn)
        seq = seq + self.modality_embed.unsqueeze(0)  # broadcast

        # --- 3. Run Transformer layers ---
        attn_weights: dict | None = None
        for layer in self.layers:
            seq = layer(seq)

        last_attn = cast(nn.MultiheadAttention, self.layers[-1].self_attn)

        modality_weights: torch.Tensor | None = None
        if return_modality_weights:
            # need_weights=True, average_attn_weights=True yields (B, 3, 3) — the
            # query/key attention matrix after averaging across heads. To collapse
            # to a (B, 3) softmax over modalities we average over the query axis,
            # producing the mean attention each modality *receives*. Rows of the
            # input already sum to 1.0, so the column-mean does too.
            with torch.no_grad():
                _, w = last_attn(seq, seq, seq, need_weights=True, average_attn_weights=True)
            # w: (B, 3, 3) -> (B, 3)
            modality_weights = w.mean(dim=1)

        if return_attention:
            with torch.no_grad():
                _, w = last_attn(seq, seq, seq, need_weights=True, average_attn_weights=False)
            attn_weights = {"cross_modal": w}  # (B, H, 3, 3)

        # --- 4. Write-project each modality back to its native dim, ---
        #        with a residual connection on the native space.
        p_attn, s_attn, g_attn = seq[:, 0], seq[:, 1], seq[:, 2]
        price_out = self.norm_price(price_emb + self.write_price(p_attn))
        semantic_out = self.norm_semantic(semantic_emb + self.write_semantic(s_attn))
        graph_out = self.norm_graph(graph_emb + self.write_graph(g_attn))

        # --- 5. Final concatenation: same shape as input concat (224-d) ---
        fused = torch.cat([price_out, semantic_out, graph_out], dim=-1)
        assert fused.size(-1) == DIM_FUSED, f"Fused dim mismatch: {fused.shape}"

        if return_modality_weights:
            return fused, modality_weights
        return fused, attn_weights
