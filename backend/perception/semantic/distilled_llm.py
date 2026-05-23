# backend/perception/semantic/distilled_llm.py
"""Distilled financial LLM encoder. Student architecture (~15M params)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel

from backend.config.constants import DIM_SEMANTIC

TEACHER_MODEL = "ProsusAI/finbert"
TEACHER_HIDDEN_DIM = 768
STUDENT_HIDDEN_DIM = 256


class DistilledFinancialEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = STUDENT_HIDDEN_DIM,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 512,
        output_dim: int = DIM_SEMANTIC,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_teacher = nn.Linear(hidden_dim, TEACHER_HIDDEN_DIM)
        self.to_embedding = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = input_ids.shape
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        h = self.layer_norm(self.token_emb(input_ids) + self.pos_emb(positions))
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)
        teacher_proj = self.to_teacher(pooled)
        embedding = self.to_embedding(pooled)
        return embedding, teacher_proj

    # ------------------------------------------------------------------
    # Integrated-Gradients attribution (Phase X.3 of the Spartan Arena
    # roadmap). Captum is imported lazily so the module-load cost stays
    # zero for the live perception service that never calls this.
    # ------------------------------------------------------------------
    _ig_tokenizer: Any = None
    """Lazily-initialised tokenizer; shared across calls within a process."""

    def integrated_gradients(
        self,
        text: str,
        target_dim: int | None = None,
        n_steps: int = 50,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-K token attributions for the encoder's embedding.

        Parameters
        ----------
        text
            Raw text to attribute. Tokenized with the FinBERT teacher's
            tokenizer (shared vocab with the student).
        target_dim
            Index of the 64-d output to attribute. If ``None``, aggregate
            attribution over the full embedding by summing ``|grad|`` across
            dimensions — useful for general saliency.
        n_steps
            Integration steps. The default of 50 matches Captum's docs.
        top_k
            Maximum number of tokens to return; sorted by ``|score|``
            descending.

        Notes
        -----
        This is called *on demand* — at most once per pivotal divergence
        point in an arena run — so the cost of allocating embeddings with
        ``requires_grad_=True`` is amortised across the whole arena trace.
        """
        from captum.attr import IntegratedGradients
        from transformers import AutoTokenizer

        if DistilledFinancialEncoder._ig_tokenizer is None:
            DistilledFinancialEncoder._ig_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
        tokenizer = DistilledFinancialEncoder._ig_tokenizer

        device = next(self.parameters()).device
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids: torch.Tensor = enc["input_ids"].to(device)
        attention_mask: torch.Tensor = enc["attention_mask"].to(device)

        was_training = self.training
        self.eval()

        token_embeds = self.token_emb(input_ids).detach()

        def forward_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
            b, t, _ = embeds.shape
            positions = torch.arange(t, device=embeds.device).unsqueeze(0).expand(b, t)
            h = self.layer_norm(embeds + self.pos_emb(positions))
            pad_mask = ~attention_mask.bool()
            h = self.transformer(h, src_key_padding_mask=pad_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            embedding = self.to_embedding(pooled)
            if target_dim is None:
                # Scalarise by L2 norm of the full output embedding.
                return embedding.norm(dim=-1)
            return embedding[:, target_dim]

        baseline = torch.zeros_like(token_embeds)
        ig = IntegratedGradients(forward_from_embeds)
        attributions = ig.attribute(token_embeds, baselines=baseline, n_steps=n_steps)
        token_scores = attributions.detach().sum(dim=-1).squeeze(0)

        token_ids = input_ids.squeeze(0).tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        pairs: list[tuple[str, float]] = [
            (t, float(s)) for t, s in zip(tokens, token_scores, strict=False)
        ]
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)

        self.train(was_training)
        return pairs[:top_k]


def load_teacher(device: str = "cuda") -> AutoModel:
    teacher = AutoModel.from_pretrained(TEACHER_MODEL).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
