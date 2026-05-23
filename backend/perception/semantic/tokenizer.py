# backend/perception/semantic/tokenizer.py
"""Cached FinBERT tokenizer wrapper.

We sit on top of ``AutoTokenizer.from_pretrained("ProsusAI/finbert")``
and add two layers of caching:

1. **Module-level singleton.** Hugging Face's ``AutoTokenizer`` loads
   ~5 MB of vocab and merges on every call; we load it once.

2. **LRU cache on the *encode* call.** During backfills we tokenise the
   same dedupe-deduplicated news multiple times (different downstream
   consumers). Caching turns the expensive operation into a hash lookup.

The tokenizer is thread-safe (Hugging Face confirms this) so the
singleton + LRU pattern is sound under both threaded and async loads.

Returns ``torch.Tensor`` rather than ``numpy`` because the only consumer
is the GPU-resident student model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER: PreTrainedTokenizerBase | None = None
_DEFAULT_MAX_LEN: int = 256


def get_tokenizer() -> PreTrainedTokenizerBase:
    """Return the cached FinBERT tokenizer, loading it on first call."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return _TOKENIZER


def encode(text: str, max_len: int = _DEFAULT_MAX_LEN) -> dict[str, torch.Tensor]:
    """Tokenise a single string. Always returns padded length ``max_len``.

    The result has shape (1, max_len) for both ``input_ids`` and
    ``attention_mask`` so it can be passed directly to the student
    model without an extra ``.unsqueeze(0)``.
    """
    tok = get_tokenizer()
    enc: dict[str, Any] = tok(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


@lru_cache(maxsize=4096)
def _encode_cached_tuple(text: str, max_len: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """LRU-cached version returning *tuples* (hashable, picklable).

    The functools LRU cache cannot store torch.Tensors directly because
    they are not hashable; we therefore cache the lists and reconstruct
    tensors at the call site. Empirically this is ~10x cheaper than
    re-tokenising for the typical 200-character financial headline.
    """
    tok = get_tokenizer()
    enc = tok(text, padding="max_length", truncation=True, max_length=max_len)
    return tuple(enc["input_ids"]), tuple(enc["attention_mask"])


def encode_cached(text: str, max_len: int = _DEFAULT_MAX_LEN) -> dict[str, torch.Tensor]:
    """Cached single-string encoder. Same shape as :func:`encode`."""
    ids, mask = _encode_cached_tuple(text, max_len)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(mask, dtype=torch.long).unsqueeze(0),
    }


def encode_batch(texts: list[str], max_len: int = _DEFAULT_MAX_LEN) -> dict[str, torch.Tensor]:
    """Batch tokenise. Shape (B, max_len) for both outputs.

    We do not cache here — batch identity is unlikely to repeat. For
    workloads with many repeats, call :func:`encode_cached` per text and
    stack the results.
    """
    tok = get_tokenizer()
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
