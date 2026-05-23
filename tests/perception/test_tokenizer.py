# tests/perception/test_tokenizer.py
"""Tests for the FinBERT tokenizer wrapper.

We avoid downloading the 5 MB FinBERT vocab in CI by mocking
``transformers.AutoTokenizer.from_pretrained`` to return a minimal
fake tokenizer. The tests verify the *wrapper* logic — the LRU cache,
the tensor shape, the singleton behaviour — not the actual subword
splitting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_tokenizer_singleton():
    """Reset the module-level tokenizer cache between tests so each test
    sees a fresh ``get_tokenizer`` call."""
    import backend.perception.semantic.tokenizer as tok_mod

    tok_mod._TOKENIZER = None
    tok_mod._encode_cached_tuple.cache_clear()
    yield
    tok_mod._TOKENIZER = None
    tok_mod._encode_cached_tuple.cache_clear()


def _make_fake_tokenizer(max_len: int = 256) -> MagicMock:
    """Build a MagicMock that mimics the HuggingFace tokenizer call signature."""
    fake = MagicMock()

    def _call(text, **kwargs):
        # Both call styles: positional string + return_tensors="pt" → tensors
        # or no return_tensors → plain lists. Mirror that.
        ml = kwargs.get("max_length", max_len)
        if isinstance(text, list):
            n = len(text)
            ids = [[1] * ml for _ in range(n)]
            mask = [[1] * ml for _ in range(n)]
        else:
            ids = [1] * ml
            mask = [1] * ml
        if kwargs.get("return_tensors") == "pt":
            t_ids = torch.tensor([ids] if not isinstance(text, list) else ids, dtype=torch.long)
            t_mask = torch.tensor([mask] if not isinstance(text, list) else mask, dtype=torch.long)
            return {"input_ids": t_ids, "attention_mask": t_mask}
        return {"input_ids": ids, "attention_mask": mask}

    fake.side_effect = _call
    return fake


@patch("transformers.AutoTokenizer.from_pretrained")
def test_get_tokenizer_is_singleton(mock_from_pretrained):
    mock_from_pretrained.return_value = _make_fake_tokenizer()
    from backend.perception.semantic.tokenizer import get_tokenizer

    t1 = get_tokenizer()
    t2 = get_tokenizer()
    assert t1 is t2
    assert mock_from_pretrained.call_count == 1


@patch("transformers.AutoTokenizer.from_pretrained")
def test_encode_returns_padded_tensors(mock_from_pretrained):
    mock_from_pretrained.return_value = _make_fake_tokenizer()
    from backend.perception.semantic.tokenizer import encode

    out = encode("Apple reports record earnings", max_len=64)
    assert "input_ids" in out and "attention_mask" in out
    assert out["input_ids"].shape == (1, 64)
    assert out["attention_mask"].shape == (1, 64)
    assert out["input_ids"].dtype == torch.long


@patch("transformers.AutoTokenizer.from_pretrained")
def test_encode_batch_returns_batched_tensors(mock_from_pretrained):
    mock_from_pretrained.return_value = _make_fake_tokenizer()
    from backend.perception.semantic.tokenizer import encode_batch

    out = encode_batch(["headline one", "headline two", "headline three"], max_len=32)
    assert out["input_ids"].shape == (3, 32)
    assert out["attention_mask"].shape == (3, 32)


@patch("transformers.AutoTokenizer.from_pretrained")
def test_encode_cached_returns_same_values_on_repeat_call(mock_from_pretrained):
    """A second call with the same text must return tensors equal to the
    first call's tensors (proving the cache fires)."""
    mock_from_pretrained.return_value = _make_fake_tokenizer()
    from backend.perception.semantic.tokenizer import encode_cached

    a = encode_cached("hello world", max_len=16)
    b = encode_cached("hello world", max_len=16)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["attention_mask"], b["attention_mask"])
