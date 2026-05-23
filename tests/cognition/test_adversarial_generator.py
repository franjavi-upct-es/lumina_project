# tests/cognition/test_adversarial_generator.py
"""Smoke tests for AdversarialGenerator (the six-warp injector).

After the Phase-1 restructure, the adversarial generator lives at
``backend.simulation.generators.adversarial`` (NOT
``backend.cognition.simulation.adversarial_generator``); this test was
updated accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.simulation.generators.adversarial import AdversarialGenerator, WarpType


class _DummyEpisodeGen:
    """Minimal generator that yields a flat synthetic episode.

    All fields match the schema expected by ``LuminaTradingEnv``.
    """

    def __iter__(self):
        return self

    def __next__(self):
        n = 100
        return {
            "prices": np.linspace(100.0, 110.0, n).astype(np.float32),
            "market_states": np.zeros((n, NEXUS_OUTPUT_DIM), dtype=np.float32),
            "volatility": np.full(n, 0.01, dtype=np.float32),
            "uncertainties": np.full(n, 0.30, dtype=np.float32),
        }


@pytest.mark.parametrize("warp", list(WarpType))
def test_each_warp_mutates_the_episode(warp: WarpType) -> None:
    """Every warp must (a) leave the price array shape unchanged, and
    (b) tag the episode with its warp_type."""
    base = _DummyEpisodeGen()
    adv = AdversarialGenerator(base, rng=np.random.default_rng(0))
    pristine_prices = next(iter(base))["prices"].copy()

    episode = adv.generate(warp)

    assert episode["warp_type"] == warp.value
    assert episode["prices"].shape == pristine_prices.shape
    # FLASH_CRASH and SUSTAINED_CRASH must lower the final price.
    if warp in (WarpType.FLASH_CRASH, WarpType.SUSTAINED_CRASH):
        assert episode["prices"][-1] <= pristine_prices[-1]


def test_random_warp_returns_a_known_type() -> None:
    base = _DummyEpisodeGen()
    adv = AdversarialGenerator(base, rng=np.random.default_rng(123))
    episode = adv.random_warp()
    assert episode["warp_type"] in {w.value for w in WarpType}
