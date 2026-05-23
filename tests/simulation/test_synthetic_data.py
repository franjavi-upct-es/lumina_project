# tests/simulation/test_synthetic_data.py
"""Tests for the synthetic episode generators (GBM + jump diffusion).

We pin schema, shape, finiteness, and the iterator protocol — not the
statistical properties of the random output (which would be flaky).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.simulation.generators.synthetic_data import (
    SyntheticEpisodeGenerator,
    gbm_episode,
    jump_diffusion_episode,
)


def test_gbm_episode_keys_and_shapes() -> None:
    rng = np.random.default_rng(0)
    ep = gbm_episode(n_steps=100, rng=rng)
    assert set(ep.keys()) >= {"prices", "market_states", "volatility", "uncertainties"}
    assert ep["prices"].shape == (100,)
    assert ep["market_states"].shape == (100, NEXUS_OUTPUT_DIM)
    assert ep["volatility"].shape == (100,)
    assert ep["uncertainties"].shape == (100,)


def test_gbm_prices_are_finite_and_positive() -> None:
    rng = np.random.default_rng(0)
    ep = gbm_episode(n_steps=500, rng=rng)
    assert np.all(np.isfinite(ep["prices"]))
    assert np.all(ep["prices"] > 0)


def test_jump_diffusion_runs_and_returns_correct_shape() -> None:
    """Jump-diffusion has the same output schema as GBM."""
    rng = np.random.default_rng(0)
    ep = jump_diffusion_episode(n_steps=2000, rng=rng)
    assert ep["prices"].shape == (2000,)
    assert np.all(np.isfinite(ep["prices"]))
    assert np.all(ep["prices"] > 0)


def test_generator_iterator_protocol() -> None:
    gen = SyntheticEpisodeGenerator(n_steps=50, rng=np.random.default_rng(0))
    it = iter(gen)
    ep = next(it)
    assert ep["prices"].shape == (50,)


def test_generator_unknown_process_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown process"):
        SyntheticEpisodeGenerator(process="not_a_process")
