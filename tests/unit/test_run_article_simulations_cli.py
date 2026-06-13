from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from backend.config.settings import get_settings
from backend.simulation.article_simulation import ArticleSimulationConfig
from scripts import run_article_simulations as cli


def test_host_compatible_timescale_url_rewrites_compose_host() -> None:
    url = "postgresql://lumina:lumina@timescale:5432/lumina"

    assert cli._host_compatible_timescale_url(url) == (
        "postgresql://lumina:lumina@localhost:5432/lumina"
    )


def test_host_compatible_timescale_url_preserves_host_urls() -> None:
    url = "postgresql://lumina:lumina@localhost:5432/lumina"

    assert cli._host_compatible_timescale_url(url) == url


@pytest.mark.asyncio
async def test_auto_data_source_falls_back_to_synthetic_when_timescale_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "TIMESCALE_URL",
        "postgresql://lumina:lumina@timescale:5432/lumina",
    )
    seen_urls: list[str] = []

    async def fail_load():
        seen_urls.append(settings.TIMESCALE_URL)
        raise OSError("offline")

    monkeypatch.setattr(cli, "load_market_data_from_timescale", fail_load)
    args = Namespace(data_source="auto", synthetic_tickers=4, synthetic_days=20)
    config = ArticleSimulationConfig(output_root=tmp_path)

    market_data, inventory = await cli._load_market_data_for_run(args, config)

    assert seen_urls == ["postgresql://lumina:lumina@localhost:5432/lumina"]
    assert inventory is None
    assert len(market_data.tickers) == 4
