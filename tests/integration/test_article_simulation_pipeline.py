"""Smoke test for the article simulation bundle on tiny synthetic data."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.simulation.article_simulation import (
    ArticleSimulationConfig,
    run_article_pipeline,
    synthetic_market_data,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_article_pipeline_writes_report_bundle(tmp_path) -> None:
    data = synthetic_market_data(n_tickers=5, n_days=420, start=datetime(2020, 1, 1, tzinfo=UTC))
    config = ArticleSimulationConfig(
        output_root=tmp_path / "article",
        checkpoint_root=tmp_path / "models",
        train_end=datetime(2020, 5, 31, tzinfo=UTC),
        val_start=datetime(2020, 6, 1, tzinfo=UTC),
        val_end=datetime(2020, 9, 30, tzinfo=UTC),
        test_start=datetime(2020, 10, 1, tzinfo=UTC),
        window_days=20,
        horizon_days=5,
        stride_days=5,
        scenario_window_days=50,
        scenario_step_days=10,
        nexus_epochs=1,
        hard_epochs=1,
        policy_bc_epochs=1,
        feedback_policy_epochs=1,
        n_trajectories=3,
        max_train_samples=96,
        max_eval_samples=48,
        max_policy_samples=96,
        batch_size=32,
        device="cpu",
        write_checkpoints=False,
    )

    out_dir = await run_article_pipeline(config, market_data=data)

    assert (out_dir / "data_inventory.json").exists()
    assert (out_dir / "split_manifest.json").exists()
    assert (out_dir / "nexus_metrics.json").exists()
    assert (out_dir / "arena_metrics.json").exists()
    assert (out_dir / "article_summary.md").exists()
    assert (out_dir / "figures" / "learning_curves.png").exists()
    assert (out_dir / "figures" / "directional_accuracy.png").exists()
    assert list((out_dir / "figures").glob("equity_*.png"))
    assert list((out_dir / "figures").glob("action_divergence_*.png"))
