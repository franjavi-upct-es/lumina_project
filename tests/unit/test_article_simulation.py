"""Unit tests for the article simulation helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from backend.simulation.article_simulation import (
    ArticleSimulationConfig,
    _bad_action_distance_stats,
    _rows_to_nested_metrics,
    backup_existing_checkpoints,
    build_samples,
    detect_cadence,
    determine_status,
    select_universe_scenarios,
    synthetic_market_data,
)


def test_detect_cadence_daily() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=i) for i in range(10)]
    assert detect_cadence(times) == "1d"


def test_split_builder_prevents_temporal_leakage() -> None:
    data = synthetic_market_data(n_tickers=5, n_days=260, start=datetime(2020, 1, 1, tzinfo=UTC))
    config = ArticleSimulationConfig(
        window_days=20,
        horizon_days=5,
        stride_days=5,
        train_end=datetime(2020, 4, 30, tzinfo=UTC),
        val_start=datetime(2020, 5, 1, tzinfo=UTC),
        val_end=datetime(2020, 7, 31, tzinfo=UTC),
        test_start=datetime(2020, 8, 1, tzinfo=UTC),
    )
    samples = build_samples(data, config)

    assert samples["train"]
    assert samples["val"]
    assert samples["test"]
    assert all(s.target_date <= config.train_end for s in samples["train"])
    assert all(s.window_start >= config.val_start for s in samples["val"])
    assert all(s.target_date <= config.val_end for s in samples["val"])
    assert all(s.window_start >= config.test_start for s in samples["test"])


def test_universe_scenario_selector_is_deterministic() -> None:
    data = synthetic_market_data(n_tickers=6, n_days=1200, start=datetime(2013, 1, 1, tzinfo=UTC))
    config = ArticleSimulationConfig(
        scenario_window_days=45,
        scenario_step_days=9,
        train_end=datetime(2014, 6, 30, tzinfo=UTC),
        val_start=datetime(2014, 7, 1, tzinfo=UTC),
        val_end=datetime(2015, 6, 30, tzinfo=UTC),
        test_start=datetime(2015, 7, 1, tzinfo=UTC),
    )

    first = select_universe_scenarios(data, config)
    second = select_universe_scenarios(data, config)

    assert [s.to_json() for s in first] == [s.to_json() for s in second]
    assert [s.name for s in first] == [
        "train_trend_sanity",
        "val_chop_calibration",
        "test_drawdown_stress",
        "test_trend_followthrough",
    ]


def test_default_universe_scenarios_use_article_windows_when_covered() -> None:
    data = synthetic_market_data(n_tickers=6, n_days=13_500, start=datetime(1989, 1, 1, tzinfo=UTC))
    scenarios = select_universe_scenarios(data, ArticleSimulationConfig())

    assert [(s.name, s.start.date().isoformat(), s.end.date().isoformat()) for s in scenarios] == [
        ("train_trend_sanity", "1990-11-01", "1991-10-30"),
        ("val_chop_calibration", "2018-09-05", "2019-09-05"),
        ("test_drawdown_stress", "2020-02-06", "2021-02-04"),
        ("test_trend_followthrough", "2023-11-07", "2024-11-06"),
    ]


def test_checkpoint_backup_preserves_non_empty_files(tmp_path: Path) -> None:
    root = tmp_path / "models"
    non_empty = root / "agent" / "final.pt"
    zero_byte = root / "fusion" / "best.pt"
    non_empty.parent.mkdir(parents=True)
    zero_byte.parent.mkdir(parents=True)
    non_empty.write_bytes(b"model")
    zero_byte.write_bytes(b"")

    backup = backup_existing_checkpoints(root, timestamp="fixed")

    assert backup is not None
    assert (backup / "agent" / "final.pt").read_bytes() == b"model"
    assert not (backup / "fusion" / "best.pt").exists()


def test_status_marks_learned_when_three_evidence_layers_pass() -> None:
    nexus_rows = [
        {
            "phase": "initial",
            "split": "val",
            "loss": 0.10,
            "directional_accuracy": 0.50,
            "naive_directional_accuracy": 0.50,
        },
        {
            "phase": "hard_example",
            "split": "val",
            "loss": 0.05,
            "directional_accuracy": 0.60,
            "naive_directional_accuracy": 0.50,
        },
        {
            "phase": "hard_example",
            "split": "test",
            "loss": 0.05,
            "directional_accuracy": 0.62,
            "naive_directional_accuracy": 0.51,
        },
    ]
    arena_rows = [
        {"phase": "baseline_eval", "scenario": "a", "split": "test", "mean_sharpe": 0.1},
        {"phase": "post_feedback", "scenario": "a", "split": "test", "mean_sharpe": 0.2},
    ]

    assert determine_status(nexus_rows, arena_rows, n_pairs=1) == "learned"
    assert determine_status(nexus_rows, arena_rows, n_pairs=0) == "failed"


def test_status_marks_partial_learning_when_two_evidence_layers_pass() -> None:
    nexus_rows = [
        {"phase": "initial", "split": "val", "loss": 0.10},
        {"phase": "hard_example", "split": "val", "loss": 0.11},
        {
            "phase": "hard_example",
            "split": "test",
            "directional_accuracy": 0.57,
            "naive_directional_accuracy": 0.52,
        },
    ]
    arena_rows = [
        {"phase": "baseline_eval", "scenario": "a", "split": "test", "mean_sharpe": 0.1},
        {"phase": "post_feedback", "scenario": "a", "split": "test", "mean_sharpe": 0.2},
    ]

    assert determine_status(nexus_rows, arena_rows, n_pairs=1) == "partial_learning"


def test_status_keeps_single_signal_inconclusive() -> None:
    nexus_rows = [
        {"phase": "initial", "split": "val", "loss": 0.10},
        {"phase": "hard_example", "split": "val", "loss": 0.09},
        {
            "phase": "hard_example",
            "split": "test",
            "directional_accuracy": 0.49,
            "naive_directional_accuracy": 0.52,
        },
    ]
    arena_rows = [
        {"phase": "baseline_eval", "scenario": "a", "split": "test", "mean_sharpe": 0.2},
        {"phase": "post_feedback", "scenario": "a", "split": "test", "mean_sharpe": 0.1},
    ]

    assert determine_status(nexus_rows, arena_rows, n_pairs=1) == "inconclusive"


def test_bad_action_distance_stats_report_percentiles() -> None:
    actions = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.5, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    ]
    bad_actions = [np.asarray([0.0, 0.0], dtype=np.float32)]

    stats = _bad_action_distance_stats(actions, bad_actions, threshold=0.25)

    assert stats["bad_action_repeat_rate"] == 1 / 3
    assert stats["bad_action_distance_min"] == 0.0
    assert stats["bad_action_distance_p50"] == 0.5
    assert stats["bad_action_distance_p95"] > stats["bad_action_distance_p50"]


def test_nested_metrics_merge_duplicate_phase_split_rows() -> None:
    nested = _rows_to_nested_metrics(
        [
            {"phase": "hard_example", "split": "train", "loss": 0.1},
            {"phase": "hard_example", "split": "train", "hard_example_count": 3.0},
        ]
    )

    assert nested["hard_example"]["train"]["loss"] == 0.1
    assert nested["hard_example"]["train"]["hard_example_count"] == 3.0


def test_metric_inputs_tolerate_flat_arrays() -> None:
    data = synthetic_market_data(n_tickers=3, n_days=90)
    config = ArticleSimulationConfig(window_days=10, horizon_days=2, stride_days=2)
    samples = build_samples(data, config)
    all_targets = np.asarray([s.target_return for rows in samples.values() for s in rows])
    assert np.isfinite(all_targets).all()
