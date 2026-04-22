from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi import HTTPException

from backend.config.settings import Settings


class DummyFrame:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class DummyCollector:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames

    def collect_with_retry(self, ticker, start_date, end_date):
        return self._frames[ticker].copy()


class DummyFeatureEngineer:
    def create_all_features(self, data):
        return DummyFrame(data)


def _patch_backtest_runtime(monkeypatch, frames, signals):
    from backend.workers import backtest_tasks

    monkeypatch.setattr(backtest_tasks, "YFinanceCollector", lambda: DummyCollector(frames))
    monkeypatch.setattr(backtest_tasks, "FeatureEngineer", lambda: DummyFeatureEngineer())
    monkeypatch.setattr(backtest_tasks, "run_async", lambda value: value)
    monkeypatch.setattr(
        backtest_tasks,
        "_resolve_strategy",
        lambda **kwargs: lambda data, features: list(signals),
    )
    monkeypatch.setattr(
        backtest_tasks.run_backtest_task, "update_state", lambda *args, **kwargs: None
    )
    return backtest_tasks


@pytest.mark.unit
def test_settings_parse_release_debug_and_quoted_origins():
    settings = Settings(
        _env_file=None,
        SECRET_KEY="x" * 32,
        DATABASE_URL="postgresql://lumina:lumina_password@localhost:5435/lumina_db",
        POSTGRES_USER="lumina",
        DEBUG="release",
        ALLOWED_ORIGINS='"http://localhost:3000,http://localhost:8000"',
    )

    assert settings.DEBUG is False
    assert settings.ALLOWED_ORIGINS == [
        "http://localhost:3000",
        "http://localhost:8000",
    ]


@pytest.mark.unit
def test_backtesting_package_imports_cleanly():
    import backend.backtesting as backtesting

    assert backtesting.MonteCarloConfig is not None
    assert backtesting.MonteCarloSimulator is not None


@pytest.mark.unit
def test_async_engine_kwargs_skip_pool_size_for_null_pool():
    from sqlalchemy.pool import NullPool

    from backend.db.models import _build_async_engine_kwargs

    kwargs = _build_async_engine_kwargs(
        "postgresql+asyncpg://lumina:lumina_password@localhost:5435/lumina_db",
        use_null_pool=True,
    )

    assert kwargs["poolclass"] is NullPool
    assert "pool_size" not in kwargs
    assert "max_overflow" not in kwargs
    assert "pool_timeout" not in kwargs


@pytest.mark.unit
def test_lstm_trainer_train_handles_current_torch_scheduler_signature(monkeypatch):
    from backend.ml_engine.models.lstm_advanced import AdvancedLSTM, LSTMTrainer

    trainer = LSTMTrainer(
        AdvancedLSTM(
            input_dim=2,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
            output_horizon=1,
            bidirectional=False,
        ),
        device="cpu",
    )

    monkeypatch.setattr(
        trainer,
        "train_epoch",
        lambda train_loader, optimizer: {
            "total_loss": 1.0,
            "price_loss": 1.0,
            "volatility_loss": 0.0,
            "regime_loss": 0.0,
        },
    )
    monkeypatch.setattr(
        trainer,
        "validate",
        lambda val_loader: {
            "total_loss": 0.5,
            "price_loss": 0.5,
            "volatility_loss": 0.0,
            "regime_loss": 0.0,
        },
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda path: None)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda path: None)

    history = trainer.train([], [], num_epochs=1, learning_rate=0.001, early_stopping_patience=1)

    assert history["train_loss"] == [1.0]
    assert history["val_loss"] == [0.5]


@pytest.mark.unit
def test_time_series_dataset_scales_features_and_uses_relative_return_targets():
    import torch

    from backend.ml_engine.models.lstm_advanced import TimeSeriesDataset

    data = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, None, 30.0, 40.0],
            "close": [100.0, 102.0, 104.0, 106.0],
        }
    )

    scaler = TimeSeriesDataset.build_feature_scaler(data, ["feature_a", "feature_b"])
    dataset = TimeSeriesDataset(
        data=data,
        feature_columns=["feature_a", "feature_b"],
        sequence_length=2,
        prediction_horizon=1,
        feature_scaler=scaler,
        target_mode="relative_returns",
    )

    features, targets = dataset[0]

    assert torch.isfinite(features).all()
    assert targets["last_close"].item() == pytest.approx(102.0)
    assert targets["future_prices"][0].item() == pytest.approx(104.0)
    assert targets["price"][0].item() == pytest.approx((104.0 / 102.0) - 1.0, rel=1e-5)


@pytest.mark.unit
def test_lstm_trainer_predict_decodes_relative_returns_to_price_levels():
    import torch

    from backend.ml_engine.models.lstm_advanced import LSTMTrainer

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return {
                "price": torch.tensor([[0.02, -0.01]], dtype=torch.float32).repeat(batch_size, 1),
                "volatility": torch.tensor([[0.1]], dtype=torch.float32).repeat(batch_size, 1),
                "regime_logits": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(
                    batch_size, 1
                ),
                "regime_probs": torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32).repeat(
                    batch_size, 1
                ),
                "uncertainty": torch.tensor([[0.01, 0.02]], dtype=torch.float32).repeat(
                    batch_size, 1
                ),
                "attention_weights": torch.ones((batch_size, x.shape[1]), dtype=torch.float32),
            }

    trainer = LSTMTrainer(DummyModel(), device="cpu", target_mode="relative_returns")
    trainer.prediction_context = {"last_close": 100.0, "target_mode": "relative_returns"}

    predictions = trainer.predict(torch.zeros((1, 3, 2), dtype=torch.float32))

    assert predictions["price"][0][0] == pytest.approx(102.0)
    assert predictions["price"][0][1] == pytest.approx(99.0)
    assert predictions["uncertainty"][0][0] == pytest.approx(1.0)
    assert predictions["uncertainty"][0][1] == pytest.approx(2.0)
    assert predictions["price_lower"][0][0] == pytest.approx(100.04)
    assert predictions["price_upper"][0][0] == pytest.approx(103.96)


@pytest.mark.asyncio
async def test_yfinance_collector_accepts_isoformat_date_strings(monkeypatch):
    from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

    collector = YFinanceCollector()
    captured: dict[str, object] = {}

    def fake_fetch(ticker, start_date, end_date, interval):
        captured["ticker"] = ticker
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["interval"] = interval
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [1000, 1200],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D", name="Date"),
        )

    monkeypatch.setattr(collector, "_fetch_yfinance_data", fake_fetch)

    data = await collector.collect_with_retry(
        ticker="AAPL",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-02T00:00:00",
    )

    start_date = captured["start_date"]
    end_date = captured["end_date"]

    assert data is not None
    assert captured["ticker"] == "AAPL"
    assert start_date.isoformat() == "2024-01-01T00:00:00"
    assert end_date.isoformat() == "2024-01-02T00:00:00"


@pytest.mark.asyncio
async def test_risk_endpoints_handle_polars_dataframes_without_truthiness_checks(monkeypatch):
    from backend.api.routes import risk

    sample_data = pl.DataFrame({"close": [100.0, 101.5, 103.0, 102.0, 104.5]})

    class DummyCollector:
        async def collect_with_retry(self, ticker, start_date, end_date):
            return sample_data

    monkeypatch.setattr(risk, "YFinanceCollector", lambda: DummyCollector())

    var_response = await risk.calculate_var(
        risk.VaRCalculationRequest(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-05T00:00:00",
        )
    )
    stress_response = await risk.stress_test(
        risk.StressTestRequest(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            scenarios={"custom_shock": -0.2},
        )
    )

    assert "95%" in var_response.var_metrics
    assert stress_response.worst_case["scenario"] in stress_response.scenarios


@pytest.mark.unit
def test_celery_task_routes_match_registered_worker_prefixes():
    from backend.workers.celery_app import celery_app

    router = celery_app.amqp.Router(celery_app.amqp.queues)

    expected_routes = {
        "workers.data_tasks.collect_market_data_task": "data",
        "workers.ml_tasks.train_model_task": "ml",
        "workers.backtest_tasks.run_backtest_task": "backtest",
    }

    for task_name, expected_queue in expected_routes.items():
        route = router.route({}, task_name, args=(), kwargs={})
        assert route["queue"].name == expected_queue

    debug_route = router.route({}, "backend.workers.celery_app.debug_task", args=(), kwargs={})
    assert debug_route["queue"].name == "default"


@pytest.mark.unit
def test_worker_tasks_bind_to_configured_celery_app():
    from backend.workers.backtest_tasks import run_backtest_task
    from backend.workers.celery_app import celery_app
    from backend.workers.ml_tasks import predict_task, train_model_task

    assert train_model_task.app is celery_app
    assert predict_task.app is celery_app
    assert run_backtest_task.app is celery_app


@pytest.mark.unit
def test_predict_task_uses_time_column_for_prediction_dates(monkeypatch):
    import torch

    from backend.ml_engine.models import lstm_advanced
    from backend.workers import ml_tasks

    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.output_horizon = kwargs.get("output_horizon")

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    class DummyTrainer:
        def __init__(self, model):
            self.model = model

        def predict(self, input_sequence):
            assert input_sequence.shape == (1, 2, 2)
            return {
                "price": np.array([[102.5, 103.5]], dtype=float),
                "price_lower": np.array([[101.0, 102.0]], dtype=float),
                "price_upper": np.array([[104.0, 105.0]], dtype=float),
                "uncertainty": np.array([[0.1, 0.2]], dtype=float),
                "regime_probs": np.array([[0.2, 0.3, 0.5]], dtype=float),
                "volatility": np.array([[0.12]], dtype=float),
            }

    class DummyFeatureEngineer:
        def create_all_features(self, data, add_lags=True, add_rolling=True):
            return data.with_columns(
                [
                    pl.Series("feature_a", [0.1, 0.2, 0.3]),
                    pl.Series("feature_b", [1.0, 1.1, 1.2]),
                ]
            )

    monkeypatch.setattr(ml_tasks.predict_task, "update_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ml_tasks,
        "_collect_data",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "close": [103.0, 100.0, 101.0],
            }
        ),
    )
    monkeypatch.setattr(ml_tasks, "FeatureEngineer", DummyFeatureEngineer)
    monkeypatch.setattr(ml_tasks.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "model_state_dict": {},
            "meta_data": {
                "input_dim": 2,
                "hidden_dim": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "sequence_length": 2,
                "prediction_horizon": 2,
                "feature_columns": ["feature_a", "feature_b"],
            },
        },
    )
    monkeypatch.setattr(lstm_advanced, "AdvancedLSTM", DummyModel)
    monkeypatch.setattr(lstm_advanced, "LSTMTrainer", DummyTrainer)

    result = ml_tasks.predict_task.run(ticker="AAPL", model_id="demo_model", days_ahead=2)

    assert result["current_price"] == 103.0
    assert result["current_date"] == "2024-01-03T00:00:00"
    assert [prediction["date"] for prediction in result["predictions"]] == [
        "2024-01-04T00:00:00",
        "2024-01-05T00:00:00",
    ]


@pytest.mark.asyncio
async def test_model_details_rejects_invalid_uuid_before_db_lookup(monkeypatch):
    import backend.db.models as db_models
    from backend.api.routes import ml

    monkeypatch.setattr(
        db_models,
        "get_async_engine",
        lambda: pytest.fail("database lookup should not run for malformed model ids"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await ml.get_model_details("test_model_id")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"


@pytest.mark.asyncio
async def test_backtest_results_reject_invalid_uuid_before_db_lookup():
    from backend.api.routes import backtest

    class UnusedDB:
        async def execute(self, *args, **kwargs):
            pytest.fail("database lookup should not run for malformed backtest ids")

    with pytest.raises(HTTPException) as exc_info:
        await backtest.get_backtest_results("test_backtest_id", db=UnusedDB())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Backtest not found"


@pytest.mark.unit
def test_xgboost_feature_selection_skips_metadata_columns():
    from backend.workers.ml_tasks import _select_xgboost_feature_columns

    enriched_pd = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "source": ["yfinance", "yfinance", "yfinance"],
            "collected_at": pd.date_range("2024-01-10", periods=3, freq="D"),
            "returns": [0.01, 0.02, 0.03],
            "feature_flag": [True, False, True],
        }
    )

    assert _select_xgboost_feature_columns(enriched_pd) == ["returns", "feature_flag"]


@pytest.mark.unit
def test_backtest_marks_open_positions_to_market_and_closes_at_end(monkeypatch):
    frames = {
        "AAA": pd.DataFrame(
            {"close": [100.0, 110.0, 120.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
    }
    backtest_tasks = _patch_backtest_runtime(monkeypatch, frames, ["BUY", "HOLD", "HOLD"])

    result = backtest_tasks.run_backtest_task.run(
        job_id="bt-mark-to-market",
        strategy_name="buy_and_hold",
        config={
            "tickers": ["AAA"],
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-03T00:00:00",
            "initial_capital": 1000.0,
            "position_size": 0.5,
            "max_positions": 1,
            "commission": 0.0,
            "slippage": 0.0,
            "strategy_params": {},
        },
    )

    assert result["equity_curve"][0]["equity"] == pytest.approx(1000.0)
    assert result["equity_curve"][-1]["equity"] == pytest.approx(1100.0)
    assert result["final_capital"] == pytest.approx(1100.0)
    assert result["trades"][-1]["exit_reason"] == "end_of_backtest"


@pytest.mark.unit
def test_backtest_stop_loss_only_applies_to_current_ticker(monkeypatch):
    frames = {
        "AAA": pd.DataFrame(
            {"close": [100.0, 80.0, 80.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
        "BBB": pd.DataFrame(
            {"close": [200.0, 200.0, 210.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
    }
    backtest_tasks = _patch_backtest_runtime(monkeypatch, frames, ["BUY", "HOLD", "HOLD"])

    result = backtest_tasks.run_backtest_task.run(
        job_id="bt-stop-loss-isolation",
        strategy_name="custom",
        config={
            "tickers": ["AAA", "BBB"],
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-03T00:00:00",
            "initial_capital": 1000.0,
            "position_size": 0.5,
            "max_positions": 2,
            "commission": 0.0,
            "slippage": 0.0,
            "stop_loss": 0.1,
            "strategy_params": {},
        },
    )

    trade_by_ticker = {trade["ticker"]: trade for trade in result["trades"]}

    assert trade_by_ticker["AAA"]["exit_reason"] == "stop_loss"
    assert trade_by_ticker["BBB"]["exit_reason"] == "end_of_backtest"
    assert trade_by_ticker["BBB"]["exit_price"] == pytest.approx(210.0)
