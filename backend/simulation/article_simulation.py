"""Article-ready daily simulations for the Lumina backend.

This module is intentionally separate from the production arena worker:
it builds a reproducible research bundle from the data already present in
TimescaleDB, even when the live minute/news/graph inputs are incomplete.
The default path uses daily OHLCV, zero semantic embeddings, and a
correlation graph, then runs a two-stage "learn from mistakes" demo:

1. Train the daily perception/fusion stack on train windows, then run a
   hard-example pass.
2. Run Spartan Arena simulations, build counterfactual pairs from
   train/validation scenarios, and fine-tune the policy from those pairs.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.training.behavioral_cloning import BehavioralCloningTrainer
from backend.config.constants import (
    ACTION_DIM,
    DIM_SEMANTIC,
    NEXUS_OUTPUT_DIM,
    NUM_SECTORS,
    TARGET_TICKERS,
)
from backend.config.settings import get_settings
from backend.data_engine.storage.timescale import TimescaleStore
from backend.fusion.nexus import DeepFusionNexus
from backend.perception.semantic.distilled_llm import DistilledFinancialEncoder
from backend.perception.structural.gat_model import GraphEncoder
from backend.perception.temporal.preprocessor import get_sector_one_hot
from backend.perception.temporal.tft_model import TemporalFusionTransformer
from backend.simulation.arena.runner import ArenaRunner, make_random_seeds
from backend.simulation.arena.schemas import ArenaRunMetadata, CounterfactualPair
from backend.simulation.environments.base_env import EnvConfig, LuminaTradingEnv
from backend.simulation.feedback.counterfactual_pairs import build_pairs, write_pairs_jsonl
from backend.simulation.feedback.replay_buffer_writer import BCDatasetWriter

DATE_FORMAT = "%Y-%m-%d"
REGIME_NAMES = ("crash", "bear", "flat", "bull")
TFT_FEATURE_NAMES = ("open", "high", "low", "close", "volume")
DEFAULT_SCENARIO_WINDOWS: dict[str, tuple[str, str, str]] = {
    "train_trend_sanity": ("train", "1990-11-01", "1991-10-30"),
    "val_chop_calibration": ("val", "2018-09-05", "2019-09-05"),
    "test_drawdown_stress": ("test", "2020-02-06", "2021-02-04"),
    "test_trend_followthrough": ("test", "2023-11-07", "2024-11-06"),
}
BAD_ACTION_REPEAT_THRESHOLD = 0.25


@dataclass(slots=True)
class ArticleSimulationConfig:
    """Runtime knobs for the article simulation pipeline."""

    output_root: Path = Path("artifacts/article_sims")
    checkpoint_root: Path = Path("models")
    train_end: datetime = datetime(2015, 12, 31, tzinfo=UTC)
    val_start: datetime = datetime(2016, 1, 1, tzinfo=UTC)
    val_end: datetime = datetime(2019, 12, 31, tzinfo=UTC)
    test_start: datetime = datetime(2020, 1, 1, tzinfo=UTC)
    window_days: int = 240
    horizon_days: int = 20
    stride_days: int = 5
    scenario_window_days: int = 252
    scenario_step_days: int = 21
    batch_size: int = 128
    nexus_epochs: int = 10
    hard_epochs: int = 5
    policy_bc_epochs: int = 8
    feedback_policy_epochs: int = 6
    hard_early_stop_patience: int = 8
    hard_min_delta: float = 1e-6
    n_trajectories: int = 8
    arena_playback_multiplier: float = 1.0
    max_train_samples: int = 0
    max_eval_samples: int = 0
    max_policy_samples: int = 8192
    extra_test_scenarios: int = 0
    seed: int = 7
    device: str = "auto"
    write_checkpoints: bool = True
    scenario_names: tuple[str, ...] = (
        "train_trend_sanity",
        "val_chop_calibration",
        "test_drawdown_stress",
        "test_trend_followthrough",
    )

    def resolved_device(self) -> str:
        if self.device != "auto":
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable; using CPU for article run")
                return "cpu"
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class DailyBar:
    time: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class DailyMarketData:
    bars_by_ticker: dict[str, list[DailyBar]]

    @property
    def tickers(self) -> list[str]:
        return sorted(self.bars_by_ticker)


@dataclass(slots=True)
class ArticleSample:
    ticker: str
    ticker_idx: int
    split: str
    window_start: datetime
    window_end: datetime
    target_date: datetime
    x: np.ndarray
    target_return: float
    naive_return: float
    regime: int


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    split: str
    start: datetime
    end: datetime
    total_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    chop: float

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["start"] = self.start.date().isoformat()
        out["end"] = self.end.date().isoformat()
        return out


@dataclass(slots=True)
class GraphBundle:
    data: Data
    tickers: list[str]
    ticker_to_idx: dict[str, int]


class DailyNexusDemoModel(nn.Module):
    """Daily OHLCV -> TFT/GAT/Nexus -> return + regime demo heads."""

    def __init__(self) -> None:
        super().__init__()
        self.tft = TemporalFusionTransformer(
            num_features=len(TFT_FEATURE_NAMES),
            feature_names=list(TFT_FEATURE_NAMES),
        )
        self.graph = GraphEncoder()
        self.nexus = DeepFusionNexus()
        self.return_head = nn.Linear(NEXUS_OUTPUT_DIM, 1)
        self.regime_head = nn.Linear(NEXUS_OUTPUT_DIM, len(REGIME_NAMES))

    def encode_state(
        self,
        x: torch.Tensor,
        ticker_idx: torch.Tensor,
        graph_data: Data,
    ) -> torch.Tensor:
        price_emb, _ = self.tft(x)
        graph_nodes = self.graph(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        graph_emb = graph_nodes[ticker_idx]
        semantic_emb = torch.zeros(
            (x.size(0), DIM_SEMANTIC),
            dtype=x.dtype,
            device=x.device,
        )
        return self.nexus(price_emb, semantic_emb, graph_emb)["market_state"]

    def forward(
        self,
        x: torch.Tensor,
        ticker_idx: torch.Tensor,
        graph_data: Data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encode_state(x, ticker_idx, graph_data)
        pred_return = self.return_head(state).squeeze(-1)
        regime_logits = self.regime_head(state)
        return state, pred_return, regime_logits


class ArticlePolicyAgent:
    """Deterministic Arena adapter for article policy evaluation."""

    def __init__(self, policy: PolicyNetwork, device: str) -> None:
        self.policy = policy.to(device)
        self.device = device

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, float, float, float, bool]:
        del deterministic
        was_training = self.policy.training
        self.policy.eval()
        tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        params = self.policy.actor(tensor)
        if self.policy.distribution == "gaussian":
            mean, _log_std = params
            action = torch.tanh(mean)
        else:
            alpha_raw, beta_raw = params
            alpha = F.softplus(alpha_raw) + 1.0
            beta = F.softplus(beta_raw) + 1.0
            action = 2.0 * (alpha / (alpha + beta)) - 1.0
        value = self.policy.critic(tensor)
        self.policy.train(was_training)
        return (
            action.cpu().numpy().squeeze(0).astype(np.float32),
            0.0,
            float(value.item()),
            0.0,
            False,
        )


async def load_market_data_from_timescale(
    tickers: Sequence[str] | None = None,
) -> tuple[DailyMarketData, dict[str, Any]]:
    """Load daily OHLCV rows and a conservative DB inventory.

    Broad aggregate queries can exhaust the small local Timescale shared
    memory segment, so this function inventories one table/ticker at a time.
    """

    store = TimescaleStore()
    await store.connect()
    try:
        async with store._conn() as conn:
            await conn.execute("SET max_parallel_workers_per_gather = 0")
            rows = await conn.fetch("SELECT DISTINCT ticker FROM ohlcv_1m ORDER BY ticker")
            available = [str(r["ticker"]) for r in rows]
            selected = sorted(set(tickers or [*TARGET_TICKERS, "^GSPC"]) & set(available))

            table_counts: dict[str, int] = {}
            for table in (
                "ohlcv_1m",
                "news_events",
                "social_posts",
                "supply_chain_edges",
                "arena_runs",
                "arena_decision_records",
                "arena_divergence_points",
                "arena_counterfactual_pairs",
                "portfolio_history",
                "backtest_runs",
            ):
                try:
                    table_counts[table] = int(await conn.fetchval(f"SELECT count(*) FROM {table}"))
                except Exception as exc:
                    table_counts[table] = -1
                    logger.warning("Inventory count failed for {}: {}", table, exc)

            bars_by_ticker: dict[str, list[DailyBar]] = {}
            ticker_coverage: dict[str, dict[str, Any]] = {}
            for ticker in selected:
                rows = await conn.fetch(
                    """
                    SELECT time, ticker, open, high, low, close, volume
                    FROM ohlcv_1m
                    WHERE ticker = $1
                    ORDER BY time ASC
                    """,
                    ticker,
                )
                bars = [
                    DailyBar(
                        time=_ensure_utc(r["time"]),
                        ticker=str(r["ticker"]),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                    )
                    for r in rows
                ]
                if bars:
                    bars_by_ticker[ticker] = bars
                    ticker_coverage[ticker] = {
                        "rows": len(bars),
                        "start": bars[0].time.isoformat(),
                        "end": bars[-1].time.isoformat(),
                        "cadence": detect_cadence([b.time for b in bars]),
                    }

        market_data = DailyMarketData(bars_by_ticker)
        inventory = build_data_inventory(
            market_data,
            table_counts=table_counts,
            available_tickers=available,
            ticker_coverage=ticker_coverage,
        )
        return market_data, inventory
    finally:
        await store.disconnect()


def build_data_inventory(
    market_data: DailyMarketData,
    *,
    table_counts: dict[str, int] | None = None,
    available_tickers: Sequence[str] | None = None,
    ticker_coverage: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    all_times = [bar.time for bars in market_data.bars_by_ticker.values() for bar in bars]
    news_count = int((table_counts or {}).get("news_events", 0))
    supply_chain_count = int((table_counts or {}).get("supply_chain_edges", 0))
    return {
        "table_counts": table_counts or {},
        "available_tickers": list(available_tickers or market_data.tickers),
        "used_tickers": market_data.tickers,
        "n_used_tickers": len(market_data.tickers),
        "start": min(all_times).isoformat() if all_times else None,
        "end": max(all_times).isoformat() if all_times else None,
        "detected_cadence": detect_cadence(sorted(all_times)[:5000]),
        "ticker_coverage": ticker_coverage or _coverage_from_bars(market_data),
        "missing_modalities": {
            "news_events": news_count == 0,
            "supply_chain_edges": supply_chain_count == 0,
        },
        "modality_notes": [
            "Semantic/news embeddings are zero-filled when news_events is empty.",
            "Graph embeddings are correlation-derived when supply_chain_edges is empty.",
        ],
    }


def detect_cadence(times: Sequence[datetime]) -> str:
    if len(times) < 3:
        return "unknown"
    ordered = sorted(_ensure_utc(t) for t in times)
    deltas = np.asarray(
        [(b - a).total_seconds() for a, b in pairwise(ordered)],
        dtype=np.float64,
    )
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return "unknown"
    median_s = float(np.median(deltas))
    if 60 * 0.5 <= median_s <= 60 * 2:
        return "1m"
    if 24 * 3600 * 0.5 <= median_s <= 24 * 3600 * 4:
        return "1d"
    return f"{median_s:.0f}s"


def build_samples(
    market_data: DailyMarketData,
    config: ArticleSimulationConfig,
) -> dict[str, list[ArticleSample]]:
    samples: dict[str, list[ArticleSample]] = {"train": [], "val": [], "test": []}
    ticker_to_idx = {ticker: i for i, ticker in enumerate(market_data.tickers)}

    for ticker in market_data.tickers:
        bars = market_data.bars_by_ticker[ticker]
        if len(bars) < config.window_days + config.horizon_days:
            continue
        for end_idx in range(
            config.window_days - 1,
            len(bars) - config.horizon_days,
            config.stride_days,
        ):
            window = bars[end_idx - config.window_days + 1 : end_idx + 1]
            target_bar = bars[end_idx + config.horizon_days]
            split = assign_split(window[0].time, target_bar.time, config)
            if split is None:
                continue
            close_now = bars[end_idx].close
            close_start = window[0].close
            target_return = _clip_return((target_bar.close - close_now) / close_now)
            naive_return = _clip_return((close_now - close_start) / close_start)
            samples[split].append(
                ArticleSample(
                    ticker=ticker,
                    ticker_idx=ticker_to_idx[ticker],
                    split=split,
                    window_start=window[0].time,
                    window_end=window[-1].time,
                    target_date=target_bar.time,
                    x=_window_to_features(window),
                    target_return=target_return,
                    naive_return=naive_return,
                    regime=regime_label(target_return),
                )
            )
    return samples


def assign_split(
    window_start: datetime,
    target_date: datetime,
    config: ArticleSimulationConfig,
) -> str | None:
    window_start = _ensure_utc(window_start)
    target_date = _ensure_utc(target_date)
    if target_date <= config.train_end:
        return "train"
    if window_start >= config.val_start and target_date <= config.val_end:
        return "val"
    if window_start >= config.test_start:
        return "test"
    return None


def select_universe_scenarios(
    market_data: DailyMarketData,
    config: ArticleSimulationConfig,
) -> list[ScenarioSpec]:
    returns_by_date = _universe_returns_by_date(market_data)
    fixed = _fixed_article_scenarios(returns_by_date, config)
    if fixed:
        return _with_extra_test_scenarios(fixed, returns_by_date, config)

    windows = _candidate_scenario_windows(returns_by_date, config)
    picks: list[ScenarioSpec] = []
    _append_pick(picks, "train_trend_sanity", windows, "train", "sharpe", reverse=True)
    _append_pick(picks, "val_chop_calibration", windows, "val", "chop", reverse=True)
    _append_pick(picks, "test_drawdown_stress", windows, "test", "max_drawdown", reverse=False)
    _append_pick(picks, "test_trend_followthrough", windows, "test", "sharpe", reverse=True)
    return _with_extra_test_scenarios(picks, returns_by_date, config)


def build_static_graph(
    market_data: DailyMarketData, config: ArticleSimulationConfig
) -> GraphBundle:
    tickers = market_data.tickers
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    returns_by_ticker = {
        ticker: _returns_by_date([b for b in bars if b.time <= config.train_end])
        for ticker, bars in market_data.bars_by_ticker.items()
    }
    spy_returns = returns_by_ticker.get("SPY")
    x = np.zeros((len(tickers), 32), dtype=np.float32)
    for i, ticker in enumerate(tickers):
        rdict = returns_by_ticker.get(ticker, {})
        arr = np.asarray(list(rdict.values()), dtype=np.float64)
        x[i, :NUM_SECTORS] = get_sector_one_hot(ticker)
        x[i, NUM_SECTORS] = min(1.0, math.log10(max(_latest_close(market_data, ticker), 1.0)) / 4)
        x[i, NUM_SECTORS + 1] = float(arr.std(ddof=0) * math.sqrt(252.0)) if arr.size else 0.0
        if spy_returns:
            dates = sorted(set(rdict) & set(spy_returns))
            if len(dates) > 30:
                a = np.asarray([rdict[d] for d in dates], dtype=np.float64)
                b = np.asarray([spy_returns[d] for d in dates], dtype=np.float64)
                var_b = float(np.var(b))
                x[i, NUM_SECTORS + 2] = float(np.cov(a, b, ddof=0)[0, 1] / var_b) if var_b else 0.0

    edge_pairs: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []
    for i, src in enumerate(tickers):
        for j, tgt in enumerate(tickers):
            if i == j:
                continue
            corr = _overlap_corr(returns_by_ticker.get(src, {}), returns_by_ticker.get(tgt, {}))
            if abs(corr) >= 0.40:
                edge_pairs.append((i, j))
                edge_attrs.append([corr, abs(corr), 0.0, max(corr, 0.0)])
    if not edge_pairs and len(tickers) > 1:
        for i in range(len(tickers)):
            j = (i + 1) % len(tickers)
            edge_pairs.append((i, j))
            edge_attrs.append([0.0, 0.0, 0.0, 0.0])

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_attrs, dtype=torch.float32)
        if edge_attrs
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    data = Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr, ticker=tickers)
    return GraphBundle(data=data, tickers=tickers, ticker_to_idx=ticker_to_idx)


def train_daily_nexus(
    samples_by_split: dict[str, list[ArticleSample]],
    graph: GraphBundle,
    config: ArticleSimulationConfig,
) -> tuple[DailyNexusDemoModel, list[dict[str, float | str]], list[dict[str, float | str]]]:
    rng = np.random.default_rng(config.seed)
    device = config.resolved_device()
    model = DailyNexusDemoModel().to(device)
    graph_data = graph.data.to(device)
    train_samples = _limit_samples(
        samples_by_split["train"], config.max_train_samples, rng, chronological=False
    )
    history: list[dict[str, float | str]] = []
    metric_rows: list[dict[str, float | str]] = []

    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history.extend(
        _train_epochs(
            model,
            graph_data,
            train_samples,
            config,
            optim,
            epochs=config.nexus_epochs,
            phase="initial",
            sample_weights=None,
        )
    )
    for split in ("train", "val", "test"):
        eval_samples = _eval_samples(samples_by_split[split], config, rng)
        metric_rows.append(
            _evaluate_nexus(model, graph_data, eval_samples, config, split, "initial")
        )

    errors = _sample_errors(model, graph_data, train_samples, config)
    hard_weights = _hard_example_weights(errors)
    hard_monitor_samples = _eval_samples(samples_by_split["val"], config, rng)
    hard_history, hard_summary = _train_hard_example_epochs(
        model,
        graph_data,
        train_samples,
        hard_monitor_samples,
        config,
        optim,
        sample_weights=hard_weights,
    )
    history.extend(hard_history)
    for split in ("train", "val", "test"):
        eval_samples = _eval_samples(samples_by_split[split], config, rng)
        metric_rows.append(
            _evaluate_nexus(model, graph_data, eval_samples, config, split, "hard_example")
        )
    metric_rows.append(
        {
            "phase": "hard_example",
            "split": "train",
            "hard_example_count": float(np.sum(hard_weights > 1.0)),
        }
    )
    metric_rows.append(hard_summary)
    return model, history, metric_rows


def build_policy_bc_dataset(
    model: DailyNexusDemoModel,
    graph: GraphBundle,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 11)
    selected = _limit_samples(list(samples), config.max_policy_samples, rng, chronological=True)
    states = encode_samples(model, graph, selected, config, include_portfolio=True)
    actions = np.zeros((len(selected), ACTION_DIM), dtype=np.float32)
    for i, sample in enumerate(selected):
        actions[i] = _oracle_action_from_return(sample.target_return)
    return states, actions


def encode_samples(
    model: DailyNexusDemoModel,
    graph: GraphBundle,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
    *,
    include_portfolio: bool,
) -> np.ndarray:
    device = config.resolved_device()
    model.eval()
    graph_data = graph.data.to(device)
    states: list[np.ndarray] = []
    loader = _sample_loader(samples, config.batch_size, shuffle=False, weights=None)
    with torch.no_grad():
        for x, ticker_idx, _target, _regime, _naive, _weights in loader:
            x = x.to(device)
            ticker_idx = ticker_idx.to(device)
            state = model.encode_state(x, ticker_idx, graph_data).cpu().numpy().astype(np.float32)
            states.append(state)
    out = np.vstack(states) if states else np.zeros((0, NEXUS_OUTPUT_DIM), dtype=np.float32)
    if include_portfolio:
        out = np.concatenate([out, np.zeros((out.shape[0], 4), dtype=np.float32)], axis=1)
    return out


async def run_article_pipeline(
    config: ArticleSimulationConfig,
    market_data: DailyMarketData | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_local_mlflow(run_dir)

    if market_data is None:
        market_data, inventory = await load_market_data_from_timescale()
    else:
        inventory = build_data_inventory(
            market_data,
            table_counts={"ohlcv_1m": sum(len(v) for v in market_data.bars_by_ticker.values())},
        )

    scenarios = select_universe_scenarios(market_data, config)
    samples = build_samples(market_data, config)
    graph = build_static_graph(market_data, config)
    split_manifest = {
        "train_end": config.train_end.date().isoformat(),
        "val_start": config.val_start.date().isoformat(),
        "val_end": config.val_end.date().isoformat(),
        "test_start": config.test_start.date().isoformat(),
        "ticker_inclusion": market_data.tickers,
        "sample_counts": {split: len(rows) for split, rows in samples.items()},
        "scenarios": [s.to_json() for s in scenarios],
    }

    _write_json(run_dir / "data_inventory.json", inventory)
    _write_json(run_dir / "split_manifest.json", split_manifest)

    model, nexus_history, nexus_metrics = train_daily_nexus(samples, graph, config)
    _write_csv(run_dir / "nexus_learning_curve.csv", nexus_history)
    _write_metric_rows(run_dir / "nexus_metrics.csv", nexus_metrics)
    _write_json(run_dir / "nexus_metrics.json", _rows_to_nested_metrics(nexus_metrics))

    policy_states, policy_actions = build_policy_bc_dataset(model, graph, samples["train"], config)
    policy = PolicyNetwork(state_dim=NEXUS_OUTPUT_DIM + 4, action_dim=ACTION_DIM)
    bc_metrics = BehavioralCloningTrainer(
        policy_states,
        policy_actions,
        expert_weights=None,
        device=config.resolved_device(),
        val_fraction=0.10,
    ).fit(policy, epochs=config.policy_bc_epochs)
    baseline_policy_state = copy.deepcopy(policy.state_dict())

    feedback_dir = run_dir / "feedback"
    feedback_pairs: list[CounterfactualPair] = []
    feedback_metrics: list[dict[str, Any]] = []
    feedback_sources = [s for s in scenarios if s.split in {"train", "val"}]
    for scenario in feedback_sources:
        metrics, pairs = await _run_arena_for_scenario(
            model,
            graph,
            policy,
            scenario,
            market_data,
            config,
            run_dir,
            phase="feedback_source",
            bad_actions=[],
        )
        feedback_metrics.append(metrics)
        feedback_pairs.extend(pairs)

    pairs_path = feedback_dir / "counterfactual_pairs.jsonl"
    write_pairs_jsonl(feedback_pairs, pairs_path)
    bc_writer = BCDatasetWriter(feedback_dir)
    added_feedback_samples = bc_writer.append_pairs(feedback_pairs, run_dir / "arena_artifacts")
    feedback_dataset_path = bc_writer.finalize()

    if added_feedback_samples:
        feedback_data = np.load(feedback_dataset_path)
        BehavioralCloningTrainer(
            feedback_data["states"],
            feedback_data["actions"],
            expert_weights=feedback_data.get("weights"),
            device=config.resolved_device(),
            val_fraction=0.20,
        ).fit(policy, epochs=config.feedback_policy_epochs)

    bad_actions = [np.asarray(p.bad_action_vector, dtype=np.float32) for p in feedback_pairs]
    arena_rows: list[dict[str, Any]] = [*feedback_metrics]
    for scenario in scenarios:
        baseline_policy = PolicyNetwork(state_dim=NEXUS_OUTPUT_DIM + 4, action_dim=ACTION_DIM)
        baseline_policy.load_state_dict(baseline_policy_state)
        metrics, _pairs = await _run_arena_for_scenario(
            model,
            graph,
            baseline_policy,
            scenario,
            market_data,
            config,
            run_dir,
            phase="baseline_eval",
            bad_actions=bad_actions,
        )
        arena_rows.append(metrics)
        metrics, _pairs = await _run_arena_for_scenario(
            model,
            graph,
            policy,
            scenario,
            market_data,
            config,
            run_dir,
            phase="post_feedback",
            bad_actions=bad_actions,
        )
        arena_rows.append(metrics)

    _write_csv(run_dir / "arena_metrics.csv", arena_rows)
    _write_json(run_dir / "arena_metrics.json", arena_rows)
    _write_json(
        run_dir / "feedback_summary.json",
        {
            "behavioral_cloning": bc_metrics,
            "counterfactual_pairs": len(feedback_pairs),
            "feedback_samples_added": added_feedback_samples,
            "pairs_path": str(pairs_path),
        },
    )

    status = determine_status(nexus_metrics, arena_rows, len(feedback_pairs))
    _write_article_summary(
        run_dir / "article_summary.md",
        status=status,
        inventory=inventory,
        split_manifest=split_manifest,
        nexus_metrics=nexus_metrics,
        arena_rows=arena_rows,
        feedback_pairs=feedback_pairs,
    )
    _write_top_counterfactuals(run_dir / "top_counterfactual_examples.md", feedback_pairs)
    _write_figures(run_dir, nexus_history, nexus_metrics, arena_rows)

    if config.write_checkpoints:
        backup_dir = backup_existing_checkpoints(config.checkpoint_root)
        write_demo_checkpoints(model, policy, config, backup_dir=backup_dir, run_dir=run_dir)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "status": status,
            "config": _jsonable_config(config),
            "output_dir": str(run_dir),
        },
    )
    return run_dir


def backup_existing_checkpoints(
    checkpoint_root: Path,
    *,
    timestamp: str | None = None,
) -> Path | None:
    timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = checkpoint_root / "backups" / f"article_demo_{timestamp}"
    targets = _checkpoint_targets(checkpoint_root)
    copied = False
    for path in targets.values():
        if path.exists() and path.stat().st_size > 0:
            rel = path.relative_to(checkpoint_root)
            dest = backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied = True
    return backup_dir if copied else None


def write_demo_checkpoints(
    model: DailyNexusDemoModel,
    policy: PolicyNetwork,
    config: ArticleSimulationConfig,
    *,
    backup_dir: Path | None,
    run_dir: Path,
) -> None:
    targets = _checkpoint_targets(config.checkpoint_root)
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.tft.state_dict(),
            "metadata": _checkpoint_metadata(run_dir, "daily_article_tft", backup_dir),
        },
        targets["temporal"],
    )
    torch.save(
        {
            "model": DistilledFinancialEncoder().state_dict(),
            "metadata": _checkpoint_metadata(run_dir, "no_news_zero_semantic", backup_dir),
        },
        targets["semantic"],
    )
    torch.save(
        {
            "model": model.graph.state_dict(),
            "metadata": _checkpoint_metadata(run_dir, "daily_correlation_graph", backup_dir),
        },
        targets["structural"],
    )
    torch.save(
        {
            "model": model.nexus.state_dict(),
            "return_head": model.return_head.state_dict(),
            "regime_head": model.regime_head.state_dict(),
            "metadata": _checkpoint_metadata(run_dir, "daily_article_nexus", backup_dir),
        },
        targets["fusion_best"],
    )
    torch.save(model.nexus.state_dict(), targets["fusion_best_nexus"])
    torch.save(policy.state_dict(), targets["agent_final"])


def determine_status(
    nexus_metrics: Sequence[dict[str, float | str]],
    arena_rows: Sequence[dict[str, Any]],
    n_pairs: int,
) -> str:
    if n_pairs <= 0:
        return "failed"
    evidence = _status_evidence(nexus_metrics, arena_rows)
    if evidence["evidence_count"] == evidence["evidence_total"]:
        return "learned"
    if evidence["evidence_count"] >= 2:
        return "partial_learning"
    return "inconclusive"


def _status_evidence(
    nexus_metrics: Sequence[dict[str, float | str]],
    arena_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    initial_val = _metric_value(nexus_metrics, "initial", "val", "loss")
    hard_val = _metric_value(nexus_metrics, "hard_example", "val", "loss")
    test_dir = _metric_value(nexus_metrics, "hard_example", "test", "directional_accuracy")
    test_naive = _metric_value(
        nexus_metrics,
        "hard_example",
        "test",
        "naive_directional_accuracy",
    )
    improved_scenarios = 0
    compared_scenarios = 0
    by_key = {(r["phase"], r["scenario"]): r for r in arena_rows}
    for (_phase, scenario), post in by_key.items():
        if post["phase"] != "post_feedback":
            continue
        baseline = by_key.get(("baseline_eval", scenario))
        if baseline is None or post["split"] not in {"val", "test"}:
            continue
        compared_scenarios += 1
        if float(post["mean_sharpe"]) > float(baseline["mean_sharpe"]):
            improved_scenarios += 1
    nexus_val_improved = _finite_lt(hard_val, initial_val)
    test_beats_naive = _finite_gt(test_dir, test_naive)
    arena_majority_improved = bool(
        compared_scenarios and improved_scenarios >= math.ceil(compared_scenarios / 2)
    )
    flags = [nexus_val_improved, test_beats_naive, arena_majority_improved]
    return {
        "nexus_val_improved": nexus_val_improved,
        "test_beats_naive": test_beats_naive,
        "arena_majority_improved": arena_majority_improved,
        "arena_improved_scenarios": improved_scenarios,
        "arena_compared_scenarios": compared_scenarios,
        "evidence_count": sum(1 for flag in flags if flag),
        "evidence_total": len(flags),
    }


def synthetic_market_data(
    *,
    n_tickers: int = 6,
    n_days: int = 900,
    seed: int = 1,
    start: datetime = datetime(2000, 1, 3, tzinfo=UTC),
) -> DailyMarketData:
    rng = np.random.default_rng(seed)
    tickers = ["SPY", "QQQ", *[f"T{i}" for i in range(max(0, n_tickers - 2))]][:n_tickers]
    bars_by_ticker: dict[str, list[DailyBar]] = {}
    for j, ticker in enumerate(tickers):
        price = 100.0 + 5.0 * j
        bars: list[DailyBar] = []
        for i in range(n_days):
            day = start.timestamp() + i * 24 * 3600
            ts = datetime.fromtimestamp(day, tz=UTC)
            ret = 0.0004 + 0.012 * rng.normal() + 0.0002 * math.sin(i / 25)
            open_ = price
            close = max(1.0, price * (1.0 + ret))
            high = max(open_, close) * (1.0 + abs(rng.normal(0, 0.003)))
            low = min(open_, close) * (1.0 - abs(rng.normal(0, 0.003)))
            bars.append(
                DailyBar(
                    time=ts,
                    ticker=ticker,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=float(1_000_000 + rng.integers(0, 50_000)),
                )
            )
            price = close
        bars_by_ticker[ticker] = bars
    return DailyMarketData(bars_by_ticker)


# ---------------------------------------------------------------------------
# Arena helpers
# ---------------------------------------------------------------------------
async def _run_arena_for_scenario(
    model: DailyNexusDemoModel,
    graph: GraphBundle,
    policy: PolicyNetwork,
    scenario: ScenarioSpec,
    market_data: DailyMarketData,
    config: ArticleSimulationConfig,
    run_dir: Path,
    *,
    phase: str,
    bad_actions: Sequence[np.ndarray],
) -> tuple[dict[str, Any], list[CounterfactualPair]]:
    settings = get_settings()
    artifact_root = run_dir / "arena_artifacts"
    settings.arena.artifact_dir = artifact_root
    episode = build_universe_episode(model, graph, scenario, market_data, config)
    seeds = make_random_seeds(config.n_trajectories, rng=np.random.default_rng(config.seed))
    metadata = ArenaRunMetadata(
        ticker="UNIVERSE",
        start_date=scenario.start,
        end_date=scenario.end,
        n_trajectories=config.n_trajectories,
        mc_seeds=seeds,
        playback_multiplier=config.arena_playback_multiplier,
    )

    def factory(seed: int) -> LuminaTradingEnv:
        class _Gen:
            def __iter__(self):
                return self

            def __next__(self):
                return _perturb_episode(episode, seed)

        return LuminaTradingEnv(_Gen(), EnvConfig())

    agent = ArticlePolicyAgent(policy=policy, device=config.resolved_device())
    runner = ArenaRunner(
        run_metadata=metadata,
        agent=agent,
        env_factory=factory,
        timescale=None,
        policy_uses_full_observation=True,
        divergence_annualization_periods=252.0,
    )
    await runner.run()
    divergences = runner.divergence_analyzer.all_divergences()
    pairs = build_pairs(metadata.run_id, divergences, runner._records_by_trajectory)
    if not pairs:
        pairs = _fallback_pairs_from_records(metadata.run_id, runner, episode)
    _write_scenario_run_figures(run_dir, phase, scenario, runner)
    phase_dir = artifact_root / "pairs" / phase
    write_pairs_jsonl(pairs, phase_dir / f"{scenario.name}.jsonl")
    return _arena_metrics(phase, scenario, runner, pairs, bad_actions), pairs


def build_universe_episode(
    model: DailyNexusDemoModel,
    graph: GraphBundle,
    scenario: ScenarioSpec,
    market_data: DailyMarketData,
    config: ArticleSimulationConfig,
) -> dict[str, np.ndarray | bool | str]:
    returns_by_date = _universe_returns_by_date(market_data)
    scenario_dates = [d for d in sorted(returns_by_date) if scenario.start <= d <= scenario.end]
    if len(scenario_dates) < 4:
        raise RuntimeError(f"Scenario {scenario.name} has too few usable dates")

    returns = np.asarray([returns_by_date[d] for d in scenario_dates], dtype=np.float32)
    prices = np.empty(len(returns), dtype=np.float32)
    prices[0] = 100.0
    for i in range(1, len(prices)):
        prices[i] = prices[i - 1] * (1.0 + returns[i])
    states = _scenario_states(model, graph, market_data, scenario_dates, config)
    vol = np.abs(returns).astype(np.float32)
    uncertainty = np.clip(0.15 + vol / (np.quantile(vol, 0.9) + 1e-6) * 0.2, 0.05, 0.75)
    volume = np.full(len(prices), 1_000_000.0, dtype=np.float32)
    high = prices * (1.0 + vol)
    low = prices * (1.0 - vol)
    return {
        "prices": prices,
        "open": prices,
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": prices,
        "volume": volume,
        "market_states": states,
        "volatility": vol,
        "uncertainties": uncertainty.astype(np.float32),
        "ticker": "UNIVERSE",
        "synthetic": False,
    }


# ---------------------------------------------------------------------------
# Training/evaluation helpers
# ---------------------------------------------------------------------------
def _train_epochs(
    model: DailyNexusDemoModel,
    graph_data: Data,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
    optim: torch.optim.Optimizer,
    *,
    epochs: int,
    phase: str,
    sample_weights: np.ndarray | None,
) -> list[dict[str, float | str]]:
    history: list[dict[str, float | str]] = []
    for epoch in range(epochs):
        train_loss = _train_one_epoch(
            model,
            graph_data,
            samples,
            config,
            optim,
            sample_weights=sample_weights,
        )
        history.append(
            {
                "phase": phase,
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
            }
        )
    return history


def _train_hard_example_epochs(
    model: DailyNexusDemoModel,
    graph_data: Data,
    train_samples: Sequence[ArticleSample],
    monitor_samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
    optim: torch.optim.Optimizer,
    *,
    sample_weights: np.ndarray,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    history: list[dict[str, float | str]] = []
    pre_hard_metrics = _evaluate_nexus(
        model,
        graph_data,
        monitor_samples,
        config,
        "monitor",
        "hard_example",
    )
    pre_hard_val_loss = float(pre_hard_metrics["loss"])
    best_val_loss = pre_hard_val_loss
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    epochs_without_improvement = 0
    stopped_epoch = 0

    for epoch in range(config.hard_epochs):
        stopped_epoch = epoch + 1
        train_loss = _train_one_epoch(
            model,
            graph_data,
            train_samples,
            config,
            optim,
            sample_weights=sample_weights,
        )
        monitor_metrics = _evaluate_nexus(
            model,
            graph_data,
            monitor_samples,
            config,
            "monitor",
            "hard_example",
        )
        val_loss = float(monitor_metrics["loss"])
        if _loss_improved(val_loss, best_val_loss, config.hard_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "phase": "hard_example",
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "monitor_val_loss": val_loss,
                "best_monitor_val_loss": best_val_loss,
                "best_epoch": float(best_epoch),
            }
        )
        if (
            config.hard_early_stop_patience > 0
            and epochs_without_improvement >= config.hard_early_stop_patience
        ):
            break

    model.load_state_dict(best_state)
    return history, {
        "phase": "hard_example",
        "split": "monitor",
        "pre_hard_val_loss": pre_hard_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": float(best_epoch),
        "stopped_epoch": float(stopped_epoch),
        "early_stopped": float(stopped_epoch < config.hard_epochs),
        "restored_best_checkpoint": 1.0,
    }


def _train_one_epoch(
    model: DailyNexusDemoModel,
    graph_data: Data,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
    optim: torch.optim.Optimizer,
    *,
    sample_weights: np.ndarray | None,
) -> float:
    device = config.resolved_device()
    model.train()
    total_loss = 0.0
    total_n = 0
    loader = _sample_loader(samples, config.batch_size, shuffle=True, weights=sample_weights)
    for x, ticker_idx, target, regime, _naive, weights in loader:
        x = x.to(device)
        ticker_idx = ticker_idx.to(device)
        target = target.to(device)
        regime = regime.to(device)
        weights = weights.to(device)
        _state, pred, logits = model(x, ticker_idx, graph_data)
        mse = (pred - target).pow(2)
        loss = (mse * weights).mean() + 0.20 * F.cross_entropy(logits, regime)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _loss_improved(candidate: float, best: float, min_delta: float) -> bool:
    if not math.isfinite(candidate):
        return False
    if not math.isfinite(best):
        return True
    return candidate < best - min_delta


def _finite_lt(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and left < right


def _finite_gt(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and left > right


@torch.no_grad()
def _evaluate_nexus(
    model: DailyNexusDemoModel,
    graph_data: Data,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
    split: str,
    phase: str,
) -> dict[str, float | str]:
    device = config.resolved_device()
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    regimes: list[np.ndarray] = []
    regime_preds: list[np.ndarray] = []
    naives: list[np.ndarray] = []
    for x, ticker_idx, target, regime, naive, _weights in _sample_loader(
        samples,
        config.batch_size,
        shuffle=False,
        weights=None,
    ):
        x = x.to(device)
        ticker_idx = ticker_idx.to(device)
        _state, pred, logits = model(x, ticker_idx, graph_data)
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
        regimes.append(regime.numpy())
        regime_preds.append(logits.argmax(dim=-1).cpu().numpy())
        naives.append(naive.numpy())
    if not preds:
        return {
            "phase": phase,
            "split": split,
            "loss": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "directional_accuracy": math.nan,
            "naive_directional_accuracy": math.nan,
            "regime_macro_f1": math.nan,
            "n_samples": 0.0,
        }
    pred_arr = np.concatenate(preds)
    target_arr = np.concatenate(targets)
    regime_arr = np.concatenate(regimes)
    regime_pred_arr = np.concatenate(regime_preds)
    naive_arr = np.concatenate(naives)
    err = pred_arr - target_arr
    return {
        "phase": phase,
        "split": split,
        "loss": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "directional_accuracy": _directional_accuracy(pred_arr, target_arr),
        "naive_directional_accuracy": _directional_accuracy(naive_arr, target_arr),
        "regime_macro_f1": _macro_f1(regime_arr, regime_pred_arr, len(REGIME_NAMES)),
        "n_samples": float(target_arr.size),
    }


@torch.no_grad()
def _sample_errors(
    model: DailyNexusDemoModel,
    graph_data: Data,
    samples: Sequence[ArticleSample],
    config: ArticleSimulationConfig,
) -> np.ndarray:
    device = config.resolved_device()
    errors: list[np.ndarray] = []
    model.eval()
    for x, ticker_idx, target, _regime, _naive, _weights in _sample_loader(
        samples, config.batch_size, shuffle=False, weights=None
    ):
        x = x.to(device)
        ticker_idx = ticker_idx.to(device)
        _state, pred, _logits = model(x, ticker_idx, graph_data)
        errors.append((pred.cpu() - target).abs().numpy())
    return np.concatenate(errors) if errors else np.zeros(0, dtype=np.float32)


def _sample_loader(
    samples: Sequence[ArticleSample],
    batch_size: int,
    *,
    shuffle: bool,
    weights: np.ndarray | None,
) -> DataLoader:
    if not samples:
        empty = TensorDataset(
            torch.zeros((0, 1, len(TFT_FEATURE_NAMES)), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
        return DataLoader(empty, batch_size=batch_size)
    xs = torch.from_numpy(np.stack([s.x for s in samples]).astype(np.float32))
    ticker_idx = torch.tensor([s.ticker_idx for s in samples], dtype=torch.long)
    target = torch.tensor([s.target_return for s in samples], dtype=torch.float32)
    regime = torch.tensor([s.regime for s in samples], dtype=torch.long)
    naive = torch.tensor([s.naive_return for s in samples], dtype=torch.float32)
    weight_t = torch.ones(len(samples), dtype=torch.float32)
    if weights is not None:
        weight_t = torch.from_numpy(weights.astype(np.float32))
    return DataLoader(
        TensorDataset(xs, ticker_idx, target, regime, naive, weight_t),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def regime_label(value: float) -> int:
    if value <= -0.08:
        return 0
    if value < -0.02:
        return 1
    if value <= 0.02:
        return 2
    return 3


def _window_to_features(window: Sequence[DailyBar]) -> np.ndarray:
    arr = np.asarray(
        [[b.open, b.high, b.low, b.close, b.volume] for b in window],
        dtype=np.float32,
    )
    mn = arr.min(axis=0, keepdims=True)
    mx = arr.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) > 1e-9, mx - mn, 1.0)
    return ((arr - mn) / rng).astype(np.float32)


def _limit_samples(
    samples: list[ArticleSample],
    max_count: int,
    rng: np.random.Generator,
    *,
    chronological: bool,
) -> list[ArticleSample]:
    if max_count <= 0 or len(samples) <= max_count:
        return list(samples)
    idx = rng.choice(len(samples), size=max_count, replace=False)
    if chronological:
        idx = np.sort(idx)
    return [samples[int(i)] for i in idx]


def _eval_samples(
    samples: list[ArticleSample],
    config: ArticleSimulationConfig,
    rng: np.random.Generator,
) -> list[ArticleSample]:
    return _limit_samples(samples, config.max_eval_samples, rng, chronological=True)


def _hard_example_weights(errors: np.ndarray) -> np.ndarray:
    if errors.size == 0:
        return np.ones(0, dtype=np.float32)
    threshold = float(np.quantile(errors, 0.75))
    weights = np.ones(errors.size, dtype=np.float32)
    weights[errors >= threshold] = 3.0
    return weights


def _directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return math.nan
    return float((np.sign(pred) == np.sign(target)).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores: list[float] = []
    for cls in range(n_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def _clip_return(value: float, lim: float = 0.30) -> float:
    return float(max(-lim, min(lim, value)))


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coverage_from_bars(market_data: DailyMarketData) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ticker, bars in market_data.bars_by_ticker.items():
        if not bars:
            continue
        out[ticker] = {
            "rows": len(bars),
            "start": bars[0].time.isoformat(),
            "end": bars[-1].time.isoformat(),
            "cadence": detect_cadence([b.time for b in bars]),
        }
    return out


def _fixed_article_scenarios(
    returns_by_date: dict[datetime, float],
    config: ArticleSimulationConfig,
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    for name in config.scenario_names:
        window = DEFAULT_SCENARIO_WINDOWS.get(name)
        if window is None:
            return []
        split, start_s, end_s = window
        start = _scenario_boundary(start_s, end=False)
        end = _scenario_boundary(end_s, end=True)
        scenario = _scenario_from_returns(name, split, start, end, returns_by_date)
        if scenario is None:
            return []
        if _scenario_split(scenario.start, scenario.end, config) != split:
            return []
        scenarios.append(scenario)
    return scenarios


def _scenario_boundary(value: str, *, end: bool) -> datetime:
    date = datetime.strptime(value, DATE_FORMAT).replace(tzinfo=UTC)
    if end:
        return date.replace(hour=23, minute=59, second=59, microsecond=999999)
    return date


def _scenario_from_returns(
    name: str,
    split: str,
    start: datetime,
    end: datetime,
    returns_by_date: dict[datetime, float],
) -> ScenarioSpec | None:
    dates = [d for d in sorted(returns_by_date) if start <= d <= end]
    if len(dates) < 4:
        return None
    returns = np.asarray([returns_by_date[d] for d in dates], dtype=np.float64)
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    volatility = float(returns.std(ddof=0) * math.sqrt(252.0))
    sharpe = float((returns.mean() / (returns.std(ddof=0) + 1e-12)) * math.sqrt(252.0))
    max_drawdown = _max_drawdown(equity)
    chop = float(volatility / (abs(total_return) + 0.02))
    return ScenarioSpec(
        name=name,
        split=split,
        start=start,
        end=end,
        total_return=total_return,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        chop=chop,
    )


def _scenario_split(
    start: datetime,
    end: datetime,
    config: ArticleSimulationConfig,
) -> str | None:
    if end <= config.train_end:
        return "train"
    if start >= config.val_start and end <= config.val_end:
        return "val"
    if start >= config.test_start:
        return "test"
    return None


def _candidate_scenario_windows(
    returns_by_date: dict[datetime, float],
    config: ArticleSimulationConfig,
) -> list[ScenarioSpec]:
    dates = sorted(returns_by_date)
    windows: list[ScenarioSpec] = []
    n = config.scenario_window_days
    for start_i in range(0, max(0, len(dates) - n + 1), config.scenario_step_days):
        win_dates = dates[start_i : start_i + n]
        returns = np.asarray([returns_by_date[d] for d in win_dates], dtype=np.float64)
        split = _scenario_split(win_dates[0], win_dates[-1], config)
        if split is None:
            continue
        equity = np.cumprod(1.0 + returns)
        total_return = float(equity[-1] - 1.0)
        volatility = float(returns.std(ddof=0) * math.sqrt(252.0))
        sharpe = float((returns.mean() / (returns.std(ddof=0) + 1e-12)) * math.sqrt(252.0))
        max_drawdown = _max_drawdown(equity)
        chop = float(volatility / (abs(total_return) + 0.02))
        windows.append(
            ScenarioSpec(
                name="candidate",
                split=split,
                start=win_dates[0],
                end=win_dates[-1],
                total_return=total_return,
                volatility=volatility,
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                chop=chop,
            )
        )
    return windows


def _with_extra_test_scenarios(
    scenarios: Sequence[ScenarioSpec],
    returns_by_date: dict[datetime, float],
    config: ArticleSimulationConfig,
) -> list[ScenarioSpec]:
    picks = list(scenarios)
    if config.extra_test_scenarios <= 0:
        return picks

    candidates = sorted(
        [w for w in _candidate_scenario_windows(returns_by_date, config) if w.split == "test"],
        key=lambda w: (abs(w.max_drawdown), w.chop, abs(w.sharpe), w.start),
        reverse=True,
    )
    added = 0
    for candidate in candidates:
        if any(candidate.start <= p.end and candidate.end >= p.start for p in picks):
            continue
        added += 1
        picks.append(
            ScenarioSpec(
                name=f"test_extra_{added:02d}",
                split=candidate.split,
                start=candidate.start,
                end=candidate.end,
                total_return=candidate.total_return,
                volatility=candidate.volatility,
                sharpe=candidate.sharpe,
                max_drawdown=candidate.max_drawdown,
                chop=candidate.chop,
            )
        )
        if added >= config.extra_test_scenarios:
            break
    return picks


def _append_pick(
    picks: list[ScenarioSpec],
    name: str,
    windows: Sequence[ScenarioSpec],
    split: str,
    attr: str,
    *,
    reverse: bool,
) -> None:
    candidates = sorted(
        [w for w in windows if w.split == split],
        key=lambda w: float(getattr(w, attr)),
        reverse=reverse,
    )
    for candidate in candidates:
        if not any(
            p.split == candidate.split and candidate.start <= p.end and candidate.end >= p.start
            for p in picks
        ):
            picks.append(
                ScenarioSpec(
                    name=name,
                    split=candidate.split,
                    start=candidate.start,
                    end=candidate.end,
                    total_return=candidate.total_return,
                    volatility=candidate.volatility,
                    sharpe=candidate.sharpe,
                    max_drawdown=candidate.max_drawdown,
                    chop=candidate.chop,
                )
            )
            return
    if candidates:
        c = candidates[0]
        picks.append(
            ScenarioSpec(
                name=name,
                split=c.split,
                start=c.start,
                end=c.end,
                total_return=c.total_return,
                volatility=c.volatility,
                sharpe=c.sharpe,
                max_drawdown=c.max_drawdown,
                chop=c.chop,
            )
        )


def _universe_returns_by_date(market_data: DailyMarketData) -> dict[datetime, float]:
    grouped: dict[datetime, list[float]] = defaultdict(list)
    min_coverage = max(3, len(market_data.tickers) // 2)
    for bars in market_data.bars_by_ticker.values():
        for prev, cur in pairwise(bars):
            if prev.close > 0:
                grouped[cur.time].append((cur.close - prev.close) / prev.close)
    return {
        date: float(np.mean(values))
        for date, values in grouped.items()
        if len(values) >= min_coverage
    }


def _returns_by_date(bars: Sequence[DailyBar]) -> dict[datetime, float]:
    out: dict[datetime, float] = {}
    for prev, cur in pairwise(bars):
        if prev.close > 0:
            out[cur.time] = (cur.close - prev.close) / prev.close
    return out


def _overlap_corr(a: dict[datetime, float], b: dict[datetime, float]) -> float:
    dates = sorted(set(a) & set(b))
    if len(dates) < 60:
        return 0.0
    av = np.asarray([a[d] for d in dates], dtype=np.float64)
    bv = np.asarray([b[d] for d in dates], dtype=np.float64)
    if av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _latest_close(market_data: DailyMarketData, ticker: str) -> float:
    bars = market_data.bars_by_ticker.get(ticker) or []
    return float(bars[-1].close) if bars else 1.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min())


@torch.no_grad()
def _scenario_states(
    model: DailyNexusDemoModel,
    graph: GraphBundle,
    market_data: DailyMarketData,
    dates: Sequence[datetime],
    config: ArticleSimulationConfig,
) -> np.ndarray:
    device = config.resolved_device()
    model.eval()
    graph_data = graph.data.to(device)
    bars_by_time = {
        ticker: {bar.time: i for i, bar in enumerate(bars)}
        for ticker, bars in market_data.bars_by_ticker.items()
    }
    out: list[np.ndarray] = []
    for date in dates:
        batch_x: list[np.ndarray] = []
        batch_idx: list[int] = []
        for ticker in graph.tickers:
            pos = bars_by_time.get(ticker, {}).get(date)
            if pos is None or pos < config.window_days - 1:
                continue
            bars = market_data.bars_by_ticker[ticker]
            window = bars[pos - config.window_days + 1 : pos + 1]
            batch_x.append(_window_to_features(window))
            batch_idx.append(graph.ticker_to_idx[ticker])
        if not batch_x:
            out.append(np.zeros(NEXUS_OUTPUT_DIM, dtype=np.float32))
            continue
        states = []
        for start in range(0, len(batch_x), config.batch_size):
            x = torch.from_numpy(np.stack(batch_x[start : start + config.batch_size])).float()
            idx = torch.tensor(batch_idx[start : start + config.batch_size], dtype=torch.long)
            state = model.encode_state(x.to(device), idx.to(device), graph_data)
            states.append(state.cpu().numpy().astype(np.float32))
        out.append(np.vstack(states).mean(axis=0))
    return np.vstack(out).astype(np.float32)


def _perturb_episode(episode: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for key, value in episode.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value
    if len(out["prices"]) > 2:
        returns = np.diff(np.log(out["prices"]), prepend=np.log(out["prices"][0]))
        returns = returns + rng.normal(0.0, 0.0025, size=returns.shape)
        out["prices"] = (float(out["prices"][0]) * np.exp(np.cumsum(returns))).astype(np.float32)
        for key in ("open", "high", "low", "close"):
            out[key] = out["prices"].copy()
    out["market_states"] = (
        out["market_states"] + rng.normal(0.0, 0.01, size=out["market_states"].shape)
    ).astype(np.float32)
    return out


def _arena_metrics(
    phase: str,
    scenario: ScenarioSpec,
    runner: ArenaRunner,
    pairs: Sequence[CounterfactualPair],
    bad_actions: Sequence[np.ndarray],
) -> dict[str, Any]:
    total_rewards = np.asarray([sum(r) for r in runner._returns_history], dtype=np.float64)
    sharpes = np.asarray([_sharpe_from_returns(r) for r in runner._returns_history])
    final_equities = np.asarray([getattr(env, "_equity", 0.0) for env in runner._envs])
    actions = [
        np.asarray(record.action_vector, dtype=np.float32)
        for records in runner._records_by_trajectory.values()
        for record in records
    ]
    bad_stats = _bad_action_distance_stats(actions, bad_actions)
    return {
        "phase": phase,
        "scenario": scenario.name,
        "split": scenario.split,
        "start": scenario.start.date().isoformat(),
        "end": scenario.end.date().isoformat(),
        "mean_total_return": float(total_rewards.mean()) if total_rewards.size else 0.0,
        "mean_final_equity": float(final_equities.mean()) if final_equities.size else 0.0,
        "mean_sharpe": float(sharpes.mean()) if sharpes.size else 0.0,
        "max_drawdown": float(
            min(
                (_max_drawdown(np.cumprod(1.0 + np.asarray(r))) for r in runner._returns_history),
                default=0.0,
            )
        ),
        "divergence_count": len(runner.divergence_analyzer.all_divergences()),
        "counterfactual_pair_count": len(pairs),
        "n_records": sum(len(v) for v in runner._records_by_trajectory.values()),
        **bad_stats,
    }


def _sharpe_from_returns(returns: Sequence[float]) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float((arr.mean() / (arr.std(ddof=0) + 1e-9)) * math.sqrt(252.0))


def _bad_action_distance_stats(
    actions: Sequence[np.ndarray],
    bad_actions: Sequence[np.ndarray],
    *,
    threshold: float = BAD_ACTION_REPEAT_THRESHOLD,
) -> dict[str, float]:
    base = {
        "bad_action_repeat_rate": 0.0,
        "bad_action_repeat_count": 0.0,
        "bad_action_repeat_threshold": threshold,
        "bad_action_reference_count": float(len(bad_actions)),
        "bad_action_distance_min": 0.0,
        "bad_action_distance_p05": 0.0,
        "bad_action_distance_p25": 0.0,
        "bad_action_distance_p50": 0.0,
        "bad_action_distance_p75": 0.0,
        "bad_action_distance_p95": 0.0,
        "bad_action_distance_mean": 0.0,
    }
    if not actions or not bad_actions:
        return base
    distances = np.asarray(
        [min(float(np.linalg.norm(action - bad)) for bad in bad_actions) for action in actions],
        dtype=np.float64,
    )
    repeat_count = float(np.sum(distances <= threshold))
    return {
        **base,
        "bad_action_repeat_rate": repeat_count / float(distances.size),
        "bad_action_repeat_count": repeat_count,
        "bad_action_distance_min": float(np.min(distances)),
        "bad_action_distance_p05": float(np.quantile(distances, 0.05)),
        "bad_action_distance_p25": float(np.quantile(distances, 0.25)),
        "bad_action_distance_p50": float(np.quantile(distances, 0.50)),
        "bad_action_distance_p75": float(np.quantile(distances, 0.75)),
        "bad_action_distance_p95": float(np.quantile(distances, 0.95)),
        "bad_action_distance_mean": float(np.mean(distances)),
    }


def _bad_action_repeat_rate(
    actions: Sequence[np.ndarray], bad_actions: Sequence[np.ndarray]
) -> float:
    return _bad_action_distance_stats(actions, bad_actions)["bad_action_repeat_rate"]


def _oracle_action_from_return(value: float) -> np.ndarray:
    direction = float(np.clip(value / 0.05, -1.0, 1.0))
    size_factor = float(np.clip(abs(value) / 0.10, 0.05, 0.5))
    return np.asarray(
        [
            direction,
            0.0,
            2.0 * size_factor - 1.0,
            -0.2 if abs(value) > 0.05 else 0.2,
        ],
        dtype=np.float32,
    )


def _fallback_pairs_from_records(
    run_id: str,
    runner: ArenaRunner,
    episode: dict[str, Any],
    *,
    horizon: int = 10,
    max_pairs: int = 12,
) -> list[CounterfactualPair]:
    prices = np.asarray(episode["prices"], dtype=np.float32)
    if prices.size < 2:
        return []
    pairs: list[CounterfactualPair] = []
    used_steps: set[int] = set()
    records = [
        record
        for by_traj in runner._records_by_trajectory.values()
        for record in by_traj
        if record.state_artifact_path
    ]
    records.sort(key=lambda record: (record.step_index, record.trajectory_id))
    for record in records:
        if record.step_index in used_steps:
            continue
        future_idx = min(record.step_index + horizon, prices.size - 1)
        current_price = float(prices[record.step_index])
        if future_idx <= record.step_index or current_price <= 0:
            continue
        future_return = float((prices[future_idx] - current_price) / current_price)
        good_action = _oracle_action_from_return(future_return)
        bad_action = np.asarray(record.action_vector, dtype=np.float32)
        distance = float(np.linalg.norm(good_action - bad_action))
        if distance < 0.25:
            continue
        confidence = float(np.clip(abs(future_return) / 0.06 + distance / 4.0, 0.05, 1.0))
        outcome = max(0.05, abs(future_return) * 10.0)
        pairs.append(
            CounterfactualPair(
                run_id=run_id,
                divergence_step_index=record.step_index,
                sim_timestamp=record.sim_timestamp,
                state_artifact_path=record.state_artifact_path,
                good_action_vector=good_action.tolist(),
                bad_action_vector=bad_action.tolist(),
                good_outcome_sharpe=outcome,
                bad_outcome_sharpe=-outcome,
                confidence_score=confidence,
            )
        )
        used_steps.add(record.step_index)
        if len(pairs) >= max_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        base = {"phase": row.get("phase"), "split": row.get("split")}
        for key, value in row.items():
            if key in {"phase", "split"}:
                continue
            flat.append({**base, "metric": key, "value": value})
    _write_csv(path, flat)


def _rows_to_nested_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for row in rows:
        phase = str(row.get("phase"))
        split = str(row.get("split"))
        target = nested.setdefault(phase, {}).setdefault(split, {})
        target.update({key: value for key, value in row.items() if key not in {"phase", "split"}})
    return nested


def _write_article_summary(
    path: Path,
    *,
    status: str,
    inventory: dict[str, Any],
    split_manifest: dict[str, Any],
    nexus_metrics: Sequence[dict[str, Any]],
    arena_rows: Sequence[dict[str, Any]],
    feedback_pairs: Sequence[CounterfactualPair],
) -> None:
    initial_val = _metric_value(nexus_metrics, "initial", "val", "loss")
    hard_val = _metric_value(nexus_metrics, "hard_example", "val", "loss")
    test_dir = _metric_value(nexus_metrics, "hard_example", "test", "directional_accuracy")
    test_naive = _metric_value(nexus_metrics, "hard_example", "test", "naive_directional_accuracy")
    evidence = _status_evidence(nexus_metrics, arena_rows)
    lines = [
        "# Lumina Article Simulation Summary",
        "",
        f"Run status: **{status}**",
        (
            "Evidence gates: "
            f"{evidence['evidence_count']}/{evidence['evidence_total']} "
            f"(nexus-val={'pass' if evidence['nexus_val_improved'] else 'fail'}, "
            f"test-naive={'pass' if evidence['test_beats_naive'] else 'fail'}, "
            f"arena-sharpe={evidence['arena_improved_scenarios']}/"
            f"{evidence['arena_compared_scenarios']})."
        ),
        "",
        "## Data Reality",
        "",
        f"- Used tickers: {inventory.get('n_used_tickers', 0)}",
        f"- Date range: {inventory.get('start')} to {inventory.get('end')}",
        f"- Detected cadence: {inventory.get('detected_cadence')}",
        "- News modality: zero-filled because `news_events` is empty.",
        "- Graph modality: correlation-derived because supply-chain rows are absent.",
        "",
        "## Nexus Learning",
        "",
        f"- Validation MSE moved from {initial_val:.6f} to {hard_val:.6f}.",
        f"- Test directional accuracy is {test_dir:.3f} versus naive {test_naive:.3f}.",
        "",
        "## Arena Feedback",
        "",
        f"- Counterfactual pairs generated from train/val scenarios: {len(feedback_pairs)}",
    ]
    for row in arena_rows:
        if row["phase"] in {"baseline_eval", "post_feedback"}:
            lines.append(
                f"- {row['phase']} / {row['scenario']}: "
                f"Sharpe={float(row['mean_sharpe']):.3f}, "
                f"bad-action-repeat={float(row['bad_action_repeat_rate']):.3f}, "
                f"bad-action-distance-p50="
                f"{float(row.get('bad_action_distance_p50', 0.0)):.3f}"
            )
    lines.extend(
        [
            "",
            "## Split Contract",
            "",
            f"- Train windows end by {split_manifest['train_end']}.",
            f"- Validation windows are inside {split_manifest['val_start']} to "
            f"{split_manifest['val_end']}.",
            f"- Test windows start at {split_manifest['test_start']} or later.",
            "",
            "These results are article-demo evidence, not a production trading claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_top_counterfactuals(path: Path, pairs: Sequence[CounterfactualPair]) -> None:
    lines = ["# Top Counterfactual Examples", ""]
    for pair in sorted(pairs, key=lambda p: -p.confidence_score)[:10]:
        lines.append(
            f"- Step {pair.divergence_step_index}: good={pair.good_action_vector}, "
            f"bad={pair.bad_action_vector}, confidence={pair.confidence_score:.3f}, "
            f"Sharpe delta={pair.good_outcome_sharpe - pair.bad_outcome_sharpe:.3f}"
        )
    if len(lines) == 2:
        lines.append("- No counterfactual pairs were produced.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scenario_run_figures(
    run_dir: Path,
    phase: str,
    scenario: ScenarioSpec,
    runner: ArenaRunner,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping scenario figures because matplotlib is unavailable: {}", exc)
        return

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_stem(f"{phase}_{scenario.name}")

    plt.figure(figsize=(8, 4))
    for tid, returns in enumerate(runner._returns_history):
        arr = np.asarray(returns, dtype=np.float64)
        if arr.size == 0:
            continue
        equity = np.cumprod(1.0 + arr)
        plt.plot(equity, linewidth=1.2, alpha=0.8, label=f"traj {tid}")
    plt.title(f"Scenario Equity: {phase} / {scenario.name}")
    plt.xlabel("Step")
    plt.ylabel("Equity multiple")
    if runner._returns_history:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"equity_{stem}.png")
    plt.close()

    steps: list[int] = []
    mean_distances: list[float] = []
    for step, records_by_traj in sorted(runner._records_by_step.items()):
        actions = [
            np.asarray(record.action_vector, dtype=np.float32)
            for record in records_by_traj.values()
        ]
        distances = [
            float(np.linalg.norm(actions[i] - actions[j]))
            for i in range(len(actions))
            for j in range(i + 1, len(actions))
        ]
        steps.append(step)
        mean_distances.append(float(np.mean(distances)) if distances else 0.0)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, mean_distances, color="#3B82F6", linewidth=1.5)
    plt.title(f"Action Divergence: {phase} / {scenario.name}")
    plt.xlabel("Step")
    plt.ylabel("Mean pairwise L2")
    plt.tight_layout()
    plt.savefig(fig_dir / f"action_divergence_{stem}.png")
    plt.close()


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _write_figures(
    run_dir: Path,
    history: Sequence[dict[str, Any]],
    nexus_metrics: Sequence[dict[str, Any]],
    arena_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping figures because matplotlib is unavailable: {}", exc)
        return
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if history:
        plt.figure(figsize=(8, 4))
        plt.plot([r["train_loss"] for r in history])
        plt.title("Nexus Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(fig_dir / "learning_curves.png")
        plt.close()

    hard_rows = [r for r in nexus_metrics if r.get("phase") == "hard_example"]
    if hard_rows:
        labels = [str(r["split"]) for r in hard_rows if "directional_accuracy" in r]
        values = [
            float(r["directional_accuracy"]) for r in hard_rows if "directional_accuracy" in r
        ]
        naive = [
            float(r["naive_directional_accuracy"])
            for r in hard_rows
            if "naive_directional_accuracy" in r
        ]
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.2, values, width=0.4, label="Nexus")
        plt.bar(x + 0.2, naive, width=0.4, label="Naive")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.title("Directional Accuracy")
        plt.tight_layout()
        plt.savefig(fig_dir / "directional_accuracy.png")
        plt.close()

    eval_rows = [r for r in arena_rows if r.get("phase") in {"baseline_eval", "post_feedback"}]
    if eval_rows:
        scenarios = sorted({r["scenario"] for r in eval_rows})
        baseline = [
            float(
                next(
                    (
                        r["mean_sharpe"]
                        for r in eval_rows
                        if r["scenario"] == s and r["phase"] == "baseline_eval"
                    ),
                    0.0,
                )
            )
            for s in scenarios
        ]
        post = [
            float(
                next(
                    (
                        r["mean_sharpe"]
                        for r in eval_rows
                        if r["scenario"] == s and r["phase"] == "post_feedback"
                    ),
                    0.0,
                )
            )
            for s in scenarios
        ]
        x = np.arange(len(scenarios))
        plt.figure(figsize=(10, 4))
        plt.bar(x - 0.2, baseline, width=0.4, label="Baseline")
        plt.bar(x + 0.2, post, width=0.4, label="Post feedback")
        plt.xticks(x, scenarios, rotation=20, ha="right")
        plt.ylabel("Mean Sharpe")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "arena_sharpe.png")
        plt.close()


def _metric_value(
    rows: Sequence[dict[str, Any]],
    phase: str,
    split: str,
    metric: str,
) -> float:
    for row in rows:
        if row.get("phase") == phase and row.get("split") == split and metric in row:
            return float(row[metric])
    return math.nan


def _checkpoint_targets(root: Path) -> dict[str, Path]:
    return {
        "temporal": root / "temporal" / "best.pt",
        "semantic": root / "semantic" / "best.pt",
        "structural": root / "structural" / "best.pt",
        "fusion_best": root / "fusion" / "best.pt",
        "fusion_best_nexus": root / "fusion" / "best_nexus.pt",
        "agent_final": root / "agent" / "final.pt",
    }


def _checkpoint_metadata(run_dir: Path, kind: str, backup_dir: Path | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "article_run_dir": str(run_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "backup_dir": None if backup_dir is None else str(backup_dir),
    }


def _jsonable_config(config: ArticleSimulationConfig) -> dict[str, Any]:
    out = asdict(config)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, datetime):
            out[key] = value.isoformat()
    return out


def _json_default(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _set_local_mlflow(run_dir: Path) -> None:
    settings = get_settings()
    # Use sqlite instead of FileStore to avoid "Run not found" race conditions and deprecation warnings.
    db_path = (run_dir / "mlflow.db").resolve()
    settings.MLFLOW_TRACKING_URI = f"sqlite:///{db_path}"
