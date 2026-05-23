# LUMINA V3 — DEVELOPMENT ROADMAP

> "Chimera" Architecture · 12-Month Horizon · 8 Phases

---

## Phase Summary

| Phase | Name                              | Duration    | Quarter |
| ----- | --------------------------------- | ----------- | ------- |
| **0** | Foundations and Infrastructure    | Weeks 1–3   | Q1      |
| **1** | Data Engine & Feature Store       | Weeks 4–7   | Q1      |
| **2** | Perception Layer                  | Weeks 8–16  | Q2      |
| **3** | Deep Fusion Nexus                 | Weeks 17–20 | Q2–Q3   |
| **4** | Cognition & Spartan Training      | Weeks 21–30 | Q3–Q4   |
| **5** | Execution & Safety System         | Weeks 27–34 | Q3–Q4   |
| **6** | API, Simulation & Integration     | Weeks 31–36 | Q4      |
| **7** | React + TypeScript Dashboard      | Weeks 33–38 | Q4      |
| **8** | Testing, Paper Trading & Final QA | Weeks 37–42 | Q4      |

---

## PHASE 0 — Foundations and Infrastructure

**Duration:** Weeks 1–3 | **Quarter:** Q1

> Objective: Get the project skeleton standing. Nobody writes ML code on sand.
> Everything built in later phases depends on this being solid.

### Root Files

| File                 | Task                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`     | Define all project dependencies (PyTorch, StableBaselines3, pytorch-geometric, transformers, fastapi, redis, timescaledb-connector, loguru, pydantic-settings, pytest). Use Poetry or UV.                |
| `Makefile`           | Shortcuts for `make dev`, `make train`, `make test`, `make docker-up`, `make lint`.                                                                                                                      |
| `.env`               | Template with all required variables: `ALPACA_API_KEY`, `NEWSAPI_KEY`, `REDIS_URL`, `TIMESCALE_URL`, `POLYGON_KEY`, `UNCERTAINTY_THRESHOLD`, `MAX_DRAWDOWN_LIMIT`. Never committed; only `.env.example`. |
| `docker-compose.yml` | Full orchestration: `api`, `perception`, `brain`, `data`, `redis`, `timescaledb` services. Define internal networks, persistent volumes, and health checks.                                              |

### System Configuration

| File                          | Task                                                                                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/config/settings.py`  | `Pydantic BaseSettings` that loads `.env`. Defines `Settings` with all typed configuration parameters. Singleton with `@lru_cache`.                                    |
| `backend/config/constants.py` | Immutable constants: target tickers, market hours (NYSE open/close), temporal window sizes, embedding dimensions (`DIM_PRICE=128`, `DIM_SEMANTIC=64`, `DIM_GRAPH=32`). |
| `backend/config/logging.py`   | Configure Loguru to emit structured JSON. Define levels, log rotation, and sinks to file and stdout. Compatible with ELK/Splunk.                                       |

### Docker & Services

| File                             | Task                                                                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `docker/Dockerfile.api`          | Python slim image for FastAPI. No GPU. Installs only API and configuration dependencies.                                            |
| `docker/Dockerfile.data`         | Image for scrapers/collectors. No GPU. Includes websocket and scraping dependencies.                                                |
| `docker/Dockerfile.perception`   | Image with CUDA support. Installs PyTorch + GPU, transformers, pytorch-geometric. The heaviest one.                                 |
| `docker/Dockerfile.brain`        | Lightweight image with PyTorch CPU/GPU for production RL-agent inference.                                                           |
| `docker/services/redis.conf`     | Redis tuning: `maxmemory-policy allkeys-lru`, RDB persistence disabled for speed, configure `maxmemory` according to available RAM. |
| `docker/services/timescale.conf` | PostgreSQL + TimescaleDB tuning: `shared_buffers`, `work_mem`, `checkpoint_completion_target`. Enable the `timescaledb` extension.  |

**✅ Phase 0 Milestone:** `docker-compose up` starts all services without
errors. Logs appear in JSON format. Redis and TimescaleDB respond to health
checks.

---

## PHASE 1 — Data Engine & Feature Store

**Duration:** Weeks 4–7 | **Quarter:** Q1

> Objective: The system can ingest real-world data and store it. The Phase 2
> encoders cannot be trained without quality historical data, and the agent
> cannot operate without real-time data. This phase is the organism's blood
> supply.

### Data Collectors

| File                                               | Task                                                                                                                                                                                                                                      |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/data_engine/collectors/price_stream.py`   | WebSocket connection to the Alpaca Market Data API (or Polygon.io). Subscribe to 1-minute ticks for all tickers defined in `constants.py`. Implement automatic reconnection with exponential backoff. Publish each tick to Redis pub/sub. |
| `backend/data_engine/collectors/news_stream.py`    | WebSocket connection to NewsAPI/Benzinga for real-time financial news. Filter by relevant tickers. Publish each raw article to a Redis queue with a 24h TTL.                                                                              |
| `backend/data_engine/collectors/social_scraper.py` | Twitter/Reddit ingestion pipeline (via official API or Pushshift). Collect posts that mention the target tickers. Filter spam with basic rules. Batch to TimescaleDB every 5 minutes.                                                     |
| `backend/data_engine/collectors/chain_scrapers.py` | Scraping of supply-chain data (supplier/customer relationships) from sources such as SEC EDGAR or specialized APIs. Generates the initial static graph for the GNN. Periodic execution (weekly).                                          |

### Processing Pipelines

| File                                         | Task                                                                                                                                                                                                      |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/data_engine/pipelines/cleaning.py`  | Detection and correction of outliers in price series (IQR or Z-score method). Handling of missing candles (forward-fill with time limit). Normalization of news text (lowercase, strip HTML, truncation). |
| `backend/data_engine/pipelines/ingestion.py` | Asynchronous ETL pipelines (asyncio + asyncpg). Consumes from Redis pub/sub and batch-writes to TimescaleDB. Deduplication logic to avoid double inserts. Ingestion latency logging.                      |

### Storage

| File                                         | Task                                                                                                                                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/data_engine/storage/timescale.py`   | CRUD wrapper around TimescaleDB. Functions: `insert_ohlcv()`, `get_historical_window()`, `insert_news_event()`, `query_news_by_ticker()`. Create hypertables in `CREATE TABLE`.    |
| `backend/data_engine/storage/redis_cache.py` | Wrapper around `redis-py` with async connection. Functions: `set_embedding()`, `get_embedding()`, `get_latest_news_vector()`. Implement TTL by data type (price: 5min, news: 24h). |

### Feature Store

| File                                   | Task                                                                                                                                                                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/feature_store/definitions.py` | Formal definitions for each feature: name, source, TTL, dimension, update frequency. Acts as the contract between the Data Engine and the Encoders.                                                                          |
| `backend/feature_store/online.py`      | Redis adapter for sub-millisecond retrieval. Implements batched `MGET` to fetch all embeddings for a ticker in a single call.                                                                                                |
| `backend/feature_store/offline.py`     | TimescaleDB adapter for training. Generates batches of historical temporal windows. Implements a PyTorch-compatible DataLoader.                                                                                              |
| `backend/feature_store/client.py`      | Unified client. The Cognition layer NEVER calls Redis or TimescaleDB directly; it always uses this client. Transparently decides whether to go to the Hot Store or Cold Store depending on context (inference vs. training). |

**✅ Phase 1 Milestone:** 10 years of 1-minute OHLCV data loaded into
TimescaleDB. The system ingests real-time ticks with latency < 50ms. Embeddings
can be stored in and retrieved from Redis correctly.

---

## PHASE 2 — Perception Layer (The Encoders)

**Duration:** Weeks 8–16 | **Quarter:** Q2

> Objective: Build and train the three encoders independently. Each one must
> produce embeddings of verifiable quality before they are connected. They are
> the system's "eyes" — if they see poorly, everything else fails.

### Temporal Encoder (TFT)

| File                                          | Task                                                                                                                                                                                                                                                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/perception/temporal/preprocessor.py` | Normalization of OHLCV series (MinMaxScaler per window, not global, to avoid data leakage). Calculation of technical indicators (RSI, MACD, Bollinger Bands). Construction of input tensors with static covariates (sector, market cap tier).                    |
| `backend/perception/temporal/encoder.py`      | Implementation of the Temporal Fusion Transformer. Modules: Variable Selection Networks (VSN), Static Covariate Encoders, Gated Residual Networks, Multi-Head Attention. Output: dim=128 vector. Use `pytorch-forecasting` as a base or a custom implementation. |
| `backend/perception/temporal/inference.py`    | Real-time inference service. Background worker that subscribes to price in Redis, generates the TFT embedding every minute, and publishes it back to the Feature Store with TTL=90s. Inference latency logging.                                                  |

### Semantic Encoder (Distilled LLM)

| File                                           | Task                                                                                                                                                                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `backend/perception/semantic/tokenizer.py`     | Financial tokenizer. Extends the DistilRoBERTa vocabulary with domain terms (tickers, financial jargon, SEC filing entities). Handles multi-sentence context for long news articles.                                                                   |
| `backend/perception/semantic/llm_distilled.py` | Loading and wrapper for the quantized model (DistilRoBERTa-financial or Llama-3-8B in 4-bit with bitsandbytes). Implement Knowledge Distillation pipeline if a teacher model is used. Target latency < 100ms per text.                                 |
| `backend/perception/semantic/vectorizer.py`    | Vectorization service. Consumes from the Redis news queue, tokenizes each article, generates a dim=64 embedding (pooling from the [CLS] token), and publishes it to the Feature Store. Implement batching to process multiple simultaneous news items. |

### Structural Encoder (GNN)

| File                                             | Task                                                                                                                                                                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/perception/structural/graph_builder.py` | Adjacency-matrix builder. Reads price correlations from TimescaleDB (30-day rolling window), supply-chain relationships from `chain_scrapers.py`, and sector memberships. Generates the `HeteroData` object from `pytorch-geometric`. |
| `backend/perception/structural/dynamic_edges.py` | Updates dynamic edge weights. Scheduled task (daily cron) that recalculates correlations and updates the stored graph. Ensure the graph in Redis is always up to date without interrupting inference.                                 |
| `backend/perception/structural/gnn_model.py`     | Implementation of Graph Attention Network v2 (`GATv2Conv` from pytorch-geometric). Multi-layer with residual connections. Message-passing mechanism. Output: dim=32 embedding per node (asset).                                       |

**✅ Phase 2 Milestone:** t-SNE visualization of the embeddings. "Crash" states
(2008, 2020, 2022) must visually cluster separately from "Bull" states. If no
separation is visible, debug the encoders before continuing. Document in
`notebooks/`.

---

## PHASE 3 — Deep Fusion Nexus

**Duration:** Weeks 17–20 | **Quarter:** Q2–Q3

> Objective: Connect the three encoders into a single market-state
> representation. This is V3's central innovation: fusion produces something
> greater than the sum of its parts. Cross-Modal Attention is what allows the
> system to "understand" which modality to trust.

### Fusion Modules

| File                              | Task                                                                                                                                                                                                                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/fusion/concatenation.py` | Concatenation of the three embedding vectors: `[price_emb(128) \| news_emb(64) \| graph_emb(32)]` → dim=224 super-vector. Include post-concatenation normalization layer (LayerNorm).                                                                                                                     |
| `backend/fusion/attention.py`     | Cross-Modal Attention block. Implement a Transformer encoder block where the three modalities can "attend" to each other. The learned attention weights determine how much weight each modality has depending on the market regime. Use PyTorch's `nn.MultiheadAttention`. Output: refined latent vector. |
| `backend/fusion/state_builder.py` | Main orchestrator. Sequentially calls the three encoders (fetch from Feature Store), runs concatenation and attention, and returns the **Super-State** ready for the agent. This is the module invoked by the inference loop on each tick. Total state-construction latency logging.                      |

**✅ Phase 3 Milestone (Parity Check):** Connect the Super-State to a simplified
version of the agent (no RL, just a linear head) and compare its ability to
predict price movement against the V2 LSTM. If the new system does not
outperform V2, debug the attention block before continuing with RL training.

---

## PHASE 4 — Cognition & Spartan Training

**Duration:** Weeks 21–30 | **Quarter:** Q3–Q4

> Objective: Forge the agent. First we teach it to imitate (Cloning), then to
> survive chaos (Domain Randomization), and finally to discover its own Alpha
> (Pure RL). The three phases are sequential and must not be skipped.

### Simulation Environments

| File                                                  | Task                                                                                                                                                                                                                                                      |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/simulation/environments/base_env.py`         | Base environment compatible with the Gymnasium interface (`reset()`, `step()`, `render()`). Manages portfolio state (cash, open positions, PnL). Implement `observation_space` (Super-State vector) and `action_space` (continuous 4-dimensional vector). |
| `backend/simulation/environments/live_shadow.py`      | Mirror environment of the real market for paper trading. Connects to live Feature Store data instead of historical data. Used in Q4 for final validation before live deployment.                                                                          |
| `backend/simulation/environments/reward_functions.py` | Implementation of reward functions: differential Sharpe Ratio (reward per step), Sortino Ratio (penalizes only downside volatility), Calmar Ratio (reward/max_drawdown). Liquidation penalty function (extreme drawdown = very negative reward).          |
| `backend/simulation/generators/synthetic_data.py`     | Synthetic data generators: Geometric Brownian Motion (GBM) for normal regime, Jump Diffusion to incorporate tail events. Calibrate parameters (μ, σ, λ) against the real historical distribution.                                                         |
| `backend/simulation/generators/adversarial.py`        | "Nightmare Scenarios" generator: volatility multiplication (2x, 3x, 5x VIX), artificial spread widening, data outages (missing candles), forced correlation decoupling. Configure as probabilistic augmentation during training.                          |
| `backend/simulation/generators/scenario_loader.py`    | Loader for specific historical crashes for test sets: 2008 (Lehman), 2010 (Flash Crash), 2015 (China), 2020 (COVID), 2022 (Bear + FTX). Enables deterministic replay for policy comparison.                                                               |

### RL Agent

| File                                        | Task                                                                                                                                                                                                                                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/cognition/policy/networks.py`      | Definition of Actor and Critic networks in PyTorch. The Actor receives the Super-State (dim=224 after fusion) and emits the Action Vector (4 floats). The Critic estimates expected reward. Implement a shared feature extractor between both networks for efficiency.                             |
| `backend/cognition/policy/distributions.py` | Probability distributions for the continuous action space. Beta distribution for bounded actions ([−1, 1], [0, 1]). Gaussian for sizing. Implement `log_prob()` for policy-gradient calculation.                                                                                                   |
| `backend/cognition/agent/ppo_continuous.py` | Implementation of the PPO (Proximal Policy Optimization) algorithm adapted for continuous action. Probability-ratio clipping (ε=0.2). Implement value function clipping and entropy bonus for exploration. Can use StableBaselines3 as a base with custom networks.                                |
| `backend/cognition/agent/sac_agent.py`      | Alternative implementation of SAC (Soft Actor-Critic). Useful for exploration in high-uncertainty regimes because of its maximum-entropy principle. Keep both (PPO and SAC) and select based on metrics during training.                                                                           |
| `backend/cognition/agent/uncertainty.py`    | Epistemic uncertainty estimator. Implement Monte Carlo Dropout: perform N=10 forward passes with `model.train()` active (dropout enabled) and calculate the variance of the Action Vector. Alternatively, Deep Ensembles with 3 lightweight policies in parallel. Output: `entropy_score` (float). |

### Training Loop

| File                                       | Task                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/cognition/training/curriculum.py` | Controller for the 3-phase training curriculum: **Phase A** (Behavioral Cloning — imitates V2 LSTM + MA crossover signals), **Phase B** (Domain Randomization — trains on "warped" episodes from the adversarial generator), **Phase C** (Pure RL — maximizes Sharpe Ratio without the restriction of imitating the teacher). Manage transitions between phases based on metrics (win rate, Sharpe). |
| `backend/cognition/training/trainer.py`    | Main training loop. Integrates: Feature Store (offline), Simulation Environments, the PPO/SAC agent, and checkpoint saving in `models/`. Metrics logging to W&B or TensorBoard. Resume-from-checkpoint support.                                                                                                                                                                                      |

**✅ Phase 4 Milestone:** The agent trained on the COVID-2020 crash scenario
does not trigger the kill switch due to maximum drawdown, preserving capital
through autonomous decisions. Sharpe Ratio > 1.5 out-of-sample.

---

## PHASE 5 — Execution & Safety System

**Duration:** Weeks 27–34 | **Quarter:** Q3–Q4 (parallel with Phase 4)

> Objective: The "Motor Cortex" and the "Amygdala". Regardless of how
> intelligent the agent is, if execution fails or safety has holes, capital
> evaporates. This phase runs in parallel with Phase 4 because the Safety System
> must be ready before the agent enters paper trading.

### Brokers

| File                                 | Task                                                                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/execution/broker/alpaca.py` | Adapter for the Alpaca Trading API. Implement: `submit_order()`, `cancel_order()`, `get_position()`, `get_account()`. Handle paper trading vs. live trading through a flag. Rate limiting and retry logic. |
| `backend/execution/broker/ibkr.py`   | Adapter for Interactive Brokers (future-proofing). Use `ib_insync` or `ib_async`. Implement the same abstract interface as `alpaca.py` for transparent interchangeability.                                 |

### Execution Logic

| File                                        | Task                                                                                                                                                                                                                                                             |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/execution/logic/order_manager.py`  | Order manager. Translates the agent's Action Vector into real orders: if `urgency < 0.5` → Limit Order at the bid; if `urgency >= 0.5` → Market Order. Implement order chasing (update the limit if not filled in N seconds). Track pending orders.              |
| `backend/execution/logic/position_sizer.py` | Position-size calculator. Implement adjusted Kelly Criterion (fractional Kelly, typically 25%–50% to reduce volatility). Volatility scaling (ATR-based). The agent's Action Vector `sizing` is multiplied by this module's output as an additional safety layer. |

### Safety System

| File                                           | Task                                                                                                                                                                                                                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `backend/execution/safety/risk_gate.py`        | **The Uncertainty Gate.** Implements the `execute_step()` flow described in the architecture: 1) Generate Super-State, 2) Calculate `entropy_score`, 3) If it exceeds `CRITICAL_THRESHOLD` → Safety Protocol, 4) Get action from agent, 5) Pass through Arbitrator. First line of defense. |
| `backend/execution/safety/circuit_breakers.py` | Hardware circuit breakers. Hard-coded rules that cannot be overridden by the agent: `MAX_DAILY_LOSS_PCT`, `MAX_DRAWDOWN_PCT`, `MAX_POSITION_SIZE_PCT`, `LEVERAGE_LIMIT`, `MARKET_HOURS_CHECK`. If any triggers → immediate zero position.                                                  |
| `backend/execution/safety/arbitrator.py`       | **Final Arbitrator.** Last instance before sending any order to the broker. Receives the agent's proposed action and validates it against all active risk rules. If there is a violation → override to defensive action. Full logging of each veto with justification.                     |

---

## PHASE 6 — API, Simulation & Integration

**Duration:** Weeks 31–36 | **Quarter:** Q4

> Objective: Expose the system to the outside world with a robust API and
> connect all modules in the full end-to-end inference loop.

### FastAPI API

| File                               | Task                                                                                                                                                                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/api/main.py`              | FastAPI entry point. Configure CORS, logging and authentication middleware. Register all routers. Lifespan events for Redis and DB connections on startup/shutdown.                                                         |
| `backend/api/deps.py`              | Dependency Injection. DB session providers, Redis pool, Feature Store client instance. Used with `Depends()` in endpoints.                                                                                                  |
| `backend/api/routes/agent.py`      | Agent monitoring endpoints: `GET /agent/status` (current action, confidence, entropy score), `GET /agent/action-history` (log of last N decisions). WebSocket `/agent/live` for real-time streaming to the dashboard.       |
| `backend/api/routes/backtest.py`   | Simulation endpoints: `POST /backtest/run` (launches a backtest with parameters), `GET /backtest/{id}/results` (retrieves metrics: Sharpe, Sortino, Max Drawdown, Win Rate).                                                |
| `backend/api/routes/data.py`       | Data inspection endpoints: `GET /data/ohlcv/{ticker}` (historical data), `GET /data/embeddings/{ticker}` (current Feature Store embeddings), `GET /data/news/{ticker}` (latest ingested news).                              |
| `backend/api/routes/monitoring.py` | Health checks and metrics: `GET /health` (liveness probe), `GET /metrics` (Prometheus format with latencies, throughput, errors by module). Grafana integration via metrics endpoint.                                       |
| `backend/api/routes/risk.py`       | Manual risk controls: `POST /risk/kill-switch` (liquidates all positions immediately), `PUT /risk/threshold` (adjust `UNCERTAINTY_THRESHOLD` hot), `GET /risk/status` (state of the Uncertainty Gate and circuit breakers). |

**✅ Phase 6 Milestone:** Functional end-to-end inference loop: market tick →
Feature Store → Perception → Fusion → Cognition → Safety → Broker (paper). Total
latency < 50ms measured end-to-end.

---

## PHASE 7 — React + TypeScript Dashboard

**Duration:** Weeks 33–38 | **Quarter:** Q4 (parallel with Phase 6)

> Objective: Replace the Streamlit dashboard with a professional React +
> TypeScript interface that enables real-time monitoring of the full system. The
> dashboard is the human operator's "cockpit."

### Frontend Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── main.tsx                     # Vite + React entry point
│   ├── App.tsx                      # Main router + Layout
│   ├── vite-env.d.ts
│   │
│   ├── api/
│   │   ├── client.ts                # Axios instance with interceptors
│   │   ├── agent.ts                 # Calls to /agent/*
│   │   ├── risk.ts                  # Calls to /risk/*
│   │   ├── backtest.ts              # Calls to /backtest/*
│   │   └── data.ts                  # Calls to /data/*
│   │
│   ├── types/
│   │   ├── agent.types.ts           # AgentStatus, ActionVector, DecisionLog
│   │   ├── market.types.ts          # OHLCV, Ticker, MarketRegime
│   │   ├── risk.types.ts            # RiskStatus, CircuitBreakerState, UncertaintyScore
│   │   └── backtest.types.ts        # BacktestResult, PerformanceMetrics
│   │
│   ├── hooks/
│   │   ├── useAgentStream.ts        # WebSocket hook for real-time agent data
│   │   ├── usePerceptionHealth.ts   # Polling encoder status
│   │   ├── usePortfolio.ts          # Portfolio state and P&L
│   │   └── useRiskGate.ts           # Real-time Uncertainty Gate state
│   │
│   ├── store/
│   │   ├── agentSlice.ts            # Zustand/Redux slice for agent state
│   │   ├── riskSlice.ts             # Risk manager state
│   │   └── marketSlice.ts           # Active market data
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx          # Side navigation
│   │   │   ├── TopBar.tsx           # Global status bar (latency, connection)
│   │   │   └── GridLayout.tsx       # Responsive grid system for panels
│   │   │
│   │   ├── panels/
│   │   │   ├── AgentStatusPanel.tsx       # Current action, confidence, action vector
│   │   │   ├── UncertaintyGatePanel.tsx   # Entropy meter with visual threshold
│   │   │   ├── PerceptionHealthPanel.tsx  # TFT / BERT / GNN status with latencies
│   │   │   ├── FusionStatePanel.tsx       # Cross-Modal Attention weight visualization
│   │   │   ├── PnLPanel.tsx               # Equity curve (Recharts AreaChart)
│   │   │   ├── PositionsPanel.tsx         # Open positions with PnL per trade
│   │   │   ├── MarketRegimePanel.tsx      # Regime indicator (Bull/Bear/Crisis/Sideways)
│   │   │   ├── TrainingPhasePanel.tsx     # Current curriculum phase (A/B/C)
│   │   │   └── ActionLogPanel.tsx         # Real-time log of agent decisions
│   │   │
│   │   └── shared/
│   │       ├── MetricCard.tsx       # Metric card with delta indicator
│   │       ├── StatusDot.tsx        # Status indicator (live/warning/offline)
│   │       ├── GaugeChart.tsx       # Reusable gauge for uncertainty/confidence
│   │       └── KillSwitchButton.tsx # Emergency button with double confirmation
│   │
│   ├── pages/
│   │   ├── DashboardPage.tsx        # Main view with all panels
│   │   ├── BacktestPage.tsx         # Launch and visualize backtests
│   │   ├── DataInspectorPage.tsx    # Explore Feature Store data
│   │   └── SettingsPage.tsx         # Adjust risk parameters hot
│   │
│   └── styles/
│       ├── globals.css              # Global CSS variables + reset
│       └── theme.ts                 # Design tokens (colors, typography, spacing)
│
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.ts
```

### Frontend Technology Stack

| Decision     | Technology                          | Rationale                                                          |
| ------------ | ----------------------------------- | ------------------------------------------------------------------ |
| Framework    | React 18 + TypeScript               | Strict typing, mature ecosystem                                    |
| Build tool   | Vite                                | Instant HMR, fast builds                                           |
| Global state | Zustand                             | Simpler than Redux for this case                                   |
| Charts       | Recharts + D3.js                    | Recharts for PnL/area, D3 for custom visualizations (Fusion State) |
| Styling      | Tailwind CSS                        | Utility-first for speed                                            |
| WebSocket    | `@tanstack/react-query` + native WS | Query for REST, native WS for streaming                            |
| HTTP Client  | Axios                               | Interceptors for auth and error handling                           |
| Testing      | Vitest + React Testing Library      | Same API as Jest, integrated with Vite                             |

**✅ Phase 7 Milestone:** Dashboard with real agent data in paper trading. The
Uncertainty Gate meter shows the entropy score in real time. The Kill Switch
works end-to-end.

---

## PHASE 8 — Testing, Paper Trading & Final QA

**Duration:** Weeks 37–42 | **Quarter:** Q4

> Objective: Validate the full system under real conditions before any real
> capital. This phase is not optional — it is the difference between an
> experiment and a production system.

### Test Suite

| Directory/File                             | Task                                                                                                                                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/unit/test_encoders.py`              | Unit tests for TFT, LLM vectorizer, and GNN. Verify output dimensions, behavior with null inputs, and reproducibility of the embedding given the same input.                                           |
| `tests/unit/test_fusion.py`                | Tests for the fusion module. Verify that concatenation produces dim=224 and that the attention block does not produce NaN on extreme inputs.                                                           |
| `tests/unit/test_safety.py`                | Critical tests for the safety system. Verify that the circuit breaker activates correctly, that the Arbitrator vetoes actions that violate constraints, and that the Kill Switch liquidates positions. |
| `tests/integration/test_inference_loop.py` | End-to-end loop test. Input: synthetic price tick. Assert: a valid action is generated in < 50ms. Verify that the Uncertainty Gate correctly intercepts high-entropy inputs.                           |
| `tests/integration/test_data_pipeline.py`  | Data Engine integration tests. Verify TimescaleDB writes and reads, Redis TTL, and consistency between Hot and Cold Store.                                                                             |
| `tests/e2e/test_backtest.py`               | End-to-end backtest test. Launch simulation on 2020 data and verify that calculated metrics are coherent (Sharpe not infinite, drawdown < 100%).                                                       |

### Research Notebooks

| File                                    | Purpose                                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `notebooks/01_data_exploration.ipynb`   | EDA of historical data. Return distributions, outlier identification, data gaps.                                    |
| `notebooks/02_encoder_validation.ipynb` | t-SNE visualization of embeddings from the 3 encoders. Confirmation of the Phase 2 milestone.                       |
| `notebooks/03_fusion_ablation.ipynb`    | Ablation study: compare full Super-State vs. TFT only vs. TFT+BERT only. Quantify the contribution of each encoder. |
| `notebooks/04_training_analysis.ipynb`  | Analysis of training curves. Learning curves, reward evolution by phase, PPO vs. SAC comparison.                    |
| `notebooks/05_backtest_analysis.ipynb`  | Detailed backtest analysis. Trade breakdown, drawdown periods, behavior during historical crashes.                  |

### DevOps Scripts

| File                             | Purpose                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `scripts/init_db.py`             | Initialization script: creates hypertables in TimescaleDB, configures indexes, loads initial historical data. |
| `scripts/download_historical.py` | Bulk download of historical data from Alpaca/Polygon. 10 years, 1 minute, all tickers.                        |
| `scripts/train.sh`               | Shell script that launches the full training pipeline with the correct parameters.                            |
| `scripts/deploy.sh`              | Deployment script: build Docker images, push to registry, update compose.                                     |

### Documentation

| File                     | Purpose                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `docs/architecture.md`   | Reference architecture document (based on the Deep Fusion Architecture doc).                                   |
| `docs/api_reference.md`  | Full REST API reference generated from FastAPI's OpenAPI spec.                                                 |
| `docs/runbook.md`        | Operational runbook: how to start the system, how to respond to alerts, how to use the Kill Switch.            |
| `docs/training_guide.md` | Guide to the Spartan Training Curriculum. Recommended parameters, how to interpret the metrics for each phase. |

**✅ Final Milestone (Paper Trading):** The system operates for 30 days in paper
trading with Alpaca. Target metrics: Sharpe Ratio > 1.5, Max Drawdown < 15%, Win
Rate > 52%. The agent navigates at least one elevated-volatility event
(VIX > 25) without triggering the kill switch.

---

## Summary of Dependencies Between Phases

```
Phase 0 (Infra) ──────────────────────────────────────────────►
Phase 1 (Data) ──────────────────────────────────────────────►
               Phase 2 (Perception) ────────────────────────►
                                    Phase 3 (Fusion) ───────►
                                                    Phase 4 (Cognition) ──►
               Phase 5 (Execution) ─────────────────────────────────────►
                                                    Phase 6 (API) ───────►
                                                    Phase 7 (Dashboard) ─►
                                                                  Phase 8 ►
```

> **Golden rule:** Never start Phase N+1 if the Phase N milestone has not been
> met. Phases 5 and 7 can run in parallel with 4 and 6 respectively.
