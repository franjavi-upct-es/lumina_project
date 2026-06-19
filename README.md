<!-- README.md -->
<!--  -->

# Lumina V3 — The "Chimera" Deep-Fusion Trading System

> _"We choose to go to the moon in this decade and do the other things, not
> because they are easy, but because they are hard."_ — JFK
>
> Reference document:
> [`docs/Lumina_V3_Deep_Fusion_Architecture.md`](./docs/Lumina_V3_Deep_Fusion_Architecture.md)
> Project structure spec:
> [`docs/LUMINA_V3_PROJECT_STRUCTURE.md`](./docs/LUMINA_V3_PROJECT_STRUCTURE.md)

This repository implements the _cognitive autonomous agent_ described in the
architecture document. It is **not** a backtesting framework: the goal is a
single neural network that _perceives_, _fuses_, _reasons_, and _acts_ — with
explicit uncertainty quantification and adversarial training.

---

## 1. Scientific Overview

### 1.1 The paradigm shift

A classical algorithmic trading system separates concerns linearly: _data
ingestion → feature engineering → forecasting model → rule-based execution_.
Each stage operates in its own representational space and the boundaries between
them are loss-prone. Lumina V2 followed this pattern and worked well in calm
regimes; the V3 plan replaces it with **deep sensor fusion** — a single,
end-to-end differentiable computation graph whose internal representation is
_holographic_: every component "sees" all modalities simultaneously, and the
agent acts on the fused latent state, not on individual signals.

The biological analogy is the **thalamus**. A driver does not vote between
vision, hearing, and balance separately; sensory streams are merged before the
prefrontal cortex makes a decision. Lumina V3 replicates this: the Cross-Modal
Attention block fuses three encoder outputs into a single latent state vector,
and only then does the cognitive agent act.

### 1.2 The Chimera architecture

```
                     ┌───────────────────┐
               OHLCV │  TFT Encoder      │── 128-d ─┐
                     └───────────────────┘          │
                     ┌───────────────────┐          │   ┌───────────────────┐
    News + filings   │ Distilled LLM     │── 64-d ──┼──▶│ Cross-Modal       │
                     └───────────────────┘          │   │ Attention Nexus   │── 256-d ─┐
                     ┌───────────────────┐          │   └───────────────────┘          │
Correlation graph    │ GATv2             │── 32-d ──┘             │                    │
                     └───────────────────┘                MC-Dropout uncertainty       │
                                                                                       ▼
                                                                           ┌──────────────────────┐
                                                                           │ PPO/SAC Agent        │
                                                                           │  • 4-D action vector │
                                                                           │  • Uncertainty Gate  │
                                                                           └──────────────────────┘
                                                                                       │
                                                                           ┌──────────────────────┐
                                                                           │ Safety Arbitrator    │
                                                                           │  (hard-rule veto)    │
                                                                           └──────────────────────┘
                                                                                       │
                                                                                       ▼
                                                                                   Broker API
```

### 1.3 Dimensional contract

| Symbol             | Value | Source                                        |
| ------------------ | ----- | --------------------------------------------- |
| `DIM_PRICE`        | 128   | TFT output                                    |
| `DIM_SEMANTIC`     | 64    | Distilled financial LLM output                |
| `DIM_GRAPH`        | 32    | GATv2 output                                  |
| `DIM_FUSED`        | 224   | = `DIM_PRICE + DIM_SEMANTIC + DIM_GRAPH`      |
| `NEXUS_OUTPUT_DIM` | 256   | Latent state given to the agent               |
| `ACTION_DIM`       | 4     | `[direction, urgency, sizing, stop_distance]` |

These dimensions are **fixed** in `backend/config/constants.py`; changing any of
them requires retraining all encoders.

---

## 2. Mathematical Specifications

### 2.1 Temporal Fusion Transformer (TFT)

Lim et al. (2021). Replaces a vanilla LSTM. For an input window **X** ∈
ℝ<sup>T×F</sup> (T = `OHLCV_WINDOW_MINUTES`, F = OHLCV+indicators):

1. **Variable Selection Network (VSN)**:

$$
\mathbf{w}_t = \mathrm{softmax}\big(\mathrm{GRN}(\mathbf{x}_t)\big), \qquad
\tilde{\mathbf{x}}_t = \sum_{i} w_{t,i} \cdot \mathrm{GRN}_i(x_{t,i})
$$

where GRN = Gated Residual Network. VSN automatically silences features that are
not informative for the current regime (e.g. ignores volume in low-liquidity
hours).

1. **Static covariate encoders** condition the LSTM/attention layers on metadata
   such as sector and market-cap tier.

2. **Multi-head causal self-attention** lets the network attend to fractal
   patterns _months_ in the past:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\tfrac{Q K^\top}{\sqrt{d_k}}\Big) V
$$

with a causal mask preventing leakage from the future.

1. **Output**: a 128-d vector representing the price-action regime, plus an
   _interpretable_ attention map over the input window.

### 2.2 Distilled Quant-LLM

Replaces sentiment scores. Teacher = `ProsusAI/finbert` (110M params); student =
our compact 4-layer Transformer (~15M params), trained with the composite loss:

$$
\mathcal{L}_{\text{distill}}
\;=\;
\alpha_{\text{MSE}} \, \big\| s_\theta(x) - t(x) \big\|^2
\;+\;
\alpha_{\text{cos}} \, \big(1 - \cos\angle\!\big(s_\theta(x),\, t(x)\big)\big)
\;+\;
\alpha_{\text{task}} \, \mathrm{CE}\!\big(h_\phi(s_\theta(x)),\, y\big)
$$

with default weights
$(\alpha_\text{MSE}, \alpha_\text{cos}, \alpha_\text{task}) = (0.5, 0.3, 0.2)$.
The cosine term aligns the _direction_ of the embedding (semantic meaning), not
just its magnitude. The task term keeps the student useful when the teacher is
wrong.

Latency budget: **< 100 ms per inference** on a single CPU core (architecture
spec §3.B).

### 2.3 Graph Attention Network v2 (GATv2)

Brody et al. (2022). For each node $i$ and layer $\ell$:

$$
e_{ij}^{(\ell)} \;=\; \mathbf{a}^\top \, \mathrm{LeakyReLU}\!\Big(W_s\, h_i^{(\ell)} + W_t\, h_j^{(\ell)} + W_e\, e_{ij}\Big)
$$

$$
\alpha_{ij} \;=\; \mathrm{softmax}_j(e_{ij}), \qquad
h_i^{(\ell+1)} \;=\; \sigma\!\Big(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W_v\, h_j^{(\ell)}\Big)
$$

Key difference vs the original GAT: attention weights depend on _both_
endpoints, not only the source. This fixes the "static attention" pathology
proven by Brody et al.

Edges combine:

- **Static edges**: sector membership, supply-chain (10-K-derived), ETF
  co-membership.
- **Dynamic edges**: rolling 60-day Pearson correlation, refreshed daily.

Output: 32-d "structural pressure" vector per node.

### 2.4 Cross-Modal Attention Nexus

Tsai et al. (2019). Fuses the three modality vectors via a multi-head
Transformer block whose "sequence" has length 3 (one token per modality). The
block re-weights each modality's contribution and a learnable sigmoid gate
further suppresses or amplifies regions of the 224-d super-vector. The MLP head
maps to 256-d:

$$
\mathbf{z}_{\text{fused}} = \big[\, \mathbf{z}_{\text{price}} \;\big|\; \mathbf{z}_{\text{sem}} \;\big|\; \mathbf{z}_{\text{graph}} \,\big] \;\in\; \mathbb{R}^{224}
$$

$$
\mathbf{z}' = \mathrm{CrossModalAttn}(\mathbf{z}_{\text{fused}}), \qquad
\mathbf{g} = \sigma(W_g \, \mathbf{z}'), \qquad
\mathbf{s} = \mathrm{MLP}(\mathbf{g} \odot \mathbf{z}') \;\in\; \mathbb{R}^{256}
$$

### 2.5 Cognitive Core (PPO with continuous 4-D actions)

Action vector $a \in [-1, 1]^4$:

- $a_0$ = portfolio direction (full short ↔ full long)
- $a_1$ = order urgency (limit/maker ↔ market/taker)
- $a_2$ = position sizing (Kelly-fraction multiplier)
- $a_3$ = stop-distance (ATR multiplier 0.5–4.0)

The squashed-Gaussian policy emits $\mu \in \mathbb{R}^4$,
$\sigma \in \mathbb{R}^4$; we sample $\xi \sim \mathcal{N}(\mu, \sigma^2)$ and
squash $a = \tanh(\xi)$. The change-of-variables correction (Haarnoja et al.,
2018, Appendix C):

$$
\log \pi(a \mid s) \;=\; \log p(\xi \mid s) \;-\; \sum_{i=1}^{4} \log\!\big(1 - a_i^2 + \varepsilon\big).
$$

PPO clipped objective (Schulman et al., 2017):

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta)
\;=\;
\mathbb{E}_t \Big[
\min\!\big(r_t(\theta)\, \hat{A}_t,\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, \hat{A}_t\big)\Big],
$$

$$
r_t(\theta) \;=\; \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},
\qquad
\hat{A}_t \;=\; \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l} \quad \text{(GAE)}.
$$

Default $\epsilon = 0.2$, $\gamma = 0.99$, $\lambda = 0.95$. Approximate KL
early-stopping at $1.5 \times 0.02$.

### 2.6 Epistemic uncertainty (Monte Carlo Dropout)

For each decision step we run $N = 10$ stochastic forward passes with dropout
enabled (Gal & Ghahramani, 2016). Per-dimension epistemic standard deviation:

$$
\sigma_d \;=\; \mathrm{std}_{i=1..N}\!\big(a_d^{(i)}\big), \quad d = 0..3
$$

$$
u_t \;=\; \tfrac{1}{4} \sum_{d=0}^{3} \sigma_d, \qquad
\bar{u}_t \;=\; \tfrac{1}{W} \sum_{k=t-W+1}^{t} u_k \quad (W = 10).
$$

The Uncertainty Gate uses **hysteresis** to avoid chattering:

- if not vetoing **and** $\bar{u}_t > \tau_{\text{high}}\,(=0.85)$ → engage veto
- if vetoing **and** $\bar{u}_t < \tau_{\text{low}}\,(=0.50)$ → release veto

When vetoed, the action vector is replaced by $[0, 0, 0, 0]$ (flat, passive,
zero size). The fallback is **NOT** recorded in the rollout buffer so the policy
keeps learning what it would have done.

### 2.7 Spartan Curriculum (training stages)

| Phase | Name                 | Method                                            | Acceptance gate                       |
| ----- | -------------------- | ------------------------------------------------- | ------------------------------------- |
| A     | Behavioural cloning  | Imitate V2 / MA-crossover oracle (MSE on actions) | Direction-accuracy ≥ 0.55             |
| B     | Domain randomisation | PPO on warped episodes (6 warp types, see §2.8)   | Mean episode reward ≥ 10              |
| C     | Sharpe optimisation  | PPO with Sharpe-shaped reward, full daily horizon | Out-of-sample annualised Sharpe ≥ 1.0 |

### 2.8 Adversarial warps (Phase B)

Six perturbation types injected by `simulation/generators/adversarial.py`:

1. `FLASH_CRASH` — instantaneous −5..−15% gap over 5 bars.
2. `SUSTAINED_CRASH` — geometric drift −20% over the second half.
3. `VOL_SPIKE` — 20-bar window of vol × 10.
4. `CORRELATION_BREAK` — Gaussian noise (σ=0.5) added to the latent state.
5. `SEMANTIC_PANIC` — additive negative drift on the semantic channels +
   uncertainty spike.
6. `SILENT_DRIFT` — log-linear drift of −0.1%/bar with normal volatility.

The agent must remain solvent on a uniformly-random mix of these.

---

## 3. Repository Layout

```bash
.
├── backend/
│   ├── api/                 # FastAPI server
│   ├── cognition/           # Agent + curriculum (Layer 3)
│   │   ├── agent/           # PPO, uncertainty gate, policy network
│   │   ├── policy/          # Networks, action distributions
│   │   └── training/        # BC, DR, Sharpe optimiser, curriculum orchestrator
│   ├── config/              # Constants (dim contract!), settings, logging
│   ├── data_engine/         # Collectors (Polygon + yfinance), pipelines, storage
│   ├── execution/           # Broker adapters, sizing, safety, kill switch
│   ├── feature_store/       # Hot/cold store unified client
│   ├── fusion/              # Cross-modal attention + Nexus (Layer 2)
│   ├── integration/         # End-to-end loop with latency budget
│   ├── paper_trading/       # Production runner + reports
│   ├── perception/          # TFT, distilled LLM, GATv2 (Layer 1)
│   └── simulation/          # Episode generators + Gymnasium env
├── alembic/                 # TimescaleDB migrations
├── docker/                  # Per-service Dockerfiles
├── frontend/                # React + TypeScript dashboard
├── notebooks/               # `#%%`-format notebooks (see notebooks/README.md)
├── scripts/                 # CLI utilities (backfill, benchmarks, drills)
├── tests/                   # pytest suite
└── docs/                    # Architecture + structure specs
```

---

## 4. Free vs Paid Data Sources

The architecture spec calls for Polygon.io ($29 / month for the Starter plan) as
the production source. **Before paying, prove the model can learn anything** on
the free `yfinance` data. The repository supports both transparently:

| Source     | Access                                                 | Granularity                  | Use                           |
| ---------- | ------------------------------------------------------ | ---------------------------- | ----------------------------- |
| yfinance   | `backend/data_engine/collectors/yfinance_collector.py` | Daily (10y), 1-min (last 7d) | Phase 0–2 prototyping         |
| Polygon.io | `backend/data_engine/collectors/price_stream.py`       | 1-min (10y), tick stream     | Phase 3+ production           |
| NewsAPI    | `backend/data_engine/collectors/news_stream.py`        | 100 req/day free tier        | News ingestion (Phase 1+)     |
| SEC EDGAR  | `backend/data_engine/collectors/chain_scrapers.py`     | 10-K filings (free)          | Supply-chain graph (Phase 1+) |

**Recommended path**:

1. Run `notebooks/01_data_exploration_yfinance.py` to validate the data
   pipeline.
2. Run `notebooks/02_tft_prototype.py` to confirm the TFT can learn.
3. Backfill via:

    ```bash
    python -m scripts.backfill_historical --source yfinance --start 2018-01-01 --end 2024-12-31
    ```

4. Subscribe to Polygon **only** after the parity check (notebook 05) passes.

---

## 5. Quickstart

```bash
# 0. Clone & enter
git clone https://github.com/franjavi-upct-es/lumina_project && cd lumina_project

# 1. Configure
cp .env.example .env     # then edit POLYGON_API_KEY etc. (or leave empty for yfinance)
make install             # pip install -e ".[dev]"

# 2. Spin up Redis + TimescaleDB
make up
make migrate             # apply alembic 001_phase1_schemas.py

# 3. Sanity-check the pipeline
python -m scripts.backfill_historical --source yfinance --start 2023-01-01 --end 2024-01-01

# 4. Run the unit tests
make test

# 5. Open the notebooks
code notebooks/

# 6. Start the API + dashboard
make dev                         # uvicorn on :8000
cd frontend && npm install && npm run dev   # Vite on :5173
```

---

## 6. Implementation Status

All modules listed in the dimensional contract and architecture spec are
implemented. Lint (`ruff`), formatting (`ruff format`), and types (`mypy`,
`tsc`) are clean across the full codebase. The test suite contains 91 passing
unit tests covering the most critical modules.

### 6.1 Module map

| Module                                                                                                   | Notes                                                  |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `config/{constants,settings,logging}`                                                                    | Dimensional contract enforced here                     |
| `data_engine/storage/{timescale,redis_cache}`                                                            | Async, batched, health-checked                         |
| `data_engine/pipelines/{cleaning,ingestion,metrics}`                                                     | DLQ, backpressure, Prometheus metrics                  |
| `data_engine/collectors/*` (Polygon, yfinance, NewsAPI, EDGAR)                                           | yfinance is the free prototyping path                  |
| `feature_store/*`                                                                                        | Hot/cold unified client; news + OHLCV cold features    |
| `perception/temporal/{tft_model,preprocessor,dataset}`                                                   | VSN + GRN + causal attention + MC-Dropout              |
| `perception/semantic/{distilled_llm,tokenizer}`                                                          | 4-layer Transformer student + cached FinBERT tokenizer |
| `perception/structural/{gat_model,graph_builder}`                                                        | 3-layer GATv2 + correlation/supply-chain graph         |
| `fusion/{cross_attention,nexus,parity_check}`                                                            | 224-d concat, Phase-3 parity gate                      |
| `cognition/policy/{networks,distributions}`                                                              | Squashed Gaussian + Beta                               |
| `cognition/agent/{policy_network,uncertainty_gate,ppo_agent}`                                            | 4-D action, MC-Dropout, GAE, KL early-stop             |
| `cognition/training/{behavioral_cloning,`<br>`domain_randomization,sharpe_optimizer,curriculum,trainer}` | Full Spartan curriculum                                |
| `simulation/environments/base_env`                                                                       | Gymnasium-compatible, 4-D action                       |
| `simulation/generators/{scenario_loader,adversarial,synthetic_data}`                                     | Six adversarial warp types                             |
| `execution/{broker,safety,sizing,orchestrator}`                                                          | Alpaca + paper broker, 7 atomic safety rules           |
| `api/*`                                                                                                  | FastAPI + WebSocket + Prometheus + portfolio route     |
| `integration/e2e_loop`                                                                                   | Latency-budgeted full reflex arc                       |
| `paper_trading/runner.py`                                                                                | Async lifecycle scaffolding + heartbeat                |
| `frontend/src/*`                                                                                         | Hash-router, dashboard, backtest, settings pages       |

### 6.2 Deferred deliverables

A few items in the architecture spec require external dependencies and are
therefore not exercised end-to-end in this repository. They are listed here so
the gap between "what is in the repo" and "what is needed for production" is
explicit:

| Deliverable                            | What is missing                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| Phase-3 parity check (V2 vs V3 Sharpe) | The V2 LSTM baseline checkpoint is in the parent project, not this repo.           |
| Phase-8 crisis-drill end-to-end test   | Requires a trained agent checkpoint at `models/agent/final.pt`.                    |
| Live social-media ingestion            | `data_engine/collectors/social_scraper.py` is a no-op until the NLP filter exists. |
| Polygon production backfill            | Requires a paid Polygon Starter API key.                                           |
| Live Alpaca paper-trading              | Requires Alpaca API credentials in `.env`.                                         |

---

## 7. Notebooks Index

| #   | File                                        | Purpose                                                | Phase |
| --- | ------------------------------------------- | ------------------------------------------------------ | ----- |
| 01  | `01_data_exploration_yfinance.ipynb`        | Free-data sanity checks; sector correlation heatmap    | 0     |
| 02  | `02_tft_prototype.ipynb`                    | TFT over-fit smoke test                                | 2     |
| 03  | `03_semantic_distillation_smoke.ipynb`      | FinBERT → distilled student smoke test                 | 2     |
| 04  | `04_graph_construction.ipynb`               | GATv2 input graph from yfinance correlations           | 2     |
| 05  | `05_fusion_parity_check.ipynb`              | Phase-3 milestone: V2 vs V3 parity                     | 3     |
| 06  | `06_uncertainty_calibration.ipynb`          | Calibrate the τ_high threshold of the Uncertainty Gate | 4     |
| 07  | `07_end_to_end_paper_trading_dry_run.ipynb` | Full-stack composition + latency budget                | 8     |

Each notebook ends with an explicit pass/fail criterion. **Do not move on to the
next phase until the relevant notebook passes on your dataset.**

---

## 8. References

The architecture is built on standard, peer-reviewed work. Key references:

- Lim et al. (2021), _Temporal Fusion Transformers for Interpretable
  Multi-Horizon Time Series Forecasting_, IJF.
- Brody, Alon, Yahav (2022), _How Attentive Are Graph Attention Networks?_,
  ICLR.
- Tsai et al. (2019), _Multimodal Transformer for Unaligned Multimodal Language
  Sequences_, ACL.
- Schulman et al. (2017), _Proximal Policy Optimization Algorithms_,
  arXiv:1707.06347.
- Schulman et al. (2016), _High-Dimensional Continuous Control Using Generalized
  Advantage Estimation_, ICLR.
- Haarnoja et al. (2018), _Soft Actor-Critic_, ICML.
- Gal & Ghahramani (2016), _Dropout as a Bayesian Approximation_, ICML.
- Moody & Saffell (2001), _Learning to Trade via Direct Reinforcement_, IEEE
  TNN.
- Engstrom et al. (2020), _Implementation Matters in Deep Policy Gradients_,
  ICLR.

---

## 9. License & Disclaimer

This is a research project. Trading carries substantial risk; this codebase
makes **no guarantees** about profitability and should not be used with capital
you cannot afford to lose. Every component has been written for instructional
and experimental purposes; production deployment requires extensive additional
validation.
