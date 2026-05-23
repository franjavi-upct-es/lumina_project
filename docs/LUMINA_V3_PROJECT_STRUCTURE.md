# **LUMINA V3: FINAL PROJECT STRUCTURE SPECIFICATION**

This document outlines the target directory structure for the Lumina V3
"Chimera" architecture. It modularizes the system into distinct cognitive layers
while maintaining operational robustness.

## **📂 Root Directory**

```
.
├── backend/                 # The core logic of the autonomous agent
├── docker/                  # Containerization for microservices
├── docs/                    # Documentation
├── frontend/                # Streamlit/React dashboards
├── notebooks/               # Research & Experimentation sandboxes
├── scripts/                 # DevOps and utility scripts
├── tests/                   # Pytest suite
├── .env                     # Environment variables (Secrets)
├── docker-compose.yml       # Orchestration
├── Makefile                 # Shortcuts for build/run commands
└── pyproject.toml           # Dependency management (Poetry/UV)
```

## **🧠 Backend: The Cognitive Core**

The backend is refactored to move away from generic "ml.engine" naming to
semantic architectural layers.

### **1. API & Configuration (/api, /config)**

Standard interfaces for external communication and system settings.

```
backend/
├── api/
│   ├── routes/
│   │   ├── agent.py           # Endpoints to monitor RL agent decisions live
│   │   ├── backtest.py        # Trigger simulation runs
│   │   ├── data.py            # Raw data inspection
│   │   ├── monitoring.py      # Health checks & Prometheus metrics
│   │   └── risk.py            # Manual override for Risk Gate (Kill Switch)
│   ├── deps.py                # Dependency injection (DB sessions, Redis pool)
│   └── main.py                # FastAPI entry point
├── config/
│   ├── constants.py           # Trading constants (market hours, tickers)
│   ├── logging.py             # Loguru configuration (JSON logs for Splunk/ELK)
│   └── settings.py            # Pydantic BaseSettings (loads .env)
```

### **2. Data Engine (/data_engine)**

**Role:** The raw ingestion layer. It gathers data but does not interpret it.

```
backend/data.engine/
├── collectors/
│   ├── chain.scrapers.py      # On-chain data (if crypto) or Supply Chain graph scraping
│   ├── news.stream.py         # Websocket connection to NewsAPI/Benzinga
│   ├── price.stream.py        # Websocket connection to Alpaca/Polygon.io
│   └── social.scraper.py      # Twitter/Reddit ingestion pipelines
├── pipelines/
│   ├── cleaning.py            # Outlier detection and null handling
│   └── ingestion.py           # Async ETL pipelines writing to TimescaleDB
└── storage/
    ├── timescale.py           # CRUD wrapper for historical data (Cold Store)
    └── redis.cache.py         # Wrapper for real-time deduplication
```

### **3. The Feature Store (/feature_store)**

**Role:** The "Nervous System." It decouples heavy computation from instant
decision-making using a Hot/Cold architecture.

```
backend/feature.store/
├── client.py                  # Unified client to fetch features (transparently handles Redis vs DB)
├── definitions.py             # Formal definitions of features (TTL, update frequency)
├── online.py                  # Redis adapter for sub-millisecond retrieval (Hot Store)
└── offline.py                 # TimescaleDB adapter for training batches (Cold Store)
```

### **4. Layer 1: Perception (/perception)**

**Role:** The "Encoders." Converts raw noisy data into dense embedding vectors.

```
backend/perception/
├── temporal/                  # The "Visual Cortex" (Price/Volume)
│   ├── encoder.py             # TFT (Temporal Fusion Transformer) model definition
│   ├── preprocessor.py        # Normalization and windowing for time-series
│   └── inference.py           # Real-time vector generation service
├── semantic/                  # The "Broca's Area" (Language/News)
│   ├── llm.distilled.py       # Quantized DistilRoBERTa/Llama wrapper
│   ├── tokenizer.py           # Custom tokenizer for financial lexicon
│   └── vectorizer.py          # Converts text stream to 64d context embeddings
└── structural/                # The "Spatial Senses" (Graph/Correlations)
│   ├── gnn.model.py           # Graph Attention Network (GATv2) definition
│   ├── graph.builder.py       # Constructs adjacency matrix from correlation/sectors
│   └── dynamic.edges.py       # Updates edge weights based on rolling correlation
```

### **5. Layer 2: Deep Fusion (/fusion)**

**Role:** The "Thalamus." Merges modalities and attends to the relevant signal.

```
backend/fusion/
├── attention.py               # Cross-Modal Attention implementation (Transformer block)
├── concatenation.py           # Logic to merge (Price | News | Graph) vectors
└── state.builder.py           # Orchestrator: Calls Perception layers \-\> Fuses \-\> Returns Super-State
```

### **6. Layer 3: Cognition (/cognition)**

**Role:** The "Prefrontal Cortex." The RL Agent making decisions.

```
backend/cognition/
├── agent/
│   ├── ppo.continuous.py      # Proximal Policy Optimization with continuous action space
│   ├── sac.agent.py           # Soft Actor-Critic alternative (for higher entropy exploration)
│   └── uncertainty.py         # Monte Carlo Dropout logic for Epistemic Uncertainty
├── policy/
│   ├── networks.py            # PyTorch definitions for Actor and Critic networks
│   └── distributions.py       # Beta/Gaussian distribution logic for actions
└── training/
    ├── curriculum.py          # Logic for Phase A/B/C training progression
    └── trainer.py             # Main training loop (interacts with Gym env)
```

### **7. Simulation & Adversarial (/simulation)**

**Role:** The "Dream State." Environments for training the agent.

```
backend/simulation/
├── environments/
│   ├── base.env.py            # Gymnasium interface (reset, step, render)
│   ├── live.shadow.py         # Environment that mirrors live market (for paper trading)
│   └── reward.functions.py    # Sharpe, Sortino, and Calmar ratio reward logic
└── generators/
    ├── adversarial.py         # Generates "Nightmare" scenarios (noise injection)
    ├── scenario.loader.py     # Replays specific historical crashes (2008, 2020\)
    └── synthetic.data.py      # Geometric Brownian Motion \+ Jump Diffusion generators
```

### **8. Execution & Safety (/execution)**

**Role:** The "Motor Cortex" & "Amygdala." Executes trades and enforces
survival.

```
backend/execution/
├── broker/
│   ├── alpaca.py              # Alpaca API Adapter
│   └── ibkr.py                # Interactive Brokers Adapter (future proofing)
├── logic/
│   ├── order.manager.py       # Handles limit order placement and chasing
│   └── position.sizer.py      # Kelly Criterion & Volatility scaling logic
└── safety/
    ├── risk.gate.py           # The "Uncertainty Gate" implementation (Veto logic)
    ├── circuit.breakers.py    # Hard stops (Max Drawdown, Daily Loss Limit)
    └── arbitrator.py          # Final authority: Checks Agent Action vs Risk Rules
```

## **🛠 Infrastructure & DevOps**

### **Docker Services (/docker)**

Expanded to support the microservices architecture.

```
docker/
├── Dockerfile.perception      # GPU-optimized image for TFT/BERT/GNN inference
├── Dockerfile.brain           # Lightweight image for RL Agent inference
├── Dockerfile.data            # Image for scrapers/collectors
├── Dockerfile.api             # FastAPI server
└── services/
    ├── redis.conf             # Optimized Redis config for vector storage
    └── timescale.conf         # DB tuning for time-series
```

## **📝 Key Functional Workflows**

### **1. The Inference Loop (Real-Time)**

1. **Market Tick** hits data.engine/collectors.
2. Data is pushed to **Redis** via feature.store/online.py.
3. fusion/state.builder.py pulls latest embeddings from Redis.
4. cognition/agent receives Super-State vector.
5. cognition/agent/uncertainty.py calculates Entropy.
6. execution/safety/risk.gate.py decides: **Agent** or **Safety Override**?
7. Action sent to execution/broker.

### **2. The Training Loop (Offline)**

1. simulation/generators creates a batch of "Warped" episodes.
2. feature.store/offline.py provides historical embeddings.
3. cognition/training/trainer.py runs PPO updates on GPU.
4. Weights are saved to models/ registry.

## **📌 Summary of Changes from V2**

1. **Dissolution of generic ml.engine:** Replaced by specific perception,
   fusion, and cognition modules to enforce the architecture.
2. **Introduction of feature.store:** Critical for low-latency handling of
   multimodal data.
3. **Expansion of simulation:** Now includes adversarial generators, not just
   backtesting logic.
4. **Hardening of execution:** The safety module is now a first-class citizen
   with veto power over the AI.
