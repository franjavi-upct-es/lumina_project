# backend/config/constants.py
"""Immutable project-wide constants — single source of truth.

This module exists to prevent dimension-mismatch bugs that are hard to debug
in deep neural pipelines. EVERY component that needs to know an embedding
dimension or a window length MUST import it from here.

Why this matters
----------------
The "Chimera" architecture in Lumina_V3_Deep_Fusion_Architecture.md fixes the
following dimensions explicitly:

    DIM_PRICE    = 128   # output of the Temporal Fusion Transformer (TFT)
    DIM_SEMANTIC = 64    # output of the distilled financial LLM
    DIM_GRAPH    = 32    # output of the GATv2 graph encoder
    -------------------
    DIM_FUSED    = 224   # = 128 + 64 + 32, the "Super-Vector"

The Deep Fusion Nexus performs RAW CONCATENATION of these three vectors
(NOT a projection to a shared dimension). This preserves the native
information capacity of each modality before the Cross-Modal Attention block
re-weights them. See section 4 of the architecture document.

Action space
------------
The cognitive agent operates in a 4-dimensional CONTINUOUS action space
(see section 5 of the architecture spec):

    action[0] = direction      ∈ [-1, 1]  (full short ↔ full long)
    action[1] = urgency        ∈ [-1, 1]  (limit/maker ↔ market/taker)
    action[2] = sizing         ∈ [-1, 1]  (fraction of risk capital, |·|)
    action[3] = stop_distance  ∈ [-1, 1]  (tight ↔ wide stop, in ATR multiples)

The agent emits raw values; downstream code (`backend.execution.logic`) is
responsible for mapping these to broker-level instructions.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Embedding dimensions (DO NOT CHANGE without retraining all encoders)
# ----------------------------------------------------------------------
DIM_PRICE: int = 128
"""Dimension of the temporal embedding produced by the TFT."""

DIM_SEMANTIC: int = 64
"""Dimension of the semantic embedding produced by the distilled LLM."""

DIM_GRAPH: int = 32
"""Dimension of the structural embedding produced by the GATv2."""

DIM_FUSED: int = DIM_PRICE + DIM_SEMANTIC + DIM_GRAPH  # 224
"""Total dimension after raw concatenation, before the fusion attention block."""

# ----------------------------------------------------------------------
# Cognitive layer
# ----------------------------------------------------------------------
ACTION_DIM: int = 4
"""Continuous action vector dimensionality.

Index convention:
    0 → direction       (short/long)
    1 → urgency         (limit ↔ market)
    2 → sizing          (Kelly-fraction multiplier)
    3 → stop_distance   (ATR multiplier)
"""

NEXUS_OUTPUT_DIM: int = 256
"""Output dimension of the Deep Fusion Nexus (the latent state given to the agent).

This is INDEPENDENT of DIM_FUSED. The Nexus projects the 224-d concatenated
super-vector through a Cross-Modal Attention block and emits a 256-d latent
state, similar in spirit to a Transformer hidden size.
"""

# ----------------------------------------------------------------------
# Temporal windowing
# ----------------------------------------------------------------------
OHLCV_WINDOW_MINUTES: int = 240
"""Length of the look-back window fed to the TFT, in 1-minute bars (= 4 hours)."""

OHLCV_HORIZON_MINUTES: int = 60
"""Forecast horizon for the self-supervised TFT pre-training task (in minutes)."""

NEWS_WINDOW_HOURS: int = 24
"""Hours of news context to aggregate for a single semantic embedding."""

GRAPH_REBUILD_HOURS: int = 24
"""Frequency at which the supply-chain + correlation graph is rebuilt."""

# ----------------------------------------------------------------------
# Universe — initial 32-ticker subset (expand to 50 in Phase 2)
# ----------------------------------------------------------------------
TARGET_TICKERS: frozenset[str] = frozenset(
    {
        # Mega-cap tech (high liquidity, high signal)
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        # Semis / mid-cap tech
        "AMD",
        "INTC",
        "AVGO",
        "CRM",
        "ORCL",
        "ADBE",
        "NFLX",
        # Financials
        "JPM",
        "BAC",
        "GS",
        "MS",
        "V",
        "MA",
        # Energy
        "XOM",
        "CVX",
        "COP",
        # Healthcare
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        # Consumer
        "WMT",
        "COST",
        "HD",
        # Indices (used as macro context features)
        "SPY",
        "QQQ",
    }
)

# ----------------------------------------------------------------------
# Sector mapping — used as static covariate for the TFT and node feature
# for the GATv2.
# ----------------------------------------------------------------------
TICKER_TO_SECTOR: dict[str, str] = {
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "tech",
    "GOOGL": "tech",
    "META": "tech",
    "AMZN": "tech",
    "TSLA": "consumer_discretionary",
    "AMD": "tech",
    "INTC": "tech",
    "AVGO": "tech",
    "CRM": "tech",
    "ORCL": "tech",
    "ADBE": "tech",
    "NFLX": "tech",
    "JPM": "financials",
    "BAC": "financials",
    "GS": "financials",
    "MS": "financials",
    "V": "financials",
    "MA": "financials",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "JNJ": "healthcare",
    "PFE": "healthcare",
    "UNH": "healthcare",
    "ABBV": "healthcare",
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    "HD": "consumer_discretionary",
    "SPY": "index",
    "QQQ": "index",
}

# Distinct sectors, alphabetised. Used to size the one-hot encoding.
SECTORS: tuple[str, ...] = (
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "financials",
    "healthcare",
    "index",
    "tech",
)
NUM_SECTORS: int = len(SECTORS)

# ----------------------------------------------------------------------
# Market hours — NYSE regular session, simplified to UTC.
# ----------------------------------------------------------------------
MARKET_OPEN_HOUR_UTC: int = 13
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR_UTC: int = 20
MARKET_CLOSE_MINUTE: int = 0

# ----------------------------------------------------------------------
# Uncertainty gate — defaults referenced from architecture spec §8.
# ----------------------------------------------------------------------
UNCERTAINTY_CRITICAL_THRESHOLD: float = 0.85
"""Above this rolling-window uncertainty value, the gate vetoes the agent.

Calibration note: uncertainty is the std-dev of N=10 Monte-Carlo Dropout
forward passes, averaged over the action vector dimensions. The threshold
0.85 is the *initial* value; it must be re-calibrated on validation data
(see notebooks/06_uncertainty_calibration.py).
"""
MC_DROPOUT_SAMPLES: int = 10
"""Number of stochastic forward passes per uncertainty estimate."""

# ----------------------------------------------------------------------
# Spartan Arena constants
# ----------------------------------------------------------------------
ARENA_MIN_TRAJECTORIES: int = 3
ARENA_MAX_TRAJECTORIES: int = 16
ARENA_DEFAULT_TRAJECTORIES: int = 8

ARENA_DIVERGENCE_HORIZON_BARS: int = 30
"""Sliding window (in bars) used by the divergence analyzer to evaluate
subsequent PnL after a decision point. Used to label "good" vs "bad" action."""

ARENA_DIVERGENCE_ACTION_THRESHOLD: float = 0.25
"""Minimum L2 distance between two trajectories' action vectors at the same
timestep to be considered a "divergence" worth analyzing."""

ARENA_PIVOTAL_SHARPE_DELTA: float = 0.30
"""Minimum Sharpe-ratio differential between the best and worst trajectories'
subsequent windows to consider a divergence "pivotal" (worth feedback)."""

ARENA_FEEDBACK_TOP_K: int = 2
ARENA_FEEDBACK_BOTTOM_K: int = 2
"""Top-K and Bottom-K trajectory ranks used by feedback/counterfactual_pairs.py."""

ARENA_STEP_HARD_CEILING_MS: float = 5000.0
"""Hard upper bound on wall-clock seconds for a single arena step. If exceeded,
log a WARNING but do not interrupt the run."""

ARENA_TIMING_WINDOW_SIZE: int = 100
"""Rolling window size (number of recent steps) used by time_controller.py
to compute average and p95 step durations."""
