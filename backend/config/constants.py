# backend/config/constants.py
"""
V3 Architecture Constants
=========================

Defines all critical constants for the Lumina V3 Deep Fusion Architecture.
These values are based on the architectural specifications in:
- Lumina_V3_Deep_Fusion_Architecture.md
- LUMINA_V3_PROJECT_ARQUITECTURE.md

Critical Constants:
- Embedding dimensions for each perception encoder
- Redis TTL values for hot storage
- Safety thresholds for risk management
- Training hyperparameters

Author: Lumina Quant Lab
Version: 3.0.0
"""

from enum import Enum

# ============================================================================
# PERCEPTION LAYER - EMBEDDING DIMENSIONS
# ============================================================================


class EmbeddingDimensions:
    """
    Fixed embedding dimensions for V3 perception encoders

    These dimensions are architectural constants and should NOT be changed
    without retraining all models.
    """

    # Individual encoder outputs
    TEMPORAL = 128  # Temporal Fusion Transformer (TFT) output
    SEMANTIC = 64  # Distilled LLM semantic embedding
    STRUCTURAL = 32  # Graph Neural Network (GNN) output

    # Fused super-state (concatenation of all encoders)
    SUPER_STATE = TEMPORAL + SEMANTIC + STRUCTURAL  # 224 dimensions

    # Action space (continuous)
    ACTION = 4  # [direction, urgency, sizing, stop_distance]


# ============================================================================
# REDIS TTL - HOT STORAGE CACHE LIFETIMES
# ============================================================================


class RedisTTL:
    """
    Time-to-live values for Redis hot storage (in seconds)

    Optimized for the trade-off between freshness and computational cost.
    """

    # Embedding cache lifetimes
    PRICE_EMBEDDING = 120  # 2 minutes (TFT updates frequently)
    NEWS_EMBEDDING = 3600  # 1 hour (news changes less frequently)
    GRAPH_EMBEDDING = 7200  # 2 hours (correlations update hourly)
    FUSED_EMBEDDING = 60  # 1 minute (super-state for immediate inference)

    # Raw feature cache
    LATEST_FEATURES = 300  # 5 minutes
    LATEST_PRICE = 60  # 1 minute

    # Metadata cache
    TICKER_INFO = 86400  # 24 hours
    MARKET_STATUS = 300  # 5 minutes


# ============================================================================
# SAFETY LAYER - RISK THRESHOLDS
# ============================================================================


class SafetyThresholds:
    """
    Hard-coded safety limits (non-negotiable)

    These are the "Amygdala" - the risk manager that can override the AI.
    """

    # Uncertainty thresholds
    UNCERTAINTY_CRITICAL = 0.8  # Entropy threshold to veto AI decisions
    UNCERTAINTY_WARNING = 0.6  # Warning level for high uncertainty

    # Position limits
    MAX_POSITION_SIZE = 0.10  # 10% of portfolio per position
    MAX_PORTFOLIO_LEVERAGE = 1.0  # No leverage (1.0 = 100% of capital)
    MAX_DAILY_TRADES = 50  # Circuit breaker for overtrading

    # Loss limits (automatic liquidation)
    MAX_DRAWDOWN_LIMIT = 0.10  # 10% max drawdown (hard stop)
    DAILY_LOSS_LIMIT = 0.03  # 3% daily loss limit
    POSITION_LOSS_LIMIT = 0.05  # 5% stop-loss per position

    # Profit taking
    PROFIT_TARGET = 0.20  # 20% profit → liquidate (V3 rule)
    TRAILING_STOP_ACTIVATION = 0.10  # Activate trailing stop at +10%
    TRAILING_STOP_DISTANCE = 0.05  # 5% trailing stop

    # Kill switch
    KILL_SWITCH_ACTIVE = False  # Emergency stop all trading
    CIRCUIT_BREAKER_COOLDOWN = 3600  # 1 hour cooldown after circuit breaker


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================


class TrainingConfig:
    """
    Default hyperparameters for V3 model training
    """

    # PPO (Proximal Policy Optimization)
    PPO_LEARNING_RATE = 3e-4
    PPO_BATCH_SIZE = 256
    PPO_EPOCHS = 10
    PPO_CLIP_RANGE = 0.2
    PPO_GAM = 0.99
    PPO_GAE_LAMBDA = 0.95

    # SAC (Soft Actor-Critic)
    SAC_LEARNING_RATE = 3e-4
    SAC_BATCH_SIZE = 256
    SAC_TAU = 0.005  # Soft update coefficient
    SAC_ALPHA = 0.2  # Entropy regularization

    # General training
    MAX_EPISODE_LENGTH = 1440  # 1 day (1440 minutes)
    REPLAY_BUFFER_SIZE = 100000
    TRAIN_FREQUENCY = 4  # Train every 4 steps

    # Curriculum learning phases
    PHASE_A_EPISODES = 10000  # Behavioral cloning
    PHASE_B_EPISODES = 50000  # Domain randomization
    PHASE_C_EPISODES = 20000  # Pure RL fine-tuning


# ============================================================================
# DATA COLLECTION
# ============================================================================


class DataConfig:
    """
    Data collection and storage configuration
    """

    # Collection frequencies
    PRICE_UPDATE_INTERVAL = 60  # Seconds (1-minute bars)
    NEWS_UPDATE_INTERVAL = 300  # 5 minutes
    SENTIMENT_UPDATE_INTERVAL = 600  # 10 minutes
    CORRELATION_UPDATE_INTERVAL = 3600  # 1 hour

    # Historical data
    MIN_HISTORY_DAYS = 365  # Minimum 1 year of data
    RECOMMENDED_HISTORY_DAYS = 1825  # 5 years recommended
    MAX_HISTORY_DAYS = 3650  # 10 years maximum

    # Feature engineering
    MAX_FEATURES = 200  # Maximum features per ticker
    FEATURE_LOOKBACK_PERIODS = [5, 10, 14, 20, 50, 200]  # Common periods

    # Data quality
    MAX_MISSING_RATIO = 0.05  # 5% max missing data
    OUTLIER_STD_THRESHOLD = 5.0  # Z-score threshold


# ============================================================================
# SIMULATION & BACKTESTING
# ============================================================================


class SimulationConfig:
    """
    Simulation environment configuration
    """

    # Initial conditions
    INITIAL_CAPITAL = 100000.0  # $100K starting capital
    RISK_CAPITAL_RATIO = 1.0  # Use 100% for trading

    # Transaction costs
    COMMISSION_RATE = 0.001  # 0.1% commission
    SLIPPAGE_RATE = 0.0005  # 0.05% slippage
    SPREAD_BPS = 1.0  # 1 basis point spread

    # Execution
    MIN_ORDER_SIZE = 1  # Minimum 1 share
    MAX_ORDER_SIZE = 10000  # Maximum 10K shares per order
    ORDER_FILL_DELAY = 0  # Assume instant fill (conservative)

    # Adversarial scenarios (Phase B training)
    VOLATILITY_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0]
    SPREAD_MULTIPLIERS = [1.0, 2.0, 5.0, 10.0]
    CRASH_SCENARIOS = [-0.05, -0.10, -0.20, -0.30]  # Synthetic crashes


# ============================================================================
# MARKET REGIMES
# ============================================================================


class MarketRegime(str, Enum):
    """
    Market regime classifications for context awareness
    """

    BULL_LOW_VOL = "bull_low_vol"  # Goldilocks
    BULL_HIGH_VOL = "bull_high_vol"  # Melt-up
    BEAR_LOW_VOL = "bear_low_vol"  # Slow bleed
    BEAR_HIGH_VOL = "bear_high_vol"  # Panic
    SIDEWAYS = "sideways"  # Range-bound
    CRASH = "crash"  # Black swan
    RECOVERY = "recovery"  # Post-crash bounce


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================


class PerformanceMetrics:
    """
    Target performance metrics for V3
    """

    # Returns
    TARGET_ANNUAL_RETURN = 0.15  # 15% annual return
    MIN_ACCEPTABLE_RETURN = 0.08  # 8% minimum

    # Risk
    TARGET_SHARPE_RATIO = 1.5  # 1.5 Sharpe
    MAX_ACCEPTABLE_DRAWDOWN = 0.15  # 15% max drawdown
    TARGET_WIN_RATE = 0.55  # 55% win rate

    # Robustness
    MIN_SORTINO_RATIO = 2.0  # Downside-focused
    MAX_CALMAR_RATIO = 1.0  # Return / Max Drawdown


# ============================================================================
# LOGGING & MONITORING
# ============================================================================


class MonitoringConfig:
    """
    Logging and monitoring configuration
    """

    LOG_LEVEL_PRODUCTION = "INFO"
    LOG_LEVEL_DEVELOPMENT = "DEBUG"
    LOG_RETENTION_DAYS = 30

    # Metrics collection
    METRICS_COLLECTION_INTERVAL = 60  # 1 minute
    PROMETHEUS_PORT = 9090

    # Alerts
    ALERT_ON_DRAWDOWN = 0.05  # Alert at 5% drawdown
    ALERT_ON_UNCERTAINTY = 0.7  # Alert at high uncertainty


# ============================================================================
# API & WEB
# ============================================================================


class APIConfig:
    """
    API and web interface configuration
    """

    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000

    # Timeouts
    REQUEST_TIMEOUT = 30  # 30 seconds
    STREAMING_TIMEOUT = 300  # 5 minutes for streams

    # Pagination
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 1000


# ============================================================================
# EXPORT ALL CONSTANTS
# ============================================================================

__all__ = [
    "EmbeddingDimensions",
    "RedisTTL",
    "SafetyThresholds",
    "TrainingConfig",
    "DataConfig",
    "SimulationConfig",
    "MarketRegime",
    "PerformanceMetrics",
    "MonitoringConfig",
    "APIConfig",
]
