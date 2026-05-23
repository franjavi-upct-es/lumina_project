# backend/simulation/arena/schemas.py
"""Pydantic schemas for the Spartan Arena subsystem.

Every record crossing a module boundary (arena -> xai -> feedback -> api ->
frontend) is one of the models defined here. Do not duplicate these
shapes elsewhere in the codebase.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


class ActionKind(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"


class ArenaRunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class VSNWeight(BaseModel):
    """One entry of the Temporal Fusion Transformer's Variable Selection Network."""

    feature: str = Field(..., description="Feature name, e.g. 'rsi_14', 'volume', 'close'.")
    weight: float = Field(..., ge=0.0, le=1.0)
    state: str = Field(..., description="Human label, e.g. 'overbought', 'neutral'.")


class GATEdgeCoefficient(BaseModel):
    """One edge of the GATv2 graph at this decision step."""

    source_ticker: str
    target_ticker: str
    coefficient: float = Field(..., ge=0.0, le=1.0)


class CrossModalWeights(BaseModel):
    """Weights assigned by the Deep Fusion Nexus to each modality.

    The three weights must sum to 1.0 (the attention block applies softmax).
    """

    price: float = Field(..., ge=0.0, le=1.0)
    news: float = Field(..., ge=0.0, le=1.0)
    graph: float = Field(..., ge=0.0, le=1.0)


class AttributionPayload(BaseModel):
    """All attribution signals extracted at a single decision step."""

    cross_modal: CrossModalWeights
    tft_vsn_top: list[VSNWeight] = Field(default_factory=list, max_length=5)
    gat_edges_top: list[GATEdgeCoefficient] = Field(default_factory=list, max_length=5)
    # LLM token importance is optional because Integrated Gradients is expensive
    # and may be disabled per-step. None = "not computed for this step".
    llm_top_tokens: list[tuple[str, float]] | None = None


class DecisionRecord(BaseModel):
    """One decision taken by the agent in one trajectory at one timestep.

    Persisted to TimescaleDB table `arena_decision_records`. The 224-d
    super-state vector is stored out-of-band in the artifact directory,
    not in the database row; only its path is kept here.
    """

    model_config = ConfigDict(frozen=True)

    record_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    trajectory_id: int = Field(..., ge=0)
    step_index: int = Field(..., ge=0)
    sim_timestamp: datetime
    wall_timestamp: datetime

    ticker: str
    ohlcv: dict[str, float]

    action_kind: ActionKind
    action_vector: list[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)

    realized_reward: float | None = None
    state_artifact_path: str

    attribution: AttributionPayload
    mc_seed: int


class DivergencePoint(BaseModel):
    """A step where trajectories diverged AND subsequent outcomes differed."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    step_index: int
    sim_timestamp: datetime

    best_trajectory_id: int
    worst_trajectory_id: int
    best_action_vector: list[float] = Field(..., min_length=4, max_length=4)
    worst_action_vector: list[float] = Field(..., min_length=4, max_length=4)

    action_l2_distance: NonNegativeFloat

    best_subsequent_sharpe: float
    worst_subsequent_sharpe: float
    sharpe_delta: float


class CounterfactualPair(BaseModel):
    """A training-grade triplet derived from a DivergencePoint.

    Written to disk as JSONL and consumed by feedback/replay_buffer_writer.py
    to build BC datasets for the next training run.
    """

    model_config = ConfigDict(frozen=True)

    pair_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    divergence_step_index: int
    sim_timestamp: datetime

    state_artifact_path: str
    good_action_vector: list[float] = Field(..., min_length=4, max_length=4)
    bad_action_vector: list[float] = Field(..., min_length=4, max_length=4)
    good_outcome_sharpe: float
    bad_outcome_sharpe: float

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident we are that this pair represents a real pivot, "
            "derived from sharpe_delta and action_l2_distance."
        ),
    )


class ArenaRunMetadata(BaseModel):
    """The control record for an entire arena run."""

    run_id: UUID = Field(default_factory=uuid4)
    status: ArenaRunStatus = ArenaRunStatus.PENDING
    ticker: str
    start_date: datetime
    end_date: datetime
    n_trajectories: int = Field(..., ge=3, le=16)
    mc_seeds: list[int] = Field(..., min_length=3, max_length=16)
    playback_multiplier: float = Field(default=1.0, ge=1.0, le=1000.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    failure_reason: str | None = None


class StepExplanation(BaseModel):
    """A rendered terminal-style explanation of a single DecisionRecord.

    Produced by xai/step_explainer.py. The `text` field is the formatted
    multi-line block; `tags` is a structured index for search/filter.
    """

    record_id: UUID
    text: str
    tags: list[
        Literal[
            "overbought",
            "oversold",
            "panic",
            "contagion",
            "low_confidence",
            "kill_switch_near",
        ]
    ]


class RunSummary(BaseModel):
    """End-of-run narrative produced by xai/run_summarizer.py."""

    run_id: UUID
    narrative: str
    best_trajectory_id: int
    worst_trajectory_id: int
    n_divergences: int
    n_pivotal_divergences: int
    summary_method: Literal["template", "slm"]
