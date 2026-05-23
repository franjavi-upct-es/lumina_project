// frontend/src/types/arena.types.ts
//
// TypeScript counterparts of the Pydantic schemas in
// backend/simulation/arena/schemas.py. Field names stay snake_case to
// match the JSON returned by FastAPI verbatim — no client-side casing
// conversion. Keep this file in lockstep with the backend schemas.

export type ActionKind = "BUY" | "SELL" | "HOLD" | "REDUCE" | "INCREASE";

export type ArenaRunStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export type StepExplanationTag =
  | "overbought"
  | "oversold"
  | "panic"
  | "contagion"
  | "low_confidence"
  | "kill_switch_near";

export interface VSNWeight {
  feature: string;
  weight: number;
  state: string;
}

export interface GATEdgeCoefficient {
  source_ticker: string;
  target_ticker: string;
  coefficient: number;
}

export interface CrossModalWeights {
  price: number;
  news: number;
  graph: number;
}

export interface AttributionPayload {
  cross_modal: CrossModalWeights;
  tft_vsn_top: VSNWeight[];
  gat_edges_top: GATEdgeCoefficient[];
  llm_top_tokens: Array<[string, number]> | null;
}

export interface DecisionRecord {
  record_id: string;
  run_id: string;
  trajectory_id: number;
  step_index: number;
  sim_timestamp: string;
  wall_timestamp: string;
  ticker: string;
  ohlcv: Record<string, number>;
  action_kind: ActionKind;
  action_vector: number[];
  confidence: number;
  uncertainty: number;
  realized_reward: number | null;
  state_artifact_path: string;
  attribution: AttributionPayload;
  mc_seed: number;
}

export interface DivergencePoint {
  run_id: string;
  step_index: number;
  sim_timestamp: string;
  best_trajectory_id: number;
  worst_trajectory_id: number;
  best_action_vector: number[];
  worst_action_vector: number[];
  action_l2_distance: number;
  best_subsequent_sharpe: number;
  worst_subsequent_sharpe: number;
  sharpe_delta: number;
}

export interface CounterfactualPair {
  pair_id: string;
  run_id: string;
  divergence_step_index: number;
  sim_timestamp: string;
  state_artifact_path: string;
  good_action_vector: number[];
  bad_action_vector: number[];
  good_outcome_sharpe: number;
  bad_outcome_sharpe: number;
  confidence_score: number;
}

export interface ArenaRunMetadata {
  run_id: string;
  status: ArenaRunStatus;
  ticker: string;
  start_date: string;
  end_date: string;
  n_trajectories: number;
  mc_seeds: number[];
  playback_multiplier: number;
  created_at: string;
  completed_at: string | null;
  failure_reason: string | null;
}

export interface StepExplanation {
  record_id: string;
  text: string;
  tags: StepExplanationTag[];
}

export interface RunSummary {
  run_id: string;
  narrative: string;
  best_trajectory_id: number;
  worst_trajectory_id: number;
  n_divergences: number;
  n_pivotal_divergences: number;
  summary_method: "template" | "slm";
}

export interface ArenaRunRequest {
  ticker: string;
  start_date: string;
  end_date: string;
  n_trajectories?: number;
  mc_seeds?: number[] | null;
  playback_multiplier?: number;
}

export interface ArenaRunCreatedResponse {
  run_id: string;
  status: ArenaRunStatus;
}
