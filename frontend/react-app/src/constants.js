// frontend/react-app/src/constants.js
// Central constants for the Lumina V3 Deep Fushion dashboard.
// Embedding dimensions and thesholds must match backend/config/constants.py

// --- V3 Architecture Dimensions (mirros backend/config/constants.py) ---
export const EMBEDDING_DIM_TEMPORAL = 128;
export const EMBEDDING_DIM_SEMANTIC = 64;
export const EMBEDDING_DIM_STRUCTURAL = 32;
// Super-State = Temporal(128) + Semantic(64) + Structural(32) = 224
export const SUPER_STATE_DIM = 224;

// --- Risk Thresholds (mirros execution/safety/arbitrator.py) ---
export const UNCERTAINTY_CRITICAL_THRESHOLD = 0.65;
export const PROFIT_TAKE_THRESHOLD = 0.2; // Auto-liquidate at +20%
export const MAX_DAILY_LOSS = 0.03;
export const MAX_POSITION_DRAWDOWN = 0.1;

// --- Color Palette ---
export const COLORS = {
  primary: "#1f77b4",
  secondary: "#ff7f0e",
  sucess: "#2ca02c",
  danger: "#d62728",
  warning: "#ffbb00",
  info: "#17a2b8",
  purple: "#9467bd",
  pink: "#e377c2",
  gray: "#7f7f7f",
  cyan: "#28becf",
};

// Encoder-specific colors for the Embedding Visualizer
export const ENCODER_COLORS = {
  temporal: COLORS.cyan,
  semantic: COLORS.pink,
  structural: COLORS.purple,
};

// --- Safety System Status ---
export const SAFETY_STATUS = {
  ACTIVE: "active",
  GUARDED: "guarded",
  CRITICAL: "critical",
  OFFLINE: "offline",
};

// --- API Prefixes ---
export const API_V3 = "/api/v3";

// --- WebSocket Channels ---
export const WS_CHANNELS = {
  AGENT_DECISIONS: "/ws/agent",
  PRICE_TICKS: "/ws/prices",
  SAFETY_ALERTS: "/ws/safety",
};

// --- Chart Theme Defaults ---
export const CHART_GRID_COLOR = "#2a2a35";
export const CHART_AXIS_COLOR = "#505060";
export const CHART_TOOLTIP_BG = "#1a1a20";
export const CHART_TOOLTIP_BORDER = "#2a2a35";

// --- Spartan Training Phases (mirrors cognition/training/curriculum.py) ---
export const TRAINING_PHASES = {
  A: { id: "A", label: "Phase A – Behavioral Cloning", color: COLORS.info },
  B: {
    id: "B",
    label: "Phase B – Domain Randomization",
    color: COLORS.warning,
  },
  C: { id: "C", label: "Phase C – Pure RL / Self-Play", color: COLORS.success },
};
