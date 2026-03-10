// frontend/react-app/src/services/api.js
// Axios HTTP client for the Lumina V3 FastAPI backend.
// All requests route through this singleton for consistent error handling.

import axios from "axios";
import API_V3 from "../constants";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

// Normalize error messages from FastAPI's detail format
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      "Unknown API error";
    return Promise.reject(new Error(msg));
  },
);

// --- Health ---
export const fetchHealth = () => api.get("/health");

// --- V3: Data (Data Explorer) ---
export const fetchPrices = (ticker, params) =>
  api.get(`${API_V3}/data/${ticker}/prices`, { params });
export const fetchFeatures = (ticker, params) =>
  api.get(`${API_V3}/data/${ticker}/features`, { params });
export const fetchCompanyInfo = (ticker) =>
  api.get(`${API_V3}/data/${ticker}/info`);

// --- V3: ML (Model Comparator) ---
export const fetchModels = (params) =>
  api.get(`${API_V3}/ml/models`, { params });
export const submitTraining = (body) => api.post(`${API_V3}/ml/train`, body);
export const fetchTrainingJob = (jobId) =>
  api.get(`${API_V3}/ml/jobs/${jobId}`);

// --- V3: Backtest (Strategy Lab) ---
export const submitBacktest = (body) =>
  api.post(`${API_V3}/backtest/run`, body);
export const fetchBacktestJob = (jobId) =>
  api.get(`${API_V3}/backtest/jobs/${jobId}`);

// --- V3: Risk (Risk Dashboard) ---
export const fetchVar = (body) => api.post(`${API_V3}/risk/var`, body);
export const fetchStressTest = (body) =>
  api.post(`${API_V3}/risk/stress-test`, body);

// --- V3: Sentiment (Sentiment Monitor) ---
export const fetchSentiment = (ticker, params) =>
  api.get(`${API_V3}/sentiment/${ticker}`, { params });
export const fetchSentimentAgg = (ticker, parmas) =>
  api.get(`${API_V3}/sentiment/${ticker}/aggregate`, { parmas });

// --- V3: Agent Monitor ---
export const fetchAgentStatus = () => api.get(`${API_V3}/agent/status`);
export const fetchAgentDecisions = (params) =>
  api.get(`${API_V3}/agent/decisions`, { params });
export const fetchUncertainty = (ticker) =>
  api.get(`${API_V3}/agent/uncertainty/${ticker}`);

// --- V3: Safety System ---
export const fetchSafetyStatus = () => api.get(`${API_V3}/safety/status`);
export const fetchCircuitBreakers = () =>
  api.get(`${API_V3}/safety/circuit-breakers`);
export const triggerKillSwitch = () => api.post(`${API_V3}/safety/kill-switch`);

// --- V3: Embeddings (Embedding Visualizer) ---
export const fetchEmbeddings = (ticker, params) =>
  api.get(`${API_V3}/embeddings/${ticker}`, { params });
export const fetchTSNEProjection = (params) =>
  api.get(`${API_V3}/embeddings/tsne`, { params });

export default api;
