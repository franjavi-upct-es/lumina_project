// frontend/src/api/client.ts
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
// Exported so WebSocket hooks can pass it as a `?token=` query param — browsers
// cannot set custom headers on the WS handshake (audit F2).
export const API_KEY = import.meta.env.VITE_API_KEY || "";

/** Append the API key as a `token` query param for WebSocket URLs. */
export function withWsToken(url: string): string {
  if (!API_KEY) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(API_KEY)}`;
}

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: { "x-api-key": API_KEY, "Content-Type": "application/json" },
  timeout: 10_000,
});

apiClient.interceptors.response.use(
  (r) => r,
  (err) => {
    console.error("[API]", err?.response?.status, err?.response?.data);
    return Promise.reject(err);
  },
);
