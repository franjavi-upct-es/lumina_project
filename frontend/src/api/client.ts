// frontend/src/api/client.ts
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

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
