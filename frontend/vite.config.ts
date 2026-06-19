// frontend/vite.config.ts
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Proxy the backend so the dev app is same-origin: no CORS, and the API key no
// longer has to round-trip cross-origin. The two backends mount points are
// `/api/*` (most routes) and `/arena/*` (the arena router). `ws: true` upgrades
// the agent and arena WebSocket handshakes through the same proxy.
//
// Override the target with VITE_BACKEND_ORIGIN when the backend isn't on
// localhost:8000. (VITE_API_BASE/VITE_WS_BASE still bypass the proxy entirely
// by pointing the client straight at a remote backend.)
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_BACKEND_ORIGIN || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": { target, changeOrigin: true, ws: true },
        "/arena": { target, changeOrigin: true, ws: true },
      },
    },
  };
});
