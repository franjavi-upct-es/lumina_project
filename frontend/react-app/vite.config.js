// frontend/react-app/vite.config.js
// Vite build configuration for Lumina V3 React Dashboard.
// Configures the dev server proxy to forward API and WebSocket requests
// to the FastAPI backend, avoiding CORS issues in development.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/ws": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
      "/health": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts", "d3"],
          query: ["@tanstack/react-query"],
          zustand: ["zustand"],
        },
      },
    },
  },
  resolve: {
    alias: { "@": "/src" },
  },
});
