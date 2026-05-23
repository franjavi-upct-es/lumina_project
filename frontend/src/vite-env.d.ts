// frontend/src/vite-env.d.ts
//
// TypeScript ambient declarations for Vite's import.meta.env injection.
//
// Vite replaces references to `import.meta.env.VITE_*` at build time
// with the literal values from the environment (or `.env` files). To
// stay strict-mode-friendly we declare the exact shape of the env
// object here so the compiler validates every access.

/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the backend REST API. Defaults to http://localhost:8000. */
  readonly VITE_API_BASE?: string;
  /** Base URL of the backend WebSocket. Defaults to ws://localhost:8000. */
  readonly VITE_WS_BASE?: string;
  /** Optional x-api-key header value; sent on every REST call. */
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
