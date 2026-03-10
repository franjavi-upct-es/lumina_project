// frontend/react-app/src/services/websocket.js
// WebSocket manager for real-time data channels.
// Provides automatic reconnection, typed message routing, and
// clean teardown - replacing Streamlit's polling model with push-based updates.
// Used by: Agent Monitor (live decisions), price tick overlay, safety alerts.

const DEFAULT_RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_ATTEMPTS = 10;

/*
 * Creates a managed WebSocket connection to a single channel.
 * Returns a cleanup function that the caller must invoke on unmount.
 *
 * @param {string}    url         - WebSocket endpoint (e.g. "/ws/agent")
 * @param {Function}  onMessage   - Called with the parsed JSON payload on each message
 * @param {Function}  [onStatus]  - Optional callback for connection status changes
 * @returns {Function}            - Teardown function - call this to close the socket
 *
 * @example
 * const cleanup = createWebSocket("/ws/agent", (msg) => {
 *  setDecisions((prev) => [msg, ...prev].slice(0, 100));
 * });
 * // In useEffect cleanup:
 * return cleanup;
 */

export function createWebSocket(url, onMessage, onStatus) {
  const baseWsUrl = (
    import.meta.env.VITE_WS_URL || "ws://localhost:8000"
  ).replace(/\/$/, "");
  const fullUrl = `${baseWsUrl}${url}`;

  let ws = null;
  let reconnectAttempts = 0;
  let destroyed = false;
  let reconnectTimer = null;

  function connect() {
    if (destroyed) return;

    try {
      ws = new WebSocket(fullUrl);
      onStatus?.("connecting");

      ws.onopen = () => {
        reconnectAttempts = 0;
        onStatus?.("connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch {
          // Non-JSON frames (e.g. ping) are silently ignored
        }
      };

      ws.onerror = () => {
        onStatus?.("error");
      };

      ws.onclose = () => {
        onStatus?.("disconnected");
        if (!destroyed && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          const delay =
            DEFAULT_RECONNECT_DELAY_MS * Math.pow(1.5, reconnectAttempts);
          reconnectAttempts++;
          reconnectTimer = setTimeout(connect, delay);
        }
      };
    } catch (err) {
      onStatus?.("error");
    }
  }

  connect();

  // Return teardown function
  return function cleanup() {
    destroyed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (ws) {
      ws.onclose = null; // Prevent the reconnect handler from firing
      ws.close();
    }
  };
}
