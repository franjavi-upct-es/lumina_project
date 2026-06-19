// frontend/src/hooks/usePolling.ts
import { useEffect, useRef, useState } from "react";
import axios from "axios";

export interface PollingResult<T> {
  data: T | null;
  error: Error | null;
  /** epoch ms of the last successful fetch, or null before the first one. */
  lastUpdated: number | null;
}

/**
 * Poll `fetcher` every `intervalMs`. The single polling primitive for the
 * dashboard so the lifecycle — interval, error clearing, in-flight
 * cancellation — lives in exactly one place.
 *
 * `fetcher` receives an AbortSignal; pass it to the underlying request so a
 * slow call is cancelled on unmount (and on React 18 StrictMode's double
 * mount) instead of resolving into an unmounted component. The fetcher
 * identity is read through a ref, so it is NOT an effect dependency: callers
 * may pass an inline closure without the interval tearing down every render
 * (which, since a poll triggers setState -> re-render, would otherwise spiral
 * into an unthrottled request loop).
 */
export function usePolling<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  enabled = true,
): PollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let stopped = false;

    const tick = async () => {
      try {
        const result = await fetcherRef.current(controller.signal);
        if (stopped || controller.signal.aborted) return;
        setData(result);
        setError(null);
        setLastUpdated(Date.now());
      } catch (err) {
        if (stopped || controller.signal.aborted || axios.isCancel(err)) return;
        setError(err instanceof Error ? err : new Error(String(err)));
      }
    };

    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      stopped = true;
      clearInterval(id);
      controller.abort();
    };
  }, [intervalMs, enabled]);

  return { data, error, lastUpdated };
}
