// frontend/src/components/common/ConnectionStatus.tsx
export function ConnectionStatus({ connected }: { connected: boolean }) {
  return (
    <span className={`lx-pill ${connected ? "ok" : "bad"}`}>
      <span className="lx-dot" />
      {connected ? "Live" : "Disconnected"}
    </span>
  );
}
