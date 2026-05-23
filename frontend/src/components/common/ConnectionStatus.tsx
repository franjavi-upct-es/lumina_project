// frontend/src/components/common/ConnectionStatus.tsx
export function ConnectionStatus({ connected }: { connected: boolean }) {
  return <span style={{ color: connected ? "green" : "red" }}>● {connected ? "Live" : "Disconnected"}</span>;
}
