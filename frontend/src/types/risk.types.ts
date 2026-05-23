// frontend/src/types/risk.types.ts
export type KillSwitchState = "NORMAL" | "CLOSE_ONLY" | "LIQUIDATE_ALL";

export interface KillSwitchResponse {
  state: KillSwitchState;
  set_at: string;
}
