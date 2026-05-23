// frontend/src/store/arenaSlice.ts
//
// Persists the active Arena run id and view-control state across page
// navigations. Kept intentionally small — heavy data (decisions, pairs)
// stays in component state.

import { create } from "zustand";

interface ArenaState {
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;
  playbackMultiplier: 1 | 10 | 100;
  setPlaybackMultiplier: (m: 1 | 10 | 100) => void;
  selectedTrajectoryId: number | null;
  setSelectedTrajectoryId: (id: number | null) => void;
}

export const useArenaStore = create<ArenaState>((set) => ({
  activeRunId: null,
  setActiveRunId: (id) => set({ activeRunId: id }),
  playbackMultiplier: 1,
  setPlaybackMultiplier: (m) => set({ playbackMultiplier: m }),
  selectedTrajectoryId: null,
  setSelectedTrajectoryId: (id) => set({ selectedTrajectoryId: id }),
}));
