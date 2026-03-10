// frontend/react-app/src/store/usePortfolioStore.js
// Zustand store for portfolio-level UI state shared across pages.
// Handles the global ticker selection and quick-analysis shortcut
// that mirrors the Streamlit sidebar's "Quick Analysis" feature.

import { create } from "zustand";

const usePortfolioStore = create((set) => ({
  // -- Global ticker selection ---
  selectedTicker: "APPL",
  setSelectedTicker: (ticker) =>
    set({ selectedTicker: ticker.toUpperCase().trim() }),

  // --- Date range preferences ---
  dateRange: "3 Months",
  setDateRange: (range) => set({ dateRange: range }),

  // --- API health status (polled on sidebar) ---
  apiOnline: null, // null = unknown, true = online, false = offline
  setApiOnline: (status) => set({ apiOnline: status }),

  // --- Sidebar collapse state ---
  sidebarCollapsed: false,
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}));

export default usePortfolioStore;
