# backend/perception/common/visualization.py
"""t-SNE visualization — the Phase 2 milestone."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_market_states(
    embeddings: np.ndarray,
    regime_labels: list[str],
    output_path: Path = Path("reports/market_states_tsne.png"),
    perplexity: int = 30,
) -> None:
    """Project TFT embeddings to 2D. Expect Crash/Bull/Sideways to separate."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    proj = tsne.fit_transform(embeddings)
    colors = {"crash": "red", "bull": "green", "sideways": "gray", "volatile": "orange"}
    _fig, ax = plt.subplots(figsize=(12, 10))
    for regime in set(regime_labels):
        mask = np.array([r == regime for r in regime_labels])
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=colors.get(regime, "blue"),
            label=regime,
            alpha=0.6,
            s=20,
        )
    ax.set_title("TFT Market State Embeddings (t-SNE)")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def label_regime_by_return(returns_window: np.ndarray, volatility: float) -> str:
    mean_ret = returns_window.mean()
    if abs(mean_ret) < 0.001 and volatility < 0.01:
        return "sideways"
    if mean_ret > 0.002:
        return "bull"
    if mean_ret < -0.002:
        return "crash"
    return "volatile"
