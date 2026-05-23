# backend/execution/sizing/kelly.py
"""Fractional Kelly position sizing."""

from __future__ import annotations

import math


def kelly_fraction(
    win_prob: float, avg_win: float, avg_loss: float, fraction: float = 0.25
) -> float:
    """Fractional Kelly. 25% of full Kelly by default for safety."""
    if win_prob <= 0 or win_prob >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    full = win_prob - (1 - win_prob) / b
    full = max(0.0, min(full, 1.0))
    return full * fraction


def size_from_confidence(
    confidence: float,
    max_size: float = 1.0,
    uncertainty: float = 0.0,
    kelly_fraction_pct: float = 0.25,
) -> float:
    """Map a [-1, 1] confidence scalar to position size, dampened by uncertainty."""
    direction = math.copysign(1.0, confidence) if confidence != 0 else 0.0
    magnitude = abs(confidence) * (1.0 - min(uncertainty, 1.0)) * kelly_fraction_pct
    return direction * min(magnitude, max_size)
