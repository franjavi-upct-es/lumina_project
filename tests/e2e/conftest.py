# tests/e2e/conftest.py
"""End-to-end fixtures.

The ``full_system`` fixture is the canonical entry point for end-to-end
tests that need a complete stack (Redis + Timescale + frozen encoders
+ a trained agent). It is currently a placeholder that skips dependent
tests until the Phase-8 deployment wiring is committed; this keeps the
test suite green while we iterate on the individual components.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def full_system() -> None:
    """Skip placeholder — replaced by the real fixture in Phase 8."""
    pytest.skip("full_system fixture not yet implemented (Phase 8)")
