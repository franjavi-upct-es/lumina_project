# tests/e2e/scenarios/test_crisis_2020.py
"""End-to-end milestone: agent must survive a synthetic 2020-style crisis.

This is the Phase-8 acceptance gate from
``Lumina_V3_Deep_Fusion_Architecture.md`` §9 Q4. The agent is required
to navigate a synthetic March-2020-shape episode autonomously, without
hitting the hard kill switch, while preserving the bulk of its capital.

Acceptance criteria
-------------------
    final_equity       > 0.85 * INITIAL_CAPITAL
    kill_switch_state != "LIQUIDATE_ALL"
    arbitrator_vetoes  > 10

The test is currently marked ``skip`` because the full stack — trained
encoders plus a trained agent — is not yet available end-to-end. It
will be enabled in Phase 8 once a candidate agent checkpoint exists.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Requires the full trained stack; enabled in Phase 8 QA.",
)


def test_crisis_2020_survival() -> None:
    """Run the synthetic-crisis episode through the full Chimera stack.

    The implementation is wired by the Phase-8 commit; until then the
    test is skipped.
    """
    raise NotImplementedError(
        "Phase-8 milestone — wiring is part of the QA milestone commit.",
    )
