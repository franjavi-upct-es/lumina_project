# backend/api/routes/risk.py
"""Risk management routes: kill switch control."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from backend.api.deps import get_kill_switch, require_api_key
from backend.api.schemas import KillSwitchRequest, KillSwitchResponse
from backend.execution.safety.kill_switch import KillSwitch, KillSwitchState

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get(
    "/kill-switch", response_model=KillSwitchResponse, dependencies=[Depends(require_api_key)]
)
async def get_kill_switch_state(ks: KillSwitch = Depends(get_kill_switch)) -> KillSwitchResponse:
    state = await ks.get_state()
    return KillSwitchResponse(state=state.value, set_at=datetime.now(UTC))


@router.post(
    "/kill-switch", response_model=KillSwitchResponse, dependencies=[Depends(require_api_key)]
)
async def set_kill_switch(
    req: KillSwitchRequest, ks: KillSwitch = Depends(get_kill_switch)
) -> KillSwitchResponse:
    await ks.set_state(KillSwitchState(req.state), req.reason)
    return KillSwitchResponse(state=req.state, set_at=datetime.now(UTC))
