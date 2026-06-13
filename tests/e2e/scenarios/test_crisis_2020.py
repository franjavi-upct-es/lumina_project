# tests/e2e/scenarios/test_crisis_2020.py
"""End-to-end crisis drill: survive a deterministic March-2020-style shock.

This test is the executable version of the Phase-8 acceptance gate described
in ``Lumina_V3_Deep_Fusion_Architecture.md``. It intentionally avoids Redis,
Timescale, and trained checkpoints so the gate can run in CI today while still
exercising the production contracts that matter for crisis survival:

* ``LuminaTradingEnv`` accounting and max-drawdown termination.
* ``SafetyArbitrator`` veto behavior under high uncertainty.
* The 4-D continuous action vocabulary used by the live policy.

Acceptance criteria
-------------------
    final_equity       > 0.85 * INITIAL_CAPITAL
    kill_switch_state != "LIQUIDATE_ALL"
    arbitrator_vetoes  > 10
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.config.constants import ACTION_DIM, NEXUS_OUTPUT_DIM
from backend.execution.safety.arbitrator import SafetyArbitrator, SafetyConfig, SafetyDecision
from backend.execution.safety.kill_switch import KillSwitchState
from backend.execution.safety.rules import SafetyContext
from backend.simulation.environments.base_env import EnvConfig, LuminaTradingEnv

INITIAL_CAPITAL = 100_000.0
MIN_FINAL_EQUITY = 0.85 * INITIAL_CAPITAL
MIN_ARBITRATOR_VETOES = 10


@dataclass(frozen=True)
class CrisisStep:
    step: int
    price: float
    equity: float
    position: float
    drawdown: float
    uncertainty: float
    vetoed: bool


@dataclass(frozen=True)
class CrisisRunResult:
    final_equity: float
    max_drawdown: float
    kill_switch_state: KillSwitchState
    arbitrator_vetoes: int
    truncated: bool
    trace: tuple[CrisisStep, ...]
    veto_reasons: tuple[str, ...]


@dataclass
class RecordingArbitrator(SafetyArbitrator):
    """SafetyArbitrator variant that records every veto for assertions."""

    decisions: list[SafetyDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(
            SafetyConfig(
                force_close_on_veto=True,
            )
        )

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        decision = super().evaluate(ctx)
        self.decisions.append(decision)
        return decision

    @property
    def veto_count(self) -> int:
        return sum(1 for decision in self.decisions if not decision.approved)

    @property
    def veto_reasons(self) -> tuple[str, ...]:
        return tuple(reason for decision in self.decisions for reason in decision.vetoes)


class March2020Generator:
    """Single deterministic episode shaped like the 2020 COVID crash.

    Regime map:
        0..15   calm grind higher
        16..24  cracks appear, volatility rises
        25..43  fast crash with uncertainty above the arbitrator threshold
        44..63  policy-response relief rally
        64..83  aftershock chop
    """

    def __iter__(self):
        return iter([_build_march_2020_episode()])


class CrisisPolicy:
    """Simple policy surrogate that tries to stay long through the crisis.

    This is deliberately not clever. The test is about the system safety layer:
    when uncertainty is extreme, the real arbitrator should flatten the proposed
    long exposure before the crash return is applied.
    """

    def act(self, _obs: np.ndarray) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0] = 1.0
        action[1] = -0.25
        action[2] = 0.0
        action[3] = 0.0
        return action


def test_crisis_2020_survival() -> None:
    result = _run_crisis_drill()

    assert result.final_equity > MIN_FINAL_EQUITY
    assert result.kill_switch_state != KillSwitchState.LIQUIDATE_ALL
    assert result.arbitrator_vetoes > MIN_ARBITRATOR_VETOES
    assert result.truncated
    assert any("rule_uncertainty" in reason for reason in result.veto_reasons)


def test_crisis_episode_contains_a_real_drawdown_shock() -> None:
    episode = _build_march_2020_episode()
    prices = episode["prices"]
    crash_slice = prices[25:44]

    assert prices[15] > prices[0]
    assert crash_slice[-1] / crash_slice[0] < 0.60
    assert np.count_nonzero(episode["uncertainties"] > 0.85) > MIN_ARBITRATOR_VETOES


def _run_crisis_drill() -> CrisisRunResult:
    arbitrator = RecordingArbitrator()
    env = LuminaTradingEnv(
        episode_generator=March2020Generator(),
        config=EnvConfig(
            initial_capital=INITIAL_CAPITAL,
            max_drawdown_pct=0.20,
            base_slippage_bps=1.0,
            commission_bps=0.5,
            veto_penalty=0.0,
        ),
        arbitrator=arbitrator,
    )
    policy = CrisisPolicy()
    obs, _info = env.reset(seed=2020)
    trace: list[CrisisStep] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs, _reward, terminated, truncated, info = env.step(policy.act(obs))
        trace.append(
            CrisisStep(
                step=env._t,
                price=float(info["close"]),
                equity=float(info["equity"]),
                position=float(info["position"]),
                drawdown=float(info["drawdown"]),
                uncertainty=float(info["uncertainty"]),
                vetoed=bool(info["vetoed"]),
            )
        )

    kill_switch_state = KillSwitchState.LIQUIDATE_ALL if terminated else KillSwitchState.NORMAL
    return CrisisRunResult(
        final_equity=float(env._equity),
        max_drawdown=max((step.drawdown for step in trace), default=0.0),
        kill_switch_state=kill_switch_state,
        arbitrator_vetoes=arbitrator.veto_count,
        truncated=truncated,
        trace=tuple(trace),
        veto_reasons=arbitrator.veto_reasons,
    )


def _build_march_2020_episode() -> dict[str, np.ndarray | bool]:
    prices = np.array(
        [
            *_compound_path(100.0, [0.002] * 15),
            *_compound_path(
                103.04, [-0.004, 0.003, -0.008, 0.005, -0.011, 0.002, -0.013, 0.004, -0.017]
            ),
            *_compound_path(
                99.06,
                [
                    -0.035,
                    -0.046,
                    -0.072,
                    -0.041,
                    -0.089,
                    -0.052,
                    -0.031,
                    -0.076,
                    -0.045,
                    -0.066,
                    -0.038,
                    -0.057,
                    -0.041,
                    -0.049,
                    -0.036,
                    -0.028,
                    -0.032,
                    -0.022,
                    -0.018,
                ],
            ),
            *_compound_path(
                35.48,
                [
                    0.025,
                    0.018,
                    -0.011,
                    0.033,
                    0.021,
                    0.015,
                    -0.009,
                    0.027,
                    0.019,
                    0.012,
                    0.024,
                    -0.007,
                    0.018,
                    0.016,
                    0.022,
                    -0.005,
                    0.014,
                    0.011,
                    0.017,
                    0.013,
                ],
            ),
            *_compound_path(
                50.73,
                [
                    -0.012,
                    0.009,
                    -0.018,
                    0.014,
                    0.006,
                    -0.011,
                    0.012,
                    -0.006,
                    0.010,
                    -0.014,
                    0.016,
                    0.008,
                    -0.007,
                    0.011,
                    -0.005,
                    0.009,
                    0.004,
                    -0.006,
                    0.007,
                    0.005,
                ],
            ),
        ],
        dtype=np.float32,
    )
    prices = prices[:84]
    log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    volatility = np.maximum(np.abs(log_returns), 0.004).astype(np.float32)
    uncertainties = np.concatenate(
        [
            np.full(16, 0.12, dtype=np.float32),
            np.linspace(0.35, 0.68, 9, dtype=np.float32),
            np.full(19, 0.93, dtype=np.float32),
            np.linspace(0.72, 0.44, 20, dtype=np.float32),
            np.full(20, 0.30, dtype=np.float32),
        ]
    )[: len(prices)]
    market_states = _market_states_from_path(prices, volatility, uncertainties)
    return {
        "prices": prices,
        "market_states": market_states,
        "volatility": volatility,
        "uncertainties": uncertainties,
        "synthetic": True,
    }


def _compound_path(start: float, returns: list[float]) -> list[float]:
    values: list[float] = []
    price = start
    for ret in returns:
        price *= 1.0 + ret
        values.append(price)
    return values


def _market_states_from_path(
    prices: np.ndarray,
    volatility: np.ndarray,
    uncertainties: np.ndarray,
) -> np.ndarray:
    states = np.zeros((len(prices), NEXUS_OUTPUT_DIM), dtype=np.float32)
    returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices, 1e-6)
    states[:, 0] = returns
    states[:, 1] = volatility
    states[:, 2] = uncertainties
    states[:, 3] = np.linspace(0.0, 1.0, len(prices), dtype=np.float32)
    return states
