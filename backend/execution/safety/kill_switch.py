# backend/execution/safety/kill_switch.py
"""Multi-state safety kill switch with low-latency in-process propagation.

Three collaborating objects make up the kill-switch system:

* :class:`KillSwitch` — Redis-backed *source of truth*. Read/write API
  used by the FastAPI risk route (operator-visible) and by the risk
  monitor (autonomous escalation on drawdown). State is one of
  ``NORMAL`` / ``CLOSE_ONLY`` / ``LIQUIDATE_ALL``.

* :class:`LocalKillSwitch` — a process-local ``threading.Event`` latch
  for ``LIQUIDATE_ALL``. The hot decision loop checks this on every
  iteration; the check is O(100 ns) and never touches Redis.

* :class:`LocalKillSwitchListener` — an async task that subscribes to
  the Redis pub/sub channel ``safety:kill_switch:events`` and flips the
  ``LocalKillSwitch`` latch whenever a ``LIQUIDATE_ALL`` event arrives.

The reason the latch is *only* armed for ``LIQUIDATE_ALL``
==========================================================
``CLOSE_ONLY`` is a *policy* signal (the safety arbitrator already
consults the Redis-resident value once per arbitrate call, which is in
the hot loop anyway). ``LIQUIDATE_ALL`` is an *interrupt* signal —
"stop everything right now" — so it deserves the dedicated low-latency
path. Conflating the two would make the fast path do unnecessary work.

Crash safety
============
If the listener task dies (asyncio failure, network blip, …) the
slow-path polling in :class:`backend.execution.safety.risk_monitor.RiskMonitor`
still picks up the Redis-side truth every ``check_interval_s`` seconds.
The latch is an OPTIMISATION, never the only line of defence.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from enum import StrEnum

from loguru import logger
from prometheus_client import Counter, Gauge

from backend.data_engine.storage.redis_cache import RedisCache

KILL_SWITCH_GAUGE = Gauge("kill_switch_state_numeric", "0=normal, 1=close_only, 2=liquidate")
KILL_SWITCH_TRIPS = Counter(
    "kill_switch_trips_total", "Kill switch activations", labelnames=("state",)
)
KILL_SWITCH_LATCH_HITS = Counter(
    "kill_switch_latch_hits_total",
    "Number of times the in-process latch short-circuited a hot loop iteration",
)

# Redis key carrying the canonical kill-switch state value.
_KEY = "safety:kill_switch"
# Redis pub/sub channel used by KillSwitch.set_state to notify every
# subscribed process of state transitions. Keep the name aligned with
# LocalKillSwitchListener — the two MUST agree.
_CHANNEL = "safety:kill_switch:events"


class KillSwitchState(StrEnum):
    NORMAL = "NORMAL"
    CLOSE_ONLY = "CLOSE_ONLY"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"


_NUMERIC = {
    KillSwitchState.NORMAL: 0,
    KillSwitchState.CLOSE_ONLY: 1,
    KillSwitchState.LIQUIDATE_ALL: 2,
}


class KillSwitch:
    """Redis-backed canonical kill-switch state.

    All transitions go through :meth:`set_state`, which performs both
    the Redis ``SET`` (durable) and the Redis ``PUBLISH`` (low-latency
    fan-out). The two operations are issued back-to-back; if either
    fails the other is still attempted because the alternative — a
    half-applied transition — is worse than a single dropped fan-out.
    """

    def __init__(self, redis: RedisCache):
        self._redis = redis

    async def get_state(self) -> KillSwitchState:
        raw = await self._redis.client.get(_KEY)
        if raw is None:
            return KillSwitchState.NORMAL
        try:
            state = KillSwitchState(raw.decode("utf-8"))
        except ValueError:
            state = KillSwitchState.NORMAL
        KILL_SWITCH_GAUGE.set(_NUMERIC[state])
        return state

    async def set_state(self, state: KillSwitchState, reason: str = "") -> None:
        """Persist the state in Redis and fan out a pub/sub notification.

        We catch errors on the publish step independently so a flapping
        pub/sub does NOT mask a successful SET — the slow polling path
        remains correct.
        """
        await self._redis.client.set(_KEY, state.value)
        KILL_SWITCH_GAUGE.set(_NUMERIC[state])
        try:
            await self._redis.client.publish(_CHANNEL, state.value)
        except Exception as exc:
            logger.warning(f"Kill-switch pub/sub fan-out failed: {exc}")
        if state != KillSwitchState.NORMAL:
            KILL_SWITCH_TRIPS.labels(state=state.value).inc()
            logger.critical(f"KILL SWITCH -> {state.value}. Reason: {reason}")
        else:
            logger.info("Kill switch reset to NORMAL")

    async def escalate(self, reason: str) -> KillSwitchState:
        current = await self.get_state()
        if current == KillSwitchState.NORMAL:
            new_state = KillSwitchState.CLOSE_ONLY
        elif current == KillSwitchState.CLOSE_ONLY:
            new_state = KillSwitchState.LIQUIDATE_ALL
        else:
            return current
        await self.set_state(new_state, reason)
        return new_state


# ============================================================
# In-process latch
# ============================================================
class LocalKillSwitch:
    """Process-wide ``threading.Event`` latch for ``LIQUIDATE_ALL``.

    The hot decision loop checks :meth:`is_liquidate` once per cycle.
    The call is a single atomic read on the ``Event`` flag — under
    100 nanoseconds on commodity hardware — so adding the check to the
    loop has no measurable cost.

    The latch is INTENTIONALLY a Python singleton. Multiple instances
    in the same process would defeat the purpose (which is "tell every
    consumer in the process at once with no synchronisation").
    """

    _instance: LocalKillSwitch | None = None

    def __init__(self) -> None:
        # The ``threading.Event`` API gives us a lock-free, signal-safe
        # boolean that any thread or asyncio task can read.
        self._event = threading.Event()

    @classmethod
    def instance(cls) -> LocalKillSwitch:
        """Return the singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_liquidate(self) -> bool:
        """O(100 ns) read used by hot loops."""
        result = self._event.is_set()
        if result:
            KILL_SWITCH_LATCH_HITS.inc()
        return result

    def trip(self) -> None:
        """Arm the latch. Called by :class:`LocalKillSwitchListener`."""
        if not self._event.is_set():
            logger.critical("LocalKillSwitch ARMED (in-process latch)")
            self._event.set()

    def reset(self) -> None:
        """Disarm the latch. Called when the kill switch is downgraded."""
        if self._event.is_set():
            logger.info("LocalKillSwitch DISARMED")
            self._event.clear()


# ============================================================
# Pub/sub listener
# ============================================================
class LocalKillSwitchListener:
    """Async background task that mirrors Redis kill-switch state into the latch.

    Lifecycle
    ---------
    * Construction is cheap — no I/O yet.
    * :meth:`start` (1) does an initial Redis ``GET`` to set the latch
      to the current state, then (2) launches a perpetual ``SUBSCRIBE``
      coroutine. The two-step start avoids the classic race where
      pub/sub messages issued *between* the SET and our subscribe would
      be lost.
    * :meth:`stop` cancels the subscription task and aclose()s the
      pubsub connection; idempotent.
    """

    def __init__(
        self,
        redis: RedisCache,
        latch: LocalKillSwitch | None = None,
    ):
        self._redis = redis
        self._latch = latch or LocalKillSwitch.instance()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        """Sync the latch to the current Redis state, then start listening."""
        # Initial sync (closes the race window above).
        await self._sync_from_redis()
        # Subscribe in a background task.
        self._stopping = False
        self._task = asyncio.create_task(self._listen(), name="kill_switch_listener")
        logger.info("LocalKillSwitchListener started")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _sync_from_redis(self) -> None:
        """Read Redis once and mirror its state into the local latch."""
        ks = KillSwitch(self._redis)
        state = await ks.get_state()
        self._apply(state)

    def _apply(self, state: KillSwitchState) -> None:
        """Update the latch based on a state value."""
        if state == KillSwitchState.LIQUIDATE_ALL:
            self._latch.trip()
        else:
            self._latch.reset()

    async def _listen(self) -> None:
        """Long-running pub/sub loop.

        We re-subscribe forever; a transient connection drop will throw
        out of ``pubsub.listen()`` and we re-enter the outer ``while``.
        """
        backoff_s = 0.5
        while not self._stopping:
            try:
                pubsub = self._redis.client.pubsub()
                await pubsub.subscribe(_CHANNEL)
                # Reset backoff once we successfully subscribe.
                backoff_s = 0.5
                async for msg in pubsub.listen():
                    if self._stopping:
                        break
                    if msg.get("type") != "message":
                        continue
                    raw = msg.get("data")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        state = KillSwitchState(raw)
                    except ValueError:
                        logger.warning(f"Ignoring unknown kill-switch event: {raw!r}")
                        continue
                    self._apply(state)
                await pubsub.unsubscribe(_CHANNEL)
                await pubsub.aclose()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stopping:
                    break
                logger.error(f"Kill-switch listener crashed; reconnecting: {exc}")
                await asyncio.sleep(backoff_s)
                # Cap the backoff at 5 s so a long Redis outage doesn't
                # bury the listener.
                backoff_s = min(backoff_s * 2, 5.0)
