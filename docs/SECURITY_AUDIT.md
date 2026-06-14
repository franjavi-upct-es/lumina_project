# Lumina V3 — Security Audit

**Date:** 2026-06-14 **Scope:** Full repository (`backend/`, `frontend/`,
`docker/`, `scripts/`, `alembic/`, notebooks, CI/CD, config) — 382 tracked
files. **Method:** Manual source review + pattern sweeps for secrets, injection
sinks, deserialization, auth gaps, TLS/CORS misconfiguration, container
hardening, and git-history secret leakage.

> This document records findings as of the audit date. Line numbers refer to the
> state of `main` at commit time and may drift as the code evolves.

---

## 1. Executive summary

The codebase is, in several respects, **above average for security hygiene**: no
secrets are committed or present in git history, the SQL layer is fully
parameterized, all containers run as a non-root user, and the React frontend has
no XSS sinks. The dangerous patterns that exist are mostly **configuration
footguns and broken access control**, not classic injection bugs.

The highest-impact issues are:

1. **Fail-open API authentication** — an empty `API_KEY` disables auth entirely.
2. **Unauthenticated read endpoints and WebSockets** — Arena run data and live
   agent streams leak without a key.
3. **Redis exposed on the host with no password and `CONFIG` enabled** —
   tampering / queue injection / potential RCE if the host is reachable.
4. **Insecure checkpoint deserialization** — unsafe model-artifact loading can
   become RCE if an attacker can influence checkpoint contents.

| #   | Severity | Finding                                                                                                   | Location                                                                      |
| --- | -------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| F1  | **High** | Fail-open auth: empty `API_KEY` ⇒ all "protected" routes open; weak default `change_me_in_production`     | `backend/api/deps.py:89`, `backend/config/settings.py:109`, `.env.example:68` |
| F2  | **High** | Broken access control: Arena `GET` endpoints + both WebSockets + `/metrics` require no API key            | `backend/api/routes/arena.py:155-358`, `agent.py:72`, `monitoring.py:52-94`   |
| F3  | **High** | Redis published to host with no `requirepass`, no protected-mode, `CONFIG` not disabled                   | `docker-compose.yml:28`, `docker/services/redis.conf`                         |
| F4  | **High** | `exec(strategy_code)` — arbitrary code execution in legacy Celery backtest workers                        | Removed with `backend/workers/`                                               |
| F5  | Medium   | Insecure deserialization: `torch.load(..., weights_only=False)` and `pickle.load` of checkpoints          | many; see §F5                                                                 |
| F6  | Medium   | CORS footgun: `allow_credentials=True` + wildcard methods/headers (unsafe if origins set to `*`)          | `backend/api/main.py:130-136`                                                 |
| F7  | Medium   | MLflow exposed without auth; TimescaleDB exposed with `lumina:lumina`                                     | `docker-compose.yml:37-71`                                                    |
| F8  | Medium   | Frontend API key (`VITE_API_KEY`) is baked into the browser bundle ⇒ effectively public                   | `frontend/src/api/client.ts:5`                                                |
| F9  | Low      | API-key comparison is not constant-time (timing side-channel)                                             | `backend/api/deps.py:91`                                                      |
| F10 | Low      | `:latest` image tags (timescale, mlflow, brain) — non-reproducible / supply-chain                         | `docker-compose.yml:38,58,225`                                                |
| F11 | Low      | `execute_raw_sql` relied on a regex blocklist in the legacy ORM layer                                      | Removed with `backend/db/`                                                    |
| F12 | Low      | Prometheus labels use raw `request.url.path` ⇒ unbounded cardinality (memory growth)                      | `backend/api/middleware.py:42-43`                                             |
| F13 | Low      | No rate limiting on authentication ⇒ API-key brute force possible                                         | `backend/api/middleware.py`, `deps.py`                                        |
| F14 | Info     | Local untracked artefacts (`.env`, `mlflow.db`, `.coverage`) present — gitignored; keep them out of VCS   | repo root                                                                     |
| F15 | Info     | Streamlit `unsafe_allow_html=True` watch item                                                             | Removed with `frontend/streamlit_app/`                                        |

---

## 2. What is done well (positive findings)

These were checked and found **clean** — worth preserving as the code evolves:

- **No committed secrets.** `.env` is untracked and never appeared in git
  history; the tracked `.env.example` (byte-identical to the local `.env`) has
  **all secret fields blank** (`POLYGON_API_KEY=`, `NEWSAPI_KEY=`,
  `ALPACA_API_KEY=`, `ALPACA_SECRET_KEY=`). History scan for `sk-…`, `AKIA…`,
  `AIza…`, `xox…` token shapes returned only false positives (base64 PNGs in
  notebooks).
- **SQL is fully parameterized.** All `asyncpg` calls in
  `data_engine/storage/timescale.py` and `api/routes/arena.py` use positional
  bind parameters (`$1, $2, …`). The one dynamic query (`arena.py:205-222`)
  interpolates only **computed integer parameter positions**, never user data.
- **Containers run as non-root.** Every Dockerfile creates and switches to
  `USER lumina` (uid 1000).
- **No frontend XSS sinks.** No `dangerouslySetInnerHTML`, `innerHTML`, `eval`,
  or `new Function`.
- **No secret logging.** The Alpaca adapter passes keys to the SDK constructor
  and logs only `paper=<bool>` (`alpaca_adapter.py:25-26`); no credentials are
  written to logs.
- **No `verify=False`, no `DEBUG=True`, no insecure `yaml.load`, no hardcoded
  credentials** in source.
- **API surface mostly authenticated.** Order-placing / kill-switch /
  backtest-submit / portfolio endpoints all carry `Depends(require_api_key)`.

---

## 3. Detailed findings

### F1 — Fail-open authentication (High)

`backend/api/deps.py:89-90`:

```python
if not settings.API_KEY:
    return UserContext(user_id="dev_user", tier="enterprise")
```

`Settings.API_KEY` defaults to `""` (`settings.py:109`). If `API_KEY` is unset
in the environment, **every endpoint guarded by `require_api_key` becomes fully
open** and the caller is silently granted an `enterprise` tier. The shipped
`.env.example` sets `API_KEY=change_me_in_production`, a well-known placeholder
that is dangerous if deployed unchanged.

**Impact:** Unauthenticated order placement, kill-switch toggling, backtest
submission, and data access in any deployment that forgets to set a strong key.

**Remediation:**

- Fail **closed** in non-development environments: if
  `ENVIRONMENT != "development"` and `API_KEY` is empty (or equals
  `change_me_in_production`), refuse to start (raise in a Pydantic validator).
- Require a minimum entropy / length for `API_KEY`.
- Keep the frictionless dev path only when `ENVIRONMENT == "development"`.

### F2 — Broken access control on reads & WebSockets (High)

These endpoints have **no** `require_api_key` dependency:

- `GET /arena/runs`, `/arena/runs/{id}`, `/arena/runs/{id}/decisions`,
  `/divergences`, `/explanations`, `/pairs`, `/summary` (`arena.py:155-330`)
- `WS /api/agent/stream` (`agent.py:72`) and `WS /arena/runs/{id}/live`
  (`arena.py:358`) — both call `accept()` with no key check and no `Origin`
  validation
- `GET /api/monitoring/health` and `GET /api/monitoring/metrics`
  (`monitoring.py:52, 94`)

The code comments describe a "multi-tenant SaaS" with subscription tiers, but
`require_api_key` returns a single global `admin_user`/`dev_user` — there is
**no tenant isolation**, so any run's data is globally readable, and the
unauthenticated GETs/WS make that data reachable without credentials.

**Impact:** Disclosure of trading decisions, divergence/XAI data, and live agent
actions to anyone who can reach the API. `/metrics` additionally leaks internal
operational detail.

**Remediation:**

- Apply auth uniformly. Prefer a **global** dependency:
  `FastAPI(dependencies=[Depends(require_api_key)])` (or
  `APIRouter(dependencies=…)`) and explicitly exempt only `/health`.
- Authenticate WebSockets via a `?token=` query param or subprotocol, validate
  it before `accept()`, and check the `Origin` header (CSWSH protection).
- Restrict `/metrics` to the internal network / scrape source, or require auth.
- Implement real per-tenant scoping if multi-tenancy is a goal.

### F3 — Redis exposed, unauthenticated, `CONFIG` enabled (High)

`docker-compose.yml:28-29` publishes `6379:6379` to the host (binds `0.0.0.0` by
default). `docker/services/redis.conf` sets **no `requirepass`**, does not
enable protected-mode, and does not `rename-command` the dangerous
`CONFIG`/`FLUSHALL`/`SAVE` verbs.

**Impact:** Anyone who can reach the host port can read/modify every key.
Because Redis is the job queue and control plane, this allows: injecting
`backtest:request:*` / arena jobs that workers execute, flipping the kill-switch
latch, and reading agent streams. With `CONFIG` available, the classic
`CONFIG SET dir … / dbfilename … / SAVE` (or module/cron abuse) escalation to
**RCE on the Redis host** is possible.

**Remediation:**

- Set a strong `requirepass` (and put it in `REDIS_URL`); enable
  `protected-mode yes`.
- `rename-command CONFIG ""`, `FLUSHALL ""`, `FLUSHDB ""`, `SAVE ""`,
  `DEBUG ""`.
- Do **not** publish 6379 to the host in production — keep it on the internal
  `lumina_net` only (drop the `ports:` mapping or bind to `127.0.0.1`).

### F4 — `exec(strategy_code)` arbitrary code execution (High, removed)

The vulnerable path lived in the legacy Celery backtest worker package. The V3
runtime uses Redis-keyed simulation workers under `backend/simulation/` and does
not execute user-supplied Python strategy strings. The legacy `backend/workers/`
package has been removed, so the `exec(strategy_code)` sink is gone rather than
gated.

**Residual guidance:** keep arbitrary Python strategy execution out of the
runtime. If user-defined strategies become a product requirement, implement a
constrained DSL or execute code in a locked-down external sandbox.

### F5 — Insecure deserialization of model artifacts (Medium)

`torch.load(..., weights_only=False)` at `cognition/agent/runtime.py:36`,
`fusion/state_assembler.py:387`,
`perception/{semantic,structural,temporal}/inference.py`, plus `pickle.load` at
`data_engine/transformers/regime_detection.py:329`.

`weights_only=False` (and `pickle`) execute arbitrary code embedded in the
artifact during load. This is safe only if **every** checkpoint is produced and
stored by trusted parties. The inference modules read paths from
config/`models/` mounts; if a checkpoint path can ever be influenced by a user,
or a model store is shared/downloaded, this is RCE.

**Remediation:**

- Use `torch.load(..., weights_only=True)` wherever the artifact is a plain
  `state_dict` (the `scripts/*` already do this correctly). Migrate checkpoints
  that need full objects.
- Prefer `safetensors` for weights. Replace `pickle` model persistence with a
  non-executable format.
- Verify artifact integrity (checksums / signatures) before loading.

### F6 — CORS misconfiguration footgun (Medium)

`backend/api/main.py:130-136` sets `allow_credentials=True` together with
`allow_methods=["*"]` and `allow_headers=["*"]`. The default origin is
`http://localhost:5173` (safe), but `CORS_ORIGINS` is comma-split from the
environment (`settings.py:114-119`). If anyone sets `CORS_ORIGINS=*`, Starlette
will **reflect the caller's Origin and return
`Access-Control-Allow-Credentials: true`** — i.e. allow _any_ site to make
credentialed requests.

**Remediation:** Never combine wildcard origins with credentials. Maintain an
explicit allow-list, and add a startup validator that rejects `"*"` in
`CORS_ORIGINS` when `allow_credentials` is on.

### F7 — MLflow / TimescaleDB exposure (Medium)

- MLflow is published on `5000:5000` with **no authentication** and
  `--serve-artifacts` (`docker-compose.yml:57-71`). MLflow has a history of
  path-traversal / arbitrary-file-read and unsafe-model-deserialization CVEs; an
  exposed, unauthenticated instance is a data-exfiltration and model-poisoning
  vector. (The `docker/Dockerfile.mlflow` variant does add `--allowed-hosts`,
  which is good Host-header hardening but not authentication.)
- TimescaleDB is published on `5432:5432` with credentials `lumina:lumina`
  (`docker-compose.yml:42-49`), matching the weak default baked into
  `settings.py:74`.

**Remediation:** Keep MLflow and Postgres off host-published ports in production
(internal network only) or front them with auth; replace `lumina:lumina` with
secrets injected at deploy time.

### F8 — Frontend API key is shipped to the browser (Medium)

`frontend/src/api/client.ts:5` reads `import.meta.env.VITE_API_KEY`. Any
`VITE_*` value is inlined into the static bundle at build time, so the key is
readable by anyone who loads the app. A single static `x-api-key` therefore
provides **no real authentication for browser clients**.

**Remediation:** For browser auth, use per-user sessions/JWTs minted by a login
flow, scoped tokens, and server-side rate limiting — not a build-time shared
key.

### F9 — Non-constant-time key comparison (Low)

`backend/api/deps.py:91` uses `x_api_key != settings.API_KEY`. Use
`hmac.compare_digest(...)` / `secrets.compare_digest(...)` to remove the timing
side-channel.

### F10 — Floating `:latest` image tags (Low)

`timescale/timescaledb:latest-pg16`, `ghcr.io/mlflow/mlflow:latest`,
`lumina/brain:latest` (`docker-compose.yml`). Pin to specific versions (ideally
by digest) for reproducible, auditable builds. (Application Python deps are
already pinned via `uv.lock` + `uv sync --frozen` — good.)

### F11 — `execute_raw_sql` regex blocklist (Low, removed)

The raw-SQL helper lived in the legacy SQLAlchemy ORM package. The V3 cleanup
removed `backend/db/`; live database access uses the async Timescale adapter and
hand-written Alembic migrations.

### F12 — Prometheus label cardinality (Low)

`backend/api/middleware.py:42-43` labels metrics with the raw
`request.url.path`, which includes path parameters (UUIDs, tickers). This
creates unbounded time-series cardinality and lets a caller grow the metrics
registry's memory by hitting many distinct paths. Use the matched **route
template** (`request.scope["route"].path`) instead of the concrete path.

### F13 — No authentication rate limiting (Low)

`CongestionControlMiddleware` limits total concurrency but there is no
per-client throttle on failed auth, so an exposed API permits API-key brute
forcing. Add rate limiting (e.g. per-IP / per-key) and lockout/backoff on
repeated 401s.

### F14 — Local untracked artefacts (Info)

`.env`, `mlflow.db` (~872 KB), and `.coverage` exist in the working tree but are
correctly listed in `.gitignore` (and `.dockerignore`). No action needed beyond
vigilance: never `git add -f` them, and consider relocating runtime DBs out of
the repo root.

### F15 — Streamlit raw HTML rendering (Info, removed)

The legacy Streamlit dashboard was removed during the V3 cleanup. The remaining
frontend path is the React/Vite dashboard, which was reviewed separately for
browser-side XSS sinks.

---

## 4. CI/CD added by this audit

Four GitHub configuration files were added under `.github/`:

| File                     | Purpose                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `workflows/ci.yml`       | Lint (`ruff`), format check, type-check (`mypy`), unit tests (`pytest`) via **uv**, plus the frontend `tsc + vite build`.                                       |
| `workflows/security.yml` | `gitleaks` (secrets, incl. full history), `pip-audit` (dependency CVEs from `uv.lock`), `bandit` (Python SAST), `trivy` (fs vuln + secret + misconfig → SARIF). |
| `workflows/codeql.yml`   | CodeQL taint analysis for Python and TypeScript → Security tab.                                                                                                 |
| `dependabot.yml`         | Weekly update PRs for uv/Python, npm, GitHub Actions, and Docker base images.                                                                                   |

Design choices: every workflow declares least-privilege
`permissions: contents: read` at the top level and widens to
`security-events: write` only where SARIF is uploaded. All Python steps run
through `uv` per project convention. `pip-audit` and `bandit` start
**non-blocking** so they surface findings without wedging every PR; flip
`continue-on-error: false` once the baseline is clean to make them hard gates.
For maximum supply-chain safety, pin each `uses:` to a commit SHA.

---

## 5. Recommended remediation order

1. **F1 + F2** — Make auth fail-closed in non-dev and apply it uniformly (incl.
   WebSockets). _(highest impact / lowest effort)_
2. **F3** — Lock down Redis (password, protected-mode, rename `CONFIG`, drop
   host port).
3. **F7** — Take MLflow/Postgres off host-published ports; rotate DB
   credentials.
4. **F4** — Resolved by removing the legacy Celery worker path that contained
   `exec(strategy_code)`.
5. **F5** — Switch to `weights_only=True` / `safetensors`; verify artifact
   integrity.
6. **F6, F8, F9–F13** — Hardening pass (CORS guard, real browser auth,
   constant-time compare, pinned images, route-template metrics, auth rate
   limiting).

---

## 6. Remediation applied

The following high-severity findings (plus F9) were fixed and validated
(auth-logic unit checks + `tests/api/test_routes.py` passing, frontend `tsc`
clean, `docker-compose.yml` parsing, strategy gating exercised against the real
source):

| # | Status | What changed |
|---|--------|--------------|
| **F1** | ✅ Fixed | `Settings` now fails closed: a `model_validator` rejects an empty/placeholder/`<16` char `API_KEY` when `ENVIRONMENT` is `staging`/`production`. `require_api_key` only allows an unset key in development, else returns `503`. (`config/settings.py`, `api/deps.py`) |
| **F2** | ✅ Fixed | `require_api_key` added to all Arena `GET` endpoints; both WebSockets now call `authorize_websocket()` before `accept()` — it validates `Origin` against `CORS_ORIGINS` (CSWSH) and a `?token=` query param. Frontend WS hooks updated to send the token. (`api/routes/arena.py`, `agent.py`, `api/deps.py`, `frontend/src/api/{client,agent}.ts`, `hooks/useArenaStream.ts`) |
| **F3** | ⚠️ Partial | Redis/Timescale/MLflow host ports bound to `127.0.0.1`; `redis.conf` neutralizes `CONFIG`/`MODULE`/`DEBUG`/`FLUSHALL`/`FLUSHDB`. **Still TODO for production:** set `requirepass` + `protected-mode` and inject DB secrets (documented inline). (`docker-compose.yml`, `docker/services/redis.conf`) |
| **F4** | ✅ Fixed | The `exec(strategy_code)` sink was removed with the legacy `backend/workers/` Celery package. The live V3 backtest path uses `backend/simulation/backtest_worker.py` and does not execute user-provided Python strings. |
| **F9** | ✅ Fixed | API-key comparison uses `hmac.compare_digest` (constant-time). (`api/deps.py`) |

### Pre-existing CI blockers — status

These were unrelated to the security fixes but blocked a green pipeline. The
**lint job is now green**; the **test job** still has pre-existing failures that
need a judgment call (see below).

| Issue | Status |
|-------|--------|
| **193 `ruff` violations** (`backend tests scripts`, pinned `ruff==0.15.17`) | ✅ **0 remaining.** Safe + behavior-preserving autofixes applied; ML matrix-naming rules `N803`/`N806` added to `ignore` (domain convention); 7 fixed by hand (async Redis in the health probe, collapsed conditionals, `noqa` for intentional lowercase context-managers + best-effort suppress); `tests/**` ignores `TID251`. `ruff check` + `ruff format --check .` both pass. |
| **`celery` imported but undeclared** | ✅ **Obsolete.** The legacy Celery worker package was removed, and the `celery` dev dependency/settings were dropped. |
| **`transformers/__init__.py` → missing `normalization`** | ✅ **Fixed** — dropped the dead `DataNormalizer` export (used nowhere); the package imports cleanly. |
| **`unit` pytest marker unregistered** | ✅ Registered in `[tool.pytest.ini_options].markers`. |

### V3 cleanup follow-up

The stale tests that referenced removed V2 APIs/schema were deleted or rewritten
against the live V3 modules. The remaining test suite should fail only on
current V3 regressions, not missing legacy packages.
