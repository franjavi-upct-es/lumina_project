# Getting Started: Setup and Configuration

??? note "Relevant source files"

    - [:material-github: .env.example](https://github.com/franjavi-upct-es/lumina_project/blob/main/.env.example)
    - [:material-github: .gitignore](https://github.com/franjavi-upct-es/lumina_project/blob/main/.gitignore)
    - [:material-github: .python-version](https://github.com/franjavi-upct-es/lumina_project/blob/main/.python-version)
    - [:material-github: Makefile](https://github.com/franjavi-upct-es/lumina_project/blob/main/Makefile)
    - [:material-github: backend/cognition/training/behavioral_cloning.py](https://github.com/franjavi-upct-es/lumina_project/blob/main/backend/cognition/training/behavioral_cloning.py)
    - [:material-github: backend/config/constants.py](https://github.com/franjavi-upct-es/lumina_project/blob/main/backend/config/constants.py)
    - [:material-github: backend/config/settings.py](https://github.com/franjavi-upct-es/lumina_project/blob/main/backend/config/settings.py)
    - [:material-github: docker/Dockerfile.api](https://github.com/franjavi-upct-es/lumina_project/blob/main/docker/Dockerfile.api)
    - [:material-github: pyproject.toml](https://github.com/franjavi-upct-es/lumina_project/blob/main/pyproject.toml)
    - [:material-github: uv.lock](https://github.com/franjavi-upct-es/lumina_project/blob/main/uv.lock)

This page provides a comprehensive guide for initializing the Lumina V3
"Chimera" development environment. It covers dependency management with `uv`,
configuration via Pydantic settings, and the containerized orchestration of the
multi-service architecture.

## 1. Environment Prerequisites

Lumina V3 requires a Linux or macOS environment with the following
specifications:

- **Python 3.11:** Explicitly pinned in
  [:material-github: .python-version#1](https://github.com/franjavi-upct-es/lumina_project/blob/main/.python-version#L1)
  and
  [:material-github: pyproject.toml#27](https://github.com/franjavi-upct-es/lumina_project/blob/main/pyproject.toml#L27)
- **uv:** The project uses `uv` for ultra-fast dependency resolution and virtual
  environment management
  [:material-github: Makefile#4-5](https://github.com/franjavi-upct-es/lumina_project/blob/main/Makefile#L4-L5)
- **Docker & Docker Compose:** Required for running the full stack, including
  TimescaleDB and Redis.
- **NVIDIA GPU (Optional):** Recommended for the Perception and Cognition
  layers. The stack supports CUDA 12.4 (standard) and CUDA 12.8 (Blackwell)
  [:material-github: Makefile#33-34](https://github.com/franjavi-upct-es/lumina_project/blob/main/Makefile#L33-L34)

## 2. Local Installation

The project utilizes a layered dependency strategy defined in
[:material-github: pyproject.toml#6-15](https://github.com/franjavi-upct-es/lumina_project/blob/main/pyproject.toml#L6-L15)
This allows individual services (API, Data, Brain) to install only the necessary
sub-packages.

### Step-by-Step Setup

1. **Clone and Sync:** Use the `Makefile` to synchronize the environment with
   all optional extras (dev, api, data, perception, brain, gpu).

    ```bash
    make install
    ```

    _Source:
    [:material-github: Makefile#50-51](https://github.com/franjavi-upct-es/lumina_project/blob/main/Makefile#L50-L51)_

2. **Configuration:** Copy the template environment file:

    ```bash
    cp .env.example .env
    ```

    _Source:
    [:material-github: .env.example#1-5](https://github.com/franjavi-upct-es/lumina_project/blob/main/.env.example#L1-L5)_

3. **Database Migrations:** Apply Alembic migrations to initialize the
   TimescaleDB schema (hypertables, news events, and portfolio tables).

    ```bash
    make migrate
    ```

    _Source:
    [:material-github: Makefile#50-51](https://github.com/franjavi-upct-es/lumina_project/blob/main/Makefile#L50-L51)_

## 3. Configuration & Environment Variables

Lumina V3 uses a centralized configuration system powered by
`pydantic-settings`. The `Settings` class in
[:material-github: backend/config/settings.py#39-108](https://github.com/franjavi-upct-es/lumina_project/blob/main/backend/config/settings.py#L39-L108)
acts a singleton that validates environment variables at runtime.

### Key Configuration Groups

| Group   | Variable                | Default                    | Description                                       |
| ------- | ----------------------- | -------------------------- | ------------------------------------------------- |
| Data    | `DATA_SOURCE`           | `yfinance`                 | `yfinance` (free/daily) or `polygon` (paid/1-min) |
| Broker  | `BROKER_MODE`           | `paper`                    | `paper` (local sim) or `alpaca` (Alpaca API)      |
| Storage | `REDIS_URL`             | `redis://localhost:6379/0` | Async Redis connection string                     |
| Storage | `TIMESCALE_URL`         | `postgresql://...`         | TimescaleDB connection string                     |
| Safety  | `UNCERTAINTY_THRESHOLD` | `0.85`                     | MC-Dropout threshold for the Uncertainty Gate     |
| Arena   | `ARENA_ARTIFACT_DIR`    | `./artifacts/arena`        | Storage for simulation trajectories               |

Sources:
[:material-github: backend/config/settings.py#14-108](https://github.com/franjavi-upct-es/lumina_project/blob/main/backend/config/settings.py#L14-L108)
[:material-github: .env.example#7-75](https://github.com/franjavi-upct-es/lumina_project/blob/main/.env.example#L7-L75)
