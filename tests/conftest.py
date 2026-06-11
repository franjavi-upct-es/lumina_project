# tests/conftest.py
<<<<<<< HEAD
"""Shared pytest fixtures."""
=======
"""
Pytest configuration and fixtures for Lumina Quant Lab tests.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e

import pytest
from dotenv import dotenv_values

<<<<<<< HEAD
=======
# Add backend to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BACKEND_ENV = {
    key: value
    for key, value in dotenv_values(REPO_ROOT / "backend" / ".env").items()
    if value is not None
}
TRUTHY = {"1", "true", "yes", "on", "debug", "development", "dev"}
FALSY = {"0", "false", "no", "off", "release", "production", "prod"}


def _env(name: str, default: str) -> str:
    return os.getenv(name, BACKEND_ENV.get(name, default))


def _build_http_url(base_var: str, host_var: str, port_var: str, default_port: int) -> str:
    explicit = os.getenv(base_var, BACKEND_ENV.get(base_var))
    if explicit:
        return explicit.rstrip("/")

    host = _env(host_var, "localhost")
    port = _env(port_var, str(default_port))
    return f"http://{host}:{port}"


def _normalize_debug_env() -> None:
    debug_value = os.getenv("DEBUG")
    if debug_value is None:
        if "DEBUG" in BACKEND_ENV:
            os.environ["DEBUG"] = BACKEND_ENV["DEBUG"]
        return

    normalized = debug_value.strip().lower()
    if normalized in TRUTHY | FALSY:
        return

    os.environ["DEBUG"] = BACKEND_ENV.get("DEBUG", "true")


def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    return host, port


def _is_api_healthy(base_url: str, timeout: float = 2.0) -> bool:
    import requests

    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _is_redis_reachable(redis_url: str) -> bool:
    from redis import Redis

    client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    try:
        return client.ping() is True
    except Exception:
        return False
    finally:
        client.close()


_normalize_debug_env()

for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "SECRET_KEY", "REDIS_PASSWORD"):
    if key in BACKEND_ENV:
        os.environ.setdefault(key, BACKEND_ENV[key])

# Environment configuration for tests
TEST_CONFIG = {
    # Database
    "DATABASE_URL": _env(
        "DATABASE_URL",
        "postgresql://lumina:lumina_password@localhost:5435/lumina_db",
    ),
    # Redis
    "REDIS_URL": _env("REDIS_URL", "redis://:lumina_password@localhost:6379/0"),
    "CELERY_BROKER_URL": _env(
        "CELERY_BROKER_URL",
        "redis://:lumina_password@localhost:6379/1",
    ),
    "CELERY_RESULT_BACKEND": _env(
        "CELERY_RESULT_BACKEND",
        "redis://:lumina_password@localhost:6379/2",
    ),
    # API
    "API_BASE_URL": _build_http_url("API_BASE_URL", "API_HOST", "API_PORT", 8000),
    # MLflow
    "MLFLOW_TRACKING_URI": _build_http_url(
        "MLFLOW_TRACKING_URI",
        "MLFLOW_HOST",
        "MLFLOW_PORT",
        5000,
    ),
}

# Set environment variables
for key, value in TEST_CONFIG.items():
    os.environ.setdefault(key, value)


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "docker: marks tests requiring Docker services")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


def _is_service_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP service is reachable."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Auto-skip integration/docker tests when services are not reachable
    api_up = _is_api_healthy(TEST_CONFIG["API_BASE_URL"])
    redis_up = _is_redis_reachable(TEST_CONFIG["REDIS_URL"])
    db_host, db_port = _parse_host_port(TEST_CONFIG["DATABASE_URL"], 5432)
    db_up = _is_service_reachable(db_host, db_port)

    for item in items:
        if "integration" in item.keywords or "docker" in item.keywords:
            if not (api_up and redis_up and db_up):
                item.add_marker(
                    pytest.mark.skip(reason="integration services not running (API/Redis/DB)")
                )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Return test configuration"""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def api_base_url():
    """Return API base URL"""
    return TEST_CONFIG["API_BASE_URL"]


@pytest.fixture(scope="session")
def database_url():
    """Return database URL"""
    return TEST_CONFIG["DATABASE_URL"]


@pytest.fixture(scope="session")
def redis_url():
    """Return Redis URL"""
    return TEST_CONFIG["REDIS_URL"]

>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e

@pytest.fixture
def anyio_backend():
    return "asyncio"
