# tests/conftest.py
"""
Pytest configuration and fixtures for Lumina Quant Lab tests.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment configuration for tests
TEST_CONFIG = {
    # Database
    "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost:5435/lumina_db"),
    # Redis
    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "CELERY_BROKER_URL": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    "CELERY_RESULT_BACKEND": os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    # API
    "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000"),
    # MLflow
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
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


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


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


@pytest.fixture
def date_range():
    """Return a standard date range for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date


@pytest.fixture
def extended_date_range():
    """Return an extended date range for ML training"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date


@pytest.fixture
def sample_tickers():
    """Return sample tickers for testing"""
    return ["AAPL", "GOOGL", "MSFT", "AMZN"]


@pytest.fixture
def sample_portfolio():
    """Return sample portfolio weights"""
    return {"AAPL": 0.3, "GOOGL": 0.25, "MSFT": 0.25, "AMZN": 0.2}


@pytest.fixture
def lstm_hyperparams(extended_date_range):
    """Return LSTM hyperparameters for testing"""
    start_date, end_date = extended_date_range
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "sequence_length": 20,
        "prediction_horizon": 3,
        "batch_size": 16,
        "num_epochs": 3,
        "learning_rate": 0.001,
        "early_stopping_patience": 2,
        "max_features": 10,
    }


@pytest.fixture
def xgboost_hyperparams(extended_date_range):
    """Return XGBoost hyperparameters for testing"""
    start_date, end_date = extended_date_range
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "prediction_horizon": 5,
        "max_features": 20,
    }


@pytest.fixture
def backtest_config(extended_date_range, sample_tickers):
    """Return backtest configuration for testing"""
    start_date, end_date = extended_date_range
    return {
        "strategy": "momentum",
        "tickers": sample_tickers[:3],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0005,
        "strategy_params": {"lookback_period": 20, "rebalance_frequency": "weekly"},
    }


# ============================================================================
# Service Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def celery_app():
    """Return Celery app instance"""
    try:
        from backend.workers.celery_app import celery_app

        return celery_app
    except ImportError:
        pytest.skip("Celery app not available")


@pytest.fixture
def yfinance_collector():
    """Return YFinance collector instance"""
    try:
        from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

        return YFinanceCollector(rate_limit=100)
    except ImportError:
        pytest.skip("YFinanceCollector not available")


@pytest.fixture
def feature_engineer():
    """Return Feature Engineer instance"""
    try:
        from backend.data_engine.transformers.feature_engineering import FeatureEngineer

        return FeatureEngineer()
    except ImportError:
        pytest.skip("FeatureEngineer not available")


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def db_connection(database_url):
    """Return database connection"""
    try:
        import psycopg2

        conn = psycopg2.connect(database_url)
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
def redis_client(redis_url):
    """Return Redis client"""
    try:
        from redis import Redis

        client = Redis.from_url(redis_url)
        client.ping()
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


# ============================================================================
# API Client Fixtures
# ============================================================================


@pytest.fixture
def api_client(api_base_url):
    """Return API test client"""
    try:
        from fastapi.testclient import TestClient

        from backend.api.main import app

        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI TestClient not available")


@pytest.fixture
def requests_session(api_base_url):
    """Return requests session for API testing"""
    import requests

    session = requests.Session()
    session.base_url = api_base_url

    # Test connection
    try:
        response = session.get(f"{api_base_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API not healthy")
    except requests.RequestException:
        pytest.skip("API not accessible")

    yield session
    session.close()


# ============================================================================
# Utility Functions
# ============================================================================


def wait_for_task(task, timeout=120, poll_interval=2):
    """
    Wait for a Celery task to complete.

    Args:
        task: Celery AsyncResult
        timeout: Maximum wait time in seconds
        poll_interval: Time between status checks

    Returns:
        Task result if successful

    Raises:
        TimeoutError: If task doesn't complete in time
        Exception: If task fails
    """
    import time

    start_time = time.time()

    while (time.time() - start_time) < timeout:
        if task.ready():
            if task.successful():
                return task.result
            else:
                raise Exception(f"Task failed: {task.info}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Task {task.id} did not complete within {timeout} seconds")


def check_service_health(service_name, check_func):
    """
    Check if a service is healthy.

    Args:
        service_name: Name of the service
        check_func: Function that returns True if healthy

    Returns:
        bool: True if healthy
    """
    try:
        return check_func()
    except Exception as e:
        print(f"⚠ {service_name} health check failed: {e}")
        return False
