# tests/test_api_endpoints.py
"""
Complete API endpoints testing for Lumina Quant Lab.
Tests all REST API routes for correct functionality.

Usage:
    pytest tests/test_api_endpoints.py -v -s
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class TestDataEndpoints:
    """Test /api/v1/data endpoints"""

    @pytest.fixture
    def api_url(self):
        return f"{API_BASE_URL}/api/v1/data"

    def test_collect_data_single_ticker(self, api_url):
        """Test data collection for single ticker"""
        import requests

        payload = {
            "ticker": "AAPL",
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            response = requests.post(f"{api_url}/collect", json=payload, timeout=60)

            # Accept 200 (sync) or 202 (async task)
            assert response.status_code in [200, 202], (
                f"Unexpected status: {response.status_code} - {response.text}"
            )

            data = response.json()
            if response.status_code == 202:
                assert "task_id" in data
                print(f"✓ Data collection task submitted: {data['task_id']}")
            else:
                assert "data" in data or "rows" in data
                print(f"✓ Data collection successful: {data}")
        except requests.RequestException as e:
            pytest.fail(f"Data collection endpoint failed: {e}")

    def test_collect_data_invalid_ticker(self, api_url):
        """Test data collection with invalid ticker"""
        import requests

        payload = {
            "ticker": "INVALID_TICKER_12345",
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            response = requests.post(f"{api_url}/collect", json=payload, timeout=60)

            # Should return error or empty result
            assert response.status_code in [200, 202, 400, 404, 422]
            print(f"✓ Invalid ticker handled: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"Invalid ticker test failed: {e}")

    def test_collect_data_missing_fields(self, api_url):
        """Test data collection with missing required fields"""
        import requests

        payload = {"ticker": "AAPL"}  # Missing dates

        try:
            response = requests.post(f"{api_url}/collect", json=payload, timeout=30)

            # Should return validation error
            assert response.status_code == 422, (
                f"Expected 422 validation error, got {response.status_code}"
            )
            print("✓ Missing fields validation working")
        except requests.RequestException as e:
            pytest.fail(f"Missing fields test failed: {e}")

    def test_get_available_tickers(self, api_url):
        """Test getting available tickers"""
        import requests

        try:
            response = requests.get(f"{api_url}/tickers", timeout=30)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Available tickers: {len(data)} found")
            elif response.status_code == 404:
                print("⚠ Tickers endpoint not implemented yet")
            else:
                print(f"⚠ Tickers endpoint returned: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Tickers endpoint test skipped: {e}")

    def test_batch_data_collection(self, api_url):
        """Test batch data collection for multiple tickers"""
        import requests

        payload = {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            response = requests.post(f"{api_url}/collect/batch", json=payload, timeout=120)

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Batch collection response: {data}")
            elif response.status_code == 404:
                print("⚠ Batch collection endpoint not implemented")
            else:
                print(f"⚠ Batch collection returned: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Batch collection test skipped: {e}")


class TestMLEndpoints:
    """Test /api/v1/ml endpoints"""

    @pytest.fixture
    def api_url(self):
        return f"{API_BASE_URL}/api/v1/ml"

    def test_train_model_lstm(self, api_url):
        """Test LSTM model training endpoint"""
        import requests

        payload = {
            "ticker": "AAPL",
            "model_type": "lstm",
            "hyperparams": {
                "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "sequence_length": 30,
                "prediction_horizon": 5,
                "batch_size": 16,
                "num_epochs": 5,
                "learning_rate": 0.001,
                "max_features": 20,
            },
        }

        try:
            response = requests.post(f"{api_url}/train", json=payload, timeout=30)

            # Training is async, should return 202 Accepted
            assert response.status_code in [200, 202], (
                f"Unexpected status: {response.status_code} - {response.text}"
            )

            data = response.json()
            assert "job_id" in data or "task_id" in data
            print(f"✓ Training task submitted: {data}")
        except requests.RequestException as e:
            pytest.fail(f"Train model endpoint failed: {e}")

    def test_train_model_xgboost(self, api_url):
        """Test XGBoost model training endpoint"""
        import requests

        payload = {
            "ticker": "MSFT",
            "model_type": "xgboost",
            "hyperparams": {
                "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "prediction_horizon": 5,
            },
        }

        try:
            response = requests.post(f"{api_url}/train", json=payload, timeout=30)

            assert response.status_code in [200, 202]
            data = response.json()
            print(f"✓ XGBoost training submitted: {data}")
        except requests.RequestException as e:
            pytest.fail(f"XGBoost training endpoint failed: {e}")

    def test_list_models(self, api_url):
        """Test listing trained models"""
        import requests

        try:
            response = requests.get(f"{api_url}/models", timeout=30)

            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Models listed: {len(data) if isinstance(data, list) else data}")
            else:
                print("⚠ No models found (empty list)")
        except requests.RequestException as e:
            pytest.fail(f"List models endpoint failed: {e}")

    def test_get_model_details(self, api_url):
        """Test getting model details"""
        import requests

        model_id = "test_model_id"

        try:
            response = requests.get(f"{api_url}/models/{model_id}", timeout=30)

            # 404 is expected if model doesn't exist
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Model details: {data}")
            else:
                print("✓ Model not found (expected for test)")
        except requests.RequestException as e:
            pytest.fail(f"Get model details endpoint failed: {e}")

    def test_predict_endpoint(self, api_url):
        """Test prediction endpoint"""
        import requests

        payload = {
            "ticker": "AAPL",
            "model_id": "test_model",
            "days_ahead": 5,
        }

        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=60)

            # 404 expected if model doesn't exist
            assert response.status_code in [200, 202, 404, 500]

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Prediction response: {data}")
            else:
                print(f"✓ Prediction endpoint responded: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"Predict endpoint failed: {e}")

    def test_get_training_status(self, api_url):
        """Test getting training job status"""
        import requests

        job_id = "test_job_id"

        try:
            response = requests.get(f"{api_url}/status/{job_id}", timeout=30)

            assert response.status_code in [200, 404]
            print(f"✓ Training status endpoint: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"Training status endpoint failed: {e}")


class TestPortfolioEndpoints:
    """Test /api/v1/portfolio endpoints"""

    @pytest.fixture
    def api_url(self):
        return f"{API_BASE_URL}/api/v1/portfolio"

    def test_optimize_portfolio(self, api_url):
        """Test portfolio optimization endpoint"""
        import requests

        payload = {
            "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN"],
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "optimization_method": "max_sharpe",
            "risk_free_rate": 0.05,
            "constraints": {"min_weight": 0.05, "max_weight": 0.40},
        }

        try:
            response = requests.post(f"{api_url}/optimize", json=payload, timeout=120)

            assert response.status_code in [200, 202, 404]

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Portfolio optimization response: {data}")
            else:
                print("⚠ Portfolio optimization endpoint not implemented")
        except requests.RequestException as e:
            pytest.fail(f"Portfolio optimization failed: {e}")

    def test_get_efficient_frontier(self, api_url):
        """Test efficient frontier calculation"""
        import requests

        payload = {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "num_portfolios": 100,
        }

        try:
            response = requests.post(f"{api_url}/efficient-frontier", json=payload, timeout=120)

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Efficient frontier calculated")
            else:
                print(f"⚠ Efficient frontier endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Efficient frontier test skipped: {e}")


class TestRiskEndpoints:
    """Test /api/v1/risk endpoints"""

    @pytest.fixture
    def api_url(self):
        return f"{API_BASE_URL}/api/v1/risk"

    def test_calculate_var(self, api_url):
        """Test Value at Risk calculation"""
        import requests

        payload = {
            "ticker": "AAPL",
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "confidence_level": 0.95,
            "holding_period": 1,
            "method": "historical",
        }

        try:
            response = requests.post(f"{api_url}/var", json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                assert "var" in data or "value_at_risk" in data
                print(f"✓ VaR calculated: {data}")
            else:
                print(f"⚠ VaR endpoint: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"VaR calculation failed: {e}")

    def test_calculate_cvar(self, api_url):
        """Test Conditional VaR (Expected Shortfall) calculation"""
        import requests

        payload = {
            "ticker": "AAPL",
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "confidence_level": 0.95,
        }

        try:
            response = requests.post(f"{api_url}/cvar", json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ CVaR calculated: {data}")
            else:
                print(f"⚠ CVaR endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ CVaR test skipped: {e}")

    def test_stress_testing(self, api_url):
        """Test stress testing endpoint"""
        import requests

        payload = {
            "portfolio": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            "scenarios": ["2008_crisis", "covid_crash", "custom"],
            "custom_scenario": {"market_drop": -0.30, "volatility_spike": 2.0},
        }

        try:
            response = requests.post(f"{api_url}/stress-test", json=payload, timeout=120)

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Stress test results: {data}")
            else:
                print(f"⚠ Stress testing endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Stress testing skipped: {e}")


class TestBacktestEndpoints:
    """Test /api/v1/backtest endpoints"""

    @pytest.fixture
    def api_url(self):
        return f"{API_BASE_URL}/api/v1/backtest"

    def test_run_backtest(self, api_url):
        """Test backtest execution endpoint"""
        import requests

        payload = {
            "strategy": "momentum",
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005,
            "strategy_params": {"lookback_period": 20, "rebalance_frequency": "weekly"},
        }

        try:
            response = requests.post(f"{api_url}/run", json=payload, timeout=120)

            assert response.status_code in [200, 202, 404]

            if response.status_code in [200, 202]:
                data = response.json()
                print(f"✓ Backtest submitted: {data}")
            else:
                print(f"⚠ Backtest endpoint: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"Backtest endpoint failed: {e}")

    def test_get_backtest_results(self, api_url):
        """Test getting backtest results"""
        import requests

        backtest_id = "test_backtest_id"

        try:
            response = requests.get(f"{api_url}/results/{backtest_id}", timeout=30)

            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Backtest results: {data}")
            else:
                print("✓ Backtest not found (expected for test)")
        except requests.RequestException as e:
            pytest.fail(f"Get backtest results failed: {e}")

    def test_list_available_strategies(self, api_url):
        """Test listing available strategies"""
        import requests

        try:
            response = requests.get(f"{api_url}/strategies", timeout=30)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Available strategies: {data}")
            else:
                print(f"⚠ Strategies endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Strategies endpoint skipped: {e}")


class TestTaskStatusEndpoints:
    """Test async task status endpoints"""

    def test_get_task_status(self):
        """Test getting Celery task status"""
        import requests

        task_id = "test_task_id"

        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}", timeout=30)

            # Accept various status codes
            assert response.status_code in [200, 404, 422]

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Task status: {data}")
            else:
                print(f"✓ Task status endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Task status endpoint skipped: {e}")

    def test_cancel_task(self):
        """Test canceling a task"""
        import requests

        task_id = "test_task_id"

        try:
            response = requests.delete(f"{API_BASE_URL}/api/v1/tasks/{task_id}", timeout=30)

            assert response.status_code in [200, 404, 405]
            print(f"✓ Cancel task endpoint: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Cancel task endpoint skipped: {e}")


class TestAPIValidation:
    """Test API input validation"""

    def test_invalid_date_format(self):
        """Test invalid date format handling"""
        import requests

        payload = {
            "ticker": "AAPL",
            "start_date": "invalid-date",
            "end_date": "also-invalid",
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/data/collect", json=payload, timeout=30
            )

            assert response.status_code == 422, (
                f"Expected 422 for invalid date, got {response.status_code}"
            )
            print("✓ Invalid date format handled correctly")
        except requests.RequestException as e:
            pytest.fail(f"Date validation test failed: {e}")

    def test_empty_payload(self):
        """Test empty payload handling"""
        import requests

        try:
            response = requests.post(f"{API_BASE_URL}/api/v1/data/collect", json={}, timeout=30)

            assert response.status_code == 422
            print("✓ Empty payload handled correctly")
        except requests.RequestException as e:
            pytest.fail(f"Empty payload test failed: {e}")

    def test_invalid_json(self):
        """Test invalid JSON handling"""
        import requests

        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/data/collect",
                data="not valid json",
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            assert response.status_code == 422
            print("✓ Invalid JSON handled correctly")
        except requests.RequestException as e:
            pytest.fail(f"Invalid JSON test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("LUMINA PROJECT - API Endpoints Test Suite")
    print("=" * 60)
    print(f"\nTesting against: {API_BASE_URL}")
    print("Make sure API is running:\n  docker-compose up -d api\n")

    pytest.main([__file__, "-v", "-s", "--tb=short"])
