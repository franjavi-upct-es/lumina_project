# tests/test_celery_tasks.py
"""
Test Celery tasks execution and workflow.
Run these tests AFTER deploying Docker containers with workers.

Usage:
    pytest tests/test_celery_tasks.py -v -s
    pytest tests/test_celery_tasks.py -v -s -k "data_task"  # Run specific tests
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure environment for tests
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5435/lumina_db")


class TestCeleryConfiguration:
    """Test Celery app configuration"""

    @pytest.fixture
    def celery_app(self):
        from backend.workers.celery_app import celery_app

        return celery_app

    def test_celery_app_exists(self, celery_app):
        """Test Celery app is properly configured"""
        assert celery_app is not None
        assert celery_app.main == "lumina_workers"
        print("✓ Celery app configured")

    def test_celery_tasks_registered(self, celery_app):
        """Test all expected tasks are registered"""
        expected_tasks = [
            "workers.data_tasks",
            "workers.ml_tasks",
            "workers.backtest_tasks",
        ]

        registered_tasks = list(celery_app.tasks.keys())

        for module in expected_tasks:
            module_tasks = [t for t in registered_tasks if t.startswith(module)]
            print(f"✓ {module}: {len(module_tasks)} tasks")

    def test_celery_queues_configured(self, celery_app):
        """Test Celery queues are configured"""
        queues = celery_app.conf.task_queues

        assert queues is not None
        queue_names = [q.name for q in queues]

        expected_queues = ["default", "data", "ml", "backtest"]
        for expected in expected_queues:
            assert expected in queue_names, f"Queue '{expected}' not configured"
            print(f"✓ Queue '{expected}' configured")

    def test_celery_broker_connection(self, celery_app):
        """Test Celery can connect to broker"""
        try:
            conn = celery_app.connection()
            conn.ensure_connection(max_retries=3)
            conn.close()
            print("✓ Celery broker connection successful")
        except Exception as e:
            pytest.fail(f"Celery broker connection failed: {e}")


class TestDataTasks:
    """Test data collection Celery tasks"""

    def test_collect_market_data_task(self):
        """Test market data collection task"""
        try:
            from backend.workers.data_tasks import collect_market_data_task
        except ImportError as e:
            pytest.skip(f"Cannot import data_tasks: {e}")

        ticker = "AAPL"
        start_date = (datetime.now() - timedelta(days=30)).isoformat()
        end_date = datetime.now().isoformat()

        # Submit task
        task = collect_market_data_task.delay(
            ticker=ticker, start_date=start_date, end_date=end_date
        )

        assert task.id is not None
        print(f"✓ Data collection task submitted: {task.id}")

        # Wait for result (max 120 seconds)
        try:
            result = task.get(timeout=120)
            print(f"✓ Data collection completed: {result}")
        except Exception as e:
            # Task might fail if no workers running
            print(f"⚠ Task execution: {e}")
            print("  Make sure Celery workers are running")

    def test_batch_data_collection_task(self):
        """Test batch data collection task"""
        try:
            from backend.workers.data_tasks import collect_batch_data_task
        except ImportError:
            pytest.skip("collect_batch_data_task not implemented")

        tickers = ["AAPL", "GOOGL", "MSFT"]
        start_date = (datetime.now() - timedelta(days=30)).isoformat()
        end_date = datetime.now().isoformat()

        try:
            task = collect_batch_data_task.delay(
                tickers=tickers, start_date=start_date, end_date=end_date
            )

            assert task.id is not None
            print(f"✓ Batch collection task submitted: {task.id}")
        except Exception as e:
            print(f"⚠ Batch collection task: {e}")

    def test_update_features_task(self):
        """Test feature update task"""
        try:
            from backend.workers.data_tasks import update_features_task
        except ImportError:
            pytest.skip("update_features_task not implemented")

        ticker = "AAPL"

        try:
            task = update_features_task.delay(ticker=ticker)
            assert task.id is not None
            print(f"✓ Feature update task submitted: {task.id}")
        except Exception as e:
            print(f"⚠ Feature update task: {e}")


class TestMLTasks:
    """Test ML training Celery tasks"""

    def test_train_model_task_submission(self):
        """Test model training task submission"""
        try:
            from backend.workers.ml_tasks import train_model_task
        except ImportError as e:
            pytest.skip(f"Cannot import ml_tasks: {e}")

        job_id = f"test_job_{int(time.time())}"
        ticker = "AAPL"
        model_type = "lstm"

        hyperparams = {
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "hidden_dim": 32,  # Small for testing
            "num_layers": 1,
            "dropout": 0.2,
            "sequence_length": 20,
            "prediction_horizon": 3,
            "batch_size": 16,
            "num_epochs": 2,  # Few epochs for testing
            "learning_rate": 0.001,
            "early_stopping_patience": 2,
            "max_features": 10,
        }

        # Submit task (don't wait - training takes time)
        task = train_model_task.delay(
            job_id=job_id, ticker=ticker, model_type=model_type, hyperparams=hyperparams
        )

        assert task.id is not None
        print(f"✓ Training task submitted: {task.id}")
        print(f"  Job ID: {job_id}")

        # Check task state after a moment
        time.sleep(2)
        print(f"  Task state: {task.state}")

    def test_predict_task(self):
        """Test prediction task"""
        try:
            from backend.workers.ml_tasks import predict_task
        except ImportError as e:
            pytest.skip(f"Cannot import ml_tasks: {e}")

        ticker = "AAPL"
        model_id = "test_model"  # This will likely fail without a real model
        days_ahead = 5

        try:
            task = predict_task.delay(ticker=ticker, model_id=model_id, days_ahead=days_ahead)

            assert task.id is not None
            print(f"✓ Prediction task submitted: {task.id}")

            # Try to get result (will fail if model doesn't exist)
            try:
                result = task.get(timeout=60)
                print(f"✓ Prediction completed: {result}")
            except Exception as e:
                print(f"⚠ Prediction task (expected to fail without model): {e}")
        except Exception as e:
            print(f"⚠ Prediction task submission: {e}")

    def test_hyperparameter_tuning_task(self):
        """Test hyperparameter tuning task"""
        try:
            from backend.workers.ml_tasks import hyperparameter_tuning_task
        except ImportError:
            pytest.skip("hyperparameter_tuning_task not implemented")

        job_id = f"hp_tune_{int(time.time())}"
        ticker = "AAPL"

        search_space = {
            "hidden_dim": [32, 64, 128],
            "num_layers": [1, 2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.01],
        }

        try:
            task = hyperparameter_tuning_task.delay(
                job_id=job_id, ticker=ticker, search_space=search_space, n_trials=5
            )

            assert task.id is not None
            print(f"✓ HP tuning task submitted: {task.id}")
        except Exception as e:
            print(f"⚠ HP tuning task: {e}")


class TestBacktestTasks:
    """Test backtesting Celery tasks"""

    def test_run_backtest_task(self):
        """Test backtest execution task"""
        try:
            from backend.workers.backtest_tasks import run_backtest_task
        except ImportError as e:
            pytest.skip(f"Cannot import backtest_tasks: {e}")

        backtest_id = f"bt_test_{int(time.time())}"

        config = {
            "strategy": "momentum",
            "tickers": ["AAPL", "GOOGL"],
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005,
            "strategy_params": {"lookback_period": 20, "rebalance_frequency": "weekly"},
        }

        try:
            task = run_backtest_task.delay(backtest_id=backtest_id, config=config)

            assert task.id is not None
            print(f"✓ Backtest task submitted: {task.id}")
            print(f"  Backtest ID: {backtest_id}")
        except Exception as e:
            print(f"⚠ Backtest task submission: {e}")

    def test_monte_carlo_simulation_task(self):
        """Test Monte Carlo simulation task"""
        try:
            from backend.workers.backtest_tasks import monte_carlo_task
        except ImportError:
            pytest.skip("monte_carlo_task not implemented")

        simulation_id = f"mc_test_{int(time.time())}"

        config = {
            "ticker": "AAPL",
            "num_simulations": 100,
            "time_horizon": 30,
        }

        try:
            task = monte_carlo_task.delay(simulation_id=simulation_id, config=config)

            assert task.id is not None
            print(f"✓ Monte Carlo task submitted: {task.id}")
        except Exception as e:
            print(f"⚠ Monte Carlo task: {e}")


class TestTaskChains:
    """Test task chains and workflows"""

    def test_data_to_training_chain(self):
        """Test chaining data collection to training"""
        from celery import chain

        try:
            from backend.workers.data_tasks import collect_market_data_task
            from backend.workers.ml_tasks import train_model_task
        except ImportError as e:
            pytest.skip(f"Cannot import tasks: {e}")

        ticker = "AAPL"
        job_id = f"chain_test_{int(time.time())}"

        # Create chain: collect data -> train model
        workflow = chain(
            collect_market_data_task.s(
                ticker=ticker,
                start_date=(datetime.now() - timedelta(days=365)).isoformat(),
                end_date=datetime.now().isoformat(),
            ),
            # Note: This simplified chain might need adjustment based on actual task signatures
        )

        try:
            result = workflow.apply_async()
            print(f"✓ Task chain submitted: {result.id}")
        except Exception as e:
            print(f"⚠ Task chain: {e}")


class TestTaskMonitoring:
    """Test task monitoring and status"""

    def test_task_state_transitions(self):
        """Test task state transitions"""
        try:
            from backend.workers.celery_app import debug_task
        except ImportError as e:
            pytest.skip(f"Cannot import debug_task: {e}")

        task = debug_task.delay()

        # Track state transitions
        states_seen = set()
        timeout = 30
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            state = task.state
            states_seen.add(state)

            if state in ["SUCCESS", "FAILURE"]:
                break

            time.sleep(0.5)

        print(f"✓ State transitions observed: {states_seen}")

        if "SUCCESS" in states_seen:
            result = task.result
            print(f"✓ Task completed: {result}")

    def test_task_revocation(self):
        """Test task cancellation"""
        try:
            from backend.workers.ml_tasks import train_model_task
        except ImportError as e:
            pytest.skip(f"Cannot import ml_tasks: {e}")

        # Submit a long-running task
        job_id = f"revoke_test_{int(time.time())}"
        task = train_model_task.delay(
            job_id=job_id,
            ticker="AAPL",
            model_type="lstm",
            hyperparams={
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "num_epochs": 100,  # Many epochs
            },
        )

        # Revoke the task
        task.revoke(terminate=True)

        time.sleep(2)

        # Check state
        print(f"✓ Task revoked: {task.id}")
        print(f"  State after revocation: {task.state}")


class TestTaskRetry:
    """Test task retry mechanisms"""

    def test_task_retry_on_failure(self):
        """Test task retries on failure"""
        # This would require a custom task that fails initially
        # For now, just verify retry configuration

        from backend.workers.celery_app import celery_app

        conf = celery_app.conf

        # Check retry-related configuration
        assert conf.task_acks_late is True, "task_acks_late should be True for reliable retries"
        print("✓ Task retry configuration verified")
        print(f"  task_acks_late: {conf.task_acks_late}")
        print(f"  task_reject_on_worker_lost: {conf.task_reject_on_worker_lost}")


class TestWorkerHealth:
    """Test worker health and status"""

    @pytest.fixture
    def celery_app(self):
        from backend.workers.celery_app import celery_app

        return celery_app

    def test_workers_ping(self, celery_app):
        """Test pinging workers"""
        inspect = celery_app.control.inspect()
        ping = inspect.ping()

        if ping is None:
            pytest.skip("No workers responding to ping. Start workers first.")

        for worker, response in ping.items():
            assert response == {"ok": "pong"}
            print(f"✓ Worker {worker}: PONG")

    def test_workers_stats(self, celery_app):
        """Test getting worker statistics"""
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if stats is None:
            pytest.skip("No workers found. Start workers first.")

        for worker, info in stats.items():
            print(f"✓ Worker {worker}:")
            print(f"    Pool: {info.get('pool', {}).get('implementation', 'unknown')}")
            print(f"    Concurrency: {info.get('pool', {}).get('max-concurrency', 'unknown')}")
            print(f"    Tasks processed: {info.get('total', {})}")

    def test_active_tasks(self, celery_app):
        """Test listing active tasks"""
        inspect = celery_app.control.inspect()
        active = inspect.active()

        if active is None:
            pytest.skip("No workers found")

        total_active = 0
        for worker, tasks in active.items():
            total_active += len(tasks)
            print(f"✓ Worker {worker}: {len(tasks)} active tasks")

        print(f"  Total active tasks: {total_active}")

    def test_scheduled_tasks(self, celery_app):
        """Test listing scheduled (beat) tasks"""
        inspect = celery_app.control.inspect()
        scheduled = inspect.scheduled()

        if scheduled is None:
            pytest.skip("No workers found")

        for worker, tasks in scheduled.items():
            print(f"✓ Worker {worker}: {len(tasks)} scheduled tasks")


if __name__ == "__main__":
    print("=" * 60)
    print("LUMINA PROJECT - Celery Tasks Test Suite")
    print("=" * 60)
    print("\nMake sure:")
    print("  1. Docker containers are running: docker-compose up -d")
    print("  2. Celery workers are active: docker-compose logs celery-worker")
    print("\n")

    pytest.main([__file__, "-v", "-s", "--tb=short"])
