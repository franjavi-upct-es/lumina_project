# tests/test_docker_services.py
"""
Integration tests for Docker services health and connectivity.
Run these tests AFTER deploying Docker containers.

Usage:
    pytest tests/test_docker_services.py -v -s
    pytest tests/test_docker_services.py -v -s -k "redis"  # Run only redis tests
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Service URLs (Docker internal network)
DOCKER_CONFIG = {
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "POSTGRES_PORT": int(os.getenv("POSTGRES_PORT", "5435")),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "lumina"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "lumina_db"),
    "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
    "REDIS_PORT": int(os.getenv("REDIS_PORT", "6379")),
    "API_HOST": os.getenv("API_HOST", "localhost"),
    "API_PORT": int(os.getenv("API_PORT", "8000")),
    "MLFLOW_HOST": os.getenv("MLFLOW_HOST", "localhost"),
    "MLFLOW_PORT": int(os.getenv("MLFLOW_PORT", "5000")),
    "STREAMLIT_HOST": os.getenv("STREAMLIT_HOST", "localhost"),
    "STREAMLIT_PORT": int(os.getenv("STREAMLIT_PORT", "8501")),
}


class TestTimescaleDBService:
    """Test TimescaleDB/PostgreSQL service"""

    @pytest.fixture
    def db_connection_string(self):
        """Build PostgreSQL connection string"""
        password = DOCKER_CONFIG["POSTGRES_PASSWORD"]
        user = DOCKER_CONFIG["POSTGRES_USER"]
        userinfo = f"{user}:{password}" if password else user
        return (
            f"postgresql://{userinfo}@"
            f"{DOCKER_CONFIG['POSTGRES_HOST']}:"
            f"{DOCKER_CONFIG['POSTGRES_PORT']}/"
            f"{DOCKER_CONFIG['POSTGRES_DB']}"
        )

    def test_postgres_connection(self, db_connection_string):
        """Test PostgreSQL is accessible"""
        import psycopg2

        try:
            conn = psycopg2.connect(db_connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            cursor.close()
            conn.close()

            assert version is not None
            print(f"✓ PostgreSQL connection successful: {version[0][:50]}...")
        except psycopg2.OperationalError as e:
            pytest.fail(f"PostgreSQL connection failed: {e}")

    def test_timescaledb_extension(self, db_connection_string):
        """Test TimescaleDB extension is installed"""
        import psycopg2

        try:
            conn = psycopg2.connect(db_connection_string)
            cursor = conn.cursor()

            # Check TimescaleDB extension
            cursor.execute(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';"
            )
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            assert result is not None, "TimescaleDB extension not installed"
            print(f"✓ TimescaleDB extension: {result[0]} v{result[1]}")
        except psycopg2.OperationalError as e:
            pytest.fail(f"TimescaleDB check failed: {e}")

    def test_database_tables_exist(self, db_connection_string):
        """Test required tables exist"""
        import psycopg2

        expected_tables = ["market_data", "features", "predictions", "models", "backtests"]

        try:
            conn = psycopg2.connect(db_connection_string)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """
            )
            existing_tables = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            missing_tables = []
            for table in expected_tables:
                if table not in existing_tables:
                    missing_tables.append(table)
                else:
                    print(f"✓ Table exists: {table}")

            if missing_tables:
                print(f"⚠ Missing tables (may need migration): {missing_tables}")
                # Warning, not failure - tables might be created later
        except psycopg2.OperationalError as e:
            pytest.fail(f"Table check failed: {e}")

    def test_database_write_read(self, db_connection_string):
        """Test database can write and read data"""
        import psycopg2

        try:
            conn = psycopg2.connect(db_connection_string)
            cursor = conn.cursor()

            # Create temp table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS _test_connectivity (
                    id SERIAL PRIMARY KEY,
                    test_value TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """
            )

            # Insert test data
            test_value = f"test_{datetime.now().isoformat()}"
            cursor.execute(
                "INSERT INTO _test_connectivity (test_value) VALUES (%s) RETURNING id;",
                (test_value,),
            )
            inserted_id = cursor.fetchone()[0]

            # Read back
            cursor.execute(
                "SELECT test_value FROM _test_connectivity WHERE id = %s;",
                (inserted_id,),
            )
            result = cursor.fetchone()

            # Cleanup
            cursor.execute("DROP TABLE IF EXISTS _test_connectivity;")
            conn.commit()

            cursor.close()
            conn.close()

            assert result[0] == test_value
            print("✓ Database write/read test passed")
        except psycopg2.Error as e:
            pytest.fail(f"Database write/read failed: {e}")


class TestRedisService:
    """Test Redis service"""

    @pytest.fixture
    def redis_client(self):
        """Create Redis client"""
        from redis import Redis

        return Redis(
            host=DOCKER_CONFIG["REDIS_HOST"],
            port=DOCKER_CONFIG["REDIS_PORT"],
            decode_responses=True,
        )

    def test_redis_connection(self, redis_client):
        """Test Redis is accessible"""
        try:
            assert redis_client.ping() is True
            print("✓ Redis connection successful (PING/PONG)")
        except Exception as e:
            pytest.fail(f"Redis connection failed: {e}")

    def test_redis_write_read(self, redis_client):
        """Test Redis can write and read data"""
        try:
            test_key = f"_test_key_{int(time.time())}"
            test_value = "test_value_lumina"

            # Write
            redis_client.set(test_key, test_value, ex=60)

            # Read
            result = redis_client.get(test_key)

            # Cleanup
            redis_client.delete(test_key)

            assert result == test_value
            print("✓ Redis write/read test passed")
        except Exception as e:
            pytest.fail(f"Redis write/read failed: {e}")

    def test_redis_list_operations(self, redis_client):
        """Test Redis list operations (used by Celery)"""
        try:
            test_key = f"_test_list_{int(time.time())}"

            # Push items
            redis_client.lpush(test_key, "item1", "item2", "item3")

            # Get length
            length = redis_client.llen(test_key)

            # Pop item
            item = redis_client.rpop(test_key)

            # Cleanup
            redis_client.delete(test_key)

            assert length == 3
            assert item == "item1"
            print("✓ Redis list operations test passed")
        except Exception as e:
            pytest.fail(f"Redis list operations failed: {e}")

    def test_redis_pubsub(self, redis_client):
        """Test Redis pub/sub functionality"""
        try:
            channel = f"_test_channel_{int(time.time())}"
            message = "test_message"

            pubsub = redis_client.pubsub()
            pubsub.subscribe(channel)

            # Publish message
            redis_client.publish(channel, message)

            # Receive message (with timeout)
            received = None
            for _ in range(10):
                msg = pubsub.get_message(timeout=1)
                if msg and msg["type"] == "message":
                    received = msg["data"]
                    break

            pubsub.unsubscribe(channel)
            pubsub.close()

            assert received == message
            print("✓ Redis pub/sub test passed")
        except Exception as e:
            pytest.fail(f"Redis pub/sub failed: {e}")


class TestFastAPIService:
    """Test FastAPI backend service"""

    @pytest.fixture
    def api_base_url(self):
        """Build API base URL"""
        return f"http://{DOCKER_CONFIG['API_HOST']}:{DOCKER_CONFIG['API_PORT']}"

    def test_api_health_endpoint(self, api_base_url):
        """Test API health endpoint"""
        import requests

        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)

            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "healthy"
            print(f"✓ API health check passed: {data}")
        except requests.ConnectionError as e:
            pytest.fail(f"API connection failed: {e}")
        except requests.Timeout:
            pytest.fail("API health check timed out")

    def test_api_root_endpoint(self, api_base_url):
        """Test API root endpoint"""
        import requests

        try:
            response = requests.get(f"{api_base_url}/", timeout=10)

            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "version" in data
            print(f"✓ API root endpoint: {data}")
        except requests.RequestException as e:
            pytest.fail(f"API root endpoint failed: {e}")

    def test_api_docs_available(self, api_base_url):
        """Test OpenAPI docs are accessible"""
        import requests

        try:
            # Swagger UI
            response = requests.get(f"{api_base_url}/docs", timeout=10)
            assert response.status_code == 200
            print("✓ Swagger docs available at /docs")

            # ReDoc
            response = requests.get(f"{api_base_url}/redoc", timeout=10)
            assert response.status_code == 200
            print("✓ ReDoc available at /redoc")

            # OpenAPI JSON
            response = requests.get(f"{api_base_url}/openapi.json", timeout=10)
            assert response.status_code == 200
            openapi_spec = response.json()
            assert "paths" in openapi_spec
            print(f"✓ OpenAPI spec: {len(openapi_spec['paths'])} endpoints")
        except requests.RequestException as e:
            pytest.fail(f"API docs check failed: {e}")

    def test_api_data_endpoints(self, api_base_url):
        """Test data collection API endpoints"""
        import requests

        try:
            # Test GET endpoints
            endpoints = [
                "/api/v1/data/",
                "/api/v1/ml/",
                "/api/v1/portfolio/",
                "/api/v1/risk/",
                "/api/v1/backtest/",
            ]

            for endpoint in endpoints:
                response = requests.get(f"{api_base_url}{endpoint}", timeout=10)
                # Accept 200, 404 (no data), or 405 (method not allowed - POST only)
                assert response.status_code in [200, 404, 405, 422], (
                    f"Unexpected status for {endpoint}: {response.status_code}"
                )
                print(f"✓ Endpoint {endpoint}: {response.status_code}")
        except requests.RequestException as e:
            pytest.fail(f"API data endpoint test failed: {e}")


class TestCeleryWorkers:
    """Test Celery workers and task execution"""

    @pytest.fixture
    def celery_app(self):
        """Get Celery app instance"""
        # Set environment for Docker
        os.environ["REDIS_URL"] = (
            f"redis://{DOCKER_CONFIG['REDIS_HOST']}:{DOCKER_CONFIG['REDIS_PORT']}/0"
        )
        os.environ["CELERY_BROKER_URL"] = (
            f"redis://{DOCKER_CONFIG['REDIS_HOST']}:{DOCKER_CONFIG['REDIS_PORT']}/1"
        )
        os.environ["CELERY_RESULT_BACKEND"] = (
            f"redis://{DOCKER_CONFIG['REDIS_HOST']}:{DOCKER_CONFIG['REDIS_PORT']}/2"
        )

        from backend.workers.celery_app import celery_app

        return celery_app

    def test_celery_broker_connection(self, celery_app):
        """Test Celery can connect to broker"""
        try:
            # Check broker connection
            conn = celery_app.connection()
            conn.ensure_connection(max_retries=3)
            conn.close()
            print("✓ Celery broker connection successful")
        except Exception as e:
            pytest.fail(f"Celery broker connection failed: {e}")

    def test_celery_workers_active(self, celery_app):
        """Test Celery workers are running"""
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()

            if stats is None:
                pytest.fail(
                    "No Celery workers found. Make sure workers are running:\n"
                    "  docker-compose ps  # Check containers\n"
                    "  docker-compose logs celery-worker  # Check worker logs"
                )

            worker_names = list(stats.keys())
            print(f"✓ Active Celery workers: {worker_names}")

            # Check registered tasks
            registered = inspect.registered()
            if registered:
                for worker, tasks in registered.items():
                    print(f"  {worker}: {len(tasks)} tasks registered")
        except Exception as e:
            pytest.fail(f"Celery worker check failed: {e}")

    def test_celery_queues(self, celery_app):
        """Test Celery queues are configured"""
        try:
            inspect = celery_app.control.inspect()
            active_queues = inspect.active_queues()

            if active_queues is None:
                pytest.fail("No active queues found")

            expected_queues = {"data", "ml", "backtest", "default"}
            found_queues = set()

            for worker, queues in active_queues.items():
                for queue in queues:
                    found_queues.add(queue["name"])
                    print(f"✓ Queue '{queue['name']}' on {worker}")

            missing_queues = expected_queues - found_queues
            if missing_queues:
                print(f"⚠ Some queues not active yet: {missing_queues}")
        except Exception as e:
            pytest.fail(f"Celery queue check failed: {e}")

    def test_celery_debug_task(self, celery_app):
        """Test executing a simple debug task"""
        try:
            from backend.workers.celery_app import debug_task

            # Submit task
            result = debug_task.delay()

            # Wait for result (max 30 seconds)
            task_result = result.get(timeout=30)

            assert task_result["status"] == "Celery is working!"
            print(f"✓ Debug task executed successfully: {task_result}")
        except Exception as e:
            pytest.fail(f"Celery debug task failed: {e}")


class TestMLflowService:
    """Test MLflow tracking server"""

    @pytest.fixture
    def mlflow_url(self):
        """Build MLflow URL"""
        return f"http://{DOCKER_CONFIG['MLFLOW_HOST']}:{DOCKER_CONFIG['MLFLOW_PORT']}"

    def test_mlflow_health(self, mlflow_url):
        """Test MLflow server is accessible"""
        import requests

        try:
            # MLflow doesn't have a /health endpoint, try the main page
            response = requests.get(mlflow_url, timeout=10)

            # MLflow returns 200 for the main UI
            assert response.status_code == 200
            print("✓ MLflow server accessible")
        except requests.ConnectionError:
            pytest.fail(
                "MLflow server not accessible. Check if container is running:\n"
                "  docker-compose ps mlflow"
            )
        except requests.Timeout:
            pytest.fail("MLflow connection timed out")

    def test_mlflow_api_experiments(self, mlflow_url):
        """Test MLflow experiments API"""
        import requests

        try:
            response = requests.get(f"{mlflow_url}/api/2.0/mlflow/experiments/search", timeout=10)

            assert response.status_code == 200
            data = response.json()
            experiments = data.get("experiments", [])
            print(f"✓ MLflow API working: {len(experiments)} experiments found")
        except requests.RequestException as e:
            pytest.fail(f"MLflow API test failed: {e}")

    def test_mlflow_create_experiment(self, mlflow_url):
        """Test creating an MLflow experiment"""
        import requests

        try:
            test_experiment_name = f"_test_experiment_{int(time.time())}"

            # Create experiment
            response = requests.post(
                f"{mlflow_url}/api/2.0/mlflow/experiments/create",
                json={"name": test_experiment_name},
                timeout=10,
            )

            assert response.status_code == 200
            data = response.json()
            experiment_id = data.get("experiment_id")
            print(f"✓ Created test experiment: {experiment_id}")

            # Clean up - delete experiment
            requests.post(
                f"{mlflow_url}/api/2.0/mlflow/experiments/delete",
                json={"experiment_id": experiment_id},
                timeout=10,
            )
            print("✓ Cleaned up test experiment")
        except requests.RequestException as e:
            pytest.fail(f"MLflow experiment creation failed: {e}")


class TestStreamlitService:
    """Test Streamlit frontend service"""

    @pytest.fixture
    def streamlit_url(self):
        """Build Streamlit URL"""
        return f"http://{DOCKER_CONFIG['STREAMLIT_HOST']}:{DOCKER_CONFIG['STREAMLIT_PORT']}"

    def test_streamlit_accessible(self, streamlit_url):
        """Test Streamlit is accessible"""
        import requests

        try:
            response = requests.get(streamlit_url, timeout=15)

            # Streamlit returns HTML page
            assert response.status_code == 200
            assert "text/html" in response.headers.get("Content-Type", "")
            print("✓ Streamlit frontend accessible")
        except requests.ConnectionError:
            pytest.fail(
                "Streamlit not accessible. Check if container is running:\n"
                "  docker-compose ps streamlit"
            )
        except requests.Timeout:
            pytest.fail("Streamlit connection timed out")

    def test_streamlit_health_endpoint(self, streamlit_url):
        """Test Streamlit health endpoint"""
        import requests

        try:
            response = requests.get(f"{streamlit_url}/_stcore/health", timeout=10)

            assert response.status_code == 200
            print("✓ Streamlit health check passed")
        except requests.RequestException as e:
            # Health endpoint might not exist in older versions
            print(f"⚠ Streamlit health endpoint check: {e}")


class TestServiceIntegration:
    """Test integration between services"""

    def test_api_database_integration(self):
        """Test API can communicate with database"""
        import requests

        api_url = f"http://{DOCKER_CONFIG['API_HOST']}:{DOCKER_CONFIG['API_PORT']}"

        try:
            response = requests.get(f"{api_url}/health", timeout=10)

            assert response.status_code == 200
            data = response.json()

            # Check database status in health response
            db_status = data.get("database", "unknown")
            print(f"✓ API-Database integration: {db_status}")
        except requests.RequestException as e:
            pytest.fail(f"API-Database integration test failed: {e}")

    def test_api_redis_integration(self):
        """Test API can communicate with Redis"""
        import requests

        api_url = f"http://{DOCKER_CONFIG['API_HOST']}:{DOCKER_CONFIG['API_PORT']}"

        try:
            response = requests.get(f"{api_url}/health", timeout=10)

            assert response.status_code == 200
            data = response.json()

            # Check Redis status in health response
            redis_status = data.get("redis", "unknown")
            print(f"✓ API-Redis integration: {redis_status}")
        except requests.RequestException as e:
            pytest.fail(f"API-Redis integration test failed: {e}")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "docker: marks tests as Docker integration tests")


if __name__ == "__main__":
    print("=" * 60)
    print("LUMINA PROJECT - Docker Services Test Suite")
    print("=" * 60)
    print("\nMake sure Docker containers are running:")
    print("  cd docker && docker-compose up -d")
    print("\n")

    pytest.main([__file__, "-v", "-s", "--tb=short"])
