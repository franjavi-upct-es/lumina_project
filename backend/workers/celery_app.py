# backend/workers/celery_app.py
"""
Celery application configuration for distributed task execution
"""

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue
from loguru import logger

from config.settings import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "lumina_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["workers.data_tasks", "workers.ml_tasks", "workers.backtest_tasks"],
)

# Configuration
celery_app.conf.update(
    # Serialization
    task_serialization="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task settings
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Result backend
    result_expires=86400,  # 24 hours
    result_backend_transport_options={"visibility_timeout": 3600},
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    # Task routing
    task_routes={
        "workers.data_tasks.*": {"queue": "data"},
        "workers.ml_tasks.*": {"queue": "ml"},
        "workers.backtest_tasks.*": {"queue": "backtest"},
    },
    # Task queues
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("data", Exchange("data"), routing_key="data"),
        Queue("ml", Exchange("ml"), routing_key="ml"),
        Queue("backtest", Exchange("backtest"), routing_key="backtest"),
        Queue("priority", Exchange("priority"), routing_key="priority", priority=10),
    ),
    # Beat schedule (periodic tasks)
    beat_schedule={
        # Update market data daily after market close
        "update-market-data": {
            "task": "workers.data_tasks.update_all_tickers",
            "schedule": crontab(hour=21, minute=30),  # 9:30 PM UTC (after US market close)
        },
        # Update features daily
        "update-feature": {
            "task": "workers.data_tasks.update_all_features",
            "schedule": crontab(hour=22, minute=0),
        },
        # Health check every hour
        "health-check": {
            "task": "workers.data_tasks.health_check_task",
            "schedule": crontab(minute=0),
        },
        # Cleanup old jobs weekly
        "cleanup-old-jobs": {
            "task": "workers.data_tasks.cleanup_old_results",
            "schedule": crontab(day_of_week="sunday", hour=2, minute=0),
        },
    },
)

# Logging
logger.info("Celery app configured successfully")
logger.info(f"Broker: {settings.CELERY_BROKER_URL.split('@')[1]}")
logger.info(f"Backend: {settings.CELERY_RESULT_BACKEND.split('@')[1]}")


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup"""
    logger.info(f"Request: {self.request!r}")
    return {"status": "Celery is working!", "task_id": self.request.id}


if __name__ == "__main__":
    celery_app.start()
