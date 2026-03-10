# backend/workers/celery_app.py
"""
Celery Application Configuration for V3
=======================================

Configures distributed task processing with:
- Multiple task queues (data, ml, backtest)
- Periodic tasks via Celery Beat
- Task routing and priorities
- Result backend for task tracking

Author: Lumina Quant Lab
Version: 3.0.0
"""

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue
from loguru import logger

from backend.config.settings import get_settings

settings = get_settings()

# ============================================================================
# CELERY APP INITIALIZATION
# ============================================================================

celery_app = Celery(
    "lumina_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "backend.workers.data_tasks",
        "backend.workers.ml_tasks",
        "backend.workers.backtest_tasks",
    ],
)

# ============================================================================
# CELERY CONFIGURATION
# ============================================================================

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
    worker_disable_rate_limits=False,
    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    # Task routing (route to specific queues)
    task_routes={
        "backend.workers.data_tasks.*": {"queue": "data"},
        "backend.workers.ml_tasks.*": {"queue": "ml"},
        "backend.workers.backtest_tasks.*": {"queue": "backtest"},
    },
    # Task queues
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("data", Exchange("data"), routing_key="data"),
        Queue("ml", Exchange("ml"), routing_key="ml"),
        Queue("backtest", Exchange("backtest"), routing_key="backtest"),
        Queue(
            "priority",
            Exchange("priority"),
            routing_key="priority",
            priority=10,
        ),
    ),
    # ========================================================================
    # BEAT SCHEDULE - Periodic Tasks
    # ========================================================================
    beat_schedule={
        # Update market data daily after US market close
        "update-market-data-daily": {
            "task": "backend.workers.data_tasks.update_all_tickers",
            "schedule": crontab(hour=21, minute=30),  # 9:30 PM UTC
        },
        # Update features daily
        "update-features-daily": {
            "task": "backend.workers.data_tasks.update_all_features",
            "schedule": crontab(hour=22, minute=0),  # 10:00 PM UTC
        },
        # Health check every hour
        "health-check-hourly": {
            "task": "backend.workers.data_tasks.health_check_task",
            "schedule": crontab(minute=0),
        },
        # Cleanup old results weekly
        "cleanup-old-results-weekly": {
            "task": "backend.workers.data_tasks.cleanup_old_results",
            "schedule": crontab(day_of_week="sunday", hour=2, minute=0),
        },
        # Update embeddings every 15 minutes (Phase 2+)
        "update-embeddings-15min": {
            "task": "backend.workers.ml_tasks.update_all_embeddings",
            "schedule": crontab(minute="*/15"),
        },
    },
)

logger.info("Celery app configured successfully")
logger.info(f"Broker: {settings.CELERY_BROKER_URL.split('@')[-1]}")
logger.info(f"Backend: {settings.CELERY_RESULT_BACKEND.split('@')[-1]}")


# ============================================================================
# DEBUG TASK
# ============================================================================


@celery_app.task(bind=True)
def debug_task(self):
    """
    Debug task for testing Celery setup

    Usage:
        from backend.workers import debug_task
        result = debug_task.delay()
        print(result.get())
    """
    logger.info(f"Request: {self.request!r}")
    return {
        "status": "Celery is working!",
        "task_id": self.request.id,
        "queue": self.request.delivery_info.get("routing_key"),
    }


if __name__ == "__main__":
    celery_app.start()
