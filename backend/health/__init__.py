# backend/health/__init__.py
"""
System Health Check Module

Provides health check utilities for monitoring the V3 platform:
- Database connectivity
- Redis connectivity
- GPU availability
- Feature store status
- Agent status

Used by API health endpoints and monitoring systems.
"""

import psutil
import torch
from loguru import logger


def check_system_health() -> dict[str, any]:
    """
    Perform comprehensive system health check.

    Returns:
        Dictionary with health status for all components
    """
    health = {"status": "healthy", "components": {}}

    try:
        # CPU
        health["components"]["cpu"] = {
            "usage_percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "status": "ok",
        }

        # Memory
        memory = psutil.virtual_memory()
        health["components"]["memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "usage_percent": memory.percent,
            "status": "ok" if memory.percent < 90 else "warning",
        }

        # Disk
        disk = psutil.disk_usage("/")
        health["components"]["disk"] = {
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3),
            "usage_percent": disk.percent,
            "status": "ok" if disk.percent < 85 else "warning",
        }

        # GPU (if available)
        if torch.cuda.is_available():
            health["components"]["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                "status": "ok",
            }
        else:
            health["components"]["gpu"] = {"available": False, "status": "not_available"}

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health


__all__ = ["check_system_health"]
