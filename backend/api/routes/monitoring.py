# backend/api/routes/monitoring.py
"""
Monitoring and Metrics Endpoints

This module provides comprehensive system monitoring including:
- Prometheus metrics exposition
- System health checks
- Performance metrics
- Resource utilization tracking
- Trading performance monitoring

V3 Monitoring Components:
- Agent performance metrics (Sharpe ratio, drawdown, win rate)
- Safety system metrics (circuit breaker triggers, kill switches)
- Feature store metrics (cache hits, embedding freshness)
- Infrastructure metrics (CPU, memory, GPU, database, Redis)
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_async_db, get_redis
from backend.config.settings import get_settings

router = APIRouter()
settings = get_settings()

# ============================================================================
# Response Models
# ============================================================================


class SystemMetrics(BaseModel):
    """System resource metrics"""

    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_available: bool
    gpu_usage_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None


class DatabaseMetrics(BaseModel):
    """Database performance metrics"""

    timestamp: datetime
    connection_pool_size: int
    active_connections: int
    idle_connections: int
    query_count_1min: int
    average_query_time_ms: float
    slow_queries_1min: int
    database_size_mb: float
    timescale_chunks: int


class RedisMetrics(BaseModel):
    """Redis cache metrics"""

    timestamp: datetime
    connected_clients: int
    used_memory_mb: float
    used_memory_peak_mb: float
    total_keys: int
    hit_rate_percent: float
    evicted_keys: int
    expired_keys: int
    ops_per_sec: float


class AgentPerformanceMetrics(BaseModel):
    """Agent trading performance metrics"""

    timestamp: datetime
    total_trades: int
    trades_24h: int
    win_rate: float
    average_return_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_percent: float
    current_drawdown_percent: float
    total_return_percent: float
    portfolio_value: float
    cash_balance: float
    positions_count: int


class SafetyMetrics(BaseModel):
    """Safety system metrics"""

    timestamp: datetime
    circuit_breaker_triggers_24h: int
    kill_switch_activations_24h: int
    safety_overrides_24h: int
    high_uncertainty_events_24h: int
    risk_gate_rejections_24h: int
    defensive_mode_duration_minutes: float
    average_uncertainty_score: float


class FeatureStoreMetrics(BaseModel):
    """Feature store performance metrics"""

    timestamp: datetime
    cache_hit_rate_percent: float
    cache_miss_rate_percent: float
    total_embeddings: int
    temporal_embeddings: int
    semantic_embeddings: int
    structural_embeddings: int
    average_embedding_age_seconds: float
    stale_embeddings: int
    embedding_updates_1min: int


class MonitoringDashboard(BaseModel):
    """Comprehensive monitoring dashboard"""

    timestamp: datetime
    status: str = Field(..., description="Overall system status: 'healthy', 'degraded', 'critical'")
    system_metrics: SystemMetrics
    database_metrics: DatabaseMetrics
    redis_metrics: RedisMetrics
    agent_performance: AgentPerformanceMetrics
    safety_metrics: SafetyMetrics
    feature_store_metrics: FeatureStoreMetrics
    alerts: list[dict[str, str]]


# ============================================================================
# Prometheus Metrics Endpoint
# ============================================================================


@router.get("/metrics/prometheus", response_model=PlainTextResponse)
async def prometheus_metrics(
    db: AsyncSession = Depends(get_async_db), redis: Redis = Depends(get_redis)
):
    """
    Prometheus metrics endpoint.

    Exposes metrics in Prometheus text format for scrapping.

    Metrics Categories:
    - System: CPU, memory, disk, GPU
    - Database: connections, queries, size
    - Redis: memory, keys, hit rate
    - Agent: performance, returns, drawdown
    - Safety: circuit breakers, overrides
    - Feature Store: cache hits, freshness

    Returns:
        PlainTextResponse: Prometheus formatted metrics
    """
    metrics = []

    # Add timestamp
    current_time = datetime.utcnow()

    # ============================================================================
    # System Metrics
    # ============================================================================

    try:
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append("# HELP lumina_cpu_usage_percent CPU usage percentage")
        metrics.append("# TYPE lumina_cpu_usage_percent gauge")
        metrics.append(f"lumina_cpu_usage_percent {cpu_percent}")

        # Memory
        memory = psutil.virtual_memory()
        metrics.append("# HELP lumina_memory_usage_percent Memory usage percentage")
        metrics.append("# TYPE lumina_memory_usage_percent gauge")
        metrics.append(f"lumina_memory_usage_percent {memory.percent}")
        metrics.append(f"lumina_memory_used_bytes {memory.used}")
        metrics.append(f"lumina_memory_total_bytes {memory.total}")

        # Disk
        disk = psutil.disk_usage("/")
        metrics.append("# HELP lumina_disk_usage_percent Disk usage percentage")
        metrics.append("# TYPE lumina_disk_usage_percent gauge")
        metrics.append(f"lumina_disk_usage_percent {disk.percent}")
        metrics.append(f"lumina_disk_used_bytes {disk.used}")
        metrics.append(f"lumina_disk_total_bytes {disk.total}")

    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")

    # ============================================================================
    # GPU Metrics
    # ============================================================================

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # GPU memory
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)

                metrics.append("# HELP lumina_gpu_memory_allocated_bytes GPU memory allocated")
                metrics.append("# TYPE lumina_gpu_memory_allocated_bytes gauge")
                metrics.append(f"lumina_gpu_memory_allocated_bytes{{device='{i}'}} {mem_allocated}")

                metrics.append("# HELP lumina_gpu_memory_reserved_bytes GPU memory reserved")
                metrics.append("# TYPE lumina_gpu_memory_reserved_bytes gauge")
                metrics.append(f"lumina_gpu_memory_reserved_bytes{{device='{i}'}} {mem_reserved}")
    except Exception as e:
        logger.error(f"Error collecting GPU metrics: {e}")

    # ========================================================================
    # Database Metrics
    # ========================================================================

    try:
        # Connection pool
        from backend.db.models import get_async_engine

        engine = get_async_engine()

        # Pool size
        pool_size = engine.pool.size()
        metrics.append("# HELP lumina_db_pool_size Database connection pool size")
        metrics.append("# TYPE lumina_db_pool_size gauge")
        metrics.append(f"lumina_db_pool_size {pool_size}")

        # Get database size
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT pg_database_size(current_database()) as size"))
            db_size = result.scalar()

            metrics.append("# HELP lumina_db_size_bytes Database size in bytes")
            metrics.append("# TYPE lumina_db_size_bytes gauge")
            metrics.append(f"lumina_db_size_bytes {db_size}")
    except Exception as e:
        logger.error(f"Error collecting database metrics: {e}")

    # ========================================================================
    # Redis Metrics
    # ========================================================================

    try:
        # Redis info
        info = redis.info()

        # Memory
        used_memory = info.get("used_memory", 0)
        metrics.append("# HELP lumina_redis_memory_used_bytes Redis memory used")
        metrics.append("# TYPE lumina_redis_memory_used_bytes gauge")
        metrics.append(f"lumina_redis_memory_used_bytes {used_memory}")

        # Connections
        connected_clients = info.get("connected_clients", 0)
        metrics.append("# HELP lumina_redis_connected_clients Redis connected clients")
        metrics.append("# TYPE lumina_redis_connected_clients gauge")
        metrics.append(f"lumina_redis_connected_clients {connected_clients}")

        # Keys
        total_keys = redis.dbsize()
        metrics.append("# HELP lumina_redis_keys_total Total Redis keys")
        metrics.append("# TYPE lumina_redis_keys_total gauge")
        metrics.append(f"lumina_redis_keys_total {total_keys}")

        # Hit rate
        keyspace_hits = info.get("keyspace_hits", 0)
        keyspace_misses = info.get("keyspace_misses", 0)
        total_requests = keyspace_hits + keyspace_misses
        hit_rate = (keyspace_hits / total_requests * 100) if total_requests > 0 else 0

        metrics.append("# HELP lumina_redis_hit_rate_percent Redis cache hit rate")
        metrics.append("# TYPE lumina_redis_hit_rate_percent gauge")
        metrics.append(f"lumina_redis_hit_rate_percent {hit_rate}")
    except Exception as e:
        logger.error(f"Error collecting Redis metrics: {e}")

    # ========================================================================
    # Agent Performance Metrics (Placeholder)
    # ========================================================================

    # TODO: Implement actual agent metrics retrieval
    metrics.append("# HELP lumina_agent_total_trades Total trades executed")
    metrics.append("# TYPE lumina_agent_total_trades counter")
    metrics.append("lumina_agent_total_trades 0")

    metrics.append("# HELP lumina_agent_sharpe_ratio Agent Sharpe ratio")
    metrics.append("# TYPE lumina_agent_sharpe_ratio gauge")
    metrics.append("lumina_agent_sharpe_ratio 0.0")

    metrics.append("# HELP lumina_agent_max_drawdown_percent Maximum drawdown percentage")
    metrics.append("# TYPE lumina_agent_max_drawdown_percent gauge")
    metrics.append("lumina_agent_max_drawdown_percent 0.0")

    # ========================================================================
    # Safety Metrics (Placeholder)
    # ========================================================================

    # TODO: Implement actual safety metrics retrieval
    metrics.append("# HELP lumina_safety_circuit_breaker_triggers Circuit breaker triggers (24h)")
    metrics.append("# TYPE lumina_safety_circuit_breaker_triggers counter")
    metrics.append("lumina_safety_circuit_breaker_triggers 0")

    metrics.append("# HELP lumina_safety_kill_switch_activations Kill switch activations (24h)")
    metrics.append("# TYPE lumina_safety_kill_switch_activations counter")
    metrics.append("lumina_safety_kill_switch_activations 0")

    # ========================================================================
    # Feature Store Metrics (Placeholder)
    # ========================================================================

    # TODO: Implement actual feature store metrics
    metrics.append("# HELP lumina_feature_store_cache_hit_rate Feature store cache hit rate")
    metrics.append("# TYPE lumina_feature_store_cache_hit_rate gauge")
    metrics.append("lumina_feature_store_cache_hit_rate 0.0")

    return PlainTextResponse("\n".join(metrics))


# ============================================================================
# Detailed Metrics Endpoints
# ============================================================================


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get system resource metrics.

    Returns current CPU, memory, disk, and GPU utilization.

    Returns:
        SystemMetrics: System resource metrics
    """
    import psutil

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memory
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)

    # Disk
    disk = psutil.disk_usage("/")
    disk_used_gb = disk.used / (1024**3)
    disk_total_gb = disk.total / (1024**3)

    # GPU
    gpu_available = False
    gpu_usage = None
    gpu_memory_used = None
    gpu_memory_total = None

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            # Get GPU memory for first device
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
    except Exception as e:
        logger.debug(f"GPU metrics not available: {e}")

    return SystemMetrics(
        timestamp=datetime.utcnow(),
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory.percent,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        disk_usage_percent=disk.percent,
        disk_used_gb=disk_used_gb,
        disk_total_gb=disk_total_gb,
        gpu_available=gpu_available,
        gpu_usage_percent=gpu_usage,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
    )


@router.get("/metrics/database", response_model=DatabaseMetrics)
async def get_database_metrics(
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get database performance metrics.

    Returns connection pool stats, query performance, and database size.

    Returns:
        DatabaseMetrics: Database metrics
    """
    from backend.db.models import get_async_engine

    engine = get_async_engine()

    # Connection pool
    pool_size = engine.pool.size()

    # Get database size
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT pg_database_size(current_database()) / (1024*1024) as size_mb")
        )
        db_size_mb = result.scalar()

        # Get number of TimescaleDB chunks
        try:
            result = await conn.execute(text("SELECT COUNT(*) FROM timescaledb_information.chunks"))
            chunk_count = result.scalar()
        except Exception:
            chunk_count = 0

    return DatabaseMetrics(
        timestamp=datetime.utcnow(),
        connection_pool_size=pool_size,
        active_connections=0,  # TODO: Get from pool
        idle_connections=0,  # TODO: Get from pool
        query_count_1min=0,  # TODO: Implement query tracking
        average_query_time_ms=0.0,  # TODO: Implement query tracking
        slow_queries_1min=0,  # TODO: Implement query tracking
        database_size_mb=db_size_mb,
        timescale_chunks=chunk_count,
    )


@router.get("/metrics/redis", response_model=RedisMetrics)
async def get_redis_metrics(
    redis: Redis = Depends(get_redis),
):
    """
    Get Redis cache metrics.

    Returns memory usage, hit rate, and operations per second.

    Returns:
        RedisMetrics: Redis metrics
    """
    # Get Redis info
    info = redis.info()

    # Memory
    used_memory_mb = info.get("used_memory", 0) / (1024**2)
    used_memory_peak_mb = info.get("used_memory_peak", 0) / (1024**2)

    # Keys
    total_keys = redis.dbsize()

    # Hit rate
    keyspace_hits = info.get("keyspace_hits", 0)
    keyspace_misses = info.get("keyspace_misses", 0)
    total_requests = keyspace_hits + keyspace_misses
    hit_rate = (keyspace_hits / total_requests * 100) if total_requests > 0 else 0

    # Evicted/expired
    evicted_keys = info.get("evicted_keys", 0)
    expired_keys = info.get("expired_keys", 0)

    # Operations per second
    ops_per_sec = info.get("instantaneous_ops_per_sec", 0)

    return RedisMetrics(
        timestamp=datetime.utcnow(),
        connected_clients=info.get("connected_clients", 0),
        used_memory_mb=used_memory_mb,
        used_memory_peak_mb=used_memory_peak_mb,
        total_keys=total_keys,
        hit_rate_percent=hit_rate,
        evicted_keys=evicted_keys,
        expired_keys=expired_keys,
        ops_per_sec=ops_per_sec,
    )


@router.get("/metrics/agent", response_model=AgentPerformanceMetrics)
async def get_agent_metrics(
    agent_id: str = Query("default", description="Agent identifier"),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get agent trading performance metrics.

    Returns comprehensive trading statistics including returns,
    risk metrics, and position information.

    Returns:
        AgentPerformanceMetrics: Agent performance metrics
    """
    # TODO: Implement actual agent metrics retrieval

    return AgentPerformanceMetrics(
        timestamp=datetime.utcnow(),
        total_trades=0,
        trades_24h=0,
        win_rate=0.0,
        average_return_percent=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_percent=0.0,
        current_drawdown_percent=0.0,
        total_return_percent=0.0,
        portfolio_value=100000.0,
        cash_balance=100000.0,
        positions_count=0,
    )


@router.get("/metrics/safety", response_model=SafetyMetrics)
async def get_safety_metrics(
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get safety system metrics.

    Returns statistics on circuit breaker triggers, kill switches,
    and safety overrides.

    Returns:
        SafetyMetrics: Safety system metrics
    """
    # TODO: Implement actual safety metrics retrieval

    return SafetyMetrics(
        timestamp=datetime.utcnow(),
        circuit_breaker_triggers_24h=0,
        kill_switch_activations_24h=0,
        safety_overrides_24h=0,
        high_uncertainty_events_24h=0,
        risk_gate_rejections_24h=0,
        defensive_mode_duration_minutes=0.0,
        average_uncertainty_score=0.0,
    )


@router.get("/metrics/feature-store", response_model=FeatureStoreMetrics)
async def get_feature_store_metrics(
    redis: Redis = Depends(get_redis),
):
    """
    Get feature store performance metrics.

    Returns cache hit rates, embedding counts, and freshness metrics.

    Returns:
        FeatureStoreMetrics: Feature store metrics
    """
    # TODO: Implement actual feature store metrics

    # Count embeddings by type
    temporal_keys = redis.keys("embedding:temporal:*")
    semantic_keys = redis.keys("embedding:semantic:*")
    structural_keys = redis.keys("embedding:structural:*")

    return FeatureStoreMetrics(
        timestamp=datetime.utcnow(),
        cache_hit_rate_percent=0.0,
        cache_miss_rate_percent=0.0,
        total_embeddings=len(temporal_keys) + len(semantic_keys) + len(structural_keys),
        temporal_embeddings=len(temporal_keys),
        semantic_embeddings=len(semantic_keys),
        structural_embeddings=len(structural_keys),
        average_embedding_age_seconds=0.0,
        stale_embeddings=0,
        embedding_updates_1min=0,
    )


@router.get("/dashboard", response_model=MonitoringDashboard)
async def get_monitoring_dashboard(
    db: AsyncSession = Depends(get_async_db),
    redis: Redis = Depends(get_redis),
):
    """
    Get comprehensive monitoring dashboard.

    Returns all metrics in a single response for dashboard display.

    Returns:
        MonitoringDashboard: Complete monitoring data
    """
    # Collect all metrics
    system_metrics = await get_system_metrics()
    database_metrics = await get_database_metrics(db)
    redis_metrics = await get_redis_metrics(redis)
    agent_performance = await get_agent_metrics(db=db)
    safety_metrics = await get_safety_metrics(db)
    feature_store_metrics = await get_feature_store_metrics(redis)

    # Determine overall status
    status = "healthy"
    alerts = []

    # Check for issues
    if system_metrics.cpu_usage_percent > 90:
        status = "degraded"
        alerts.append({"level": "warning", "message": "High CPU usage"})

    if system_metrics.memory_usage_percent > 90:
        status = "critical"
        alerts.append({"level": "critical", "message": "High memory usage"})

    if redis_metrics.hit_rate_percent < 50:
        alerts.append({"level": "info", "message": "Low Redis hit rate"})

    return MonitoringDashboard(
        timestamp=datetime.utcnow(),
        status=status,
        system_metrics=system_metrics,
        database_metrics=database_metrics,
        redis_metrics=redis_metrics,
        agent_performance=agent_performance,
        safety_metrics=safety_metrics,
        feature_store_metrics=feature_store_metrics,
        alerts=alerts,
    )
