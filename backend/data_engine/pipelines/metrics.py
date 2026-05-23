# backend/data_engine/pipelines/metrics.py
"""Prometheus metrics for the ingestion pipeline."""

from prometheus_client import Counter, Gauge, Histogram

INGESTION_THROUGHPUT = Counter(
    "ingestion_rows_total",
    "Total rows successfully ingested",
    labelnames=("data_type",),
)
INGESTION_ERRORS = Counter(
    "ingestion_errors_total",
    "Total errors during ingestion",
    labelnames=("data_type", "stage"),
)
INGESTION_LAG = Histogram(
    "ingestion_lag_seconds",
    "Latency from event timestamp to DB commit",
    labelnames=("data_type",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
INGESTION_BUFFER_SIZE = Gauge(
    "ingestion_buffer_size",
    "Current size of in-memory buffer",
    labelnames=("data_type",),
)
INGESTION_BACKPRESSURE = Counter(
    "ingestion_backpressure_events_total",
    "Times backpressure has been triggered",
    labelnames=("data_type",),
)
