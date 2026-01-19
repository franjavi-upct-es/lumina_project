# backend/config/logging_config.py
"""
Centralized logging configuration for Lumina Quant Lab
Uses loguru for advanced logging with rotation, formatting, and filtering
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from backend.config.settings import get_settings

settings = get_settings()


def setup_logging(
    log_level: str | None = None,
    log_to_file: bool = True,
    log_dir: Path | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    compression: str = "zip",
    json_logs: bool = False,
):
    """
    Configure logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_dir: Directory for log files (default: ./logs)
        rotation: When to rotate logs (size or time based)
        retention: How long to keep logs
        compression: Compression format for rotated logs
        json_logs: Whether to use JSON formatting
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    if log_level is None:
        log_level = settings.LOG_LEVEL

    # Console handler with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe
    )

    # File handlers
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        # Standard log format for files
        if json_logs:
            file_format = "{message}"

            def json_formatter(record):
                import json

                return (
                    json.dumps(
                        {
                            "timestamp": record["time"].isoformat(),
                            "level": record["level"].name,
                            "module": record["name"],
                            "function": record["function"],
                            "line": record["line"],
                            "message": record["message"],
                            "exception": str(record["exception"]) if record["exception"] else None,
                        }
                    )
                    + "\n"
                )

            serialize = json_formatter
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            )
            serialize = False

        # General application log
        logger.add(
            log_dir / "lumina_{time:YYYY-MM-DD}.log",
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            serialize=serialize,
        )

        # Error log (only errors and above)
        logger.add(
            log_dir / "lumina_errors_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            serialize=serialize,
        )

        # API access log
        logger.add(
            log_dir / "api_access_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            filter=lambda record: "api" in record["name"].lower(),
            serialize=serialize,
        )

        # Backtesting log
        logger.add(
            log_dir / "backtest_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            filter=lambda record: "backtest" in record["name"].lower(),
            serialize=serialize,
        )

        # ML training log
        logger.add(
            log_dir / "ml_training_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
            filter=lambda record: "ml_engine" in record["name"].lower()
            or "train" in record["name"].lower(),
            serialize=serialize,
        )

    logger.info(f"Logging configured: level={log_level}, file_logging={log_to_file}")


def setup_production_logging():
    """
    Production-optimized logging configuration
    - JSON format for easy parsing
    - Structured logging
    - More aggressive rotation
    """
    setup_logging(
        log_level="INFO",
        log_to_file=True,
        log_dir=Path("/var/log/lumina") if Path("/var/log").exists() else Path("logs"),
        rotation="50 MB",
        retention="14 days",
        compression="zip",
        json_logs=True,
    )

    logger.info("Production logging configured")


def setup_development_logging():
    """
    Development-optimized logging configuration
    - More verbose
    - Human-readable format
    - Less aggressive rotation
    """
    setup_logging(
        log_level="DEBUG",
        log_to_file=True,
        log_dir=Path("logs"),
        rotation="200 MB",
        retention="7 days",
        compression=None,
        json_logs=False,
    )

    logger.info("Development logging configured")


def get_logger(name: str = None):
    """
    Get a logger instance

    Args:
        name: Name for the logger (usually __name__)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Context managers for logging


class log_context:
    """
    Context manager for logging with additional context

    Usage:
        with log_context(user_id="123", operation="backtest"):
            logger.info("Running backtest")
    """

    def __init__(self, **kwargs):
        self.context = kwargs

    def __enter__(self):
        logger.configure(extra=self.context)
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.configure(extra={})


class timed_operation:
    """
    Context manager for timing operations

    Usage:
        with timed_operation("Data collection"):
            collect_data()
    """

    def __init__(self, operation_name: str, log_level: str = "INFO"):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger.log(self.log_level, f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            logger.log(self.log_level, f"Completed: {self.operation_name} (took {duration:.2f}s)")
        else:
            logger.error(f"Failed: {self.operation_name} after {duration:.2f}s - {exc_val}")


# Decorators for logging


def log_function_call(level: str = "DEBUG"):
    """
    Decorator to log function calls

    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise

        return wrapper

    return decorator


def log_exceptions(reraise: bool = True):
    """
    Decorator to log exceptions

    Usage:
        @log_exceptions()
        def my_function():
            raise ValueError("Error")
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                if reraise:
                    raise

        return wrapper

    return decorator


# Structured logging helpers


def log_api_request(
    method: str, path: str, status_code: int, duration_ms: float, user_id: str | None = None
):
    """
    Log API request in structured format

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: User ID if authenticated
    """
    logger.bind(
        type="api_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        user_id=user_id,
    ).info(f"{method} {path} - {status_code} ({duration_ms:.2f}ms)")


def log_backtest_run(
    strategy_name: str, tickers: list, start_date: str, end_date: str, metrics: dict
):
    """
    Log backtest run with metrics

    Args:
        strategy_name: Name of strategy
        tickers: List of tickers
        start_date: Start date
        end_date: End date
        metrics: Performance metrics
    """
    logger.bind(
        type="backtest",
        strategy=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        **metrics,
    ).info(
        f"Backtest completed: {strategy_name} | "
        f"Return: {metrics.get('total_return', 0):.2%} | "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
    )


def log_model_training(
    model_name: str,
    model_type: str,
    ticker: str,
    train_samples: int,
    val_loss: float,
    duration_seconds: float,
):
    """
    Log model training completion

    Args:
        model_name: Name of model
        model_type: Type of model
        ticker: Ticker symbol
        train_samples: Number of training samples
        val_loss: Validation loss
        duration_seconds: Training duration
    """
    logger.bind(
        type="model_training",
        model_name=model_name,
        model_type=model_type,
        ticker=ticker,
        train_samples=train_samples,
        val_loss=val_loss,
        duration_seconds=duration_seconds,
    ).info(
        f"Model trained: {model_name} ({model_type}) | "
        f"Ticker: {ticker} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Duration: {duration_seconds:.1f}s"
    )


# Initialize logging based on environment
if settings.ENVIRONMENT == "production":
    setup_production_logging()
elif settings.ENVIRONMENT == "development":
    setup_development_logging()
else:
    setup_logging()  # Default configuration


# Export commonly used items
__all__ = [
    "logger",
    "get_logger",
    "setup_logging",
    "setup_production_logging",
    "setup_development_logging",
    "log_context",
    "timed_operation",
    "log_function_call",
    "log_exceptions",
    "log_api_request",
    "log_backtest_run",
    "log_model_training",
]
