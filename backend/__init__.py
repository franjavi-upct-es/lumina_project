# backend/__init__.py
"""
Lumina Quant Lab 2.0 - Backend Package
Professional Quantitative Trading Research Platform

This package provides a comprehensive suite of tools for:
- Market data collection and processing
- Feature engineering with 100+ technical indicators
- Machine learning models (LSTM, Transformer, XGBoost)
- Backtesting engine with realistic transaction costs
- Risk analysis and portfolio optimization
- NLP-based sentiment analysis
"""

__version__ = "2.0.0"
__author__ = "Lumina Team"
__license__ = "MIT"

# Package metadata
PACKAGE_INFO = {
    "name": "lumina-backend",
    "version": __version__,
    "description": "Professional Quantitative Trading Research Platform",
    "modules": [
        "api",
        "config",
        "data_engine",
        "ml_engine",
        "quant_engine",
        "backtesting",
        "nlp_engine",
        "workers",
        "db",
    ],
}


def get_version() -> str:
    """Return the current package version."""
    return __version__


def get_package_info() -> dict:
    """Return package metadata information."""
    return PACKAGE_INFO.copy()


__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "get_version",
    "get_package_info",
    "PACKAGE_INFO",
]
