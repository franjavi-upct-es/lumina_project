# backend/ml_engine/evaluation/__init__.py
"""
ML Evaluation Module for Lumina Quant Lab

Provides model evaluation and explainability tools:

ModelMetrics:
- Regression metrics (MAE, MSE, RMSE, R2)
- Classification metrics (Accuracy, F1, AUC)
- Financial metrics (Sharpe, Sortino, Calmar)
- Directional accuracy

SHAPExplainer:
- SHAP value computation
- Feature importance rankings
- Interaction effects
- Visualization utilities

ErrorAnalyzer:
- Error analysis by time period
- Error analysis by market regime
- Outlier detection
- Prediction intervals

Usage:
    from backend.ml_engine.evaluation import ModelMetrics, SHAPExplainer

    # Compute metrics
    metrics = ModelMetrics()
    results = metrics.compute(y_true, y_pred)

    # Explain model
    explainer = SHAPExplainer(model)
    shap_values = explainer.explain(X_test)
"""

from backend.ml_engine.evaluation.error_analysis import ErrorAnalyzer
from backend.ml_engine.evaluation.metrics import FinancialMetrics as ModelMetrics
from backend.ml_engine.evaluation.shap_explainer import SHAPExplainer

__all__ = [
    "ModelMetrics",
    "SHAPExplainer",
    "ErrorAnalyzer",
]
