# tests/test_xgboost_model.py
"""
Unit tests for XGBoost model
Run with: pytest tests/test_xgboost_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml_engine.models.xgboost_model import XGBoostFinancialModel


class TestXGBoostModel:
    """Tests for XGBoost financial model"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)

        # Generate synthetic price data
        n_samples = 1000
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 10 + 100  # Price around 100

        # Split into train/val
        train_size = int(0.8 * n_samples)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]

        return X_train, y_train, X_val, y_val

    @pytest.fixture
    def model(self):
        """Create model instance"""
        return XGBoostFinancialModel(
            model_name="test_xgboost",
            hyperparameters={
                "n_estimators": 50,  # Small for testing
                "max_depth": 3,
                "learning_rate": 0.1,
            },
        )

    def test_model_initialization(self, model):
        """Test model initializes correctly"""
        assert model.model_name == "test_xgboost"
        assert model.model_type == "xgboost"
        assert model.hyperparameters["n_estimators"] == 50
        assert not model.is_trained

    def test_model_build(self, model):
        """Test model building"""
        built_model = model.build(input_shape=(100, 20))

        assert built_model is not None
        assert model.model is not None

    def test_model_fit(self, model, sample_data):
        """Test model training"""
        X_train, y_train, X_val, y_val = sample_data

        history = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=10,
            verbose=0,
        )

        assert model.is_trained
        assert "train_metrics" in history
        assert "val_metrics" in history
        assert history["train_metrics"]["train_mae"] > 0

    def test_model_predict(self, model, sample_data):
        """Test model prediction"""
        X_train, y_train, X_val, y_val = sample_data

        # Train first
        model.fit(X_train=X_train, y_train=y_train, verbose=0)

        # Predict
        predictions = model.predict(X_val)

        assert predictions is not None
        assert len(predictions) == len(X_val)
        assert predictions.dtype == np.float64

    def test_model_evaluate(self, model, sample_data):
        """Test model evaluation"""
        X_train, y_train, X_val, y_val = sample_data

        # Train
        model.fit(X_train=X_train, y_train=y_train, verbose=0)

        # Evaluate
        metrics = model.evaluate(X_val, y_val)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0

    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction"""
        X_train, y_train, X_val, y_val = sample_data

        # Create DataFrame with feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)

        # Train
        model.fit(X_train=X_train_df, y_train=y_train, verbose=0)

        # Get importance
        importance = model.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == X_train.shape[1]
        assert all(isinstance(v, float) for v in importance.values())

    def test_predict_with_uncertainty(self, model, sample_data):
        """Test prediction with uncertainty"""
        X_train, y_train, X_val, y_val = sample_data

        # Train
        model.fit(X_train=X_train, y_train=y_train, verbose=0)

        # Predict with uncertainty
        predictions, uncertainties = model.predict_with_uncertainty(X_val)

        assert predictions is not None
        assert uncertainties is not None
        assert len(predictions) == len(uncertainties)
        assert len(predictions) == len(X_val)

    def test_model_save_load(self, model, sample_data):
        """Test model saving and loading"""
        X_train, y_train, X_val, y_val = sample_data

        # Train
        model.fit(X_train=X_train, y_train=y_train, verbose=0)

        # Get predictions before save
        pred_before = model.predict(X_val)

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = model.save(tmpdir)

            # Create new model and load
            new_model = XGBoostFinancialModel(model_name="test_xgboost")
            new_model.load(tmpdir)

            # Check loaded model
            assert new_model.is_trained
            assert new_model.model is not None

            # Get predictions after load
            pred_after = new_model.predict(X_val)

            # Predictions should be identical
            np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_cross_validation(self, model, sample_data):
        """Test time series cross-validation"""
        X_train, y_train, X_val, y_val = sample_data

        # Run cross-validation
        cv_results = model.cross_validate(X=X_train, y=y_train, n_splits=3)

        assert "train_mae" in cv_results
        assert "val_mae" in cv_results
        assert "avg_val_mae" in cv_results
        assert "std_val_mae" in cv_results
        assert len(cv_results["train_mae"]) == 3

    def test_multistep_prediction(self):
        """Test multi-step prediction"""
        # Generate multi-step targets
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        n_horizons = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples, n_horizons) * 10 + 100

        # Split
        train_size = int(0.8 * n_samples)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]

        # Create model
        model = XGBoostFinancialModel(
            model_name="test_multistep",
            hyperparameters={"n_estimators": 20, "max_depth": 3},
        )

        # Train
        results = model.fit_multistep(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, verbose=0
        )

        assert model.is_trained
        assert len(model.models_per_horizon) == n_horizons
        assert "horizon_1" in results

        # Predict
        predictions = model.predict(X_val)
        assert predictions.shape == (len(X_val), n_horizons)

    def test_model_with_nan_handling(self, model):
        """Test model handles NaN values properly"""
        # Create data with some NaNs
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)

        # Add some NaNs (XGBoost should handle these)
        X_train[0, 0] = np.nan
        X_train[5, 2] = np.nan

        # This should work (XGBoost handles NaN)
        try:
            model.fit(X_train=X_train, y_train=y_train, verbose=0)
            assert model.is_trained
        except Exception as e:
            pytest.fail(f"Model should handle NaN values: {e}")

    def test_model_metadata(self, model, sample_data):
        """Test model meta_data is created correctly"""
        X_train, y_train, X_val, y_val = sample_data

        # Train
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ticker="AAPL",
            verbose=0,
        )

        # Check meta_data
        assert model.meta_data is not None
        assert model.meta_data.model_name == "test_xgboost"
        assert model.meta_data.model_type == "xgboost"
        assert model.meta_data.ticker == "AAPL"
        assert model.meta_data.training_samples == len(X_train)
        assert model.meta_data.validation_samples == len(X_val)

    def test_model_summary(self, model, sample_data):
        """Test model summary generation"""
        X_train, y_train, X_val, y_val = sample_data

        # Train
        model.fit(X_train=X_train, y_train=y_train, verbose=0)

        # Get summary
        summary = model.get_model_summary()

        assert summary is not None
        assert "model_name" in summary
        assert "model_type" in summary
        assert "is_trained" in summary
        assert summary["is_trained"] is True

    def test_hyperparameters_override(self):
        """Test hyperparameter override works"""
        custom_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.01,
            "min_child_weight": 5,
        }

        model = XGBoostFinancialModel(
            model_name="test_custom", hyperparameters=custom_params
        )

        assert model.hyperparameters["n_estimators"] == 100
        assert model.hyperparameters["max_depth"] == 10
        assert model.hyperparameters["learning_rate"] == 0.01
        assert model.hyperparameters["min_child_weight"] == 5


# Integration tests


class TestXGBoostIntegration:
    """Integration tests with real data"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_train_with_real_data(self):
        """Test training with real market data"""
        from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
        from backend.data_engine.transformers.feature_engineering import FeatureEngineer
        from datetime import datetime, timedelta

        # Collect real data
        collector = YFinanceCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        data = await collector.collect_with_retry(
            ticker="AAPL", start_date=start_date, end_date=end_date
        )

        if data is None or data.height < 100:
            pytest.skip("Not enough data collected")

        # Engineer features
        fe = FeatureEngineer()
        enriched = fe.create_all_features(data)

        # Prepare data
        enriched_pd = enriched.to_pandas()
        feature_cols = fe.get_all_feature_names()[:30]

        # Remove rows with NaN
        enriched_pd = enriched_pd.dropna(subset=feature_cols + ["close"])

        if len(enriched_pd) < 100:
            pytest.skip("Not enough valid samples")

        X = enriched_pd[feature_cols].values
        y = enriched_pd["close"].values

        # Split
        train_size = int(0.8 * len(X))
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]

        # Train model
        model = XGBoostFinancialModel(
            model_name="real_data_test",
            hyperparameters={"n_estimators": 50, "max_depth": 4},
        )

        history = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ticker="AAPL",
            verbose=0,
        )

        # Validate training worked
        assert model.is_trained
        assert history["train_metrics"]["train_mae"] > 0
        assert history["val_metrics"]["val_mae"] > 0

        # Test prediction
        predictions = model.predict(X_val)
        assert len(predictions) == len(X_val)

        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])