# backend/ml_engine/models/lstm_advanced.py
"""
Advanced LSTM model with attention mechanism for financial prediction
Multi-variate, multi-task learning with uncertainty quantification
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class Attention(nn.Module):
    """
    Attention mechanism for LSTM
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)

        return context, attention_weights.squeeze(-1)


class AdvancedLSTM(nn.Module):
    """
    Advanced LSTM with:
    - Multi-variate input (50+ features)
    - Attention mechanism
    - Multi-task output (price, volatility, regime)
    - Dropout for regularization
    - Residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        output_horizon: int = 5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon
        self.bidirectional = bidirectional

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = Attention(lstm_output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Multi-task outputs
        # 1. Price prediction (multi-step)
        self.price_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_horizon),
        )

        # 2. Volatility prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive
        )

        # 3. Regime classification (bull/bear/sideways)
        self.regime_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 3 regimes
        )

        # 4. Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_horizon),
            nn.Softplus(),  # Ensure positive
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Dictionary with predictions
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # LSTM
        lstm_out, (_, _) = self.lstm(x)  # (batch, seq_len, lstm_output_dim)

        # Attention
        context, attention_weights = self.attention(lstm_out)
        context = self.dropout(context)

        # Multi-task outputs
        price_pred = self.price_head(context)  # (batch, output_horizon)
        volatility_pred = self.volatility_head(context)  # (batch, 1)
        regime_logits = self.regime_head(context)  # (batch, 3)
        uncertainty = self.uncertainty_head(context)  # (batch, output_horizon)

        return {
            "price": price_pred,
            "volatility": volatility_pred,
            "regime_logits": regime_logits,
            "regime_probs": F.softmax(regime_logits, dim=1),
            "uncertainty": uncertainty,
            "attention_weights": attention_weights,
        }


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data
    """

    def __init__(
        self,
        data: pl.DataFrame,
        feature_columns: list[str],
        target_column: str = "close",
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        stride: int = 1,
        feature_scaler: dict[str, Any] | None = None,
        target_mode: str = "relative_returns",
    ):
        """
        Args:
            data: Polars DataFrame with features
            feature_columns: List of feature column names
            target_column: Target variable column name
            sequence_length: Length of input sequence
            prediction_horizon: Number of steps to predict ahead
            stride: Step size for creating sequences
        """
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.feature_scaler = feature_scaler
        self.target_mode = target_mode

        # Convert to numpy for faster indexing
        self.features = self.transform_features(
            data,
            feature_columns,
            feature_scaler=feature_scaler,
        )
        self.targets = data.select(target_column).to_numpy().squeeze().astype(np.float32)

        # Create indices for valid sequences
        self.indices = self._create_indices()

        logger.info(f"Created dataset with {len(self.indices)} sequences")

    @staticmethod
    def _prepare_feature_frame(data: pl.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        frame = data.to_pandas().reindex(columns=feature_columns).copy()
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    @classmethod
    def build_feature_scaler(
        cls, data: pl.DataFrame, feature_columns: list[str]
    ) -> dict[str, list[float]]:
        frame = cls._prepare_feature_frame(data, feature_columns)
        fill_values = frame.median().fillna(0.0)
        frame = frame.fillna(fill_values)
        means = frame.mean().fillna(0.0)
        stds = frame.std().replace(0.0, 1.0).fillna(1.0)

        return {
            "feature_columns": list(feature_columns),
            "fill_values": fill_values.astype(float).tolist(),
            "means": means.astype(float).tolist(),
            "stds": stds.astype(float).tolist(),
        }

    @classmethod
    def transform_features(
        cls,
        data: pl.DataFrame,
        feature_columns: list[str],
        feature_scaler: dict[str, Any] | None = None,
    ) -> np.ndarray:
        frame = cls._prepare_feature_frame(data, feature_columns)

        if feature_scaler:
            fill_values = pd.Series(
                feature_scaler.get("fill_values", []),
                index=feature_scaler.get("feature_columns", feature_columns),
                dtype=float,
            ).reindex(feature_columns)
            means = pd.Series(
                feature_scaler.get("means", []),
                index=feature_scaler.get("feature_columns", feature_columns),
                dtype=float,
            ).reindex(feature_columns)
            stds = pd.Series(
                feature_scaler.get("stds", []),
                index=feature_scaler.get("feature_columns", feature_columns),
                dtype=float,
            ).reindex(feature_columns)

            fill_values = fill_values.fillna(0.0)
            means = means.fillna(0.0)
            stds = stds.replace(0.0, 1.0).fillna(1.0)

            frame = frame.fillna(fill_values)
            frame = (frame - means) / stds
        else:
            frame = frame.fillna(0.0)

        return np.asarray(frame.to_numpy(dtype=np.float32), dtype=np.float32)

    def _encode_price_target(self, future_prices: np.ndarray, last_close: float) -> np.ndarray:
        if self.target_mode == "relative_returns":
            base_price = last_close if abs(last_close) > 1e-8 else 1.0
            return ((future_prices / base_price) - 1.0).astype(np.float32)
        return future_prices.astype(np.float32)

    def _create_indices(self) -> list[int]:
        """
        Create valid sequence start indices
        """
        max_start = len(self.features) - self.sequence_length - self.prediction_horizon
        if max_start < 0:
            return []
        return list(range(0, max_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Get a single sequence

        Returns:
            features: (sequence_length, input_dim)
            targets: Dictionary with different target types
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length

        # Get feature sequence
        X = self.features[start_idx:end_idx]

        # Get target window
        target_start = end_idx
        target_end = target_start + self.prediction_horizon
        future_prices = self.targets[target_start:target_end]
        last_close = float(self.targets[end_idx - 1])
        y_prices = self._encode_price_target(future_prices, last_close)

        # Calculate additional targets
        price_path = np.concatenate([[last_close], future_prices])
        returns = np.diff(price_path) / np.where(np.abs(price_path[:-1]) < 1e-8, 1.0, price_path[:-1])
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Regime (simplified: 0=bear, 1=sideways, 2=bull based on overall trend)
        base_price = last_close if abs(last_close) > 1e-8 else 1.0
        overall_return = (future_prices[-1] - last_close) / base_price
        if overall_return < -0.02:
            regime = 0  # Bear
        elif overall_return > 0.02:
            regime = 2  # Bull
        else:
            regime = 1  # Sideways

        targets = {
            "price": torch.FloatTensor(y_prices),
            "volatility": torch.FloatTensor([volatility]),
            "regime": torch.LongTensor([regime]),
            "future_prices": torch.FloatTensor(future_prices.astype(np.float32)),
            "last_close": torch.FloatTensor([last_close]),
        }

        return torch.FloatTensor(X), targets


class LSTMTrainer:
    """
    Trainer for the advanced LSTM model
    """

    def __init__(
        self,
        model: AdvancedLSTM,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_mode: str = "price",
    ):
        self.model = model.to(device)
        self.device = device
        self.target_mode = target_mode
        self.prediction_context: dict[str, Any] = {}
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_price_loss": [],
            "val_price_loss": [],  # type: ignore
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        price_loss_weight: float = 1.0,
        volatility_loss_weight: float = 0.3,
        regime_loss_weight: float = 0.2,
    ) -> dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        total_price_loss = 0.0
        total_vol_loss = 0.0
        total_regime_loss = 0.0
        num_batches = 0

        for batch_features, batch_targets in train_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            target_prices = batch_targets["price"].to(self.device)
            target_vol = batch_targets["volatility"].to(self.device)
            target_regime = batch_targets["regime"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch_features)

            # Calculate losses
            price_loss = F.mse_loss(outputs["price"], target_prices)
            vol_loss = F.mse_loss(outputs["volatility"], target_vol)
            regime_loss = F.cross_entropy(outputs["regime_logits"], target_regime.view(-1))

            # Combined loss
            loss = (
                price_loss_weight * price_loss
                + volatility_loss_weight * vol_loss
                + regime_loss_weight * regime_loss
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_price_loss += price_loss.item()
            total_vol_loss += vol_loss.item()
            total_regime_loss += regime_loss.item()
            num_batches += 1

        return {
            "total_loss": total_loss / num_batches,
            "price_loss": total_price_loss / num_batches,
            "volatility_loss": total_vol_loss / num_batches,
            "regime_loss": total_regime_loss / num_batches,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        price_loss_weight: float = 1.0,
        volatility_loss_weight: float = 0.3,
        regime_loss_weight: float = 0.2,
    ) -> dict[str, float]:
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0.0
        total_price_loss = 0.0
        total_vol_loss = 0.0
        total_regime_loss = 0.0
        num_batches = 0

        for batch_features, batch_targets in val_loader:
            batch_features = batch_features.to(self.device)
            target_prices = batch_targets["price"].to(self.device)
            target_vol = batch_targets["volatility"].to(self.device)
            target_regime = batch_targets["regime"].to(self.device)

            outputs = self.model(batch_features)

            price_loss = F.mse_loss(outputs["price"], target_prices)
            vol_loss = F.mse_loss(outputs["volatility"], target_vol)
            regime_loss = F.cross_entropy(outputs["regime_logits"], target_regime.view(-1))

            loss = (
                price_loss_weight * price_loss
                + volatility_loss_weight * vol_loss
                + regime_loss_weight * regime_loss
            )

            total_loss += loss.item()
            total_price_loss += price_loss.item()
            total_vol_loss += vol_loss.item()
            total_regime_loss += regime_loss.item()
            num_batches += 1

        return {
            "total_loss": total_loss / num_batches,
            "price_loss": total_price_loss / num_batches,
            "volatility_loss": total_vol_loss / num_batches,
            "regime_loss": total_regime_loss / num_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
    ):
        """
        Full training loop with early stopping
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)  # type: ignore

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            scheduler.step(val_metrics["total_loss"])

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}"
            )

            # Track history
            self.history["train_loss"].append(train_metrics["total_loss"])
            self.history["val_loss"].append(val_metrics["total_loss"])
            self.history["train_price_loss"].append(train_metrics["price_loss"])
            self.history["val_price_loss"].append(val_metrics["price_loss"])

            # Early stopping
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
                # Save best model
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.load_checkpoint("best_model.pt")
        return self.history

    @staticmethod
    def decode_price_outputs(
        raw_prices: np.ndarray,
        raw_uncertainty: np.ndarray | None = None,
        last_close: float | np.ndarray | None = None,
        target_mode: str = "price",
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if last_close is None or target_mode != "relative_returns":
            lower = None
            upper = None
            if raw_uncertainty is not None:
                lower = raw_prices - 1.96 * raw_uncertainty
                upper = raw_prices + 1.96 * raw_uncertainty
            return raw_prices, lower, upper, raw_uncertainty

        anchors = np.asarray(last_close, dtype=np.float32)
        if anchors.ndim == 0:
            anchors = anchors.reshape(1, 1)
        elif anchors.ndim == 1:
            anchors = anchors.reshape(-1, 1)

        anchors = np.where(np.abs(anchors) < 1e-8, 1.0, anchors)

        prices = anchors * (1.0 + raw_prices)

        if raw_uncertainty is None:
            return prices, None, None, None

        lower = anchors * (1.0 + raw_prices - 1.96 * raw_uncertainty)
        upper = anchors * (1.0 + raw_prices + 1.96 * raw_uncertainty)
        price_uncertainty = np.abs(anchors * raw_uncertainty)

        return prices, lower, upper, price_uncertainty

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Make predictions with uncertainty

        Args:
            features: (batch, seq_len, input_dim) or (seq_len, input_dim)

        Returns:
            Dictionary with predictions and uncertainties
        """
        self.model.eval()

        # Handle single sequence
        if features.dim() == 2:
            features = features.unsqueeze(0)

        features = features.to(self.device)
        outputs = self.model(features)

        raw_prices = outputs["price"].cpu().numpy()
        raw_uncertainty = outputs["uncertainty"].cpu().numpy()
        target_mode = self.prediction_context.get("target_mode", self.target_mode)
        last_close = self.prediction_context.get("last_close")
        prices, lower, upper, uncertainty = self.decode_price_outputs(
            raw_prices,
            raw_uncertainty,
            last_close=last_close,
            target_mode=target_mode,
        )

        return {
            "price": prices,
            "price_lower": lower if lower is not None else prices,
            "price_upper": upper if upper is not None else prices,
            "volatility": outputs["volatility"].cpu().numpy(),
            "regime_probs": outputs["regime_probs"].cpu().numpy(),
            "uncertainty": uncertainty if uncertainty is not None else raw_uncertainty,
            "raw_price": raw_prices,
            "raw_uncertainty": raw_uncertainty,
            "attention_weights": outputs["attention_weights"].cpu().numpy(),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({"model_state_dict": self.model.state_dict(), "history": self.history}, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
