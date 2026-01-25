# backend/ml_engine/models/transformer.py
"""
Transformer model for financial time series prediction
Uses temporal self-attention for capturing long-range dependencies.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from backend.ml_engine.models.base_model import BaseModel, ModelMetadata


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_horizon: int = 5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_horizon = output_horizon

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Output heads
        self.dropout = nn.Dropout(dropout)

        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1),
            nn.Softplus(),
        )

        # Attention pooling
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dictionary with predictions
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)

        # Attention pooling to get sequence representation
        attention_weights = F.softmax(self.attention_pool(encoded), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attention_weights * encoded, dim=1)  # (batch, d_model)

        # Multi-task outputs
        price_pred = self.price_head(context)  # (batch, output_horizon)
        volatility_pred = self.volatility_head(context)  # (batch, 1)

        return {
            "price": price_pred,
            "volatility": volatility_pred,
            "attention_weights": attention_weights.squeeze(-1),
        }


class TransformerFinancialModel(BaseModel):
    """
    Transformer-based financial time series model

    Features:
    - Temporal self-attention
    - Multi-task learning (price + volatility)
    - Attention visualization
    - Handles variable-length sequences
    """

    def __init__(self, model_name: str, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize Transformer model

        Args:
            model_name: Name of model
            hyperparameters: Model hyperparameters
        """
        default_params = {
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "sequence_length": 60,
            "output_horizon": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        super().__init__(
            model_name=model_name, model_type="transformer", hyperparameters=default_params
        )

        self.model: TimeSeriesTransformer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None

        logger.info(f"Initialized Transformer model on {self.device}")

    def build(self, input_shape: tuple[int, ...], **kwargs) -> TimeSeriesTransformer:
        """
        Build Transformer model

        Args:
            input_shape: (batch, seq_len, input_dim)
            **kwargs: Additional parameters

        Returns:
            Transformer model
        """
        input_dim = input_shape[-1]

        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=self.hyperparameters["d_model"],
            nhead=self.hyperparameters["nhead"],
            num_encoder_layers=self.hyperparameters["num_encoder_layers"],
            dim_feedforward=self.hyperparameters["dropout"],
            dropout=self.hyperparameters["dropout"],
            output_horizon=self.hyperparameters["output_horizon"],
        ).to(self.device)

        logger.info(
            f"Built Transformer model with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

        return self.model

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train Transformer model

        Args:
            X_train: Training features (2D or 3D)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters

        Returns:
            Training history
        """
        logger.info("Training Transformer model")

        # Convert to numpy
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Build model if not already built
        if self.model is None:
            # Assume 2D input, reshape to 3D for sequence
            seq_len = self.hyperparameters["sequence_length"]
            n_features = X_train.shape[1]
            self.build(input_shape=(1, seq_len, n_features))

        # Prepare data loaders
        train_dataset = TransformerDataset(
            X_train_np,
            y_train_np,
            sequence_length=self.hyperparameters["sequence_length"],
            output_horizon=self.hyperparameters["output_horizon"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val

            val_dataset = TransformerDataset(
                X_train_np,
                y_val_np,
                sequence_length=self.hyperparameters["sequence_length"],
                output_horizon=self.hyperparameters["output_horizon"],
            )

            val_loader = DataLoader(val_dataset, batch_size=self.hyperparameters["batch_size"])

        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparameters["learning_rate"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        num_epochs = kwargs.get("num_epochs", 50)
        early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

                scheduler.step(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(kwargs.get("checkpoint_path", "best_transformer.pt"))
                else:
                    patiente_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        # Load best model
        if val_loader is not None:
            self.load_checkpoint(kwargs.get("checkpoint_path", "best_transformer.pt"))

        self.is_trained = True
        self.training_history = history

        # Calculate final metrics
        train_metrics = {"train_loss": history["train_loss"][-1]}
        val_metrics = {"val_loss": history["val_loss"][-1]} if val_loader else {}

        # Store metadata
        self.meta_data = ModelMetadata(
            model_id=self.model_name,
            model_name=self.model_name,
            model_type=self.model_type,
            version="1.0",
            ticker=kwargs.get("ticker", "UNKNOWN"),
            training_samples=len(train_dataset),
            validation_samples=len(val_dataset),
            hyperparameters=self.hyperparameters,
            feature_names=self.feature_names,
            num_features=len(self.feature_names),
            prediction_horizon=self.hyperparameters["output_horizon"],
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
        )

        logger.success("Transformer training complete")

        return history

    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(batch_X)

            # Loss (MSE on price prediction)
            loss = F.mse_loss(outputs["price"], batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, X: np.ndarray | pd.DataFrame, **kwargs) -> np.ndarray | dict[str, np.ndarray]:
        """
        Make predictions

        Args:
            X: Input features
            **kwargs: Additional parameters

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        # Create dataset
        dataset = TransformerDataset(
            X_np,
            np.zeros(len(X_np)),  # Dummy targets
            sequence_length=self.hyperparameters["sequence_length"],
            output_horizon=self.hyperparameters["output_horizon"],
        )

        loader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"])

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs["price"].cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def evaluate(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, **kwargs
    ) -> dict[str, float]:
        """Evaluate model"""
        predictions = self.predict(X)

        # Handle multi-step predictions
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]  # Use first horizon

        return self.compute_metrics(y, predictions)

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hyperparameters": self.hyperparameters,
                "feature_names": self.feature_names,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.hyperparameters = checkpoint["hyperparameters"]
        self.feature_names = checkpoint["feature_names"]

    def _save_model(self, path: Path):
        """Save model"""
        self.save_checkpoint(str(path))

    def _load_model(self, path: Path):
        """Load model"""
        self.load_checkpoint(str(path))

    def _log_model_to_mlflow(self):
        """Log to MLflow"""
        import mlflow

        mlflow.pytorch.log_model(self.model, "model")


class TransformerDataset(Dataset):
    """Dataset for Transformer"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 60,
        output_horizon: int = 5,
    ):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.output_horizon = output_horizon

    def __len__(self):
        return len(self.features) - self.sequence_length - self.output_horizon + 1

    def __getitem__(self, idx):
        # Input sequence
        X = self.features[idx : idx + self.sequence_length]

        # Target (future values)
        y_start = idx + self.sequence_length
        y = self.targets[y_start : y_start + self.output_horizon]

        return torch.FloatTensor(X), torch.FloatTensor(y)
