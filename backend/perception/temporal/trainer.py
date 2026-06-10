# backend/perception/temporal/trainer.py
"""TFT self-supervised trainer."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.perception.temporal.tft_model import TemporalFusionTransformer


class TFTTrainer:
    def __init__(
        self,
        model: TemporalFusionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        device: str = "cuda",
        checkpoint_dir: Path = Path("models/temporal"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.pred_head = nn.Linear(model.output_head.out_features, 1).to(device)
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.pred_head.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            x, y = batch["x"].to(self.device), batch["target_return"].to(self.device)
            self.optimizer.zero_grad()
            emb, _ = self.model(x)
            pred = self.pred_head(emb).squeeze(-1)
            loss = self.loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / n

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            x, y = batch["x"].to(self.device), batch["target_return"].to(self.device)
            emb, _ = self.model(x)
            pred = self.pred_head(emb).squeeze(-1)
            total += self.loss_fn(pred, y).item() * x.size(0)
            n += x.size(0)
        return total / n

    def fit(self, epochs: int = 50, early_stop_patience: int = 5) -> None:
        mlflow.set_experiment("tft_encoder")
        with mlflow.start_run():
            best_val, patience = float("inf"), 0
            for epoch in range(epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                self.scheduler.step()
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                logger.info(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
                if val_loss < best_val:
                    best_val, patience = val_loss, 0
                    ckpt = self.checkpoint_dir / "best.pt"
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "pred_head": self.pred_head.state_dict(),
                        },
                        ckpt,
                    )
                    mlflow.log_artifact(str(ckpt))
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
