# backend/fusion/trainer.py
"""Supervised trainer for the DeepFusionNexus.

Trains the Nexus to correctly identify market regimes (Bull, Bear, Crash)
based on signatures in the concatenated super-vector. This "warms up" the
Cross-Modal Attention weights so the RL agent receives a stable latent
representation from day one.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.fusion.nexus import DeepFusionNexus


class NexusTrainer:
    def __init__(
        self,
        model: DeepFusionNexus,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 5e-4,
        device: str = "cuda",
        checkpoint_dir: Path = Path("models/fusion"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.ckpt_dir = checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Temporary classification head for the regime task.
        # We don't save this head; we only care about the Nexus backbone.
        self.regime_head = nn.Linear(model.head[-2].out_features, 4).to(device)
        
        self.optim = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.regime_head.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    def fit(self, epochs: int = 50) -> None:
        mlflow.set_experiment("fusion_nexus_warmup")
        with mlflow.start_run():
            best_val_acc = 0.0
            for epoch in range(epochs):
                self.model.train()
                train_loss, train_correct, n_total = 0.0, 0, 0
                
                for p, s, g, regimes in tqdm(self.train_loader, desc=f"Nexus Epoch {epoch}"):
                    p, s, g, regimes = [x.to(self.device) for x in (p, s, g, regimes)]
                    
                    self.optim.zero_grad()
                    out = self.model(p, s, g)
                    logits = self.regime_head(out["market_state"])
                    
                    loss = F.cross_entropy(logits, regimes)
                    loss.backward()
                    self.optim.step()
                    
                    train_loss += loss.item() * p.size(0)
                    train_correct += (logits.argmax(1) == regimes).sum().item()
                    n_total += p.size(0)

                val_loss, val_acc = self._validate()
                mlflow.log_metrics({
                    "train_loss": train_loss / n_total,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

                logger.info(f"Nexus Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.2%}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), self.ckpt_dir / "best_nexus.pt")
                    logger.success(f"Nexus checkpoint saved (acc={val_acc:.2%})")

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for p, s, g, regimes in self.val_loader:
            p, s, g, regimes = [x.to(self.device) for x in (p, s, g, regimes)]
            out = self.model(p, s, g)
            logits = self.regime_head(out["market_state"])
            total_loss += F.cross_entropy(logits, regimes).item() * p.size(0)
            correct += (logits.argmax(1) == regimes).sum().item()
            n += p.size(0)
        return total_loss / n, correct / n
