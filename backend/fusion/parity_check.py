# backend/fusion/parity_check.py
"""Parity Check: prove V3 fusion matches or exceeds V2's simple LSTM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from backend.fusion.nexus import NEXUS_OUTPUT_DIM, DeepFusionNexus

if TYPE_CHECKING:
    from backend.perception.semantic.distilled_llm import DistilledFinancialEncoder
    from backend.perception.structural.gat_model import GraphEncoder
    from backend.perception.temporal.tft_model import TemporalFusionTransformer


@dataclass
class ParityResult:
    v2_sharpe: float
    v3_sharpe: float
    v2_hit_rate: float
    v3_hit_rate: float
    v2_loss: float
    v3_loss: float
    passed: bool

    def report(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"{status}\n"
            f"  V2 Sharpe:    {self.v2_sharpe:.3f}   V3 Sharpe:    {self.v3_sharpe:.3f}\n"
            f"  V2 Hit rate:  {self.v2_hit_rate:.3f}   V3 Hit rate:  {self.v3_hit_rate:.3f}\n"
            f"  V2 Val loss:  {self.v2_loss:.4f}   V3 Val loss:  {self.v3_loss:.4f}"
        )


class V3FusionHead(nn.Module):
    def __init__(self, input_dim: int = NEXUS_OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ParityCheck:
    def __init__(
        self,
        tft: TemporalFusionTransformer,
        semantic: DistilledFinancialEncoder,
        graph: GraphEncoder,
        nexus: DeepFusionNexus,
        v2_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        min_sharpe_improvement: float = 0.0,
    ):
        self.device = device
        self.tft = tft.to(device).eval()
        self.semantic = semantic.to(device).eval()
        self.graph = graph.to(device).eval()
        self.nexus = nexus.to(device).eval()
        for m in (self.tft, self.semantic, self.graph, self.nexus):
            for p in m.parameters():
                p.requires_grad = False
        self.v2 = v2_model.to(device).eval()
        for p in self.v2.parameters():
            p.requires_grad = False
        self.head = V3FusionHead().to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.min_sharpe_improvement = min_sharpe_improvement

    def _v3_forward(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            price_emb, _ = self.tft(batch["ohlcv_window"].to(self.device))
            semantic_emb, _ = self.semantic(
                batch["news_input_ids"].to(self.device),
                batch["news_attention_mask"].to(self.device),
            )
            graph_emb = batch["graph_emb"].to(self.device)
            nexus_out = self.nexus(price_emb, semantic_emb, graph_emb)
        return self.head(nexus_out["market_state"])

    def train_head(self, epochs: int = 10, lr: float = 1e-3) -> float:
        optim = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.HuberLoss(delta=1.0)
        best_val = float("inf")
        for epoch in range(epochs):
            self.head.train()
            for batch in self.train_loader:
                optim.zero_grad()
                pred = self._v3_forward(batch)
                loss = loss_fn(pred, batch["target_return"].to(self.device))
                loss.backward()
                optim.step()
            self.head.eval()
            val_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    pred = self._v3_forward(batch)
                    val_losses.append(loss_fn(pred, batch["target_return"].to(self.device)).item())
            val = float(np.mean(val_losses))
            if val < best_val:
                best_val = val
            logger.info(f"Parity head epoch {epoch}: val_loss={val:.4f}")
        return best_val

    @torch.no_grad()
    def evaluate(self) -> ParityResult:
        v2_preds, v3_preds, targets = [], [], []
        loss_fn = nn.HuberLoss(delta=1.0)
        v2_losses, v3_losses = [], []
        self.head.eval()
        for batch in self.val_loader:
            target = batch["target_return"].to(self.device)
            v2_pred = self.v2(batch["ohlcv_window"].to(self.device))
            if v2_pred.dim() > 1:
                v2_pred = v2_pred.squeeze(-1)
            v3_pred = self._v3_forward(batch)
            v2_preds.append(v2_pred.cpu().numpy())
            v3_preds.append(v3_pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            v2_losses.append(loss_fn(v2_pred, target).item())
            v3_losses.append(loss_fn(v3_pred, target).item())
        v2_arr = np.concatenate(v2_preds)
        v3_arr = np.concatenate(v3_preds)
        y = np.concatenate(targets)

        def sharpe_from_preds(preds, actuals):
            signal = np.sign(preds)
            pnl = signal * actuals
            return float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252 * 390))

        def hit_rate(preds, actuals):
            return float(np.mean(np.sign(preds) == np.sign(actuals)))

        result = ParityResult(
            v2_sharpe=sharpe_from_preds(v2_arr, y),
            v3_sharpe=sharpe_from_preds(v3_arr, y),
            v2_hit_rate=hit_rate(v2_arr, y),
            v3_hit_rate=hit_rate(v3_arr, y),
            v2_loss=float(np.mean(v2_losses)),
            v3_loss=float(np.mean(v3_losses)),
            passed=False,
        )
        result.passed = result.v3_sharpe >= (result.v2_sharpe + self.min_sharpe_improvement)
        return result

    def run(self, head_epochs: int = 10) -> ParityResult:
        mlflow.set_experiment("fusion_parity_check")
        with mlflow.start_run():
            logger.info("Training V3 task head on frozen fusion stack")
            final_loss = self.train_head(epochs=head_epochs)
            logger.info(f"Final head val loss: {final_loss:.4f}")
            result = self.evaluate()
            mlflow.log_metrics(
                {
                    "v2_sharpe": result.v2_sharpe,
                    "v3_sharpe": result.v3_sharpe,
                    "v2_hit_rate": result.v2_hit_rate,
                    "v3_hit_rate": result.v3_hit_rate,
                    "v2_val_loss": result.v2_loss,
                    "v3_val_loss": result.v3_loss,
                    "parity_passed": int(result.passed),
                }
            )
            print(result.report())
            return result


def save_parity_report(result: ParityResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Phase 3 Parity Check Report\n\n")
        f.write(f"**Status:** {'PASSED' if result.passed else 'FAILED'}\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | V2 | V3 | Delta |\n|---|---|---|---|\n")
        f.write(
            f"| Sharpe Ratio | {result.v2_sharpe:.3f} | {result.v3_sharpe:.3f} | {result.v3_sharpe - result.v2_sharpe:+.3f} |\n"
        )
        f.write(
            f"| Hit Rate | {result.v2_hit_rate:.3f} | {result.v3_hit_rate:.3f} | {result.v3_hit_rate - result.v2_hit_rate:+.3f} |\n"
        )
        f.write(
            f"| Val Loss | {result.v2_loss:.4f} | {result.v3_loss:.4f} | {result.v2_loss - result.v3_loss:+.4f} |\n"
        )
