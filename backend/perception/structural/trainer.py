# backend/perception/structural/trainer.py
"""Link-prediction self-supervised trainer for the GATv2 encoder."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from backend.perception.structural.gat_model import GraphEncoder


class GraphTrainer:
    def __init__(
        self,
        model: GraphEncoder,
        data: Data,
        lr: float = 1e-3,
        device: str = "cuda",
        checkpoint_dir: Path = Path("models/structural"),
        val_ratio: float = 0.1,
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.ckpt_dir = checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        num_edges = data.edge_index.size(1)
        perm = torch.randperm(num_edges)
        split = int(num_edges * (1 - val_ratio))
        self.train_edges = data.edge_index[:, perm[:split]]
        self.val_edges = data.edge_index[:, perm[split:]]
        self.train_edge_attr = data.edge_attr[perm[:split]]

    @staticmethod
    def _score(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return (z[edges[0]] * z[edges[1]]).sum(dim=-1)

    def fit(self, epochs: int = 200) -> None:
        mlflow.set_experiment("graph_encoder")
        with mlflow.start_run():
            best = float("inf")
            for epoch in range(epochs):
                self.model.train()
                self.optim.zero_grad()
                z = self.model(self.data.x, self.train_edges, self.train_edge_attr)
                neg_edges = negative_sampling(
                    edge_index=self.train_edges,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=self.train_edges.size(1),
                )
                pos = self._score(z, self.train_edges)
                neg = self._score(z, neg_edges)
                loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos, neg]),
                    torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        z = self.model(self.data.x, self.train_edges, self.train_edge_attr)
                        val_neg = negative_sampling(
                            self.val_edges,
                            self.data.num_nodes,
                            num_neg_samples=self.val_edges.size(1),
                        )
                        val_loss = F.binary_cross_entropy_with_logits(
                            torch.cat([self._score(z, self.val_edges), self._score(z, val_neg)]),
                            torch.cat(
                                [
                                    torch.ones(self.val_edges.size(1), device=self.device),
                                    torch.zeros(val_neg.size(1), device=self.device),
                                ]
                            ),
                        ).item()
                    mlflow.log_metrics({"train": loss.item(), "val": val_loss}, step=epoch)
                    logger.info(f"Epoch {epoch}: train={loss.item():.4f} val={val_loss:.4f}")
                    if val_loss < best:
                        best = val_loss
                        torch.save({"model": self.model.state_dict()}, self.ckpt_dir / "best.pt")
