# backend/perception/semantic/distillation.py
"""Knowledge distillation: FinBERT teacher -> compact student."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.perception.semantic.distilled_llm import DistilledFinancialEncoder, load_teacher


class DistillationTrainer:
    def __init__(
        self,
        student: DistilledFinancialEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 5e-5,
        device: str = "cuda",
        checkpoint_dir: Path = Path("models/semantic"),
        alpha_mse: float = 0.5,
        alpha_cos: float = 0.3,
        alpha_task: float = 0.2,
    ):
        self.student = student.to(device)
        self.teacher = load_teacher(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.ckpt_dir = checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_mse, self.alpha_cos, self.alpha_task = alpha_mse, alpha_cos, alpha_task
        self.task_head = nn.Linear(student.to_embedding.out_features, 3).to(device)
        self.optim = torch.optim.AdamW(
            list(student.parameters()) + list(self.task_head.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

    def _step(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=mask)
            t_pooled = t_out.last_hidden_state[:, 0]
        s_emb, s_teacher_proj = self.student(input_ids, mask)
        loss_mse = F.mse_loss(s_teacher_proj, t_pooled)
        loss_cos = 1.0 - F.cosine_similarity(s_teacher_proj, t_pooled).mean()
        loss_task = F.cross_entropy(self.task_head(s_emb), labels)
        return self.alpha_mse * loss_mse + self.alpha_cos * loss_cos + self.alpha_task * loss_task

    def fit(self, epochs: int = 10) -> None:
        mlflow.set_experiment("semantic_distillation")
        with mlflow.start_run():
            best = float("inf")
            for epoch in range(epochs):
                self.student.train()
                total, n = 0.0, 0
                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                    self.optim.zero_grad()
                    loss = self._step(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.optim.step()
                    total += loss.item() * batch["input_ids"].size(0)
                    n += batch["input_ids"].size(0)
                train_loss = total / n
                self.student.eval()
                with torch.no_grad():
                    v_total, v_n = 0.0, 0
                    for batch in self.val_loader:
                        v_total += self._step(batch).item() * batch["input_ids"].size(0)
                        v_n += batch["input_ids"].size(0)
                val_loss = v_total / v_n
                mlflow.log_metrics({"train": train_loss, "val": val_loss}, step=epoch)
                logger.info(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
                if val_loss < best:
                    best = val_loss
                    torch.save({"model": self.student.state_dict()}, self.ckpt_dir / "best.pt")
