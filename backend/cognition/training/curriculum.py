# backend/cognition/training/curriculum.py
"""Spartan Curriculum: BC -> Domain Randomization -> Sharpe optimization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mlflow
from loguru import logger

from backend.cognition.agent.ppo_agent import PPOAgent


class Stage(StrEnum):
    BEHAVIORAL_CLONING = "bc"
    DOMAIN_RANDOMIZATION = "dr"
    SHARPE_OPTIMIZATION = "sharpe"


@dataclass
class CurriculumConfig:
    bc_epochs: int = 20
    bc_min_accuracy: float = 0.55
    dr_episodes: int = 500
    dr_min_mean_reward: float = -50.0
    sharpe_episodes: int = 1000
    sharpe_min_ratio: float = -25.0
    checkpoint_dir: Path = Path("models/agent")


class SpartanCurriculum:
    def __init__(self, agent: PPOAgent, config: CurriculumConfig = CurriculumConfig()):
        self.agent = agent
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_stage = Stage.BEHAVIORAL_CLONING

    def run(self, bc_trainer, dr_runner, sharpe_optimizer) -> None:
        mlflow.set_experiment("spartan_curriculum")
        with mlflow.start_run():
            logger.info("=== STAGE 1: Behavioral Cloning ===")
            bc_metrics = bc_trainer.fit(self.agent.policy, epochs=self.config.bc_epochs)
            mlflow.log_metrics({f"bc_{k}": v for k, v in bc_metrics.items()})
            if bc_metrics["accuracy"] < self.config.bc_min_accuracy:
                raise RuntimeError(
                    f"BC gate failed: {bc_metrics['accuracy']:.3f} < {self.config.bc_min_accuracy}"
                )
            self._save("bc.pt")

            logger.info("=== STAGE 2: Domain Randomization ===")
            self.current_stage = Stage.DOMAIN_RANDOMIZATION
            dr_metrics = dr_runner.run(self.agent, episodes=self.config.dr_episodes)
            mlflow.log_metrics({f"dr_{k}": v for k, v in dr_metrics.items()})
            if dr_metrics["mean_reward"] < self.config.dr_min_mean_reward:
                raise RuntimeError(f"DR gate failed: {dr_metrics['mean_reward']:.2f}")
            self._save("dr.pt")

            logger.info("=== STAGE 3: Sharpe Optimization ===")
            self.current_stage = Stage.SHARPE_OPTIMIZATION
            sh_metrics = sharpe_optimizer.run(self.agent, episodes=self.config.sharpe_episodes)
            mlflow.log_metrics({f"sharpe_{k}": v for k, v in sh_metrics.items()})
            if sh_metrics["sharpe"] < self.config.sharpe_min_ratio:
                raise RuntimeError(f"Sharpe gate failed: {sh_metrics['sharpe']:.3f}")
            self._save("final.pt")
            logger.success("Spartan curriculum COMPLETE")

    def _save(self, fname: str) -> None:
        import torch

        path = self.config.checkpoint_dir / fname
        torch.save(self.agent.policy.state_dict(), path)
        mlflow.log_artifact(str(path))
