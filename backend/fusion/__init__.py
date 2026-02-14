# backend/fusion/__init__.py
"""
Deep Fusion Layer - The Thalamus

Layer 2 of the V3 Chimera Architecture

This module implements the Deep Fusion Nexus that combines multi-modal
embeddings into a unified super-state representation.

Architecture Flow:
1. Concatenation: Merge temporal (128d) + semantic (64d) + structural (32d) = 224d
2. Cross-Modal Attention: Learn relevance weights between modalities
3. State Builder: Orchestrate the fusion process and output final state

The fusion layer acts as the "Thalamus" of the AI brain, gating and prioritizing
sensory information from different perception modalities.

Example Scenarios:
- Earnings Call: Semantic embedding "screams" → suppress technical analysis
- Flash Crash: Price embedding shows anomaly → amplify despite no news
- Sector Rotation: Graph shows sector movement → highlight structural signal

Components:
- concatenation: Simple vector concatenation
- attention: Cross-modal attention mechanism (Transformer-based)
- state_builder: Orchestrates fusion pipeline
"""

from backend.fusion.attention import CrossModalAttention, MultiHeadCrossAttention
from backend.fusion.concatenation import ModalityGate, SimpleConcatenation
from backend.fusion.state_builder import (
    FusionConfig,
    FusionStateBuilder,
    ModalityInput,
)

__all__ = [
    # Concatenation
    "SimpleConcatenation",
    "ModalityGate",
    # Attention
    "CrossModalAttention",
    "MultiHeadCrossAttention",
    # State Builder
    "FusionStateBuilder",
    "FusionConfig",
    "ModalityInput",
]
