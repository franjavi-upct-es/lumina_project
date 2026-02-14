# backend/perception/semantic/__init__.py
"""
Semantic Encoder - Layer 1.B of V3 Chimera Architecture

Extracts semantic embeddings from financial text (news, social media, earnings calls)
using distilled Large Language Models.

Problem Solved:
Simple sentiment scores (0.8 Positive) lose nuance. Compare:
1. "Apple beats earnings by 10%"
2. "Apple beats earnings by 10%, but warns of supply chain catastrophe"

Both might score "Positive", but #2 implies crash. The LLM captures
conditional logic, fear, hesitation - the full semantic context.

Architecture:
- Model: DistilRoBERTa-financial (distilled from RoBERTa-large)
- Latency: <100ms via knowledge distillation
- Output: 64-dimensional semantic embedding
- Vector space: "Recall" close to "Loss", far from "Growth"

Knowledge Distillation:
Small student model mimics large teacher (GPT-4 scale), trading
tiny accuracy for massive speed gains.

Components:
- tokenizer: Financial domain-specific tokenization
- llm_distilled: Distilled transformer model
- vectorizer: Text → 64d embedding pipeline

Example:
    >>> from backend.perception.semantic import SemanticEncoder
    >>>
    >>> encoder = SemanticEncoder()
    >>> text = "Fed raises rates 25bps, signals hawkish stance"
    >>> embedding = encoder.encode(text)  # 64d vector
    >>>
    >>> # Embedding captures: rate hike + hawkish sentiment + central bank context
"""

from backend.perception.semantic.llm_distilled import (
    DistilledLLMEncoder,
    LLMConfig,
)
from backend.perception.semantic.tokenizer import (
    FinancialTokenizer,
    TokenizerConfig,
)
from backend.perception.semantic.vectorizer import (
    SemanticEncoder,
    TextVectorizer,
    VectorizerConfig,
)

__all__ = [
    # Tokenizer
    "FinancialTokenizer",
    "TokenizerConfig",
    # LLM
    "DistilledLLMEncoder",
    "LLMConfig",
    # Vectorizer
    "TextVectorizer",
    "SemanticEncoder",
    "VectorizerConfig",
]
