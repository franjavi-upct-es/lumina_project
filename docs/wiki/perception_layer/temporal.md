# Temporal Encoder: Temporal Fusion Transformer (TFT)

??? note "Relevant source files"

    - [gh:backend/cognition/training/behavioral_cloning.py]
    - [gh:backend/config/constants.py]
    - [gh:backend/perception/temporal/inference.py]
    - [gh:notebooks/02_tft_prototype.ipynb]

The **Temporal Encoder** is the first component of the Perception Layer in the
Chimera architecture. It is responsible for transforming raw multi-variate price
data (OHLCV) into a high-dimensional **Temporal Embedding** ($d=128$). It
utilizes a custom implementation of the **Temporal Fusion Transformer (TFT)**
(Lim et al., 2021), optimized for financial time-series by incorporating gated
residual networks and variable selection to handle noisy market data.

## 1. Architecture Overview

The `TemporalFusionTransformer` class
[gh:backend/perception/temporal/tft_model.py#L240-L244] implements a multi-stage
pipeline designed to extract both long-term dependencies and local patterns from
market bars.

### Key Components

- **Gated Residual Network (GRN):** The fundamental building block
  [gh:backend/perception/temporal/tft_model.py#L28-L32] It provides adaptive
  non-linear processing with a GLU (Gated Linear Unit) to skip unnecessary
  steps, preventing vanishing gradients in deep financial networks.
- **Variable Selection Network (VSN):** Performs instance-specific feature
  selection [gh:backend/perception/temporal/tft_model.py#L82-L86] It assign
  weights to different OHLCV inputs, allowing the model to ignore features
  (e.g., Volume) if they are currently non-informative.
- **LSTM Encoder:** Processes the sequence of variable-selected features to
  capture local temporal context
  [gh:backend/perception/temporal/tft_model.py#L288-L289]
- **Interpretable Multi-Head Attention:** A specialized attention mechanism that
  allows the model to focus on specific past events (e.g., a flash crash 30
  minutes ago) [gh:backend/perception/temporal/tft_model.py#L165-L169]
- **Temporal Embedding Output:** The final state is projected to a
  128-dimensional vector defined by `DIM_PRICE`
  [gh:backend/config/constants.py#L43-L44]
