# README.md
<!-- lumina-v2/README.md -->

<div align="center">

# LUMINA 2.0

## A Quantitative Research Platform for Financial Markets Analysis

### *Integrating Statistical Learning, Time Series Econometrics, and Natural Language Processing*

---

[![Version](https://img.shields.io/badge/Version-2.0.0-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)]()

</div>

---

## Abstract

**Lumina 2.0** is a comprehensive quantitative research platform designed for systematic analysis of financial markets. The platform integrates methodologies from statistical learning theory, time series econometrics, portfolio optimization, and computational linguistics to provide a unified framework for market research, strategy development, and risk assessment.

This documentation presents the theoretical foundations underlying each component of the system, with emphasis on mathematical rigor and scientific methodology. The platform implements state-of-the-art techniques from academic finance literature while maintaining practical applicability for real-world trading research.

---

## Table of Contents

1. [Theoretical Framework](#1-theoretical-framework)
2. [Time Series Analysis and Feature Engineering](#2-time-series-analysis-and-feature-engineering)
3. [Machine Learning Models](#3-machine-learning-models)
4. [Risk Analytics and Portfolio Theory](#4-risk-analytics-and-portfolio-theory)
5. [Market Regime Detection](#5-market-regime-detection)
6. [Natural Language Processing for Finance](#6-natural-language-processing-for-finance)
7. [Backtesting Methodology](#7-backtesting-methodology)
8. [Factor Models](#8-factor-models)
9. [System Architecture](#9-system-architecture)
10. [References](#10-references)

---

## 1. Theoretical Framework

### 1.1 Market Hypothesis and Assumptions

The platform operates under a framework that acknowledges varying degrees of market efficiency. While the Efficient Market Hypothesis (EMH) in its strong form suggests that prices fully reflect all available information, Lumina 2.0 is built on the premise that:

1. **Weak-form inefficiencies** may exist and can be exploited through technical analysis
2. **Behavioral anomalies** create temporary mispricings
3. **Information asymmetries** provide opportunities for alpha generation

The fundamental assumption underlying our predictive models follows from the **Adaptive Market Hypothesis** (Lo, 2004), which posits that market efficiency varies over time as market participants adapt to changing conditions.

### 1.2 Price Process Modeling

Asset prices are modeled as stochastic processes. The fundamental price dynamics follow a generalized Itô process:

$$dS_t = \mu(S_t, t)dt + \sigma(S_t, t)dW_t$$

Where:
- $S_t$ denotes the asset price at time $t$
- $\mu(S_t, t)$ represents the drift coefficient (expected return)
- $\sigma(S_t, t)$ represents the diffusion coefficient (volatility)
- $W_t$ is a standard Brownian motion

In discrete time, log-returns are computed as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

The platform assumes returns follow a generalized distribution that accounts for:
- **Fat tails** (leptokurtosis)
- **Volatility clustering** (heteroskedasticity)
- **Asymmetric responses** to positive and negative shocks

---

## 2. Time Series Analysis and Feature Engineering

### 2.1 Technical Indicators: Mathematical Foundations

The platform implements over 100 technical indicators, each grounded in mathematical theory. Below are the core formulations:

#### 2.1.1 Moving Averages

**Simple Moving Average (SMA)**:

$$SMA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}$$

**Exponential Moving Average (EMA)**:

$$EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}$$

Where $\alpha = \frac{2}{n+1}$ is the smoothing factor.

**Weighted Moving Average (WMA)**:

$$WMA_n(t) = \frac{\sum_{i=0}^{n-1}(n-i) \cdot P_{t-i}}{\sum_{i=1}^{n}i}$$

#### 2.1.2 Momentum Oscillators

**Relative Strength Index (RSI)**:

$$RSI = 100 - \frac{100}{1 + RS}$$

Where $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$ over $n$ periods.

The RSI can be interpreted as a bounded transformation of the ratio of positive to negative price movements, providing a measure of momentum bounded in $[0, 100]$.

**Stochastic Oscillator (%K)**:

$$\%K = \frac{C_t - L_n}{H_n - L_n} \times 100$$

Where:
- $C_t$ = current closing price
- $L_n$ = lowest low over $n$ periods
- $H_n$ = highest high over $n$ periods

**MACD (Moving Average Convergence Divergence)**:

$$MACD = EMA_{12}(P) - EMA_{26}(P)$$

$$Signal = EMA_9(MACD)$$

$$Histogram = MACD - Signal$$

#### 2.1.3 Volatility Measures

**Bollinger Bands**:

$$Upper = SMA_n + k \cdot \sigma_n$$

$$Lower = SMA_n - k \cdot \sigma_n$$

Where $\sigma_n$ is the $n$-period rolling standard deviation and $k$ typically equals 2.

**Average True Range (ATR)**:

$$TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$

$$ATR_n = \frac{1}{n}\sum_{i=0}^{n-1}TR_{t-i}$$

**Parkinson Volatility Estimator** (range-based):

$$\sigma_P^2 = \frac{1}{4n\ln(2)}\sum_{i=1}^{n}\left(\ln\frac{H_i}{L_i}\right)^2$$

**Garman-Klass Volatility Estimator**:

$$\sigma_{GK}^2 = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{1}{2}\left(\ln\frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_i}{O_i}\right)^2\right]$$

### 2.2 Stationarity Testing

Before applying statistical models, the platform tests for stationarity using:

**Augmented Dickey-Fuller (ADF) Test**:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p}\delta_i \Delta y_{t-i} + \varepsilon_t$$

The null hypothesis $H_0: \gamma = 0$ (unit root present) is tested against $H_1: \gamma < 0$ (stationarity).

**KPSS Test** (Kwiatkowski-Phillips-Schmidt-Shin):

Tests the null hypothesis of stationarity around a deterministic trend:

$$y_t = \xi t + r_t + \varepsilon_t$$

Where $r_t$ is a random walk.

### 2.3 Feature Normalization

To ensure numerical stability and comparability across features, the platform implements:

**Z-Score Normalization**:

$$z_i = \frac{x_i - \mu}{\sigma}$$

**Min-Max Scaling**:

$$x'_i = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$

**Robust Scaling** (resistant to outliers):

$$x'_i = \frac{x_i - Q_2}{Q_3 - Q_1}$$

Where $Q_1, Q_2, Q_3$ represent the first quartile, median, and third quartile respectively.

---

## 3. Machine Learning Models

### 3.1 Long Short-Term Memory Networks (LSTM)

LSTM networks are employed for sequential prediction tasks, addressing the vanishing gradient problem inherent in standard recurrent neural networks.

#### 3.1.1 LSTM Cell Architecture

The LSTM cell maintains a cell state $C_t$ and hidden state $h_t$ through gating mechanisms:

**Forget Gate**:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate**:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate**:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

Where $\sigma$ denotes the sigmoid activation function and $\odot$ represents element-wise multiplication.

#### 3.1.2 Multi-Variate LSTM Configuration

The platform implements a stacked LSTM architecture with:
- Multiple input features (OHLCV + technical indicators)
- Bidirectional processing for pattern recognition
- Attention mechanisms for temporal weighting

**Self-Attention Mechanism**:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.2 Transformer Architecture

The Temporal Transformer model adapts the attention mechanism for financial time series:

#### 3.2.1 Multi-Head Self-Attention

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$

Where each head is computed as:

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

#### 3.2.2 Positional Encoding

For temporal data, sinusoidal positional encodings are applied:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3.3 Gradient Boosting Machines (XGBoost)

XGBoost implements gradient boosted decision trees with regularization:

**Objective Function**:

$$\mathcal{L}(\phi) = \sum_{i=1}^{n}l(y_i, \hat{y}_i) + \sum_{k=1}^{K}\Omega(f_k)$$

Where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$$

- $T$ = number of leaves
- $w_j$ = leaf weights
- $\gamma, \lambda$ = regularization parameters

**Second-Order Taylor Expansion**:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n}\left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

Where:
- $g_i = \partial_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)})$ (first-order gradient)
- $h_i = \partial^2_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)})$ (second-order gradient)

### 3.4 Ensemble Methods

#### 3.4.1 Meta-Learning Framework

The ensemble combines predictions from multiple base models using a meta-learner:

$$\hat{y}_{ensemble} = g\left(\hat{y}_{LSTM}, \hat{y}_{Transformer}, \hat{y}_{XGBoost}, \mathbf{x}_{meta}\right)$$

Where $g$ is a learned function (typically another gradient boosting model) and $\mathbf{x}_{meta}$ contains meta-features such as prediction disagreement and confidence intervals.

#### 3.4.2 Uncertainty Quantification

Prediction uncertainty is estimated through:

**Ensemble Variance**:

$$Var(\hat{y}) = \frac{1}{M}\sum_{m=1}^{M}(\hat{y}_m - \bar{\hat{y}})^2$$

**Confidence Intervals** (assuming approximate normality):

$$CI_{1-\alpha} = \hat{y} \pm z_{\alpha/2} \cdot \sqrt{Var(\hat{y})}$$

### 3.5 Model Training Methodology

#### 3.5.1 Walk-Forward Optimization

To prevent look-ahead bias, the platform implements walk-forward cross-validation:

```
[Train Window 1][Val 1]
        [Train Window 2][Val 2]
                [Train Window 3][Val 3]
```

#### 3.5.2 Purged Cross-Validation

Following the methodology of de Prado (2018), purged cross-validation eliminates data leakage:

1. **Embargo period**: Gap between training and test sets
2. **Purging**: Removal of samples from training that overlap with test labels

**Combinatorial Purged Cross-Validation (CPCV)**:

Number of possible combinations:

$$\binom{n}{k} \cdot \binom{n-k}{k} \cdot ... $$

### 3.6 Model Evaluation Metrics

**Root Mean Square Error (RMSE)**:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Mean Absolute Percentage Error (MAPE)**:

$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Directional Accuracy**:

$$DA = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}_{sign(\Delta y_i) = sign(\Delta \hat{y}_i)}$$

**Information Coefficient (IC)**:

$$IC = Corr(r_{t+1}, \hat{r}_{t+1})$$

---

## 4. Risk Analytics and Portfolio Theory

### 4.1 Value at Risk (VaR)

VaR quantifies the maximum expected loss at a given confidence level over a specified time horizon.

#### 4.1.1 Parametric VaR (Variance-Covariance Method)

Assuming normally distributed returns:

$$VaR_\alpha = \mu - z_\alpha \cdot \sigma$$

For portfolio:

$$VaR_p = z_\alpha \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$$

Where:
- $\mathbf{w}$ = portfolio weights vector
- $\Sigma$ = covariance matrix of returns
- $z_\alpha$ = quantile of standard normal distribution

#### 4.1.2 Historical VaR

$$VaR_\alpha = -Percentile(R, \alpha \times 100)$$

Where $R$ represents the historical return distribution.

#### 4.1.3 Cornish-Fisher VaR (Modified VaR)

Adjusts for skewness and kurtosis:

$$z_{CF} = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)(K-3) - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$

Where $S$ = skewness and $K$ = kurtosis.

#### 4.1.4 Monte Carlo VaR

Generate $N$ scenarios from estimated distribution:

$$\{r_1, r_2, ..., r_N\} \sim f(r; \hat{\theta})$$

$$VaR_\alpha = -Percentile(\{r_i\}_{i=1}^{N}, \alpha \times 100)$$

### 4.2 Conditional Value at Risk (CVaR / Expected Shortfall)

CVaR measures the expected loss given that VaR is exceeded:

$$CVaR_\alpha = E[L | L > VaR_\alpha] = \frac{1}{1-\alpha}\int_\alpha^1 VaR_u \, du$$

For continuous distributions:

$$CVaR_\alpha = \frac{1}{\alpha}\int_0^\alpha VaR_u \, du$$

**Properties**:
- CVaR is a coherent risk measure (subadditive, positive homogeneous, translation invariant, monotonic)
- CVaR ≥ VaR for any confidence level

### 4.3 Drawdown Analysis

**Maximum Drawdown (MDD)**:

$$MDD = \max_{t \in [0,T]}\left[\max_{s \in [0,t]}V_s - V_t\right]$$

**Calmar Ratio**:

$$Calmar = \frac{CAGR}{|MDD|}$$

**Ulcer Index** (pain index):

$$UI = \sqrt{\frac{1}{n}\sum_{i=1}^{n}D_i^2}$$

Where $D_i$ represents the percentage drawdown at time $i$.

### 4.4 Modern Portfolio Theory

#### 4.4.1 Mean-Variance Optimization (Markowitz, 1952)

**Objective**: Minimize portfolio variance for a target return:

$$\min_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w}$$

Subject to:

$$\mathbf{w}^T \boldsymbol{\mu} = r_{target}$$

$$\mathbf{w}^T \mathbf{1} = 1$$

**Efficient Frontier**: The set of portfolios that maximize expected return for each level of risk.

#### 4.4.2 Risk Parity

Equal risk contribution from each asset:

$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

**Optimization**: Minimize the variance of risk contributions:

$$\min_{\mathbf{w}} \sum_{i=1}^{n}\left(RC_i - \frac{\sigma_p}{n}\right)^2$$

#### 4.4.3 Black-Litterman Model

Combines equilibrium returns with investor views:

**Equilibrium Returns** (reverse optimization):

$$\Pi = \delta \Sigma \mathbf{w}_{mkt}$$

**Posterior Expected Returns**:

$$E[R] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$$

Where:
- $\tau$ = scalar (uncertainty in equilibrium)
- $P$ = view matrix (which assets the views are about)
- $Q$ = view vector (expected returns according to views)
- $\Omega$ = uncertainty matrix for views

### 4.5 Performance Metrics

**Sharpe Ratio** (Sharpe, 1966):

$$SR = \frac{E[R_p] - R_f}{\sigma_p}$$

**Sortino Ratio** (downside deviation):

$$Sortino = \frac{E[R_p] - R_f}{\sigma_d}$$

Where $\sigma_d = \sqrt{E[\min(R - R_f, 0)^2]}$

**Information Ratio**:

$$IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)} = \frac{\alpha}{\sigma_\epsilon}$$

**Treynor Ratio**:

$$Treynor = \frac{E[R_p] - R_f}{\beta_p}$$

---

## 5. Market Regime Detection

### 5.1 Hidden Markov Models (HMM)

Market regimes are modeled as latent states in an HMM framework.

#### 5.1.1 Model Specification

**State Transition Matrix**:

$$A = \{a_{ij}\}, \quad a_{ij} = P(S_{t+1} = j | S_t = i)$$

**Emission Distribution** (Gaussian):

$$P(O_t | S_t = k) = \mathcal{N}(\mu_k, \sigma_k^2)$$

**Model Parameters**: $\lambda = (A, B, \pi)$
- $A$: Transition probabilities
- $B$: Emission parameters
- $\pi$: Initial state distribution

#### 5.1.2 Inference Algorithms

**Forward Algorithm** (likelihood computation):

$$\alpha_t(i) = P(O_1, ..., O_t, S_t = i | \lambda)$$

$$\alpha_t(j) = \left[\sum_{i=1}^{N}\alpha_{t-1}(i)a_{ij}\right]b_j(O_t)$$

**Viterbi Algorithm** (most likely state sequence):

$$\delta_t(j) = \max_{i}\left[\delta_{t-1}(i)a_{ij}\right]b_j(O_t)$$

**Baum-Welch Algorithm** (parameter estimation via EM):

E-step: Compute expected state occupancies
M-step: Update parameters to maximize expected log-likelihood

### 5.2 Regime Identification

The platform identifies three primary market regimes:

| Regime | Characteristics | Typical μ | Typical σ |
|--------|----------------|-----------|-----------|
| **Bull** | Positive drift, low volatility | > 0 | Low |
| **Bear** | Negative drift, high volatility | < 0 | High |
| **Sideways** | Near-zero drift, moderate volatility | ≈ 0 | Medium |

### 5.3 Volatility Regime Models

**GARCH(p,q)** (Bollerslev, 1986):

$$\sigma_t^2 = \omega + \sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2 + \sum_{j=1}^{p}\beta_j\sigma_{t-j}^2$$

**GJR-GARCH** (asymmetric effects):

$$\sigma_t^2 = \omega + \sum_{i=1}^{q}(\alpha_i + \gamma_i I_{t-i})\varepsilon_{t-i}^2 + \sum_{j=1}^{p}\beta_j\sigma_{t-j}^2$$

Where $I_{t-i} = 1$ if $\varepsilon_{t-i} < 0$ (leverage effect).

---

## 6. Natural Language Processing for Finance

### 6.1 Sentiment Analysis Framework

The platform processes textual data from multiple sources (news, social media, earnings calls) to extract sentiment signals.

#### 6.1.1 FinBERT Architecture

FinBERT is a domain-adapted BERT model pre-trained on financial corpora:

**Input Representation**:

$$E = E_{token} + E_{segment} + E_{position}$$

**Self-Attention in BERT**:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Classification Head**:

$$P(sentiment | text) = softmax(W \cdot h_{[CLS]} + b)$$

Where $h_{[CLS]}$ is the hidden state of the [CLS] token.

#### 6.1.2 Sentiment Score Aggregation

Multiple sentiment sources are combined using confidence-weighted averaging:

$$S_{agg} = \frac{\sum_{i=1}^{n}w_i \cdot c_i \cdot s_i}{\sum_{i=1}^{n}w_i \cdot c_i}$$

Where:
- $s_i$ = sentiment score from source $i$
- $c_i$ = confidence score from source $i$
- $w_i$ = source reliability weight

### 6.2 Sentiment-Price Divergence

A key signal is the divergence between sentiment and price movement:

$$Divergence_t = S_t - \frac{r_t - \bar{r}}{\sigma_r}$$

Strong divergence may indicate:
- **Positive divergence**: Bullish sentiment with negative returns → potential reversal
- **Negative divergence**: Bearish sentiment with positive returns → potential reversal

### 6.3 Information Decay

Sentiment information decays over time following an exponential model:

$$w(t) = e^{-\lambda t}$$

Where $\lambda$ is the decay rate parameter, typically calibrated to half-life of information relevance.

---

## 7. Backtesting Methodology

### 7.1 Event-Driven Backtesting

The backtesting engine simulates real market conditions through an event-driven architecture:

```
Event Queue → Strategy Handler → Order Management → Execution Simulation → Portfolio Update
```

### 7.2 Transaction Cost Modeling

#### 7.2.1 Components of Transaction Costs

**Total Cost**:

$$TC = C_{commission} + C_{spread} + C_{market impact} + C_{slippage}$$

**Bid-Ask Spread Cost**:

$$C_{spread} = \frac{1}{2}\left(\frac{P_{ask} - P_{bid}}{P_{mid}}\right)$$

**Market Impact** (Square-root model):

$$MI = \sigma \cdot sign(Q) \cdot \sqrt{\frac{|Q|}{V}} \cdot \gamma$$

Where:
- $Q$ = order size
- $V$ = average daily volume
- $\sigma$ = daily volatility
- $\gamma$ = market impact coefficient

#### 7.2.2 Slippage Modeling

Slippage is modeled as a function of order size and market conditions:

$$Slippage = \alpha + \beta \cdot \frac{Order Size}{ADV} + \varepsilon$$

### 7.3 Walk-Forward Optimization

To ensure out-of-sample validity:

1. **In-Sample**: Optimize strategy parameters
2. **Out-of-Sample**: Test optimized parameters
3. **Roll Forward**: Repeat with new data

**Anchored Walk-Forward**: Training window grows over time
**Rolling Walk-Forward**: Training window remains constant

### 7.4 Monte Carlo Analysis

Generate synthetic price paths to assess strategy robustness:

**Geometric Brownian Motion**:

$$S_{t+\Delta t} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}Z\right]$$

Where $Z \sim N(0,1)$.

**Bootstrap Resampling**:

$$r_t^* \sim \{r_1, r_2, ..., r_T\}$$ (with replacement)

---

## 8. Factor Models

### 8.1 Capital Asset Pricing Model (CAPM)

$$E[R_i] = R_f + \beta_i(E[R_m] - R_f)$$

Where:

$$\beta_i = \frac{Cov(R_i, R_m)}{Var(R_m)}$$

### 8.2 Fama-French Three-Factor Model (1993)

$$R_i - R_f = \alpha_i + \beta_i^{MKT}(R_m - R_f) + \beta_i^{SMB} \cdot SMB + \beta_i^{HML} \cdot HML + \varepsilon_i$$

**Factor Definitions**:
- **SMB** (Small Minus Big): Size premium
- **HML** (High Minus Low): Value premium (Book-to-Market)

### 8.3 Fama-French Five-Factor Model (2015)

Extends the three-factor model with profitability and investment factors:

$$R_i - R_f = \alpha_i + \beta_i^{MKT}MKT + \beta_i^{SMB}SMB + \beta_i^{HML}HML + \beta_i^{RMW}RMW + \beta_i^{CMA}CMA + \varepsilon_i$$

**Additional Factors**:
- **RMW** (Robust Minus Weak): Profitability premium
- **CMA** (Conservative Minus Aggressive): Investment premium

### 8.4 Carhart Four-Factor Model (1997)

Adds momentum to the Fama-French three-factor model:

$$R_i - R_f = \alpha_i + \beta_i^{MKT}MKT + \beta_i^{SMB}SMB + \beta_i^{HML}HML + \beta_i^{MOM}MOM + \varepsilon_i$$

**Momentum Factor (MOM/UMD)**:
- Long: Top 30% past 12-month returns (excluding last month)
- Short: Bottom 30% past 12-month returns

### 8.5 Principal Component Analysis (PCA)

PCA extracts orthogonal factors that explain maximum variance:

**Eigenvalue Decomposition**:

$$\Sigma = V \Lambda V^T$$

Where:
- $\Lambda$ = diagonal matrix of eigenvalues
- $V$ = matrix of eigenvectors (principal components)

**Variance Explained**:

$$VE_k = \frac{\lambda_k}{\sum_{i=1}^{n}\lambda_i}$$

**Factor Loadings**:

$$L = V\sqrt{\Lambda}$$

### 8.6 Risk Factor Attribution

**Factor Risk Decomposition**:

$$\sigma_p^2 = \mathbf{b}^T \Sigma_F \mathbf{b} + \sigma_\varepsilon^2$$

Where:
- $\mathbf{b}$ = vector of factor exposures (betas)
- $\Sigma_F$ = factor covariance matrix
- $\sigma_\varepsilon^2$ = idiosyncratic variance

**Marginal Contribution to Risk**:

$$MCR_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

---

## 9. System Architecture

### 9.1 Computational Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION                        │
│  Sources: Yahoo Finance, Alpha Vantage, FRED, Social Media  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                       │
│  Technical Indicators │ Sentiment Features │ Macro Factors  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING                          │
│         LSTM │ Transformer │ XGBoost │ Ensemble             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RISK ANALYTICS                            │
│       VaR/CVaR │ Drawdown │ Factor Models │ Optimization    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKTESTING                              │
│    Event-Driven │ Walk-Forward │ Monte Carlo │ Validation   │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Data Storage Layer

| Component | Purpose | Technology |
|-----------|---------|------------|
| Time Series DB | High-frequency price data | TimescaleDB |
| Feature Store | Computed indicators | TimescaleDB + Parquet |
| Model Registry | Trained models | MLflow |
| Cache Layer | Real-time data | Redis |

### 9.3 Processing Framework

- **Asynchronous Processing**: FastAPI + Celery for distributed task execution
- **Data Manipulation**: Polars for high-performance DataFrame operations
- **Model Training**: PyTorch with GPU acceleration
- **Numerical Computing**: NumPy + SciPy for scientific computations

---

## 10. References

### Academic Literature

1. **Bollerslev, T.** (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

2. **Black, F., & Litterman, R.** (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.

3. **Carhart, M. M.** (1997). On Persistence in Mutual Fund Performance. *The Journal of Finance*, 52(1), 57-82.

4. **de Prado, M. L.** (2018). *Advances in Financial Machine Learning*. Wiley.

5. **Fama, E. F., & French, K. R.** (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3-56.

6. **Fama, E. F., & French, K. R.** (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1-22.

7. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

8. **Lo, A. W.** (2004). The Adaptive Markets Hypothesis. *The Journal of Portfolio Management*, 30(5), 15-29.

9. **Markowitz, H.** (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.

10. **Sharpe, W. F.** (1966). Mutual Fund Performance. *Journal of Business*, 39(1), 119-138.

11. **Vaswani, A., et al.** (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

### Technical Documentation

- PyTorch Documentation: https://pytorch.org/docs/
- Statsmodels Documentation: https://www.statsmodels.org/
- Scikit-learn Documentation: https://scikit-learn.org/
- Polars Documentation: https://pola.rs/

---

<div align="center">

## License

MIT License © 2025 Lumina Quant Lab

---

**Version**: 2.0.0  
**Last Updated**: January 2025  
**Authors**: Lumina Quant Lab Research Team

</div>
