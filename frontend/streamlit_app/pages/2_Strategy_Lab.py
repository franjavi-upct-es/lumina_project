# frontend/streamlit-app/pages/2_Strategy_Lab.py
"""
Strategy Lab - Build, test, and optimize trading strategies
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import os

# Page config
st.set_page_config(
    page_title="Strategy Lab - Lumina",
    page_icon="🧪",
    layout="wide",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown(
    """
<style>
    .strategy-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .metric-negative {
        color: #f44336;
        font-weight: bold;
    }
    .code-editor {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = {}
if "active_job" not in st.session_state:
    st.session_state.active_job = None

# Header
st.title("🧪 Strategy Lab")
st.markdown("Build, test, and optimize your trading strategies")

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Strategy Builder")

    # Strategy type selection
    strategy_mode = st.radio(
        "Build Mode",
        ["Pre-built Strategies", "Custom Code", "Visual Builder"],
        help="Choose how to create your strategy",
    )

    st.markdown("---")
    st.markdown("### 📊 Backtest Settings")

    # Tickers
    tickers_input = st.text_input(
        "Tickers (comma-separated)", value="AAPL", help="e.g., AAPL,GOOGL,MSFT"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    st.markdown("---")
    st.markdown("### 💰 Capital & Risk")

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1000, max_value=10000000, value=100000, step=1000
    )

    position_size = (
        st.slider("Position Size (%)", min_value=1, max_value=100, value=10, step=1) / 100
    )

    max_positions = st.slider("Max Positions", min_value=1, max_value=20, value=5, step=1)

    # Transaction costs
    with st.expander("⚙️ Advanced Settings"):
        commission = (
            st.number_input("Commission (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            / 100
        )

        slippage = (
            st.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            / 100
        )

        stop_loss = st.number_input(
            "Stop Loss (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0
        )

        take_profit = st.number_input(
            "Take Profit (%)", min_value=0.0, max_value=200.0, value=0.0, step=5.0
        )

# Main content area
if strategy_mode == "Pre-built Strategies":
    st.markdown("### 📚 Pre-built Strategies")

    # Strategy selection
    strategies = {
        "RSI Strategy": {
            "description": "Buy when RSI < oversold, Sell when RSI > overbought",
            "params": {
                "rsi_period": {"type": "slider", "min": 7, "max": 21, "default": 14},
                "oversold": {"type": "slider", "min": 20, "max": 40, "default": 30},
                "overbought": {"type": "slider", "min": 60, "max": 80, "default": 70},
            },
        },
        "MACD Strategy": {
            "description": "Buy on bullish crossover, Sell on bearish crossover",
            "params": {},
        },
        "Moving Average Crossover": {
            "description": "Buy when fast MA crosses above slow MA",
            "params": {
                "fast_period": {"type": "slider", "min": 10, "max": 100, "default": 50},
                "slow_period": {
                    "type": "slider",
                    "min": 100,
                    "max": 300,
                    "default": 200,
                },
            },
        },
        "Bollinger Bands": {
            "description": "Buy at lower band, Sell at upper band",
            "params": {},
        },
        "Mean Reversion": {
            "description": "Buy when price is far below mean",
            "params": {
                "lookback": {"type": "slider", "min": 10, "max": 50, "default": 20},
                "std_threshold": {
                    "type": "slider",
                    "min": 1.0,
                    "max": 3.0,
                    "default": 2.0,
                    "step": 0.5,
                },
            },
        },
        "Momentum": {
            "description": "Buy when momentum is strong",
            "params": {
                "lookback": {"type": "slider", "min": 10, "max": 50, "default": 20},
                "threshold": {
                    "type": "slider",
                    "min": 0.01,
                    "max": 0.10,
                    "default": 0.02,
                    "step": 0.01,
                },
            },
        },
        "Combo Strategy": {
            "description": "Multi-indicator voting system",
            "params": {
                "min_votes": {"type": "slider", "min": 1, "max": 3, "default": 2},
            },
        },
    }

    # Display strategies
    selected_strategy = st.selectbox("Select Strategy", list(strategies.keys()))

    st.info(f"**Description:** {strategies[selected_strategy]['description']}")

    # Strategy parameters
    st.markdown("#### Strategy Parameters")

    strategy_params = {}
    params_config = strategies[selected_strategy]["params"]

    if params_config:
        cols = st.columns(len(params_config))
        for idx, (param_name, config) in enumerate(params_config.items()):
            with cols[idx]:
                if config["type"] == "slider":
                    value = st.slider(
                        param_name.replace("_", " ").title(),
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config.get("step", 1),
                    )
                    strategy_params[param_name] = value
    else:
        st.info("This strategy has no configurable parameters")

    # Generate strategy code
    if selected_strategy == "RSI Strategy":
        strategy_code = f"""
def strategy(data, features):
    signals = []
    rsi_col = 'rsi_{strategy_params.get("rsi_period", 14)}'
    
    for i in range(len(data)):
        if i < {strategy_params.get("rsi_period", 14)}:
            signals.append('HOLD')
        else:
            rsi = features[rsi_col].iloc[i] if rsi_col in features.columns else None
            
            if rsi is None or pd.isna(rsi):
                signals.append('HOLD')
            elif rsi < {strategy_params.get("oversold", 30)}:
                signals.append('BUY')
            elif rsi > {strategy_params.get("overbought", 70)}:
                signals.append('SELL')
            else:
                signals.append('HOLD')
    
    return signals
"""
    elif selected_strategy == "MACD Strategy":
        strategy_code = """
def strategy(data, features):
    signals = []
    prev_macd = None
    prev_signal = None
    
    for i in range(len(data)):
        macd = features['macd'].iloc[i] if 'macd' in features.columns else None
        signal = features['macd_signal'].iloc[i] if 'macd_signal' in features.columns else None
        
        if macd is None or signal is None or pd.isna(macd) or pd.isna(signal) or prev_macd is None:
            signals.append('HOLD')
        else:
            if prev_macd <= prev_signal and macd > signal:
                signals.append('BUY')
            elif prev_macd >= prev_signal and macd < signal:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        prev_macd = macd
        prev_signal = signal
    
    return signals
"""
    elif selected_strategy == "Moving Average Crossover":
        strategy_code = f"""
def strategy(data, features):
    signals = []
    fast_col = 'sma_{strategy_params.get("fast_period", 50)}'
    slow_col = 'sma_{strategy_params.get("slow_period", 200)}'
    prev_fast = None
    prev_slow = None
    
    for i in range(len(data)):
        fast = features[fast_col].iloc[i] if fast_col in features.columns else None
        slow = features[slow_col].iloc[i] if slow_col in features.columns else None
        
        if fast is None or slow is None or pd.isna(fast) or pd.isna(slow) or prev_fast is None:
            signals.append('HOLD')
        else:
            if prev_fast <= prev_slow and fast > slow:
                signals.append('BUY')
            elif prev_fast >= prev_slow and fast < slow:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        prev_fast = fast
        prev_slow = slow
    
    return signals
"""
    elif selected_strategy == "Bollinger Bands":
        strategy_code = """
def strategy(data, features):
    signals = []
    
    for i in range(len(data)):
        close = data['close'].iloc[i]
        bb_upper = features['bb_upper'].iloc[i] if 'bb_upper' in features.columns else None
        bb_lower = features['bb_lower'].iloc[i] if 'bb_lower' in features.columns else None
        
        if bb_upper is None or bb_lower is None or pd.isna(bb_upper) or pd.isna(bb_lower):
            signals.append('HOLD')
        elif close <= bb_lower:
            signals.append('BUY')
        elif close >= bb_upper:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
"""
    elif selected_strategy == "Mean Reversion":
        strategy_code = f"""
def strategy(data, features):
    import numpy as np
    signals = []
    closes = data['close'].values
    lookback = {strategy_params.get("lookback", 20)}
    
    for i in range(len(data)):
        if i < lookback:
            signals.append('HOLD')
            continue
        
        recent = closes[i-lookback:i]
        mean = np.mean(recent)
        std = np.std(recent)
        current = closes[i]
        
        z_score = (current - mean) / std if std > 0 else 0
        
        if z_score < -{strategy_params.get("std_threshold", 2.0)}:
            signals.append('BUY')
        elif z_score > {strategy_params.get("std_threshold", 2.0)}:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
"""
    elif selected_strategy == "Momentum":
        strategy_code = f"""
def strategy(data, features):
    signals = []
    closes = data['close'].values
    lookback = {strategy_params.get("lookback", 20)}
    
    for i in range(len(data)):
        if i < lookback:
            signals.append('HOLD')
            continue
        
        momentum = (closes[i] - closes[i-lookback]) / closes[i-lookback]
        
        if momentum > {strategy_params.get("threshold", 0.02)}:
            signals.append('BUY')
        elif momentum < -{strategy_params.get("threshold", 0.02)}:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
"""
    else:  # Combo
        strategy_code = f"""
def strategy(data, features):
    signals = []
    
    for i in range(len(data)):
        if i < 20:
            signals.append('HOLD')
            continue
        
        buy_votes = 0
        sell_votes = 0
        
        # RSI vote
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].iloc[i]
            if not pd.isna(rsi):
                if rsi < 30:
                    buy_votes += 1
                elif rsi > 70:
                    sell_votes += 1
        
        # MACD vote
        if 'macd' in features.columns and 'macd_signal' in features.columns:
            macd = features['macd'].iloc[i]
            signal = features['macd_signal'].iloc[i]
            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal:
                    buy_votes += 1
                else:
                    sell_votes += 1
        
        # Bollinger Bands vote
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            close = data['close'].iloc[i]
            bb_upper = features['bb_upper'].iloc[i]
            bb_lower = features['bb_lower'].iloc[i]
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                if close <= bb_lower:
                    buy_votes += 1
                elif close >= bb_upper:
                    sell_votes += 1
        
        # Decision
        if buy_votes >= {strategy_params.get("min_votes", 2)}:
            signals.append('BUY')
        elif sell_votes >= {strategy_params.get("min_votes", 2)}:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
"""

    # Show code
    with st.expander("📝 View Generated Code"):
        st.code(strategy_code, language="python")

elif strategy_mode == "Custom Code":
    st.markdown("### 💻 Custom Strategy Code")

    st.info(
        """
    **Instructions:**
    - Define a function called `strategy(data, features)` that returns a list of signals
    - Signals: 'BUY', 'SELL', 'HOLD'
    - `data` contains OHLCV columns
    - `features` contains engineered features (RSI, MACD, etc.)
    """
    )

    # Code editor
    default_code = """
def strategy(data, features):
    '''
    Custom trading strategy
    
    Args:
        data: DataFrame with OHLCV columns
        features: DataFrame with engineered features
    
    Returns:
        List of signals: 'BUY', 'SELL', 'HOLD'
    '''
    signals = []
    
    for i in range(len(data)):
        # Your strategy logic here
        if i < 20:
            signals.append('HOLD')
        else:
            # Example: Simple RSI strategy
            rsi = features['rsi_14'].iloc[i] if 'rsi_14' in features.columns else 50
            
            if rsi < 30:
                signals.append('BUY')
            elif rsi > 70:
                signals.append('SELL')
            else:
                signals.append('HOLD')
    
    return signals
"""

    strategy_code = st.text_area(
        "Strategy Code",
        value=st.session_state.get("custom_code", default_code),
        height=400,
        help="Write your custom strategy code here",
    )

    st.session_state.custom_code = strategy_code

    # Validate code
    if st.button("🔍 Validate Code"):
        try:
            namespace = {}
            exec(strategy_code, namespace)
            if "strategy" in namespace:
                st.success("✅ Code is valid!")
            else:
                st.error("❌ No 'strategy' function found")
        except Exception as e:
            st.error(f"❌ Code error: {e}")

else:  # Visual Builder
    st.markdown("### 🎨 Visual Strategy Builder")
    st.info("🚧 Visual builder coming soon! Use Pre-built or Custom Code for now.")
    strategy_code = None

# Backtest execution section
st.markdown("---")
st.markdown("### 🚀 Run Backtest")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    strategy_name = st.text_input(
        "Strategy Name",
        value=selected_strategy if strategy_mode == "Pre-built Strategies" else "Custom_Strategy",
    )

with col2:
    run_async = st.checkbox("Async Execution", value=True, help="Run in background")

with col3:
    st.write("")
    st.write("")
    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        if not strategy_code:
            st.error("No strategy code defined")
        else:
            # Prepare backtest request
            backtest_request = {
                "strategy_name": strategy_name,
                "strategy_code": strategy_code,
                "tickers": tickers,
                "start_date": start_date.isoformat() + "T00:00:00",
                "end_date": end_date.isoformat() + "T00:00:00",
                "initial_capital": initial_capital,
                "position_size": position_size,
                "max_positions": max_positions,
                "commission": commission,
                "slippage": slippage,
                "benchmark": "SPY",
                "async_execution": run_async,
            }

            if stop_loss > 0:
                backtest_request["stop_loss"] = stop_loss / 100
            if take_profit > 0:
                backtest_request["take_profit"] = take_profit / 100

            # Submit backtest
            try:
                with st.spinner("Submitting backtest..."):
                    response = requests.post(
                        f"{API_URL}/api/v2/backtest/run",
                        json=backtest_request,
                        timeout=10,
                    )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.active_job = result["job_id"]

                    st.success(f"✅ Backtest submitted! Job ID: {result['job_id']}")

                    if run_async:
                        st.info(
                            "⏳ Backtest running in background. Results will appear below when complete."
                        )
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error(f"❌ Error submitting backtest: {e}")

# Results section
if st.session_state.active_job:
    st.markdown("---")
    st.markdown("### 📊 Backtest Results")

    # Poll for results
    with st.spinner("Checking backtest status..."):
        try:
            response = requests.get(
                f"{API_URL}/api/v2/backtest/jobs/{st.session_state.active_job}",
                timeout=5,
            )

            if response.status_code == 200:
                job_status = response.json()

                # Status indicator
                status = job_status["status"]
                if status == "SUCCESS":
                    st.success("✅ Backtest Complete!")
                elif status == "FAILURE":
                    st.error(f"❌ Backtest Failed: {job_status.get('error', 'Unknown error')}")
                elif status in ["PENDING", "PROGRESS"]:
                    progress = job_status.get("progress", {})
                    st.info(f"⏳ Status: {status} - {progress.get('step', 'Processing...')}")

                # Display results if complete
                if status == "SUCCESS" and "result" in job_status:
                    results = job_status["result"]

                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        total_return = results.get("total_return", 0) * 100
                        color = "normal" if total_return >= 0 else "inverse"
                        st.metric(
                            "Total Return",
                            f"{total_return:.2f}%",
                            delta=f"{total_return:.2f}%",
                            delta_color=color,
                        )

                    with col2:
                        sharpe = results.get("sharpe_ratio", 0)
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    with col3:
                        max_dd = results.get("max_drawdown", 0) * 100
                        st.metric("Max Drawdown", f"{max_dd:.2f}%")

                    with col4:
                        win_rate = results.get("win_rate", 0) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                    with col5:
                        num_trades = results.get("num_trades", 0)
                        st.metric("Trades", num_trades)

                    # Detailed metrics
                    st.markdown("#### Detailed Performance")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Returns**")
                        st.write(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
                        st.write(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
                        st.write(
                            f"Annualized Return: {results.get('annualized_return', 0) * 100:.2f}%"
                        )

                    with col2:
                        st.markdown("**Risk Metrics**")
                        st.write(f"Volatility: {results.get('volatility', 0) * 100:.2f}%")
                        st.write(f"Sortino Ratio: {results.get('sortino_ratio', 0):.2f}")
                        st.write(f"Calmar Ratio: {results.get('calmar_ratio', 0):.2f}")

                    with col3:
                        st.markdown("**Trade Statistics**")
                        st.write(f"Total Trades: {num_trades}")
                        st.write(f"Avg Win: ${results.get('avg_win', 0):.2f}")
                        st.write(f"Avg Loss: ${results.get('avg_loss', 0):.2f}")

                    # Equity curve
                    if "equity_curve" in results and results["equity_curve"]:
                        st.markdown("#### Equity Curve")

                        equity_data = pd.DataFrame(results["equity_curve"])
                        equity_data["date"] = pd.to_datetime(equity_data["date"])

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=equity_data["date"],
                                y=equity_data["equity"],
                                mode="lines",
                                name="Portfolio Value",
                                line=dict(color="#4CAF50", width=2),
                                fill="tozeroy",
                            )
                        )

                        fig.update_layout(
                            title="Portfolio Equity Over Time",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode="x unified",
                            template="plotly_dark",
                            height=400,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Trades table
                    if "trades" in results and results["trades"]:
                        st.markdown("#### Recent Trades")

                        trades_df = pd.DataFrame(results["trades"][-20:])  # Last 20 trades

                        # Format columns
                        if "entry_time" in trades_df.columns:
                            trades_df["entry_time"] = pd.to_datetime(
                                trades_df["entry_time"]
                            ).dt.strftime("%Y-%m-%d")
                        if "exit_time" in trades_df.columns:
                            trades_df["exit_time"] = pd.to_datetime(
                                trades_df["exit_time"]
                            ).dt.strftime("%Y-%m-%d")

                        st.dataframe(
                            trades_df[
                                [
                                    "ticker",
                                    "direction",
                                    "entry_time",
                                    "exit_time",
                                    "entry_price",
                                    "exit_price",
                                    "pnl",
                                    "pnl_percent",
                                ]
                            ],
                            use_container_width=True,
                        )

                    # Monthly returns
                    if "monthly_returns" in results and results["monthly_returns"]:
                        st.markdown("#### Monthly Returns")

                        monthly_data = results["monthly_returns"]
                        months = list(monthly_data.keys())
                        returns = list(monthly_data.values())

                        fig = go.Figure()

                        colors = ["green" if r > 0 else "red" for r in returns]

                        fig.add_trace(
                            go.Bar(x=months, y=returns, marker_color=colors, name="Return")
                        )

                        fig.update_layout(
                            title="Monthly Returns",
                            xaxis_title="Month",
                            yaxis_title="Return (%)",
                            template="plotly_dark",
                            height=300,
                        )

                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error checking status: {e}")

# Footer
st.markdown("---")
st.markdown(
    f"**Strategy Lab** | Strategies: Pre-built & Custom | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
