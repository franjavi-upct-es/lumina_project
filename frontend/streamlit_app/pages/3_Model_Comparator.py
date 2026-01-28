# frontend/streamlit-app/pages/3_Model_Comparator.py
"""
Model Comparator - Compare and analyze ML models
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Model Comparator - Lumina",
    page_icon="🤖",
    layout="wide",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown(
    """
<style>
    .model-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .winner-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

# Header
st.title("🤖 Model Comparator")
st.markdown("Compare ML models and analyze their performance")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Training")

    # Ticker selection
    ticker = st.text_input("Ticker", value="AAPL").upper()

    # Date range
    st.markdown("#### Training Data")
    training_years = st.slider("Years of Data", min_value=1, max_value=5, value=2)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * training_years)

    st.info(f"📅 {start_date.date()} to {end_date.date()}")

    st.markdown("---")
    st.markdown("### 🔧 Model Selection")

    # Model types
    train_lstm = st.checkbox("LSTM", value=True)
    train_xgboost = st.checkbox("XGBoost", value=True)
    train_transformer = st.checkbox("Transformer", value=False, disabled=True)

    # Hyperparameters
    with st.expander("⚙️ Hyperparameters"):
        st.markdown("**LSTM**")
        lstm_epochs = st.slider("Epochs", 5, 100, 30, key="lstm_epochs")
        lstm_hidden = st.slider("Hidden Dim", 32, 256, 128, key="lstm_hidden")

        st.markdown("**XGBoost**")
        xgb_estimators = st.slider("Estimators", 50, 500, 100, key="xgb_estimators")
        xgb_depth = st.slider("Max Depth", 3, 10, 6, key="xgb_depth")

    # Train button
    if st.button("🚀 Train Models", type="primary", width="stretch"):
        models_to_train = []
        if train_lstm:
            models_to_train.append("lstm")
        if train_xgboost:
            models_to_train.append("xgboost")

        if not models_to_train:
            st.error("Select at least one model to train")
        else:
            st.session_state.training_active = True
            st.session_state.models_to_train = models_to_train

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Training", "🔍 Comparison", "💡 Insights"])

with tab1:
    st.markdown("### 📊 Model Overview")

    # Fetch available models
    try:
        response = requests.get(
            f"{API_URL}/api/v2/ml/models",
            params={"ticker": ticker, "limit": 10},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            if models:
                st.success(f"Found {len(models)} models for {ticker}")

                # Display models in grid
                cols = st.columns(3)

                for idx, model in enumerate(models[:9]):
                    with cols[idx % 3]:
                        st.markdown(
                            f"""
                        <div class="model-card">
                            <h4>{model["model_type"].upper()}</h4>
                            <p><strong>Trained:</strong> {model["trained_on"][:10]}</p>
                            <p><strong>MAE:</strong> {model.get("mae", 0):.4f}</p>
                            <p><strong>R²:</strong> {model.get("r2_score", 0):.4f}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Models table
                st.markdown("#### All Models")
                models_df = pd.DataFrame(models)

                if not models_df.empty:
                    display_cols = [
                        "model_name",
                        "model_type",
                        "trained_on",
                        "mae",
                        "rmse",
                        "r2_score",
                    ]
                    available_cols = [c for c in display_cols if c in models_df.columns]

                    st.dataframe(models_df[available_cols], width="stretch")

            else:
                st.info(f"No models found for {ticker}. Train some models first!")

        else:
            st.error(f"Error fetching models: {response.status_code}")

    except Exception as e:
        st.error(f"Error: {e}")

with tab2:
    st.markdown("### 📈 Model Training")

    if "training_active" in st.session_state and st.session_state.training_active:
        st.info("🚀 Training initiated...")

        models_to_train = st.session_state.models_to_train
        training_results = {}

        for model_type in models_to_train:
            st.markdown(f"#### Training {model_type.upper()}")

            # Prepare training request
            train_request = {
                "ticker": ticker,
                "model_type": model_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "async_training": True,
            }

            if model_type == "lstm":
                train_request.update(
                    {
                        "num_epochs": lstm_epochs,
                        "hidden_dim": lstm_hidden,
                        "num_layers": 3,
                        "batch_size": 32,
                        "sequence_length": 60,
                        "prediction_horizon": 5,
                    }
                )
            elif model_type == "xgboost":
                train_request.update(
                    {
                        "n_estimators": xgb_estimators,
                        "max_depth": xgb_depth,
                        "learning_rate": 0.05,
                        "prediction_horizon": 5,
                    }
                )

            # Submit training
            try:
                with st.spinner(f"Submitting {model_type} training..."):
                    response = requests.post(
                        f"{API_URL}/api/v2/ml/train", json=train_request, timeout=10
                    )

                if response.status_code == 200:
                    result = response.json()
                    job_id = result["job_id"]

                    st.success(f"✅ {model_type.upper()} training submitted (Job: {job_id})")

                    training_results[model_type] = {"job_id": job_id, "status": "queued"}

                    # Monitor progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(20):  # Poll for 20 iterations
                        time.sleep(3)

                        try:
                            status_response = requests.get(
                                f"{API_URL}/api/v2/ml/jobs/{job_id}", timeout=5
                            )

                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                current_status = status_data["status"]

                                progress = min((i + 1) / 20, 0.95)
                                progress_bar.progress(progress)
                                status_text.text(f"Status: {current_status}")

                                if current_status == "SUCCESS":
                                    progress_bar.progress(1.0)
                                    st.success(f"✅ {model_type.upper()} training complete!")

                                    # Store result
                                    training_results[model_type] = {
                                        "job_id": job_id,
                                        "status": "success",
                                        "model_id": status_data.get("model_id"),
                                    }
                                    break

                                elif current_status == "FAILURE":
                                    st.error(
                                        f"❌ {model_type.upper()} training failed: {status_data.get('error')}"
                                    )
                                    break

                        except Exception as e:
                            st.warning(f"Status check error: {e}")

                else:
                    st.error(f"❌ Training submission failed: {response.status_code}")

            except Exception as e:
                st.error(f"❌ Error: {e}")

        # Store results
        st.session_state.trained_models = training_results
        st.session_state.training_active = False

        # Summary
        st.markdown("---")
        st.markdown("### Training Summary")

        successful = sum(1 for r in training_results.values() if r.get("status") == "success")
        st.write(f"**Completed:** {successful}/{len(training_results)} models")

    else:
        st.info("👈 Configure and start training in the sidebar")

        # Show training history
        try:
            response = requests.get(
                f"{API_URL}/api/v2/ml/models",
                params={"ticker": ticker, "limit": 5},
                timeout=5,
            )

            if response.status_code == 200:
                recent_models = response.json().get("models", [])

                if recent_models:
                    st.markdown("#### Recent Training Jobs")

                    for model in recent_models:
                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            st.write(f"**{model['model_type'].upper()}**")
                        with col2:
                            st.write(f"Trained: {model['trained_on'][:19]}")
                        with col3:
                            if model.get("is_active"):
                                st.success("Active")
                            else:
                                st.info("Inactive")

        except Exception as e:
            st.warning(f"Could not fetch recent models: {e}")

with tab3:
    st.markdown("### 🔍 Model Comparison")

    # Model selection for comparison
    st.markdown("#### Select Models to Compare")

    try:
        response = requests.get(
            f"{API_URL}/api/v2/ml/models",
            params={"ticker": ticker, "is_active": True},
            timeout=10,
        )

        if response.status_code == 200:
            available_models = response.json().get("models", [])

            if len(available_models) >= 2:
                # Model selection
                model_options = [f"{m['model_name']} ({m['model_type']})" for m in available_models]

                selected_models = st.multiselect(
                    "Select Models (2+)",
                    model_options,
                    default=model_options[:2] if len(model_options) >= 2 else [],
                )

                if len(selected_models) >= 2:
                    # Get selected model details
                    selected_indices = [model_options.index(m) for m in selected_models]
                    models_to_compare = [available_models[i] for i in selected_indices]

                    # Comparison metrics
                    st.markdown("#### Performance Comparison")

                    # Metrics table
                    comparison_data = []
                    for model in models_to_compare:
                        comparison_data.append(
                            {
                                "Model": model["model_name"],
                                "Type": model["model_type"].upper(),
                                "MAE": model.get("mae", 0),
                                "RMSE": model.get("rmse", 0),
                                "R²": model.get("r2_score", 0),
                                "Trained": model["trained_on"][:10],
                            }
                        )

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width="stretch")

                    # Visual comparison
                    col1, col2 = st.columns(2)

                    with col1:
                        # MAE comparison
                        fig_mae = go.Figure()

                        fig_mae.add_trace(
                            go.Bar(
                                x=[m["Type"] for m in comparison_data],
                                y=[m["MAE"] for m in comparison_data],
                                text=[f"{m['MAE']:.4f}" for m in comparison_data],
                                textposition="auto",
                                marker_color=["#2196F3", "#4CAF50", "#FF9800"][
                                    : len(comparison_data)
                                ],
                            )
                        )

                        fig_mae.update_layout(
                            title="Mean Absolute Error (Lower is Better)",
                            xaxis_title="Model",
                            yaxis_title="MAE",
                            template="plotly_dark",
                            showlegend=False,
                        )

                        st.plotly_chart(fig_mae, width="stretch")

                    with col2:
                        # R² comparison
                        fig_r2 = go.Figure()

                        fig_r2.add_trace(
                            go.Bar(
                                x=[m["Type"] for m in comparison_data],
                                y=[m["R²"] for m in comparison_data],
                                text=[f"{m['R²']:.4f}" for m in comparison_data],
                                textposition="auto",
                                marker_color=["#2196F3", "#4CAF50", "#FF9800"][
                                    : len(comparison_data)
                                ],
                            )
                        )

                        fig_r2.update_layout(
                            title="R² Score (Higher is Better)",
                            xaxis_title="Model",
                            yaxis_title="R² Score",
                            template="plotly_dark",
                            showlegend=False,
                        )

                        st.plotly_chart(fig_r2, width="stretch")

                    # Winner determination
                    st.markdown("#### 🏆 Best Model")

                    # Determine winner based on multiple metrics
                    best_mae = min(comparison_data, key=lambda x: x["MAE"])
                    best_r2 = max(comparison_data, key=lambda x: x["R²"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(f"**Best MAE:** {best_mae['Model']} ({best_mae['MAE']:.4f})")

                    with col2:
                        st.success(f"**Best R²:** {best_r2['Model']} ({best_r2['R²']:.4f})")

                else:
                    st.info("Select at least 2 models to compare")

            else:
                st.warning(
                    f"Need at least 2 models to compare. Currently have {len(available_models)}."
                )
                st.info("Train more models in the Training tab")

        else:
            st.error("Error fetching models")

    except Exception as e:
        st.error(f"Error: {e}")

with tab4:
    st.markdown("### 💡 Model Insights")

    st.markdown("#### Feature Importance")

    try:
        # Get latest model
        response = requests.get(
            f"{API_URL}/api/v2/ml/models",
            params={"ticker": ticker, "limit": 1},
            timeout=5,
        )

        if response.status_code == 200:
            models = response.json().get("models", [])

            if models:
                latest_model = models[0]
                model_id = latest_model["model_id"]

                st.info(f"Analyzing: {latest_model['model_name']} ({latest_model['model_type']})")

                # Request feature importance computation
                if st.button("🔍 Compute Feature Importance"):
                    with st.spinner("Computing feature importance (this may take a minute)..."):
                        importance_response = requests.get(
                            f"{API_URL}/api/v2/ml/features/importance/{model_id}",
                            params={"num_samples": 100},
                            timeout=120,
                        )

                        if importance_response.status_code == 200:
                            st.success("✅ Feature importance computed!")

                            # This would return a task ID in real implementation
                            st.info("Feature importance computation started in background")
                        else:
                            st.error("Failed to compute feature importance")

                # Display mock feature importance for now
                st.markdown("#### Top Features")

                # Mock data
                top_features = {
                    "rsi_14": 0.15,
                    "macd": 0.12,
                    "sma_20": 0.10,
                    "volume_ma_ratio_20": 0.09,
                    "volatility_20d": 0.08,
                    "bb_position": 0.07,
                    "returns": 0.06,
                    "ema_12": 0.05,
                    "atr_14": 0.04,
                    "obv": 0.03,
                }

                # Bar chart
                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation="h",
                        marker_color="#4CAF50",
                        text=[f"{v:.2%}" for v in top_features.values()],
                        textposition="auto",
                    )
                )

                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    template="plotly_dark",
                    height=400,
                )

                st.plotly_chart(fig, width="stretch")

            else:
                st.info("No models available for analysis")

    except Exception as e:
        st.error(f"Error: {e}")

    # Model recommendations
    st.markdown("---")
    st.markdown("#### 📋 Recommendations")

    st.info(
        """
    **Model Selection Guidelines:**

    - **LSTM**: Best for time series with complex patterns and long-term dependencies
    - **XGBoost**: Fast, interpretable, works well with tabular features
    - **Transformer**: Best for very long sequences (coming soon)
    - **Ensemble**: Combines multiple models for best overall performance

    **Tips:**
    - Train multiple models and compare
    - Use XGBoost for quick iterations
    - Use LSTM when you need sequential modeling
    - Always validate on out-of-sample data
    """
    )

# Footer
st.markdown("---")
st.markdown(
    f"**Model Comparator** | Compare ML models | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
