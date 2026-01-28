# frontend/streamlit_app/pages/5_Sentiment_Monitor.py

"""
Sentiment Monitor - Multi-Source Sentiment Analysis Dashboard

This module provides a comprehensive sentiment monitoring interface that aggregates
and visualizes sentiment data from multiple sources including Reddit, Twitter, and
financial news. It integrates with the NLP engine backend to provide real-time
sentiment analysis and divergence detection.

Features:
    - Multi-source sentiment aggregation
    - Real-time sentiment tracking
    - Sentiment divergence detection
    - Historical sentiment trends
    - Source-specific sentiment breakdown
    - Correlation with price movements
    - Alert generation for significant sentiment shifts

References:
    - Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market.
      Journal of Computational Science, 2(1), 1-8.
    - Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media
      in the stock market. The Journal of Finance, 62(3), 1139-1168.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import json

# Page configuration
st.set_page_config(
    page_title="Sentiment Monitor - Lumina 2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        border-radius: 5px;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        border-radius: 5px;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


class SentimentMonitor:
    """
    Main class for sentiment monitoring and analysis.
    
    This class handles data fetching, processing, and visualization of sentiment
    data from multiple sources. It provides methods for calculating aggregate
    sentiment, detecting divergences, and generating insights.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000/api/v2"):
        """
        Initialize the Sentiment Monitor.
        
        Args:
            api_base_url: Base URL for the FastAPI backend
        """
        self.api_base_url = api_base_url
        
    def fetch_sentiment_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch sentiment data from the backend API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            sources: List of sources to include (None for all)
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            # In production, this would call the actual API
            # For now, generate sample data
            return self._generate_sample_sentiment_data(ticker, start_date, end_date, sources)
        except Exception as e:
            st.error(f"Error fetching sentiment data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_sample_sentiment_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample sentiment data for demonstration purposes.
        
        This method creates realistic sample data when the API is not available.
        In production, this would be replaced by actual API calls.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data generation
            end_date: End date for data generation
            sources: List of sources to include
            
        Returns:
            DataFrame with sample sentiment data
        """
        # Define available sources
        all_sources = ['news', 'reddit', 'twitter']
        if sources is None:
            sources = all_sources
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for date in date_range:
            for source in sources:
                # Generate sentiment with some correlation between sources
                base_sentiment = np.random.randn() * 0.3
                
                # Source-specific noise
                if source == 'news':
                    sentiment = base_sentiment + np.random.randn() * 0.1
                    confidence = np.random.uniform(0.7, 0.95)
                    volume = np.random.randint(10, 50)
                elif source == 'reddit':
                    sentiment = base_sentiment + np.random.randn() * 0.15
                    confidence = np.random.uniform(0.6, 0.85)
                    volume = np.random.randint(50, 200)
                else:  # twitter
                    sentiment = base_sentiment + np.random.randn() * 0.2
                    confidence = np.random.uniform(0.5, 0.8)
                    volume = np.random.randint(100, 500)
                
                # Clip sentiment to [-1, 1]
                sentiment = np.clip(sentiment, -1, 1)
                
                data.append({
                    'time': date,
                    'ticker': ticker,
                    'source': source,
                    'sentiment_score': sentiment,
                    'confidence': confidence,
                    'volume': volume
                })
        
        df = pd.DataFrame(data)
        return df
    
    def fetch_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch price data for correlation analysis.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with price data
        """
        try:
            # Generate sample price data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate realistic price movements
            initial_price = 100
            returns = np.random.randn(len(date_range)) * 0.02
            prices = initial_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'time': date_range,
                'ticker': ticker,
                'close': prices
            })
            
            return df
        except Exception as e:
            st.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_aggregate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate sentiment across all sources.
        
        This method combines sentiment from multiple sources using a weighted
        average based on confidence scores and volume.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            DataFrame with aggregate sentiment by date
        """
        # Weight by confidence and volume
        df['weight'] = df['confidence'] * np.log1p(df['volume'])
        
        # Calculate weighted average sentiment
        aggregate = df.groupby('time').apply(
            lambda x: np.average(x['sentiment_score'], weights=x['weight'])
        ).reset_index()
        aggregate.columns = ['time', 'sentiment_score']
        
        # Calculate aggregate confidence
        confidence = df.groupby('time')['confidence'].mean().reset_index()
        aggregate['confidence'] = confidence['confidence']
        
        # Calculate total volume
        volume = df.groupby('time')['volume'].sum().reset_index()
        aggregate['volume'] = volume['volume']
        
        return aggregate
    
    def detect_sentiment_divergence(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        window: int = 5
    ) -> pd.DataFrame:
        """
        Detect divergence between sentiment and price movements.
        
        Divergence occurs when sentiment and price move in opposite directions,
        which can signal potential reversals or market inefficiencies.
        
        Args:
            sentiment_df: DataFrame with aggregate sentiment
            price_df: DataFrame with price data
            window: Rolling window for trend detection
            
        Returns:
            DataFrame with divergence signals
        """
        # Merge sentiment and price data
        merged = pd.merge(
            sentiment_df[['time', 'sentiment_score']],
            price_df[['time', 'close']],
            on='time',
            how='inner'
        )
        
        # Calculate trends
        merged['sentiment_trend'] = merged['sentiment_score'].rolling(window).mean().diff()
        merged['price_trend'] = merged['close'].pct_change().rolling(window).mean()
        
        # Detect divergence (opposite signs)
        merged['divergence'] = (
            (merged['sentiment_trend'] > 0) & (merged['price_trend'] < 0)
        ) | (
            (merged['sentiment_trend'] < 0) & (merged['price_trend'] > 0)
        )
        
        # Calculate divergence strength
        merged['divergence_strength'] = np.abs(
            merged['sentiment_trend'] * merged['price_trend']
        )
        
        return merged
    
    def calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate key sentiment metrics.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary with sentiment metrics
        """
        metrics = {
            'avg_sentiment': df['sentiment_score'].mean(),
            'sentiment_std': df['sentiment_score'].std(),
            'sentiment_range': df['sentiment_score'].max() - df['sentiment_score'].min(),
            'positive_ratio': (df['sentiment_score'] > 0).mean(),
            'negative_ratio': (df['sentiment_score'] < 0).mean(),
            'neutral_ratio': (df['sentiment_score'] == 0).mean(),
            'avg_confidence': df['confidence'].mean(),
            'total_volume': df['volume'].sum()
        }
        
        return metrics


def create_sentiment_timeline_chart(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> go.Figure:
    """
    Create a dual-axis chart showing sentiment and price over time.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        price_df: DataFrame with price data
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Aggregate Sentiment Score', 'Price'),
        row_heights=[0.4, 0.6]
    )
    
    # Sentiment plot
    fig.add_trace(
        go.Scatter(
            x=sentiment_df['time'],
            y=sentiment_df['sentiment_score'],
            mode='lines',
            name='Sentiment',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ),
        row=1, col=1
    )
    
    # Add zero line for sentiment
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Price plot
    fig.add_trace(
        go.Scatter(
            x=price_df['time'],
            y=price_df['close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_source_comparison_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Create a chart comparing sentiment across different sources.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Get unique sources
    sources = sentiment_df['source'].unique()
    colors = {'news': '#1f77b4', 'reddit': '#ff7f0e', 'twitter': '#2ca02c'}
    
    for source in sources:
        source_data = sentiment_df[sentiment_df['source'] == source]
        
        fig.add_trace(
            go.Scatter(
                x=source_data['time'],
                y=source_data['sentiment_score'],
                mode='lines',
                name=source.capitalize(),
                line=dict(color=colors.get(source, 'gray'), width=2)
            )
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sentiment by Source",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_sentiment_distribution_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Create a distribution chart for sentiment scores.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(
        go.Histogram(
            x=sentiment_df['sentiment_score'],
            nbinsx=50,
            name='Sentiment Distribution',
            marker=dict(
                color=sentiment_df['sentiment_score'],
                colorscale='RdYlGn',
                line=dict(color='black', width=1)
            )
        )
    )
    
    fig.update_layout(
        title="Sentiment Score Distribution",
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_divergence_chart(divergence_df: pd.DataFrame) -> go.Figure:
    """
    Create a chart highlighting sentiment-price divergences.
    
    Args:
        divergence_df: DataFrame with divergence data
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Sentiment Trend', 'Price Trend', 'Divergence Signals'),
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Sentiment trend
    fig.add_trace(
        go.Scatter(
            x=divergence_df['time'],
            y=divergence_df['sentiment_trend'],
            mode='lines',
            name='Sentiment Trend',
            line=dict(color='purple', width=2)
        ),
        row=1, col=1
    )
    
    # Price trend
    fig.add_trace(
        go.Scatter(
            x=divergence_df['time'],
            y=divergence_df['price_trend'],
            mode='lines',
            name='Price Trend',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Divergence signals
    divergence_points = divergence_df[divergence_df['divergence']]
    fig.add_trace(
        go.Scatter(
            x=divergence_points['time'],
            y=divergence_points['divergence_strength'],
            mode='markers',
            name='Divergence',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            )
        ),
        row=3, col=1
    )
    
    # Add zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Sentiment Δ", row=1, col=1)
    fig.update_yaxes(title_text="Price Δ", row=2, col=1)
    fig.update_yaxes(title_text="Strength", row=3, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_volume_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing sentiment volume by source over time.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        Plotly figure object
    """
    # Aggregate volume by source and time
    volume_data = sentiment_df.groupby(['time', 'source'])['volume'].sum().reset_index()
    
    fig = px.bar(
        volume_data,
        x='time',
        y='volume',
        color='source',
        title='Sentiment Volume by Source',
        labels={'volume': 'Number of Posts', 'time': 'Date'},
        color_discrete_map={'news': '#1f77b4', 'reddit': '#ff7f0e', 'twitter': '#2ca02c'}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        barmode='stack'
    )
    
    return fig


def display_sentiment_metrics(metrics: Dict):
    """
    Display sentiment metrics in a formatted layout.
    
    Args:
        metrics: Dictionary with sentiment metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_value = metrics['avg_sentiment']
        sentiment_color = "🟢" if sentiment_value > 0.1 else "🔴" if sentiment_value < -0.1 else "🟡"
        st.metric(
            "Average Sentiment",
            f"{sentiment_value:.3f}",
            delta=sentiment_color,
            help="Overall average sentiment score across all sources"
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{metrics['avg_confidence']:.1%}",
            help="Average confidence level of sentiment predictions"
        )
    
    with col3:
        st.metric(
            "Total Volume",
            f"{metrics['total_volume']:,.0f}",
            help="Total number of posts/articles analyzed"
        )
    
    with col4:
        st.metric(
            "Volatility",
            f"{metrics['sentiment_std']:.3f}",
            help="Standard deviation of sentiment scores"
        )
    
    # Second row of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Positive %",
            f"{metrics['positive_ratio']:.1%}",
            help="Percentage of positive sentiment"
        )
    
    with col2:
        st.metric(
            "Negative %",
            f"{metrics['negative_ratio']:.1%}",
            help="Percentage of negative sentiment"
        )
    
    with col3:
        st.metric(
            "Neutral %",
            f"{metrics['neutral_ratio']:.1%}",
            help="Percentage of neutral sentiment"
        )


def main():
    """Main application function."""
    
    # Title and description
    st.title("📊 Sentiment Monitor")
    st.markdown("""
    Monitor real-time sentiment analysis from multiple sources including financial news,
    Reddit discussions, and Twitter feeds. Detect sentiment divergences and correlations
    with price movements.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker selection
    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Enter the stock ticker symbol to monitor"
    ).upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    # Source selection
    st.sidebar.subheader("Data Sources")
    sources = []
    if st.sidebar.checkbox("News", value=True):
        sources.append('news')
    if st.sidebar.checkbox("Reddit", value=True):
        sources.append('reddit')
    if st.sidebar.checkbox("Twitter", value=True):
        sources.append('twitter')
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        divergence_window = st.slider(
            "Divergence Detection Window",
            min_value=3,
            max_value=14,
            value=5,
            help="Number of days for trend calculation"
        )
        
        show_raw_data = st.checkbox(
            "Show Raw Data",
            value=False,
            help="Display raw sentiment data tables"
        )
    
    # Fetch data button
    if st.sidebar.button("Update Data", type="primary"):
        with st.spinner("Fetching sentiment data..."):
            # Initialize monitor
            monitor = SentimentMonitor()
            
            # Fetch data
            sentiment_data = monitor.fetch_sentiment_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                sources=sources if sources else None
            )
            
            price_data = monitor.fetch_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            # Store in session state
            st.session_state['sentiment_data'] = sentiment_data
            st.session_state['price_data'] = price_data
            st.session_state['ticker'] = ticker
    
    # Check if data is available
    if 'sentiment_data' not in st.session_state or st.session_state['sentiment_data'].empty:
        st.info("👈 Configure parameters and click 'Update Data' to begin monitoring")
        return
    
    sentiment_data = st.session_state['sentiment_data']
    price_data = st.session_state['price_data']
    
    # Initialize monitor for calculations
    monitor = SentimentMonitor()
    
    # Calculate aggregate sentiment
    aggregate_sentiment = monitor.calculate_aggregate_sentiment(sentiment_data)
    
    # Calculate metrics
    metrics = monitor.calculate_sentiment_metrics(aggregate_sentiment)
    
    # Display metrics
    st.subheader("📈 Key Metrics")
    display_sentiment_metrics(metrics)
    
    st.markdown("---")
    
    # Main visualizations
    st.subheader("📊 Sentiment & Price Timeline")
    timeline_chart = create_sentiment_timeline_chart(aggregate_sentiment, price_data)
    st.plotly_chart(timeline_chart, width="stretch")
    
    # Two-column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Source Comparison")
        source_chart = create_source_comparison_chart(sentiment_data)
        st.plotly_chart(source_chart, width="stretch")
    
    with col2:
        st.subheader("📊 Sentiment Distribution")
        dist_chart = create_sentiment_distribution_chart(aggregate_sentiment)
        st.plotly_chart(dist_chart, width="stretch")
    
    st.markdown("---")
    
    # Volume analysis
    st.subheader("📈 Volume Analysis")
    volume_chart = create_volume_chart(sentiment_data)
    st.plotly_chart(volume_chart, width="stretch")
    
    st.markdown("---")
    
    # Divergence analysis
    st.subheader("⚠️ Sentiment-Price Divergence Analysis")
    st.markdown("""
    Divergence occurs when sentiment and price move in opposite directions. This can signal:
    - **Bullish Divergence**: Negative sentiment with rising prices (potential reversal down)
    - **Bearish Divergence**: Positive sentiment with falling prices (potential reversal up)
    """)
    
    divergence_data = monitor.detect_sentiment_divergence(
        aggregate_sentiment,
        price_data,
        window=divergence_window
    )
    
    # Display divergence chart
    divergence_chart = create_divergence_chart(divergence_data)
    st.plotly_chart(divergence_chart, width="stretch")
    
    # Divergence alerts
    recent_divergences = divergence_data[divergence_data['divergence']].tail(5)
    
    if not recent_divergences.empty:
        st.warning(f"⚠️ {len(recent_divergences)} divergence signals detected in the selected period")
        
        with st.expander("View Divergence Details"):
            st.dataframe(
                recent_divergences[['time', 'sentiment_trend', 'price_trend', 'divergence_strength']].style.format({
                    'sentiment_trend': '{:.4f}',
                    'price_trend': '{:.4f}',
                    'divergence_strength': '{:.4f}'
                }),
                width="stretch"
            )
    
    st.markdown("---")
    
    # Raw data display (optional)
    if show_raw_data:
        st.subheader("📋 Raw Data")
        
        tab1, tab2, tab3 = st.tabs(["Sentiment Data", "Aggregate Sentiment", "Price Data"])
        
        with tab1:
            st.dataframe(sentiment_data, width="stretch")
            st.download_button(
                "Download Sentiment Data (CSV)",
                data=sentiment_data.to_csv(index=False),
                file_name=f"{ticker}_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.dataframe(aggregate_sentiment, width="stretch")
            st.download_button(
                "Download Aggregate Sentiment (CSV)",
                data=aggregate_sentiment.to_csv(index=False),
                file_name=f"{ticker}_aggregate_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.dataframe(price_data, width="stretch")
            st.download_button(
                "Download Price Data (CSV)",
                data=price_data.to_csv(index=False),
                file_name=f"{ticker}_price_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        Sentiment Monitor | Lumina 2.0 | Data sources: News, Reddit, Twitter<br>
        Powered by FinBERT and advanced NLP
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()