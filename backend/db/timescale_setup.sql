-- backend/db/timescale_setup.sql
-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Price data hypertable
CREATE TABLE IF NOT EXISTS price_data (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    adjusted_close DOUBLE PRECISION,
    dividends DOUBLE PRECISION,
    stock_splits DOUBLE PRECISION,
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('price_data', 'time', if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_data (ticker, time DESC);

-- Feature store hypertable
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION,
    feature_category VARCHAR(50),   -- 'technical', 'fundamental', 'sentiment', 'macro'
    PRIMARY KEY (ticker, time, feature_name)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_ticker ON features (ticker, feature_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_category ON features (feature_category, time DESC);

-- Sentiment data hypertable
CREATE TABLE IF NOT EXISTS sentiment_data (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    source VARCHAR(50) NOT NULL,    -- 'news', 'reddit', 'twitter', 'finbert'
    sentiment_score DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    volume INT, -- Number of mentions
    text_snippet TEXT,
    meta_data JSONB,
    PRIMARY KEY (time, ticker, source)
);

SELECT create_hypertable('sentiment_data', 'time', if_not_exists => TRUE);

-- Model predictions hypertable
CREATE TABLE IF NOT EXISTS predictions (
    prediction_time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    horizon_days INT NOT NULL,
    predicted_price DOUBLE PRECISION,
    confidence_lower DOUBLE PRECISION,
    confidence_upper DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    actual_price DOUBLE PRECISION, -- NULL until realized
    error DOUBLE PRECISION, -- Computed when actual price available
    PRIMARY KEY (prediction_time, ticker, model_name, horizon_days)
);

SELECT create_hypertable('predictions', 'prediction_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions (ticker, prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_name, prediction_time DESC);

-- Market regime detection
CREATE TABLE IF NOT EXISTS market_regimes (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    regime_type VARCHAR(20),  -- 'bull', 'bear', 'sideways'
    probability DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    expected_duration_days INT,
    meta_data JSONB,
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('market_regimes', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_regimes_ticker ON market_regimes (ticker, time DESC);

-- Backtest results (regular table)
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    tickers VARCHAR(50)[],
    initial_capital DOUBLE PRECISION,
    final_capital DOUBLE PRECISION,
    total_return DOUBLE PRECISION,
    annualized_return DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    calmar_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    num_trades INT,
    avg_trade DOUBLE PRECISION,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results (strategy_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_dates ON backtest_results (start_date, end_date);

-- Backtest trades detail
CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id UUID NOT NULL REFERENCES backtest_results(backtest_id) ON DELETE CASCADE,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    ticker VARCHAR(10) NOT NULL,
    direction VARCHAR(10),  -- 'long', 'short'
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    quantity DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    pnl_percent DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    slippage DOUBLE PRECISION,
    meta_data JSONB
);

CREATE INDEX IF NOT EXISTS idx_trades_backtest ON backtest_trades (backtest_id, entry_time);

-- Model meta_data (regular table for MLflow integration)
CREATE TABLE IF NOT EXISTS models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),     -- 'lstm', 'transformer', 'xgboost', 'ensemble'
    version VARCHAR(20),
    trained_on TIMESTAMPTZ DEFAULT NOW(),
    ticker VARCHAR(10),
    training_samples INT,
    validation_samples INT,
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    r2_score DOUBLE PRECISION,
    sharpe_bactest DOUBLE PRECISION,
    hyperparameters JSONB,
    feature_importance JSONB,
    mlflow_run_id VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_name ON models (model_name, version DESC);
CREATE INDEX IF NOT EXISTS idx_models_ticker ON models (ticker, trained_on DESC);

-- Portfolio positions (for paper trading)
CREATE TABLE IF NOT EXISTS portfolio_positions (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(10),   -- 'buy', 'sell'
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    balance_after DOUBLE PRECISION,
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_user ON portfolio_positions (user_id, executed_at DESC);

-- Portfolio balance
CREATE TABLE IF NOT EXISTS portfolio_balance (
    user_id VARCHAR(50) PRIMARY KEY,
    cash DOUBLE PRECISION NOT NULL DEFAULT 100000.0,
    equity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_value DOUBLE PRECISION DEFAULT 100000.0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Continuous aggregates for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_price_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    ticker,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM price_data
GROUP BY day, ticker
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_price_summary',
       start_offset => INTERVAL '30 days',
       end_offset => INTERVAL '1 day',
       schedule_interval => INTERVAL '1 day');

-- Aggregate for feature statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_feature_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    ticker,
    feature_name,
    avg(feature_value) AS avg_value,
    stddev(feature_value) AS std_value
    min(feature_value) AS min_value,
    max(feature_value) AS max_value
FROM features
GROUP BY day, ticker, feature_name
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_feature_stats',
       start_offset => INTERVAL '30 days',
       end_offset => INTERVAL '1 day',
       schedule_interval => INTERVAL '1 day');

-- Data retention policies (keep raw data for 1 year, aggregates forever)
SELECT add_retention_policy('price_data', INTERVAL '1 year');
SELECT add_retention_policy('features', INTERVAL '6 months');
SELECT add_retention_policy('sentiment_data', INTERVAL '3 months');

-- Compression policies
SELECT add_compression_policy('price_data', INTERVAL '7 days');
SELECT add_compression_policy('features', INTERVAL '7 days');
SELECT add_compression_policy('sentiment_data', INTERVAL '7 days');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lumina;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lumina;
