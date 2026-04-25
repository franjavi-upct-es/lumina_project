"""Phase 1: data engine schemas (ohlcv, news, social, supply chain)

Revision ID: 001_phase1_schemas
Revises:
Create Date: 2026-04-11
"""

from alembic import op

revision = "001_phase1_schemas"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # ---------- ohlcv_1m ----------
    op.execute("""
        CREATE TABLE ohlcv_1m (
            time          TIMESTAMPTZ      NOT NULL,
            ticker        TEXT             NOT NULL,
            open          NUMERIC(18, 6)   NOT NULL,
            high          NUMERIC(18, 6)   NOT NULL,
            low           NUMERIC(18, 6)   NOT NULL,
            close         NUMERIC(18, 6)   NOT NULL,
            volume        BIGINT           NOT NULL,
            vwap          NUMERIC(18, 6),
            trade_count   INTEGER,
            PRIMARY KEY (ticker, time),
            CONSTRAINT ohlcv_high_valid CHECK (high >= GREATEST(open, close)),
            CONSTRAINT ohlcv_low_valid  CHECK (low  <= LEAST(open, close)),
            CONSTRAINT ohlcv_volume_nn  CHECK (volume >= 0)
        );
    """)
    op.execute(
        "SELECT create_hypertable('ohlcv_1m', 'time', chunk_time_interval => INTERVAL '7 days');"
    )
    op.execute("CREATE INDEX idx_ohlcv_ticker_time ON ohlcv_1m (ticker, time DESC);")
    op.execute("""
        ALTER TABLE ohlcv_1m SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'ticker',
            timescaledb.compress_orderby = 'time DESC'
        );
    """)
    op.execute("SELECT add_compression_policy('ohlcv_1m', INTERVAL '7 days');")

    # ---------- news_events ----------
    op.execute("""
        CREATE TABLE news_events (
            time          TIMESTAMPTZ   NOT NULL,
            event_id      UUID          NOT NULL DEFAULT uuid_generate_v4(),
            tickers       TEXT[]        NOT NULL,
            source        TEXT          NOT NULL,
            headline      TEXT          NOT NULL,
            body          TEXT,
            url           TEXT,
            content_hash  TEXT          NOT NULL,
            raw           JSONB,
            PRIMARY KEY (event_id, time)
        );
    """)
    op.execute(
        "SELECT create_hypertable('news_events', 'time', chunk_time_interval => INTERVAL '30 days');"
    )
    op.execute("CREATE INDEX idx_news_tickers ON news_events USING GIN (tickers);")
    op.execute("CREATE UNIQUE INDEX idx_news_content_hash ON news_events (content_hash, time);")

    # ---------- social_posts ----------
    op.execute("""
        CREATE TABLE social_posts (
            time              TIMESTAMPTZ   NOT NULL,
            post_id           TEXT          NOT NULL,
            ticker            TEXT          NOT NULL,
            platform          TEXT          NOT NULL,
            author            TEXT,
            content           TEXT          NOT NULL,
            engagement_score  NUMERIC(10,4),
            PRIMARY KEY (post_id, time)
        );
    """)
    op.execute(
        "SELECT create_hypertable('social_posts', 'time', chunk_time_interval => INTERVAL '7 days');"
    )
    op.execute("CREATE INDEX idx_social_ticker_time ON social_posts (ticker, time DESC);")

    # ---------- supply_chain_edges ----------
    op.execute("""
        CREATE TABLE supply_chain_edges (
            id                  SERIAL PRIMARY KEY,
            source_ticker       TEXT          NOT NULL,
            target_ticker       TEXT          NOT NULL,
            relationship_type   TEXT          NOT NULL,
            weight              NUMERIC(6,4)  NOT NULL DEFAULT 1.0,
            valid_from          DATE          NOT NULL,
            valid_to            DATE,
            CONSTRAINT scc_no_self_loop CHECK (source_ticker <> target_ticker),
            CONSTRAINT scc_weight_range CHECK (weight >= 0 AND weight <= 1)
        );
    """)
    op.execute("CREATE INDEX idx_scc_source ON supply_chain_edges (source_ticker, valid_from);")
    op.execute("CREATE INDEX idx_scc_target ON supply_chain_edges (target_ticker, valid_from);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS supply_chain_edges CASCADE;")
    op.execute("DROP TABLE IF EXISTS social_posts CASCADE;")
    op.execute("DROP TABLE IF EXISTS news_events CASCADE;")
    op.execute("DROP TABLE IF EXISTS ohlcv_1m CASCADE;")
