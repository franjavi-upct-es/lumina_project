# alembic/versions/003_add_portfolio_and_backtest.py
"""Add portfolio_history and backtest_runs tables

Revision ID: 003_add_portfolio_and_backtest
Revises: 002_add_arena_tables
Create Date: 2026-05-25
"""
from alembic import op
import sqlalchemy as sa

revision = "003_add_portfolio_and_backtest"
down_revision = "002_add_arena_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------- portfolio_history ----------
    op.execute("""
        CREATE TABLE portfolio_history (
            time    TIMESTAMPTZ   NOT NULL,
            equity  NUMERIC(18, 2) NOT NULL,
            cash    NUMERIC(18, 2) NOT NULL
        );
    """)
    op.execute("SELECT create_hypertable('portfolio_history', 'time', chunk_time_interval => INTERVAL '1 day');")
    op.execute("CREATE INDEX idx_portfolio_history_time ON portfolio_history (time DESC);")

    # ---------- backtest_runs ----------
    op.execute("""
        CREATE TABLE backtest_runs (
            run_id        TEXT         PRIMARY KEY,
            status        TEXT         NOT NULL,
            sharpe        NUMERIC(10, 4),
            max_drawdown  NUMERIC(10, 4),
            total_return  NUMERIC(10, 4),
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX idx_backtest_runs_created ON backtest_runs (created_at DESC);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS backtest_runs CASCADE;")
    op.execute("DROP TABLE IF EXISTS portfolio_history CASCADE;")
