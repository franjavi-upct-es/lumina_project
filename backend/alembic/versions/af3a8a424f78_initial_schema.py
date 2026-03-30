"""initial_schema

Revision ID: af3a8a424f78
Revises:
Create Date: 2026-03-30 10:14:22.041389

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "af3a8a424f78"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: This migration represents the initial schema.
    # TimescaleDB hypertable conversion, continuous aggregates, compression policies,
    # and retention policies are managed in backend/db/timescale_setup.sql
    # and must be applied separately after this migration.
    #
    # The actual table creation is handled by SQLAlchemy's Base.metadata.create_all()
    # or the timescale_setup.sql script. This revision serves as the baseline
    # for future incremental migrations.
    pass


def downgrade() -> None:
    # Dropping all tables would be destructive — intentionally left empty.
    # Use timescale_setup.sql for full schema recreation.
    pass
