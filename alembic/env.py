# alembic/env.py
"""Alembic environment file for the Lumina V3 schema.

Two important deviations from a default ``alembic init`` template:

1. The database URL is resolved from the ``TIMESCALE_URL`` environment
   variable (falling back to whatever ``sqlalchemy.url`` is set to in
   ``alembic.ini``). This lets the *same* migration scripts run from a
   developer's laptop, from inside the ``migrate`` Docker container, and
   from CI without editing files.

2. ``target_metadata`` is intentionally ``None``. Every Lumina migration
   uses raw SQL (via ``op.execute``) — partly because TimescaleDB's
   hypertables and compression policies are not expressible in
   SQLAlchemy's metadata, and partly because the SQL idioms in the
   migrations (``CREATE EXTENSION``, ``SELECT create_hypertable(...)``)
   are clearer when written verbatim. The downside is that
   ``alembic revision --autogenerate`` produces empty diffs; that's a
   deliberate trade — we write migrations by hand for this schema.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Alembic Config object, providing access to the values within the
# ``alembic.ini`` file in use.
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Environment override for the SQLAlchemy URL.
_env_url = os.environ.get("TIMESCALE_URL")
if _env_url:
    config.set_main_option("sqlalchemy.url", _env_url)

# No SQLAlchemy metadata — see module docstring for the rationale.
target_metadata = None


def run_migrations_offline() -> None:
    """Generate the migration SQL without a live database connection.

    Used by ``alembic upgrade --sql`` to produce review-friendly SQL
    diffs for production change-management.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations directly against a live database.

    Used by the ``migrate`` Docker service on startup and by the local
    ``make migrate`` target.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
