# alembic/versions/002_add_arena_tables.py
"""Phase X.4: Spartan Arena schemas (runs, decision records, divergences, pairs, explanations)

Revision ID: 002_add_arena_tables
Revises: 001_phase1_schemas
Create Date: 2026-05-23
"""
from alembic import op

revision = "002_add_arena_tables"
down_revision = "001_phase1_schemas"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------- arena_runs ----------
    # Regular table; one row per arena run (metadata + status).
    op.execute("""
        CREATE TABLE arena_runs (
            run_id              UUID         PRIMARY KEY,
            status              VARCHAR(16)  NOT NULL,
            ticker              VARCHAR(16)  NOT NULL,
            start_date          TIMESTAMPTZ  NOT NULL,
            end_date            TIMESTAMPTZ  NOT NULL,
            n_trajectories      INT          NOT NULL,
            mc_seeds            BIGINT[]     NOT NULL,
            playback_multiplier REAL         NOT NULL DEFAULT 1.0,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ,
            failure_reason      TEXT,
            CONSTRAINT arena_runs_status_valid CHECK (
                status IN ('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED')
            ),
            CONSTRAINT arena_runs_n_traj CHECK (n_trajectories BETWEEN 3 AND 16)
        );
    """)
    op.execute("CREATE INDEX idx_arena_runs_status_created ON arena_runs (status, created_at DESC);")

    # ---------- arena_decision_records ----------
    # Hypertable on sim_timestamp; one row per (trajectory, step).
    op.execute("""
        CREATE TABLE arena_decision_records (
            record_id              UUID         NOT NULL,
            run_id                 UUID         NOT NULL REFERENCES arena_runs(run_id) ON DELETE CASCADE,
            trajectory_id          INT          NOT NULL,
            step_index             INT          NOT NULL,
            sim_timestamp          TIMESTAMPTZ  NOT NULL,
            wall_timestamp         TIMESTAMPTZ  NOT NULL,
            ticker                 VARCHAR(16)  NOT NULL,
            ohlcv                  JSONB        NOT NULL,
            action_kind            VARCHAR(16)  NOT NULL,
            action_vector          REAL[]       NOT NULL,
            confidence             REAL         NOT NULL,
            uncertainty            REAL         NOT NULL,
            realized_reward        REAL,
            state_artifact_path    TEXT         NOT NULL,
            attribution            JSONB        NOT NULL,
            mc_seed                BIGINT       NOT NULL,
            PRIMARY KEY (record_id, sim_timestamp),
            CONSTRAINT decision_action_vector_len CHECK (array_length(action_vector, 1) = 4),
            CONSTRAINT decision_confidence_range CHECK (confidence  BETWEEN 0 AND 1),
            CONSTRAINT decision_uncertainty_range CHECK (uncertainty BETWEEN 0 AND 1)
        );
    """)
    op.execute(
        "SELECT create_hypertable('arena_decision_records', 'sim_timestamp', "
        "chunk_time_interval => INTERVAL '7 days');"
    )
    op.execute(
        "CREATE INDEX idx_arena_decisions_run_traj_step "
        "ON arena_decision_records (run_id, trajectory_id, step_index);"
    )
    op.execute(
        "CREATE INDEX idx_arena_decisions_run_time "
        "ON arena_decision_records (run_id, sim_timestamp DESC);"
    )

    # ---------- arena_divergence_points ----------
    # Hypertable on sim_timestamp; one row per detected divergence.
    op.execute("""
        CREATE TABLE arena_divergence_points (
            run_id                  UUID         NOT NULL REFERENCES arena_runs(run_id) ON DELETE CASCADE,
            step_index              INT          NOT NULL,
            sim_timestamp           TIMESTAMPTZ  NOT NULL,
            best_trajectory_id      INT          NOT NULL,
            worst_trajectory_id     INT          NOT NULL,
            best_action_vector      REAL[]       NOT NULL,
            worst_action_vector     REAL[]       NOT NULL,
            action_l2_distance      REAL         NOT NULL,
            best_subsequent_sharpe  REAL         NOT NULL,
            worst_subsequent_sharpe REAL         NOT NULL,
            sharpe_delta            REAL         NOT NULL,
            PRIMARY KEY (run_id, step_index, sim_timestamp),
            CONSTRAINT divergence_action_vec_len CHECK (
                array_length(best_action_vector, 1) = 4
                AND array_length(worst_action_vector, 1) = 4
            ),
            CONSTRAINT divergence_l2_nn CHECK (action_l2_distance >= 0)
        );
    """)
    op.execute(
        "SELECT create_hypertable('arena_divergence_points', 'sim_timestamp', "
        "chunk_time_interval => INTERVAL '30 days');"
    )
    op.execute(
        "CREATE INDEX idx_arena_divergences_run "
        "ON arena_divergence_points (run_id, sim_timestamp DESC);"
    )

    # ---------- arena_counterfactual_pairs ----------
    # Regular table — pairs are sparse, no need to hypertable them.
    op.execute("""
        CREATE TABLE arena_counterfactual_pairs (
            pair_id                 UUID         PRIMARY KEY,
            run_id                  UUID         NOT NULL REFERENCES arena_runs(run_id) ON DELETE CASCADE,
            divergence_step_index   INT          NOT NULL,
            sim_timestamp           TIMESTAMPTZ  NOT NULL,
            state_artifact_path     TEXT         NOT NULL,
            good_action_vector      REAL[]       NOT NULL,
            bad_action_vector       REAL[]       NOT NULL,
            good_outcome_sharpe     REAL         NOT NULL,
            bad_outcome_sharpe      REAL         NOT NULL,
            confidence_score        REAL         NOT NULL,
            CONSTRAINT pair_action_vec_len CHECK (
                array_length(good_action_vector, 1) = 4
                AND array_length(bad_action_vector, 1) = 4
            ),
            CONSTRAINT pair_confidence_range CHECK (confidence_score BETWEEN 0 AND 1)
        );
    """)
    op.execute(
        "CREATE INDEX idx_arena_pairs_run_confidence "
        "ON arena_counterfactual_pairs (run_id, confidence_score DESC);"
    )

    # ---------- arena_step_explanations ----------
    # Regular table — 1:1 with arena_decision_records.
    op.execute("""
        CREATE TABLE arena_step_explanations (
            record_id  UUID         PRIMARY KEY,
            run_id     UUID         NOT NULL REFERENCES arena_runs(run_id) ON DELETE CASCADE,
            text       TEXT         NOT NULL,
            tags       TEXT[]       NOT NULL DEFAULT '{}'::TEXT[]
        );
    """)
    op.execute(
        "CREATE INDEX idx_arena_explanations_run ON arena_step_explanations (run_id);"
    )
    # GIN index makes tag-filtered queries (e.g. WHERE 'overbought' = ANY(tags))
    # fast for the explanation panel's filter chips.
    op.execute(
        "CREATE INDEX idx_arena_explanations_tags "
        "ON arena_step_explanations USING GIN (tags);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS arena_step_explanations CASCADE;")
    op.execute("DROP TABLE IF EXISTS arena_counterfactual_pairs CASCADE;")
    op.execute("DROP TABLE IF EXISTS arena_divergence_points CASCADE;")
    op.execute("DROP TABLE IF EXISTS arena_decision_records CASCADE;")
    op.execute("DROP TABLE IF EXISTS arena_runs CASCADE;")
