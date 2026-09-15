"""Initial continuity schema.

Revision ID: 20260827_0001
Revises:
Create Date: 2026-08-27 00:01:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260827_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS continuity")
    op.create_table(
        "response_indexes",
        sa.Column("scope_id", sa.UUID(), nullable=False),
        sa.Column("source_response_id", sa.Text(), nullable=False),
        sa.Column("visible_item_hashes", postgresql.ARRAY(postgresql.BYTEA()), nullable=False),
        sa.Column(
            "lookup_hash",
            postgresql.BYTEA(),
            sa.Computed("visible_item_hashes[cardinality(visible_item_hashes)]", persisted=True),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("source_response_id <> ''", name="ck_response_indexes_source_response_id_nonempty"),
        sa.CheckConstraint("cardinality(visible_item_hashes) > 0", name="ck_response_indexes_visible_item_hashes_nonempty"),
        sa.CheckConstraint("expires_at > recorded_at", name="ck_response_indexes_expiry_after_recording"),
        sa.PrimaryKeyConstraint("scope_id", "source_response_id", name="pk_response_indexes"),
        schema="continuity",
    )
    op.create_index(
        "ix_response_indexes_lookup",
        "response_indexes",
        ["scope_id", "lookup_hash"],
        schema="continuity",
    )
    op.create_index(
        "ix_response_indexes_expiry",
        "response_indexes",
        ["expires_at"],
        schema="continuity",
    )


def downgrade() -> None:
    op.drop_index("ix_response_indexes_expiry", table_name="response_indexes", schema="continuity")
    op.drop_index("ix_response_indexes_lookup", table_name="response_indexes", schema="continuity")
    op.drop_table("response_indexes", schema="continuity")
    op.execute("DROP SCHEMA continuity")
