"""Create attendance sessions.

Revision ID: e7a4d91f2b63
Revises: d4f6a8b2c9e1
Create Date: 2026-05-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e7a4d91f2b63"
down_revision = "d4f6a8b2c9e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "attendance_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("starts_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_attendance_sessions_user_id"), "attendance_sessions", ["user_id"])
    op.create_index(op.f("ix_attendance_sessions_name"), "attendance_sessions", ["name"])
    op.create_index(op.f("ix_attendance_sessions_status"), "attendance_sessions", ["status"])
    op.create_index(op.f("ix_attendance_sessions_starts_at"), "attendance_sessions", ["starts_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_attendance_sessions_starts_at"), table_name="attendance_sessions")
    op.drop_index(op.f("ix_attendance_sessions_status"), table_name="attendance_sessions")
    op.drop_index(op.f("ix_attendance_sessions_name"), table_name="attendance_sessions")
    op.drop_index(op.f("ix_attendance_sessions_user_id"), table_name="attendance_sessions")
    op.drop_table("attendance_sessions")
