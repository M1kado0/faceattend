"""Rename biometric tables for attendance domain.

Revision ID: d4f6a8b2c9e1
Revises: c33b1b771202
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4f6a8b2c9e1"
down_revision = "c33b1b771202"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("enrollments", "face_registrations")
    op.rename_table("matches", "attendance_records")

    op.alter_column(
        "attendance_records",
        "image_id",
        new_column_name="face_registration_id",
        existing_type=sa.String(),
    )
    op.alter_column(
        "attendance_records",
        "crawled_at",
        new_column_name="checked_in_at",
        existing_type=sa.DateTime(),
    )
    op.add_column("attendance_records", sa.Column("session_id", sa.String(), nullable=True))
    op.drop_column("attendance_records", "source_url")
    op.drop_column("attendance_records", "source_page")

    op.alter_column(
        "takedowns",
        "match_id",
        new_column_name="attendance_record_id",
        existing_type=sa.String(),
    )

    op.execute(
        "ALTER INDEX IF EXISTS ix_enrollments_user_id RENAME TO ix_face_registrations_user_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_enrollments_embedding_id "
        "RENAME TO ix_face_registrations_embedding_id"
    )
    op.execute("ALTER INDEX IF EXISTS ix_matches_user_id RENAME TO ix_attendance_records_user_id")
    op.execute(
        "ALTER INDEX IF EXISTS ix_matches_image_id "
        "RENAME TO ix_attendance_records_face_registration_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_takedowns_match_id RENAME TO ix_takedowns_attendance_record_id"
    )
    op.create_index(
        op.f("ix_attendance_records_session_id"),
        "attendance_records",
        ["session_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_attendance_records_session_id"), table_name="attendance_records")
    op.execute(
        "ALTER INDEX IF EXISTS ix_takedowns_attendance_record_id RENAME TO ix_takedowns_match_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_attendance_records_face_registration_id "
        "RENAME TO ix_matches_image_id"
    )
    op.execute("ALTER INDEX IF EXISTS ix_attendance_records_user_id RENAME TO ix_matches_user_id")
    op.execute(
        "ALTER INDEX IF EXISTS ix_face_registrations_embedding_id "
        "RENAME TO ix_enrollments_embedding_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_face_registrations_user_id RENAME TO ix_enrollments_user_id"
    )

    op.alter_column(
        "takedowns",
        "attendance_record_id",
        new_column_name="match_id",
        existing_type=sa.String(),
    )

    op.add_column("attendance_records", sa.Column("source_page", sa.String(), nullable=True))
    op.add_column("attendance_records", sa.Column("source_url", sa.String(), nullable=True))
    op.drop_column("attendance_records", "session_id")
    op.alter_column(
        "attendance_records",
        "checked_in_at",
        new_column_name="crawled_at",
        existing_type=sa.DateTime(),
    )
    op.alter_column(
        "attendance_records",
        "face_registration_id",
        new_column_name="image_id",
        existing_type=sa.String(),
    )

    op.rename_table("attendance_records", "matches")
    op.rename_table("face_registrations", "enrollments")
