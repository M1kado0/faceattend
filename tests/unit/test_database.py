"""Behavior tests for the local SQLite lifecycle boundary."""

from pathlib import Path

import pytest

from faceattend.persistence.database import SQLiteDatabase


def test_initialize_applies_migrations_once_and_enforces_foreign_keys(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "faceattend.sqlite3")

    database.initialize()
    database.initialize()

    with database.connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert versions == [(1,), (2,), (3,)]


def test_transaction_rolls_back_all_writes_on_failure(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "faceattend.sqlite3")
    database.initialize()

    with pytest.raises(RuntimeError, match="stop"), database.transaction() as connection:
        connection.execute(
            "INSERT INTO people (id, display_name, created_at) VALUES (?, ?, ?)",
            ("person-1", "Ada", "2026-09-07T12:00:00Z"),
        )
        raise RuntimeError("stop")

    with database.connection() as connection:
        count = connection.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    assert count == 0
