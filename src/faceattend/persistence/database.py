"""SQLite connection, migration, and transaction lifecycle."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SQLiteDatabase:
    """Own local database setup while keeping connections thread-local.

    A connection is opened for one operation and never shared between camera,
    inference, persistence, or GUI threads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._migrations = Path(__file__).with_name("migrations")

    def initialize(self) -> None:
        """Create the database and apply each numbered migration once."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied = {
                row[0] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for migration in sorted(self._migrations.glob("[0-9][0-9][0-9]_*.sql")):
                version = int(migration.name.split("_", 1)[0])
                if version in applied:
                    continue
                sql = migration.read_text(encoding="utf-8")
                try:
                    connection.executescript(
                        "BEGIN IMMEDIATE;\n"
                        f"{sql}\n"
                        "INSERT INTO schema_migrations(version, name) "
                        f"VALUES ({version}, '{migration.name}');\n"
                        "COMMIT;"
                    )
                except sqlite3.Error:
                    connection.execute("ROLLBACK")
                    raise

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open one configured connection for the calling thread."""
        connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit all writes together or roll all of them back."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
