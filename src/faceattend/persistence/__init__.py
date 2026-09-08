"""Local persistence boundaries for the desktop application."""

from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.records import (
    AttendanceWrite,
    EnrollmentWrite,
    ErasureResult,
    LivenessAttempt,
)
from faceattend.persistence.repositories import LocalRepository, PersistenceError

__all__ = [
    "AttendanceWrite",
    "EnrollmentWrite",
    "ErasureResult",
    "LivenessAttempt",
    "LocalRepository",
    "PersistenceError",
    "SQLiteDatabase",
]
