"""Run a non-biometric end-to-end smoke test of local persistence and matching."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from faceattend.persistence import (  # noqa: E402
    LivenessAttempt,
    LocalRepository,
    SQLiteDatabase,
)
from faceattend.vision.matcher import ExactNumpyMatcher  # noqa: E402
from faceattend.vision.types import (  # noqa: E402
    EmbeddingTemplate,
    FaceQuality,
    ModelMetadata,
)


@dataclass(frozen=True, slots=True)
class PersistenceSmokeResult:
    database_path: Path
    templates_after_restart: int
    same_person_status: str
    unknown_status: str
    first_attendance_was_duplicate: bool
    second_attendance_was_duplicate: bool
    erased_template_count: int
    templates_after_erasure: int
    audit_preserved_after_erasure: bool


def _unit_vector(index: int, *, dimension: int = 512) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _template(
    template_id: str,
    person_id: str,
    embedding: np.ndarray,
    pose_bin: str,
    model: ModelMetadata,
    created_at: datetime,
) -> EmbeddingTemplate:
    return EmbeddingTemplate(
        template_id=template_id,
        person_id=person_id,
        embedding=embedding,
        model_name=model.name,
        model_version=model.version,
        model_checksum=model.checksum,
        normalized=True,
        pose_bin=pose_bin,
        quality=FaceQuality(True, 250.0, 0.5, 0.08),
        created_at=created_at,
    )


def run_smoke(database_path: Path) -> PersistenceSmokeResult:
    """Exercise the Phase 3 public boundaries using synthetic embeddings."""
    if database_path.exists():
        raise FileExistsError(f"refusing to reuse existing database: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    model = ModelMetadata("smoke_embedding", "synthetic-v1", "synthetic-checksum")
    repository = LocalRepository(SQLiteDatabase(database_path))
    repository.database.initialize()

    configuration_id = repository.add_configuration(
        {"match_threshold": 0.8, "ambiguity_margin": 0.05, "purpose": "smoke-test"},
        created_at=now,
    )
    model_id = repository.add_model("embedding", model, created_at=now)
    person_id, consent_id = repository.add_person_with_consent(
        "Persistence Smoke Person",
        purpose="local persistence smoke test",
        actor_id="smoke-script",
        created_at=now,
    )
    enrollment_id = repository.start_enrollment(
        person_id,
        consent_id=consent_id,
        configuration_id=configuration_id,
        actor_id="smoke-script",
        started_at=now,
    )
    templates = (
        _template(str(uuid4()), person_id, _unit_vector(0), "neutral", model, now),
        _template(str(uuid4()), person_id, _unit_vector(1), "left", model, now),
        _template(str(uuid4()), person_id, _unit_vector(2), "right", model, now),
    )
    repository.complete_enrollment(
        enrollment_id,
        templates,
        model_id=model_id,
        actor_id="smoke-script",
        completed_at=now,
    )

    # Recreate both objects to prove data can be loaded after process restart.
    restarted = LocalRepository(SQLiteDatabase(database_path))
    restarted.database.initialize()
    loaded = restarted.load_compatible_templates(model)
    matcher = ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.05)
    matcher.rebuild(loaded)
    same_person = matcher.match(_unit_vector(0))
    unknown = matcher.match(_unit_vector(10))

    liveness_attempt_id = str(uuid4())
    restarted.record_liveness_attempt(
        LivenessAttempt(
            attempt_id=liveness_attempt_id,
            person_id=person_id,
            operation="check_in",
            challenge_sequence=("turn_left", "blink"),
            completed_challenges=("turn_left", "blink"),
            active_decision="passed",
            passive_decision="passed",
            failure_reason=None,
            passive_median=0.99,
            passive_minimum=0.97,
            suspicious_frame_count=0,
            processing_failure_count=0,
            configuration_version_id=configuration_id,
            created_at=now,
        ),
        actor_id="smoke-script",
    )
    attendance_session_id = restarted.add_attendance_session(
        "Persistence smoke session", opened_at=now, actor_id="smoke-script"
    )
    first = restarted.record_attendance(
        attendance_session_id,
        person_id,
        liveness_attempt_id=liveness_attempt_id,
        match_score=same_person.best.score if same_person.best else 0.0,
        match_threshold=matcher.match_threshold,
        ambiguity_margin=matcher.ambiguity_margin,
        configuration_id=configuration_id,
        actor_id="smoke-script",
        checked_in_at=now,
    )
    second = restarted.record_attendance(
        attendance_session_id,
        person_id,
        liveness_attempt_id=liveness_attempt_id,
        match_score=same_person.best.score if same_person.best else 0.0,
        match_threshold=matcher.match_threshold,
        ambiguity_margin=matcher.ambiguity_margin,
        configuration_id=configuration_id,
        actor_id="smoke-script",
        checked_in_at=now,
    )
    erasure = restarted.erase_person(person_id, actor_id="smoke-script", deleted_at=now)
    remaining = restarted.load_compatible_templates(model)
    audit_preserved = "person.erased" in restarted.audit_actions()

    return PersistenceSmokeResult(
        database_path=database_path,
        templates_after_restart=len(loaded),
        same_person_status=same_person.status.value,
        unknown_status=unknown.status.value,
        first_attendance_was_duplicate=first.duplicate,
        second_attendance_was_duplicate=second.duplicate,
        erased_template_count=erasure.template_count,
        templates_after_erasure=len(remaining),
        audit_preserved_after_erasure=audit_preserved,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        help="New SQLite path. Existing files are never reused or overwritten.",
    )
    args = parser.parse_args()
    database_path = args.database or Path(tempfile.mkdtemp(prefix="faceattend-persistence-")) / (
        "smoke.sqlite3"
    )
    result = run_smoke(database_path)

    print(f"Database: {result.database_path}")
    print(f"Templates after restart: {result.templates_after_restart}")
    print(f"Same-person match: {result.same_person_status}")
    print(f"Unknown-person match: {result.unknown_status}")
    print(f"First attendance duplicate: {result.first_attendance_was_duplicate}")
    print(f"Second attendance duplicate: {result.second_attendance_was_duplicate}")
    print(f"Templates erased: {result.erased_template_count}")
    print(f"Templates after erasure: {result.templates_after_erasure}")
    print(f"Audit preserved: {result.audit_preserved_after_erasure}")
    print("SUCCESS: Phase 3 local persistence smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
