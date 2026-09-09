"""Composition root joining Qt inputs to the headless local application core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from faceattend.application.attendance_service import AttendanceCoordinator, AttendanceRequest
from faceattend.application.attendance_session import AttendanceDesktopSessionProcessor
from faceattend.application.local_config import LocalAppConfig
from faceattend.application.registration_service import RegistrationCoordinator, RegistrationRequest
from faceattend.application.registration_session import RegistrationDesktopSessionProcessor
from faceattend.application.runtime import DesktopMode, SessionProcessor
from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.repositories import LocalRepository
from faceattend.vision.active_liveness import ActiveLivenessChallengeEvaluator
from faceattend.vision.challenge_session import ChallengeSession
from faceattend.vision.face_analyzer import HeadlessFaceAnalyzer, create_face_analyzer
from faceattend.vision.matcher import ExactNumpyMatcher
from faceattend.vision.mediapipe_runtime import MediaPipeLivenessSession
from faceattend.vision.passive_liveness import (
    MiniFASNetPassiveLivenessDetector,
    TemporalPassiveLivenessSession,
)
from faceattend.vision.types import Frame, FrameEvidence


@dataclass(frozen=True, slots=True)
class RegistrationInput:
    display_name: str
    consent_granted: bool


class _DeferredLivenessSession:
    """Create one timestamped liveness session on its first camera frame."""

    def __init__(self, analyzer: HeadlessFaceAnalyzer) -> None:
        self.analyzer = analyzer
        self._session: Any = None

    def _get(self, frame: Frame) -> Any:
        if self._session is None:
            challenge = ChallengeSession()
            # ``ChallengeSession`` deliberately requires every observed frame
            # timestamp to be strictly later than its start timestamp.  The
            # first camera frame also starts and is immediately evaluated by
            # this session, so start one millisecond earlier.  Otherwise a
            # normal first frame is incorrectly rejected as non-monotonic.
            challenge.start(_start_before_first_frame(frame.captured_at_ns))
            evaluator = ActiveLivenessChallengeEvaluator(session=challenge)
            if not isinstance(self.analyzer.passive, MiniFASNetPassiveLivenessDetector):
                raise RuntimeError("local runtime requires the MiniFASNet adapter")
            self._session = MediaPipeLivenessSession(
                evaluator,
                TemporalPassiveLivenessSession(self.analyzer.passive),
                processor=self.analyzer,
            )
        return self._session

    def __call__(self, frame: Frame) -> FrameEvidence:
        return cast(FrameEvidence, self._get(frame)(frame))

    @property
    def phase(self) -> Any:
        return self._require().phase

    @property
    def result(self) -> Any:
        return self._require().result

    @property
    def failure_reason(self) -> str | None:
        return cast(str | None, self._require().failure_reason)

    @property
    def active(self) -> Any:
        return self._require().active

    @property
    def instruction(self) -> str:
        return cast(str, self._require().instruction)

    def take_embedding_candidates(self, **kwargs: int) -> tuple[FrameEvidence, ...]:
        return cast(tuple[FrameEvidence, ...], self._require().take_embedding_candidates(**kwargs))

    def take_embedding_candidate(self) -> FrameEvidence:
        return cast(FrameEvidence, self._require().take_embedding_candidate())

    def cancel(self) -> object:
        if self._session is not None:
            return self._session.cancel()
        return None

    def close(self) -> None:
        self.cancel()
        self.analyzer.close()

    def _require(self) -> Any:
        if self._session is None:
            raise RuntimeError("liveness session has not received a camera frame")
        return self._session


class LocalApplication:
    """Own local configuration and create a fresh worker-owned attempt per mode."""

    def __init__(self, config: LocalAppConfig) -> None:
        self.config = config
        self.repository = LocalRepository(SQLiteDatabase(config.database_path))
        self.repository.database.initialize()
        now = datetime.now(UTC)
        self.configuration_id = self.repository.add_configuration(
            self._configuration_payload(), created_at=now
        )
        self.embedding_model_id = self.repository.add_model(
            "embedding", config.embedding.metadata, created_at=now
        )
        self._registration_input: RegistrationInput | None = None
        self._attendance_session_id: str | None = None

    def configure_registration(self, data: RegistrationInput) -> None:
        if not data.display_name.strip() or not data.consent_granted:
            raise ValueError("name and explicit consent are required")
        self._registration_input = data

    def create_attendance_session(self, name: str) -> str:
        return self.repository.add_attendance_session(
            name, opened_at=datetime.now(UTC), actor_id="operator"
        )

    def configure_attendance(self, attendance_session_id: str) -> None:
        if not attendance_session_id.strip():
            raise ValueError("select or create an attendance session")
        self._attendance_session_id = attendance_session_id

    def processor_factory(self, mode: DesktopMode) -> SessionProcessor:
        analyzer = create_face_analyzer(
            self.config.manifest, passive_threshold=self.config.passive_threshold
        )
        session = _DeferredLivenessSession(analyzer)
        if mode is DesktopMode.REGISTRATION:
            if self._registration_input is None:
                session.close()
                raise RuntimeError("registration name and consent are required")
            registration_coordinator = RegistrationCoordinator(
                RegistrationRequest(
                    self._registration_input.display_name,
                    self._registration_input.consent_granted,
                    "attendance",
                    "operator",
                    self.configuration_id,
                    self.embedding_model_id,
                    self.config.embedding.metadata,
                    self.config.landmarker.metadata.version,
                ),
                session=session,
                face_analyzer=analyzer,
                matcher=ExactNumpyMatcher(
                    match_threshold=self.config.match_threshold,
                    ambiguity_margin=self.config.ambiguity_margin,
                ),
                repository=self.repository,
            )
            return RegistrationDesktopSessionProcessor(
                registration_coordinator, close=session.close
            )
        if mode is DesktopMode.ATTENDANCE:
            if self._attendance_session_id is None:
                session.close()
                raise RuntimeError("select an attendance session")
            attendance_coordinator = AttendanceCoordinator(
                AttendanceRequest(
                    self._attendance_session_id,
                    "operator",
                    self.configuration_id,
                    self.config.embedding.metadata,
                    self.config.landmarker.metadata.version,
                ),
                session=session,
                face_analyzer=analyzer,
                matcher=ExactNumpyMatcher(
                    match_threshold=self.config.match_threshold,
                    ambiguity_margin=self.config.ambiguity_margin,
                ),
                repository=self.repository,
            )
            return AttendanceDesktopSessionProcessor(attendance_coordinator, close=session.close)
        raise ValueError("home is not a capture mode")

    def _configuration_payload(self) -> dict[str, object]:
        return {
            "camera_index": self.config.camera_index,
            "passive_threshold": self.config.passive_threshold,
            "match_threshold": self.config.match_threshold,
            "ambiguity_margin": self.config.ambiguity_margin,
            "models": {
                "detector": self.config.detector.metadata.checksum,
                "embedding": self.config.embedding.metadata.checksum,
                "passive": self.config.passive.metadata.checksum,
                "landmarker": self.config.landmarker.metadata.checksum,
            },
        }


def _start_before_first_frame(captured_at_ns: int) -> int:
    """Return a millisecond start time strictly before the first frame."""
    timestamp_ms = captured_at_ns // 1_000_000
    return max(0, timestamp_ms - 1)
