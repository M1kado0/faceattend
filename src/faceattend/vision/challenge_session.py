"""Short-lived randomized active-liveness challenge sessions."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ChallengeAction(StrEnum):
    """Actions that can be requested from a participant."""

    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LOOK_UP = "look_up"
    SMILE = "smile"


class ChallengeSessionStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CHALLENGE_TIMEOUT = "challenge_timeout"
    SESSION_TIMEOUT = "session_timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RandomSource(Protocol):
    def randint(self, start: int, stop: int) -> int: ...

    def sample(self, population: list[ChallengeAction], k: int) -> list[ChallengeAction]: ...


@dataclass(frozen=True, slots=True)
class ChallengeSessionConfig:
    """Timing and sequence limits for one active-liveness session."""

    min_challenges: int = 2
    max_challenges: int = 3
    challenge_timeout_ms: int = 8_000
    session_timeout_ms: int = 30_000
    challenge_pool: tuple[ChallengeAction, ...] = (
        ChallengeAction.BLINK,
        ChallengeAction.TURN_LEFT,
        ChallengeAction.TURN_RIGHT,
    )

    def __post_init__(self) -> None:
        if self.min_challenges < 1:
            raise ValueError("min_challenges must be positive")
        if self.max_challenges < self.min_challenges:
            raise ValueError("max_challenges must be >= min_challenges")
        if len(set(self.challenge_pool)) != len(self.challenge_pool):
            raise ValueError("challenge_pool must not contain duplicates")
        if self.max_challenges > len(self.challenge_pool):
            raise ValueError("max_challenges cannot exceed challenge_pool size")
        if self.challenge_timeout_ms <= 0 or self.session_timeout_ms <= 0:
            raise ValueError("timeouts must be positive")
        if self.challenge_timeout_ms > self.session_timeout_ms:
            raise ValueError("challenge timeout cannot exceed session timeout")


@dataclass(frozen=True, slots=True)
class ChallengeSessionSnapshot:
    """Auditable public state of a challenge session."""

    status: ChallengeSessionStatus
    sequence: tuple[ChallengeAction, ...]
    completed: tuple[ChallengeAction, ...]
    current: ChallengeAction | None
    started_at_ms: int | None
    current_started_at_ms: int | None
    reason: str | None = None


class ChallengeSession:
    """Generate and track one ordered, time-bounded challenge sequence.

    CV evidence is evaluated by a separate active-liveness evaluator. This class
    owns only sequence selection, lifecycle, monotonic timestamps, and timeouts.
    """

    def __init__(
        self,
        config: ChallengeSessionConfig | None = None,
        *,
        random_source: RandomSource | None = None,
    ) -> None:
        self.config = config or ChallengeSessionConfig()
        self._random = random_source or secrets.SystemRandom()
        self.reset()

    def reset(self) -> None:
        self._sequence: tuple[ChallengeAction, ...] = ()
        self._completed: tuple[ChallengeAction, ...] = ()
        self._started_at_ms: int | None = None
        self._current_started_at_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._status = ChallengeSessionStatus.NOT_STARTED
        self._reason: str | None = None

    def fail(self, reason: str) -> ChallengeSessionSnapshot:
        if self._status is ChallengeSessionStatus.IN_PROGRESS:
            return self._fail(ChallengeSessionStatus.FAILED, reason)
        return self.snapshot

    def cancel(self) -> ChallengeSessionSnapshot:
        if self._status is ChallengeSessionStatus.IN_PROGRESS:
            return self._fail(ChallengeSessionStatus.CANCELLED, "cancelled")
        return self.snapshot

    @property
    def snapshot(self) -> ChallengeSessionSnapshot:
        current = None
        if self._status is ChallengeSessionStatus.IN_PROGRESS:
            current = self._sequence[len(self._completed)]
        return ChallengeSessionSnapshot(
            status=self._status,
            sequence=self._sequence,
            completed=self._completed,
            current=current,
            started_at_ms=self._started_at_ms,
            current_started_at_ms=self._current_started_at_ms,
            reason=self._reason,
        )

    def start(self, timestamp_ms: int) -> ChallengeSessionSnapshot:
        if self._status is ChallengeSessionStatus.IN_PROGRESS:
            raise RuntimeError("challenge session is already in progress")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        count = self._random.randint(self.config.min_challenges, self.config.max_challenges)
        self._sequence = tuple(self._random.sample(list(self.config.challenge_pool), count))
        self._completed = ()
        self._started_at_ms = timestamp_ms
        self._current_started_at_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms
        self._status = ChallengeSessionStatus.IN_PROGRESS
        self._reason = None
        return self.snapshot

    def observe(self, timestamp_ms: int) -> ChallengeSessionSnapshot:
        self._check_started()
        self._check_timestamp(timestamp_ms)
        return self._observe_timeout(timestamp_ms)

    def _observe_timeout(self, timestamp_ms: int) -> ChallengeSessionSnapshot:
        if self._status is not ChallengeSessionStatus.IN_PROGRESS:
            return self.snapshot
        assert self._started_at_ms is not None
        assert self._current_started_at_ms is not None
        if timestamp_ms - self._started_at_ms > self.config.session_timeout_ms:
            return self._fail(ChallengeSessionStatus.SESSION_TIMEOUT, "session_timeout")
        if timestamp_ms - self._current_started_at_ms > self.config.challenge_timeout_ms:
            return self._fail(ChallengeSessionStatus.CHALLENGE_TIMEOUT, "challenge_timeout")
        return self.snapshot

    def complete_current(self, timestamp_ms: int) -> ChallengeSessionSnapshot:
        self._check_started()
        # Evaluators may already have observed this frame to enforce timeout
        # on every path. Re-check the clocks without advancing time twice.
        if self._last_timestamp_ms == timestamp_ms:
            self._observe_timeout(timestamp_ms)
        else:
            self.observe(timestamp_ms)
        if self._status is not ChallengeSessionStatus.IN_PROGRESS:
            return self.snapshot
        current = self._sequence[len(self._completed)]
        self._completed = (*self._completed, current)
        if len(self._completed) == len(self._sequence):
            self._status = ChallengeSessionStatus.COMPLETED
            self._current_started_at_ms = None
        else:
            self._current_started_at_ms = timestamp_ms
        return self.snapshot

    def _check_started(self) -> None:
        if self._status is ChallengeSessionStatus.NOT_STARTED:
            raise RuntimeError("challenge session has not been started")

    def _check_timestamp(self, timestamp_ms: int) -> None:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp_ms = timestamp_ms

    def _fail(self, status: ChallengeSessionStatus, reason: str) -> ChallengeSessionSnapshot:
        self._status = status
        self._reason = reason
        return self.snapshot
