"""Active liveness challenge definitions."""

from __future__ import annotations

from enum import StrEnum


class ActiveLivenessChallenge(StrEnum):
    """Enum for active liveness challenges."""

    BLINK_TWICE = "blink_twice"
    BLINK_TURN_LEFT_RIGHT = "blink_turn_left_right"
    # LOOK_UP = "look_up"
    # LOOK_DOWN = "look_down"
