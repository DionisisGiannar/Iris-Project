"""Intent recognition for the Iris Assistant."""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """Enumerated set of supported intents."""

    DESCRIBE = "describe"
    QUIT = "quit"


STOP_KEYWORDS = {"quit", "exit", "stop app", "goodbye", "shut down", "terminate"}


def classify_intent(text: str) -> Intent:
    """Classify an utterance into a supported intent."""
    normalized = " ".join(text.lower().strip().split())
    if any(keyword in normalized for keyword in STOP_KEYWORDS):
        return Intent.QUIT
    # Default behavior: describe the current scene
    return Intent.DESCRIBE

