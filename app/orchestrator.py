"""Lightweight intent routing and task coordination for Iris."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger("iris.orchestrator")

# --------------------------------------------------------------------------- #
# Intent patterns (rule-based; keep simple and offline)

INTENT_PATTERNS: Dict[str, list[str]] = {
    "DESCRIBE_SCENE": [
        r"\bwhat (do you|d'you) see\b",
        r"\bdescribe (what'?s|what is) (ahead|in front of me)\b",
        r"\bdescribe\b",
    ],
    "FIND_OBJECT": [
        r"\bfind (.+)",
        r"\blook for (.+)",
        r"\bsearch for (.+)",
    ],
    "STOP": [
        r"\bstop\b",
        r"\bcancel\b",
        r"\benough\b",
    ],
    "PAUSE": [
        r"\bpause\b",
    ],
    "RESUME": [
        r"\bresume\b",
        r"\bcontinue\b",
    ],
    "SETTINGS": [
        r"\bverbosity (low|normal|high)\b",
    ],
    "HELP": [
        r"\bhelp\b",
        r"\bwhat can you do\b",
    ],
    "CONFIRM": [
        r"\b(yes|yeah|yep|sure|ok|okay|affirmative)\b",
        r"\b(keep|continue)\s+(looking|searching)\b",
    ],
    "DECLINE": [
        r"\b(no|nah|nope)\b",
        r"\b(don't|do not)\b",
    ],
}


WAKE_PATTERN = re.compile(r"^\s*iris\b[.,:;!\s-]*", re.IGNORECASE)


@dataclass
class TaskState:
    """Runtime task state shared across subsystems."""

    active_goal: str | None = None
    paused: bool = False
    verbosity: str = "normal"  # low|normal|high
    last_tts_ts: float = 0.0
    tts_cooldown_s: float = 2.5
    wake_word_enabled: bool = True
    pending_goal: str | None = None
    awaiting_goal_confirmation: bool = False


state = TaskState()


# --------------------------------------------------------------------------- #
def parse_intent(text: str) -> Tuple[str | None, Dict[str, str], bool]:
    """Return (intent, slots, addressed) for an utterance."""
    utterance = text.strip()

    if not utterance:
        logger.debug("Ignoring empty utterance.")
        return None, {}, False

    # Wake word currently optional; treat all utterances as addressed.
    addressed = True

    lowered = utterance.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            slots: Dict[str, str] = {}
            if intent == "FIND_OBJECT":
                for slot_pattern in INTENT_PATTERNS["FIND_OBJECT"]:
                    goal_match = re.search(slot_pattern, lowered)
                    if goal_match and goal_match.groups():
                        slots["object"] = goal_match.group(1).strip()
                        break
            elif intent == "SETTINGS":
                level_match = re.search(r"(low|normal|high)", lowered)
                if level_match:
                    slots["verbosity"] = level_match.group(1)
            logger.info("Parsed intent=%s slots=%s addressed=%s", intent, slots, addressed)
            return intent, slots, True

    # Default to describe scene if addressed but unmatched.
    logger.info("Utterance '%s' did not match any known intent.", text)
    return None, {}, True


def can_speak_now() -> bool:
    """Return True if speech should be emitted (respects pause + cooldown)."""
    if state.paused:
        logger.debug("Speech suppressed: system paused.")
        return False
    elapsed = time.time() - state.last_tts_ts
    if elapsed < state.tts_cooldown_s:
        logger.debug("Speech suppressed: cooldown %.2fs remaining.", state.tts_cooldown_s - elapsed)
        return False
    return True


def note_spoken() -> None:
    """Record a speech event."""
    state.last_tts_ts = time.time()


def clear_goal() -> None:
    """Clear any active goal with logging."""
    if state.active_goal:
        logger.info("Clearing active goal '%s'.", state.active_goal)
    state.active_goal = None


def handle_intent(
    intent: str | None,
    slots: Dict[str, str],
    *,
    describer,
    frames,
    preview: bool,
    tts,
    latest_frame=None,
) -> Dict[str, object]:
    """Execute side-effects for a parsed intent.

    Returns a dict with at least:
        handled (bool): whether an intent fired
        preview_frame (ndarray | None): optional frame to persist upstream
    """
    result: Dict[str, object] = {"handled": False, "preview_frame": None, "message": None}

    if intent is None:
        return result

    intent_upper = intent.upper()
    logger.info("Handling intent=%s slots=%s state=%s", intent_upper, slots, state)

    if intent_upper == "FIND_OBJECT":
        target = slots.get("object")
        if not target:
            logger.info("No object specified for FIND_OBJECT; ignoring.")
            return result

        target = target.strip()
        if not target:
            logger.info("Target resolved to empty string; ignoring FIND_OBJECT.")
            return result

        if state.active_goal or state.pending_goal:
            logger.info(
                "Replacing existing search (active=%s, pending=%s) with target '%s'.",
                state.active_goal,
                state.pending_goal,
                target,
            )
        clear_goal()
        state.pending_goal = None
        state.awaiting_goal_confirmation = False

        # Attempt immediate detection from the latest frame before engaging loop.
        if latest_frame is not None:
            try:
                detections = describer.detect(latest_frame)
                match = describer.select_best_match(detections, target)
            except Exception:
                logger.exception("Immediate detection attempt failed for '%s'.", target)
                match = None

            if match and match.confidence >= 0.5:
                width = latest_frame.shape[1]
                location = describer.where_in_fov(match.bbox, width)
                phrase = f"I already see a {match.label} on your {location}."
                if can_speak_now():
                    _safe_say(tts, phrase)
                else:
                    logger.info("Immediate find confirmation suppressed (cooldown/pause).")
                result["message"] = phrase
                result["handled"] = True
                logger.info(
                    "Immediate detection satisfied goal '%s' (confidence=%.2f, location=%s).",
                    match.label,
                    match.confidence,
                    location,
                )
                return result

        # Not found right away; ask user whether to keep looking.
        state.pending_goal = target
        state.awaiting_goal_confirmation = True
        logger.info("Awaiting confirmation to keep searching for '%s'.", target)
        prompt = f"I can't see {target} yet. Should I keep looking?"
        if can_speak_now():
            _safe_say(tts, prompt)
        else:
            logger.info("Search continuation prompt suppressed (cooldown/pause).")
        result["message"] = prompt
        result["handled"] = True
        return result

    if intent_upper == "DESCRIBE_SCENE":
        description, preview_frame = describer.describe(frames, preview=preview)
        result["preview_frame"] = preview_frame
        if can_speak_now():
            _safe_say(tts, description)
        else:
            logger.info("Description suppressed by speech cooldown/pause.")
        result["message"] = description
        result["handled"] = True
        return result

    if intent_upper == "STOP":
        cleared_goal = state.active_goal
        clear_goal()
        state.paused = False
        state.pending_goal = None
        state.awaiting_goal_confirmation = False
        confirmation = "Stopped."
        if can_speak_now():
            _safe_say(tts, confirmation)
        else:
            logger.info("Stop confirmation skipped due to cooldown/pause.")
        result["message"] = confirmation
        result["handled"] = True
        logger.info("Stop processed (cleared_goal=%s).", cleared_goal)
        return result

    if intent_upper == "PAUSE":
        state.paused = True
        confirmation = "Paused."
        _safe_say(tts, confirmation)
        result["message"] = confirmation
        result["handled"] = True
        logger.info("System paused.")
        return result

    if intent_upper == "RESUME":
        was_paused = state.paused
        state.paused = False
        confirmation = "Resumed."
        if can_speak_now():
            _safe_say(tts, confirmation)
        else:
            logger.info("Resume confirmation skipped due to cooldown.")
        result["message"] = confirmation
        result["handled"] = True
        logger.info("System resumed (was_paused=%s).", was_paused)
        return result

    if intent_upper == "SETTINGS":
        verbosity = slots.get("verbosity")
        if verbosity:
            state.verbosity = verbosity
            logger.info("Verbosity set to %s.", verbosity)
            confirmation = f"Verbosity set to {verbosity}."
            if can_speak_now():
                _safe_say(tts, confirmation)
            else:
                logger.info("Verbosity confirmation suppressed (cooldown/pause).")
            result["message"] = confirmation
        result["handled"] = True
        return result

    if intent_upper == "HELP":
        help_text = (
            "You can say: describe what you see, find an object, pause, resume, stop, "
            "or set verbosity low, normal, or high."
        )
        if can_speak_now():
            _safe_say(tts, help_text)
        else:
            logger.info("Help message suppressed by cooldown/pause.")
        result["message"] = help_text
        result["handled"] = True
        return result

    if intent_upper == "CONFIRM":
        if not state.awaiting_goal_confirmation or not state.pending_goal:
            logger.info("Confirmation received but no pending search goal.")
            result["handled"] = True
            return result
        target = state.pending_goal
        state.active_goal = target
        state.pending_goal = None
        state.awaiting_goal_confirmation = False
        logger.info("User confirmed persistent search for '%s'.", target)
        confirmation = f"Okay, I'll keep searching for {target}."
        if can_speak_now():
            _safe_say(tts, confirmation)
        else:
            logger.info("Search confirmation suppressed by cooldown/pause.")
        result["message"] = confirmation
        result["handled"] = True
        return result

    if intent_upper == "DECLINE":
        if not state.awaiting_goal_confirmation or not state.pending_goal:
            logger.info("Decline received without pending goal; ignoring.")
            result["handled"] = True
            return result
        target = state.pending_goal
        state.pending_goal = None
        state.awaiting_goal_confirmation = False
        confirmation = f"Okay, I won't keep searching for {target}."
        if can_speak_now():
            _safe_say(tts, confirmation)
        else:
            logger.info("Decline acknowledgement suppressed by cooldown/pause.")
        result["message"] = confirmation
        result["handled"] = True
        return result

    if intent_upper == "WAKE_ACK":
        result["handled"] = True
        logger.debug("Wake word acknowledged; awaiting command.")
        return result

    logger.debug("No handler for intent %s.", intent_upper)
    return result


def _safe_say(tts, text: str) -> None:
    """Speak via TTS with logging + cooldown note."""
    try:
        tts.say(text)
    except Exception:
        logger.exception("TTS failed for text: %s", text)
    else:
        note_spoken()
