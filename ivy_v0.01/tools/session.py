# tools/session.py
#
# Owns all session state for every active conversation.
#
# CLI (main.py):   one conversation → one session_id ("cli")
# FastAPI:         each user sends their own session_id with every request
# Telegram:        chat_id becomes the session_id automatically
#
# How tools know which session to use — ContextVar:
#   Before calling agent.run(), the API sets the active session_id once:
#       session.set_active("abc-123")
#   Every tool then calls session.get_location(), session.get_place() etc.
#   with no arguments — they all read the ContextVar internally.
#   If two requests run in parallel, each thread has its own ContextVar value,
#   so they never interfere with each other.
#
# Future evolution:
#   - swap _store for Redis  → change only this file
#   - add TTL/expiry logic   → change only this file
#   - tools.py and agent.py stay untouched in all cases

import contextvars
from logger import log

# ── Storage ───────────────────────────────────────────────────────────────────
# _store maps session_id → state dict.
# Each state dict has the same shape as the old single _state.

_store: dict[str, dict] = {}

# ── Active session ─────────────────────────────────────────────────────────────
# ContextVar gives each thread its own copy of the current session_id.
# Thread A can have "abc-123" while thread B has "def-456" simultaneously.

_active_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="cli"
)


def set_active(session_id: str) -> None:
    """
    Set the active session for the current thread.
    Call this once before agent.run() on every request.
    """
    _active_session_id.set(session_id)
    log.debug("session: active → %r", session_id)


def _current() -> dict:
    """
    Return the state dict for the active session.
    Creates a fresh slot if this session_id is new.
    """
    sid = _active_session_id.get()
    if sid not in _store:
        _store[sid] = {"location": None, "places": {}, "history": []}
        log.debug("session: new slot created for %r", sid)
    return _store[sid]


# ── Location ──────────────────────────────────────────────────────────────────
# These signatures are unchanged — tools call them with no arguments.

def get_location() -> dict | None:
    return _current()["location"]


def set_location(location: dict) -> None:
    _current()["location"] = location
    log.debug("session: location set → %s, %s",
              location.get("city"), location.get("country"))


# ── Places ────────────────────────────────────────────────────────────────────

def get_place(name: str) -> dict | None:
    """Return the full place dict for a given name, or None if not cached."""
    match = _current()["places"].get(name)
    if match:
        log.debug("session: place hit → %r", name)
    else:
        log.debug("session: place miss → %r", name)
    return match


def set_places(places: list[dict]) -> None:
    """Store the full dict for every place in the list, keyed by name."""
    state = _current()
    for place in places:
        state["places"][place["name"]] = place
    log.debug("session: stored %d places", len(places))


def all_place_names() -> list[str]:
    return list(_current()["places"].keys())


def all_places() -> list[dict]:
    return list(_current()["places"].values())


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def clear() -> None:
    """Reset the active session — useful for 'start over' commands."""
    sid = _active_session_id.get()
    _store[sid] = {"location": None, "places": {}, "history": []}
    log.info("session: cleared for %r", sid)


# ── History ───────────────────────────────────────────────────────────────────
# Conversation history lives here so each user keeps their own context.
# The API reads it before calling agent.run() and appends to it after.

def get_history() -> list[dict]:
    """Return the conversation history for the active session."""
    return _current()["history"]


def append_to_history(role: str, content: str) -> None:
    """Add one message to the conversation history."""
    _current()["history"].append({"role": role, "content": content})
    log.debug("session: history +%s (%d chars)", role, len(content))


def summary() -> dict:
    """Lightweight overview of the current session — useful for debugging."""
    state = _current()
    return {
        "session_id":    _active_session_id.get(),
        "has_location":  state["location"] is not None,
        "city":          (state["location"] or {}).get("city"),
        "places_cached": len(state["places"]),
        "place_names":   all_place_names(),
    }