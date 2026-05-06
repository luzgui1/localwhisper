# session.py
#
# Owns all session state for one conversation.
# Tools never touch _state directly — they call the functions below.
#
# Future evolution path:
#   - swap _state for a Redis client  → change only this file
#   - add TTL / expiry logic          → change only this file
#   - move to per-session-id store    → change only this file
#   tools.py and agent.py stay untouched in all cases.

from logger import log

_state: dict = {
    "location": None,   # {"lat": float, "lng": float, "city": str, "country": str}
    "places":   {},     # name (str) → full place dict
}


# ── Location ──────────────────────────────────────────────────────────────────

def get_location() -> dict | None:
    return _state["location"]


def set_location(location: dict) -> None:
    _state["location"] = location
    log.debug("session: location set → %s, %s", location.get("city"), location.get("country"))


# ── Places ────────────────────────────────────────────────────────────────────

def get_place(name: str) -> dict | None:
    """Return the full place dict for a given name, or None if not cached."""
    match = _state["places"].get(name)
    if match:
        log.debug("session: place hit → %r", name)
    else:
        log.debug("session: place miss → %r", name)
    return match


def set_places(places: list[dict]) -> None:
    """Store the full dict for every place in the list, keyed by name."""
    for place in places:
        _state["places"][place["name"]] = place
    log.debug("session: stored %d places", len(places))


def all_place_names() -> list[str]:
    """Return the names of all places currently cached."""
    return list(_state["places"].keys())


def all_places() -> list[dict]:
    """Return the full dict for every cached place."""
    return list(_state["places"].values())


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def clear() -> None:
    """Reset the session — useful for 'start over' commands."""
    _state["location"] = None
    _state["places"]   = {}
    log.info("session: cleared")


def summary() -> dict:
    """Return a lightweight overview of the current session state."""
    return {
        "has_location": _state["location"] is not None,
        "city":         (_state["location"] or {}).get("city"),
        "places_cached": len(_state["places"]),
        "place_names":  all_place_names(),
    }