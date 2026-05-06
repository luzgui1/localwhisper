# tools.py
#
# Three tools exposed to the LangChain agent.
# Session reads/writes go through session.py — never directly to _state.

import json
import math
import os
import time

import numpy as np
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from . import session
from logger import log

# ── Encoder singleton ─────────────────────────────────────────────────────────
log.info("Loading SentenceTransformer model")
_t0 = time.perf_counter()
_encoder = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Model loaded in %.2fs", time.perf_counter() - _t0)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GetUserLocationInput(BaseModel):
    pass


class SearchAndRankPlacesInput(BaseModel):
    query: str = Field(
        description=(
            "A specific search term derived from what the user wants, in their own language. "
            "Include the type of venue AND the user's mood or requirement. "
            "Examples: 'chocolate quente aconchegante', 'bar animado com música ao vivo', "
            "'restaurante italiano barato para jantar'. "
            "Never use generic terms like 'places' or 'venues'."
        )
    )
    lat: float = Field(
        description="Latitude of the user's location, taken from get_user_location output."
    )
    lng: float = Field(
        description="Longitude of the user's location, taken from get_user_location output."
    )
    radius_m: int = Field(
        default=500,
        description=(
            "Search radius in metres. Use 500 for city-centre searches. "
            "Increase to 1500–2000 if the user mentions they are willing to travel further."
        )
    )


class GetSessionPlacesInput(BaseModel):
    name: str = Field(
        default="",
        description=(
            "Name of a specific place for full details. "
            "Leave empty for questions about the full list of previously found places — "
            "use this when the user asks which are open, which are closest, "
            "which are in a specific neighbourhood, or how they compare."
        )
    )


# ── Tool 1: get_user_location ─────────────────────────────────────────────────

@tool("get_user_location", args_schema=GetUserLocationInput)
def get_user_location() -> str:
    """
    Get the user's current geographic location.
    Call this first, before any place search, when the user asks for a
    venue recommendation. Returns cached result instantly on subsequent calls.
    """
    log.info("→ get_user_location called")
    t0 = time.perf_counter()

    cached = session.get_location()
    if cached:
        log.info("← get_user_location: session hit in %.3fs", time.perf_counter() - t0)
        return json.dumps(cached)

    try:
        resp = requests.get("http://ip-api.com/json", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            msg = f"ip-api returned: {data.get('status')}"
            log.warning("  %s", msg)
            return json.dumps({"error": msg})

        location = {
            "lat":     data["lat"],
            "lng":     data["lon"],
            "city":    data.get("city", "unknown"),
            "country": data.get("country", "unknown"),
        }
        session.set_location(location)
        log.info(
            "← get_user_location: fetched in %.2fs → %s, %s",
            time.perf_counter() - t0, location["city"], location["country"],
        )
        return json.dumps(location)

    except requests.Timeout:
        log.error("  ip-api.com timed out")
        return json.dumps({"error": "ip-api.com timed out — try again"})
    except Exception as e:
        log.error("  get_user_location failed: %s", e)
        return json.dumps({"error": str(e)})


# ── Tool 2: search_and_rank_places ────────────────────────────────────────────

@tool("search_and_rank_places", args_schema=SearchAndRankPlacesInput)
def search_and_rank_places(query: str, lat: float, lng: float, radius_m: int = 500) -> str:
    """
    Search for leisure venues near a location and return the top 5 results
    ranked by relevance to the user's request. Call this after get_user_location,
    using a query that captures the user's specific mood and requirements.
    """
    log.info("→ search_and_rank_places  query=%r  lat=%.4f  lng=%.4f  radius=%dm",
             query, lat, lng, radius_m)
    t0 = time.perf_counter()

    api_key = os.getenv("GOOGLE_MAPS_API")
    if not api_key:
        return json.dumps({"error": "GOOGLE_MAPS_API env variable not set"})

    try:
        import googlemaps
        gmaps = googlemaps.Client(key=api_key, requests_kwargs={"timeout": 10})

        raw = gmaps.places(
            query=query,
            location=(lat, lng),
            radius=radius_m,
        ).get("results", [])[:20]

        log.info("  gmaps.places → %d results", len(raw))

        places = []
        for i, r in enumerate(raw):
            place_id   = r.get("place_id", "")
            place_name = r.get("name", "?")
            log.debug("  [%d/%d] details for %r", i + 1, len(raw), place_name)

            details = {}
            if place_id:
                try:
                    details = gmaps.place(
                        place_id=place_id,
                        fields=["website", "reviews", "opening_hours"],
                    ).get("result", {})
                except Exception as e:
                    log.warning("    detail fetch failed for %r: %s", place_name, e)

            review_texts = [
                rev.get("text", "")
                for rev in (details.get("reviews") or [])[:3]
                if isinstance(rev, dict) and rev.get("text")
            ]

            places.append({
                "place_id":      place_id,
                "name":          place_name,
                "address":       r.get("formatted_address") or r.get("vicinity", ""),
                "type":          (r.get("types") or ["unknown"])[0],
                "lat":           r.get("geometry", {}).get("location", {}).get("lat"),
                "lng":           r.get("geometry", {}).get("location", {}).get("lng"),
                "rating":        r.get("rating"),
                "ratings_total": r.get("user_ratings_total"),
                "price_level":   r.get("price_level"),
                "open_now":      r.get("opening_hours", {}).get("open_now"),
                "website":       details.get("website"),
                "reviews":       review_texts,
            })

        log.info("  fetched %d places in %.2fs — ranking…", len(places), time.perf_counter() - t0)

    except Exception as e:
        log.error("  search failed: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})

    if not places:
        return json.dumps([])

    # ── Rank ──────────────────────────────────────────────────────────────────

    def clamp01(x): return max(0.0, min(1.0, float(x)))
    def safe_float(x, default=None):
        try: return float(x)
        except: return default

    def haversine_km(lat1, lng1, lat2, lng2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat     = math.radians(lat2 - lat1)
        dlng     = math.radians(lng2 - lng1)
        a        = (math.sin(dlat / 2) ** 2
                    + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2)
        return round(6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)

    def proximity_score(distance_km, radius_m):
        """Closer places score higher. Normalised by the search radius."""
        if distance_km is None:
            return 0.5
        radius_km = radius_m / 1000
        return clamp01(1.0 - distance_km / radius_km)

    def price_score(p):
        v = safe_float(p)
        return 0.5 if v is None else clamp01(1.0 - v / 4.0)

    def rating_score(rating, total):
        r = safe_float(rating)
        if r is None: return 0.0
        n    = safe_float(total, 0.0) or 0.0
        conf = clamp01(math.log1p(n) / math.log1p(10_000))
        return clamp01(r / 5.0) * conf

    def place_to_text(p):
        return (
            f"Name: {p.get('name','')}\n"
            f"Type: {p.get('type','')}\n"
            f"Address: {p.get('address','')}\n"
            f"Rating: {p.get('rating','N/A')} ({p.get('ratings_total',0)} reviews)\n"
            f"Price level: {p.get('price_level','N/A')}\n"
            f"Open now: {p.get('open_now','unknown')}\n"
            f"Reviews: {' '.join(p.get('reviews') or [])}"
        ).strip()

    texts      = [place_to_text(p) for p in places]
    q_emb      = _encoder.encode([query], normalize_embeddings=True)[0]
    p_embs     = _encoder.encode(texts,   normalize_embeddings=True)
    sem_scores = np.dot(p_embs, q_emb)

    # Compute distance for every place that has coordinates
    for p in places:
        if p.get("lat") and p.get("lng"):
            p["distance_km"] = haversine_km(lat, lng, p["lat"], p["lng"])
        else:
            p["distance_km"] = None

    W_SEM, W_RATING, W_PROXIMITY, W_PRICE = 0.60, 0.15, 0.15, 0.10

    for i, p in enumerate(places):
        sem   = float(sem_scores[i])
        sem01 = clamp01((sem + 1.0) / 2.0)
        r_s   = rating_score(p.get("rating"), p.get("ratings_total"))
        pr_s  = proximity_score(p.get("distance_km"), radius_m)
        pc_s  = price_score(p.get("price_level"))

        p["final_score"] = round(
            W_SEM * sem01 + W_RATING * r_s + W_PROXIMITY * pr_s + W_PRICE * pc_s, 4
        )
        log.debug(
            "    %-30s  dist=%.2fkm  sem=%.3f  rating=%.3f  prox=%.3f  price=%.3f  → %.3f",
            p.get("name", "?")[:30],
            p.get("distance_km") or 0,
            sem01, r_s, pr_s, pc_s, p["final_score"],
        )

    ranked = sorted(places, key=lambda x: x["final_score"], reverse=True)

    # ── Write full data to session ────────────────────────────────────────────
    # All 20 full dicts go to session — not just the top 5.
    # get_place_details can then serve any of them without hitting Maps again.
    session.set_places(ranked)

    log.info(
        "← search_and_rank_places OK in %.2fs → top: %r (%.3f)",
        time.perf_counter() - t0,
        ranked[0]["name"] if ranked else "—",
        ranked[0].get("final_score", 0) if ranked else 0,
    )

    return json.dumps([
        {
            "name":         p["name"],
            "address":      p["address"],
            "rating":       p.get("rating"),
            "open_now":     p.get("open_now"),
            "website":      p.get("website"),
            "distance_km":  p.get("distance_km"),
            "final_score":  p["final_score"],
            "top_review":   p["reviews"][0] if p.get("reviews") else None,
        }
        for p in ranked[:5]
    ], ensure_ascii=False)


# ── Tool 3: get_session_places ───────────────────────────────────────────────

@tool("get_session_places", args_schema=GetSessionPlacesInput)
def get_session_places(name: str = "") -> str:
    """
    Answer any follow-up question about places already found in this conversation
    without calling Google Maps again. Use this whenever the user asks about the
    current set of results — including which are open, which are closest ('perto de mim'),
    which are in a specific neighbourhood, or details about one specific place.
    Each place includes distance_km from the user when coordinates are available.
    Only call search_and_rank_places when the user wants a completely new type of venue.
    """
    log.info("→ get_session_places  name=%r", name)
    t0 = time.perf_counter()

    # Specific place requested
    if name:
        place = session.get_place(name)
        if not place:
            available = session.all_place_names()
            log.warning("  %r not in session. available: %s", name, available)
            return json.dumps({
                "error":     f"No cached data for '{name}'.",
                "available": available,
            })
        log.info("← get_session_places (one) in %.3fs → %r", time.perf_counter() - t0, name)
        return json.dumps(place, ensure_ascii=False)

    # No name — return all cached places with distance already computed during ranking
    places = session.all_places()
    if not places:
        return json.dumps({"error": "No places cached yet. Call search_and_rank_places first."})

    slim = [
        {
            "name":        p["name"],
            "address":     p["address"],
            "rating":      p.get("rating"),
            "open_now":    p.get("open_now"),
            "price_level": p.get("price_level"),
            "website":     p.get("website"),
            "final_score": p.get("final_score"),
            "distance_km": p.get("distance_km"),
        }
        for p in places
    ]
    log.info("← get_session_places (all) in %.3fs → %d places", time.perf_counter() - t0, len(slim))
    return json.dumps(slim, ensure_ascii=False)