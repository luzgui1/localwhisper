# pipeline/tools.py
import math
import os
import numpy as np
import googlemaps
import streamlit as st
from . import logger
from streamlit_js_eval import streamlit_js_eval
from sentence_transformers import SentenceTransformer

def get_user_location():

    if "user_location" in st.session_state and st.session_state["user_location"]:
        return st.session_state["user_location"]

    if st.button("📍 Usar minha localização"):
        result = streamlit_js_eval(
            js_expressions="""
            new Promise((resolve) => {
              navigator.geolocation.getCurrentPosition(
                (pos) => resolve({
                  lat: pos.coords.latitude,
                  lng: pos.coords.longitude
                }),
                () => resolve(null)
              );
            });
            """,
            key="geo"
        )

        logger.info(f"MODULE_CALLED:'user-location-tool', DATA: '{result}'")

        if result:
            return result
        else:
            st.warning("Não foi possível obter a localização.")
            return None

    return None

def get_places(user_location: dict,search_terms: list, radius_m=250, max_places=20):
    """
    user_location: {lat: float, lng: float}
    radius_m: int
    max_places: int

    Returns dict like:
        {
        "user_id":str,
        "user_loc":{lat: float, lng: float},
        "places_nearby": [
            {"place_name":str,"place_id":str,"place_address":str,"place_type":str,"place_website":str,"place_price_level":int,"place_opening_hours":str,"place_reviews":str}
        ],
        "places_ratings":["place_id":str,"model_rating":float]
        }
    """

    api_key = os.getenv("GOOGLE_MAPS_API")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API not found in environment variables")
    
    gmaps = googlemaps.Client(key=api_key)
    lat = user_location["lat"]
    lng = user_location["lng"]

    nearby_places = gmaps.places(query=search_terms, location=(lat, lng), radius=radius_m)

    result = nearby_places.get("results", [])[:max_places]

    if not result:
        return None
    
    result_dict = []
    
    for r in result:
        place_id = r.get("place_id")

        details = gmaps.place(
            place_id = place_id,
            fields=["website","reviews","opening_hours"]
        ).get("result",{})

        result_dict.append({
            "place_name":r.get("name"),
            "place_id":r.get("place_id", ""),
            "place_address":r.get("formatted_address", "") or r.get("vicinity", ""),
            "place_type":(r.get("types", "") or [None])[0],
            "place_rating":r.get("rating", ""),
            "place_user_ratings_total":r.get("user_ratings_total"),
            "place_price_level":r.get("price_level", ""),
            "place_open_now":r.get("open_now"),
            "place_opening_hours":r.get("opening_hours", ""),
            "place_website":details.get("website"),
            "place_reviews":details.get("reviews") or [],
            "place_opening_hours":details.get("opening_hours") or {}
        })
    
    return result_dict

def rank_places(user_query: str, places_nearby: list):
    """
    Semantic ranking for a small list of nearby places.
    Input `places_nearby` is expected to be a list of dicts like `compact_places`.

    Returns:
        ranked_places: same list of dicts, sorted by semantic similarity (best first),
                       each dict gets an extra key: "semantic_score".
    """

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    if not places_nearby:
        return []

    # 1) Build a compact text representation for each place
    def place_to_text(p: dict) -> str:
        name = p.get("name", "") or ""
        address = p.get("address", "") or ""
        price = p.get("price", "")
        rating = p.get("rating", "")
        ratings_total = p.get("ratings_total", "")
        reviews = p.get("reviews") or []

        reviews_text = " ".join([r for r in reviews if isinstance(r, str)])

        return (
            f"Name: {name}\n"
            f"Address: {address}\n"
            f"Rating: {rating} (total ratings: {ratings_total})\n"
            f"Price level: {price}\n"
            f"Reviews: {reviews_text}\n"
        ).strip()

    place_texts = [place_to_text(p) for p in places_nearby]

    # 2) Encode query + places (normalize so dot product == cosine similarity)
    query_emb = encoder.encode([user_query], normalize_embeddings=True)[0]
    place_embs = encoder.encode(place_texts, normalize_embeddings=True)

    # 3) Cosine similarity (dot product because embeddings are normalized)
    scores = np.dot(place_embs, query_emb)

    # --- helpers for final_score ---
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def safe_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    # Weights
    W_SEM = 0.60
    W_RATING = 0.30
    W_PRICE = 0.10

    def price_score(price_level):
        """
        Normalization of price. The cheaper, the better.
        """
        p = safe_float(price_level, default=None)
        if p is None:
            return 0.5
        # map 0..4 -> 1..0 (cheaper better)
        return clamp01(1.0 - (p / 4.0))

    # Rating score: rating (0..5) * confidence based on ratings_total
    # confidence uses log to avoid 10k dominating too much
    def rating_score(rating, ratings_total):
        r = safe_float(rating, default=None)
        if r is None:
            return 0.0
        r_norm = clamp01(r / 5.0)

        n = safe_float(ratings_total, default=0.0) or 0.0
        # confidence 0..1 using log scale: 0 reviews -> 0, 10k -> ~1
        conf = clamp01(math.log1p(n) / math.log1p(10000))
        return r_norm * conf

    # 4) Append scores into each place dict
    for i, sem in enumerate(scores):
        p = places_nearby[i]

        sem_score = float(sem)
        # sem can be negative; for mixing we map to 0..1
        sem_01 = clamp01((sem_score + 1.0) / 2.0)

        r_score = rating_score(p.get("rating"), p.get("ratings_total"))
        pr_score = price_score(p.get("price"))

        final = (W_SEM * sem_01) + (W_RATING * r_score) + (W_PRICE * pr_score)

        p["semantic_score"] = sem_score
        p["semantic_score_01"] = sem_01
        p["rating_score"] = r_score
        p["price_score"] = pr_score
        p["final_score"] = float(final)

    return places_nearby
