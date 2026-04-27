import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.logger import get_logger

logger = get_logger("retrieval")

_GENRE_PROFILES_PATH = Path(__file__).parent.parent / "data" / "genre_profiles.json"

_FEATURE_NAMES = ["energy", "valence", "danceability", "acousticness", "tempo"]
_CLOSE_MATCH_THRESHOLD = 0.15


def _song_to_vector(song: Dict) -> List[float]:
    """Converts song numeric features to a normalized 5-dim vector."""
    tempo_norm = (float(song.get("tempo_bpm", 120)) - 40) / (220 - 40)
    return [
        float(song.get("energy", 0.5)),
        float(song.get("valence", 0.5)),
        float(song.get("danceability", 0.5)),
        float(song.get("acousticness", 0.5)),
        tempo_norm,
    ]


def _user_to_vector(user_prefs: Dict) -> List[float]:
    """Converts user preferences to the same 5-dim feature vector."""
    acoustic_val = 1.0 if user_prefs.get("likes_acoustic", False) else 0.0
    return [
        float(user_prefs.get("energy", 0.5)),
        float(user_prefs.get("valence", 0.5)),
        float(user_prefs.get("danceability", 0.5)),
        acoustic_val,
        float(user_prefs.get("tempo_normalized", 0.5)),
    ]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_similar_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> List[Tuple[Dict, float, List[str]]]:
    """
    Retrieval step: ranks songs by cosine similarity to user feature vector.
    Returns top-k as (song_dict, similarity_score, matched_feature_names).
    """
    user_vec = _user_to_vector(user_prefs)
    results = []

    for song in songs:
        song_vec = _song_to_vector(song)
        sim = _cosine_similarity(user_vec, song_vec)

        matched = [
            _FEATURE_NAMES[i]
            for i in range(len(user_vec))
            if abs(song_vec[i] - user_vec[i]) < _CLOSE_MATCH_THRESHOLD
        ]

        results.append((song, sim, matched))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:k]

    if top:
        logger.info(
            "Retrieved %d songs. Top match: '%s' (similarity=%.3f)",
            k,
            top[0][0].get("title", "?"),
            top[0][1],
        )

    return top


def generate_explanation(
    user_prefs: Dict,
    song: Dict,
    matched_features: List[str],
    similarity: float,
) -> str:
    """
    Generation step: produces a human-readable explanation citing retrieved features.
    This is the RAG 'generation' — grounded in what was actually retrieved.
    """
    acoustic_pref = "acoustic" if user_prefs.get("likes_acoustic", False) else "non-acoustic"

    feature_descriptions = {
        "energy": (
            f"energy ({song['energy']:.2f}) matches your target "
            f"({float(user_prefs.get('energy', 0.5)):.2f})"
        ),
        "valence": f"positivity/valence ({song['valence']:.2f}) aligns with your preference",
        "danceability": f"danceability ({song['danceability']:.2f}) suits your vibe",
        "acousticness": (
            f"acoustic character ({song['acousticness']:.2f}) fits your "
            f"{acoustic_pref} preference"
        ),
        "tempo": f"tempo ({song['tempo_bpm']} BPM) matches your pace",
    }

    cited = [feature_descriptions[f] for f in matched_features if f in feature_descriptions]
    if not cited:
        cited = [f"overall audio profile similarity ({similarity:.2f})"]

    return (
        f"'{song['title']}' by {song['artist']}: "
        + "; ".join(cited)
        + f". Feature similarity score: {similarity:.2f}."
    )


def rag_recommend(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Full RAG pipeline: retrieve similar songs, then generate explanations.
    Returns list of (song_dict, similarity_score, explanation_string).
    """
    retrieved = retrieve_similar_songs(user_prefs, songs, k)
    return [
        (song, sim, generate_explanation(user_prefs, song, matched, sim))
        for song, sim, matched in retrieved
    ]


# ---------------------------------------------------------------------------
# Stretch Feature: Multi-Source RAG — Genre Knowledge Base
# ---------------------------------------------------------------------------

def load_genre_profiles(path: Path = _GENRE_PROFILES_PATH) -> Dict:
    """Loads the genre knowledge base from data/genre_profiles.json."""
    try:
        with open(path, encoding="utf-8") as f:
            profiles = json.load(f)
        logger.info("Loaded genre profiles for %d genres", len(profiles))
        return profiles
    except FileNotFoundError:
        logger.warning("Genre profiles not found at %s — using empty profiles", path)
        return {}


def retrieve_genre_context(genre: str, profiles: Dict) -> Optional[Dict]:
    """
    Retrieves the knowledge-base entry for a given genre.
    Returns None if the genre isn't in the profiles.
    This is the second 'retrieval source' in the enhanced RAG pipeline.
    """
    profile = profiles.get(genre)
    if profile:
        logger.debug("Retrieved genre context for '%s'", genre)
    else:
        logger.debug("No genre context found for '%s'", genre)
    return profile


def _keyword_overlap(song: Dict, profile: Dict) -> List[str]:
    """
    Returns keywords from the genre profile that are semantically relevant
    to this song's mood and title. Used to enrich the explanation.
    """
    if not profile:
        return []
    keywords = profile.get("keywords", [])
    mood = song.get("mood", "").lower()
    title = song.get("title", "").lower()
    relevant = [kw for kw in keywords if kw in mood or kw in title]
    return relevant[:3] if relevant else keywords[:2]


def generate_enhanced_explanation(
    user_prefs: Dict,
    song: Dict,
    matched_features: List[str],
    similarity: float,
    genre_profiles: Dict,
) -> str:
    """
    Enhanced generation step: combines numeric feature evidence (Source 1)
    with genre knowledge-base context (Source 2).

    Baseline explanation: cites only matched audio features.
    Enhanced explanation: also includes what the genre is known for and
    when listeners typically reach for it — grounding the recommendation
    in real-world context, not just numbers.
    """
    acoustic_pref = "acoustic" if user_prefs.get("likes_acoustic", False) else "non-acoustic"

    feature_descriptions = {
        "energy": (
            f"energy ({song['energy']:.2f}) matches your target "
            f"({float(user_prefs.get('energy', 0.5)):.2f})"
        ),
        "valence": f"positivity/valence ({song['valence']:.2f}) aligns with your preference",
        "danceability": f"danceability ({song['danceability']:.2f}) suits your vibe",
        "acousticness": (
            f"acoustic character ({song['acousticness']:.2f}) fits your "
            f"{acoustic_pref} preference"
        ),
        "tempo": f"tempo ({song['tempo_bpm']} BPM) matches your pace",
    }

    cited = [feature_descriptions[f] for f in matched_features if f in feature_descriptions]
    if not cited:
        cited = [f"overall audio profile similarity ({similarity:.2f})"]

    # Source 1: numeric retrieval
    numeric_part = (
        f"'{song['title']}' by {song['artist']}: "
        + "; ".join(cited)
        + f". [similarity: {similarity:.2f}]"
    )

    # Source 2: genre knowledge base
    song_genre = song.get("genre", "")
    user_genre = user_prefs.get("genre", "")
    profile = retrieve_genre_context(song_genre, genre_profiles)

    context_part = ""
    if profile:
        listen_when = profile.get("listen_when", "")
        description_snippet = profile.get("description", "")[:120].rstrip()
        if song_genre == user_genre:
            context_part = (
                f" Genre context: {description_snippet}... "
                f"Best for: {listen_when}."
            )
        else:
            related = profile.get("related_genres", [])
            if user_genre in related:
                context_part = (
                    f" Note: {song_genre.title()} is closely related to {user_genre} — "
                    f"{listen_when}."
                )
            else:
                context_part = f" Included for audio diversity ({song_genre.title()}: {listen_when})."

    return numeric_part + context_part


def rag_recommend_enhanced(
    user_prefs: Dict,
    songs: List[Dict],
    genre_profiles: Dict,
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Multi-source RAG pipeline combining:
      Source 1 — numeric cosine similarity over audio features
      Source 2 — genre knowledge base (data/genre_profiles.json)

    Returns list of (song_dict, similarity_score, enhanced_explanation).
    """
    retrieved = retrieve_similar_songs(user_prefs, songs, k)
    results = []
    for song, sim, matched in retrieved:
        explanation = generate_enhanced_explanation(
            user_prefs, song, matched, sim, genre_profiles
        )
        results.append((song, sim, explanation))
    logger.info(
        "Enhanced RAG: retrieved %d songs with genre context for '%s'",
        len(results),
        user_prefs.get("genre", "?"),
    )
    return results


def compare_rag_outputs(
    user_prefs: Dict,
    songs: List[Dict],
    genre_profiles: Dict,
    k: int = 3,
) -> Dict:
    """
    Returns a side-by-side comparison of baseline vs enhanced RAG explanations.
    Used to demonstrate measurable improvement in explanation richness.
    """
    baseline = rag_recommend(user_prefs, songs, k)
    enhanced = rag_recommend_enhanced(user_prefs, songs, genre_profiles, k)

    comparison = []
    for (s_base, sim_base, exp_base), (s_enh, sim_enh, exp_enh) in zip(baseline, enhanced):
        comparison.append({
            "song": s_base.get("title"),
            "similarity": sim_base,
            "baseline_explanation": exp_base,
            "enhanced_explanation": exp_enh,
            "chars_added": len(exp_enh) - len(exp_base),
        })

    return {
        "user_genre": user_prefs.get("genre"),
        "genre_profile_loaded": user_prefs.get("genre", "") in genre_profiles,
        "comparisons": comparison,
        "avg_chars_added": round(
            sum(c["chars_added"] for c in comparison) / len(comparison), 1
        ) if comparison else 0,
    }
