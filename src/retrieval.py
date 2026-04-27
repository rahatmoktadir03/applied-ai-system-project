import math
from typing import List, Dict, Tuple
from src.logger import get_logger

logger = get_logger("retrieval")

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
