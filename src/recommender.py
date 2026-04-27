from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from src.logger import get_logger, validate_user_prefs, validate_song_row

logger = get_logger("recommender")


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_dict = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dicts = [asdict(s) for s in self.songs]
        ranked = recommend_songs(user_dict, song_dicts, k=k)
        id_to_song = {s.id: s for s in self.songs}
        return [id_to_song[rec[0]["id"]] for rec in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_dict = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        score, reasons = score_song(user_dict, asdict(song))
        return f"Recommended because: {', '.join(reasons)} (score: {score:.2f})"


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    logger.info("Loading songs from %s", csv_path)
    df = pd.read_csv(csv_path)
    numeric_cols = ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    all_rows = df.to_dict(orient="records")
    valid = [row for i, row in enumerate(all_rows) if validate_song_row(row, i)]
    logger.info("Loaded %d / %d songs from %s", len(valid), len(all_rows), csv_path)
    return valid


def confidence_score(user_prefs: Dict, song: Dict) -> float:
    """
    Returns a 0.0–1.0 confidence rating based on how many of 4 key
    factors strongly matched: genre, mood, energy proximity, acoustic fit.
    Separate from the weighted score — measures breadth of match, not depth.
    """
    matched = 0

    if song.get("genre") == user_prefs.get("genre"):
        matched += 1

    if song.get("mood") == user_prefs.get("mood"):
        matched += 1

    energy_diff = abs(float(song.get("energy", 0.5)) - float(user_prefs.get("energy", 0.5)))
    if energy_diff <= 0.20:
        matched += 1

    acousticness = float(song.get("acousticness", 0.5))
    if user_prefs.get("likes_acoustic", False) and acousticness >= 0.5:
        matched += 1
    elif not user_prefs.get("likes_acoustic", False) and acousticness < 0.5:
        matched += 1

    return round(matched / 4, 2)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Weights sum to 1.0: genre(0.35) + mood(0.25) + energy(0.20) + acousticness(0.10) + valence(0.10)
    Returns (score, reasons)
    """
    score = 0.0
    reasons = []

    if song.get("genre") == user_prefs.get("genre"):
        score += 0.35
        reasons.append("genre match")

    if song.get("mood") == user_prefs.get("mood"):
        score += 0.25
        reasons.append("mood match")

    energy_diff = abs(float(song.get("energy", 0.5)) - float(user_prefs.get("energy", 0.5)))
    energy_contribution = 0.20 * (1.0 - energy_diff)
    score += energy_contribution
    reasons.append(f"energy similarity {1.0 - energy_diff:.2f}")

    if user_prefs.get("likes_acoustic", False):
        score += 0.10 * float(song.get("acousticness", 0.5))
        reasons.append("acoustic match")
    else:
        score += 0.10 * (1.0 - float(song.get("acousticness", 0.5)))
        reasons.append("non-acoustic preference")

    valence_contribution = 0.10 * float(song.get("valence", 0.5))
    score += valence_contribution
    reasons.append(f"valence boost {float(song.get('valence', 0.5)):.2f}")

    return (round(score, 4), reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple]:
    """
    Scores and ranks all songs against user preferences.
    Returns top-k as list of (song_dict, score, explanation, confidence).
    """
    user_prefs = validate_user_prefs(user_prefs)
    logger.info("Generating recommendations for genre=%s mood=%s energy=%.2f",
                user_prefs.get("genre"), user_prefs.get("mood"), user_prefs.get("energy", 0.5))
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        conf = confidence_score(user_prefs, song)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation, conf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
