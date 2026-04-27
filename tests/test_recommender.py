import pytest
from src.recommender import Song, UserProfile, Recommender, score_song, load_songs, confidence_score
from src.logger import validate_user_prefs
from src.evaluation import EvaluationSuite

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_evaluation_suite_runs():
    suite = EvaluationSuite("data/songs.csv")
    report = suite.run_all()

    assert "summary" in report
    assert report["summary"]["avg_relevance"] > 0
    assert 0.0 <= report["summary"]["avg_diversity"] <= 1.0
    assert 0.0 <= report["summary"]["avg_consistency"] <= 1.0
    assert 0.0 <= report["summary"]["avg_confidence"] <= 1.0
    assert len(report["results"]) == 5


# --- score_song range ---

def test_score_song_returns_valid_range():
    user = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    song = {"genre": "pop", "mood": "happy", "energy": 0.8,
            "acousticness": 0.1, "valence": 0.9}
    score, _ = score_song(user, song)
    assert 0.0 <= score <= 1.0


def test_score_song_perfect_match_scores_near_one():
    user = {"genre": "pop", "mood": "happy", "energy": 1.0, "likes_acoustic": False}
    song = {"genre": "pop", "mood": "happy", "energy": 1.0,
            "acousticness": 0.0, "valence": 1.0}
    score, _ = score_song(user, song)
    assert score >= 0.95


# --- load_songs ---

def test_load_songs_returns_all_rows():
    songs = load_songs("data/songs.csv")
    assert len(songs) == 10


def test_load_songs_returns_dicts_with_required_keys():
    songs = load_songs("data/songs.csv")
    required = {"id", "title", "artist", "genre", "mood",
                "energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    for song in songs:
        assert required.issubset(song.keys())


# --- validate_user_prefs guardrails ---

def test_validate_user_prefs_clamps_energy_above_one():
    result = validate_user_prefs({"genre": "pop", "mood": "happy", "energy": 1.8})
    assert result["energy"] == 1.0


def test_validate_user_prefs_raises_on_missing_genre():
    with pytest.raises(ValueError, match="genre"):
        validate_user_prefs({"mood": "happy", "energy": 0.5})


# --- confidence scoring ---

def test_confidence_score_perfect_match_returns_one():
    user = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.1}
    assert confidence_score(user, song) == 1.0


def test_confidence_score_no_match_returns_low():
    user = {"genre": "rock", "mood": "intense", "energy": 0.9, "likes_acoustic": False}
    song = {"genre": "lofi", "mood": "chill", "energy": 0.3, "acousticness": 0.9}
    assert confidence_score(user, song) == 0.0
