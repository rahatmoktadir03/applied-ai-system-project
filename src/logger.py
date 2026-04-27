import logging
from pathlib import Path
from typing import Dict, Any

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOGS_DIR / "recommender.log"

VALID_GENRES = {"pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop"}
VALID_MOODS = {"happy", "chill", "intense", "relaxed", "moody", "focused"}
FLOAT_FIELDS = ["energy", "valence", "danceability", "acousticness"]
BPM_RANGE = (40, 220)
UNIT_RANGE = (0.0, 1.0)


def get_logger(name: str = "recommender") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def validate_user_prefs(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes a user preference dict.
    Raises ValueError for missing required fields.
    Logs warnings for soft issues and applies safe defaults.
    """
    logger = get_logger()
    sanitized = dict(prefs)

    genre = sanitized.get("genre", "")
    if not genre:
        raise ValueError("Missing required field: 'genre'")
    if genre not in VALID_GENRES:
        logger.warning(
            "Unknown genre '%s'. Valid genres: %s. Proceeding without genre match.",
            genre,
            sorted(VALID_GENRES),
        )

    mood = sanitized.get("mood", "")
    if not mood:
        raise ValueError("Missing required field: 'mood'")
    if mood not in VALID_MOODS:
        logger.warning(
            "Unknown mood '%s'. Valid moods: %s. Proceeding without mood match.",
            mood,
            sorted(VALID_MOODS),
        )

    energy = sanitized.get("energy", 0.5)
    if not isinstance(energy, (int, float)):
        raise ValueError(f"'energy' must be a number, got {type(energy).__name__}")
    energy = float(energy)
    if not (UNIT_RANGE[0] <= energy <= UNIT_RANGE[1]):
        logger.warning("energy=%.2f out of range [0, 1]. Clamping.", energy)
        energy = max(0.0, min(1.0, energy))
    sanitized["energy"] = energy

    if "likes_acoustic" not in sanitized:
        sanitized["likes_acoustic"] = False
    elif not isinstance(sanitized["likes_acoustic"], bool):
        sanitized["likes_acoustic"] = bool(sanitized["likes_acoustic"])

    logger.debug("Validated user prefs: %s", sanitized)
    return sanitized


def validate_song_row(row: Dict[str, Any], row_index: int) -> bool:
    """
    Validates a single song dict loaded from CSV.
    Returns True if valid, False if the row should be skipped.
    """
    logger = get_logger()
    for field in FLOAT_FIELDS:
        val = row.get(field)
        if val is None:
            logger.warning("Row %d missing field '%s'. Skipping song.", row_index, field)
            return False
        try:
            fval = float(val)
        except (TypeError, ValueError):
            logger.warning(
                "Row %d field '%s' is not numeric ('%s'). Skipping.", row_index, field, val
            )
            return False
        if not (UNIT_RANGE[0] <= fval <= UNIT_RANGE[1]):
            logger.warning(
                "Row %d field '%s'=%.3f out of range [0, 1]. Skipping.", row_index, field, fval
            )
            return False

    bpm = row.get("tempo_bpm")
    if bpm is None:
        logger.warning("Row %d missing 'tempo_bpm'. Skipping.", row_index)
        return False
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        logger.warning("Row %d 'tempo_bpm' is not numeric ('%s'). Skipping.", row_index, bpm)
        return False
    if not (BPM_RANGE[0] <= bpm <= BPM_RANGE[1]):
        logger.warning(
            "Row %d tempo_bpm=%.1f out of range %s. Skipping.", row_index, bpm, BPM_RANGE
        )
        return False

    return True
