import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.agent import Agent
from src.recommender import load_songs
from src.logger import get_logger

logger = get_logger("evaluation")

LOGS_DIR = Path(__file__).parent.parent / "logs"

EVAL_PROFILES = [
    {
        "name": "High-Energy Pop Fan",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "likes_acoustic": False,
    },
    {
        "name": "Chill Lofi Studier",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
    },
    {
        "name": "Rock Gym Warrior",
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
    },
    {
        "name": "Jazz Afternoon Relaxer",
        "genre": "jazz",
        "mood": "relaxed",
        "energy": 0.35,
        "likes_acoustic": True,
    },
    {
        "name": "Synthwave Night Driver",
        "genre": "synthwave",
        "mood": "moody",
        "energy": 0.74,
        "likes_acoustic": False,
    },
]


class EvaluationSuite:
    def __init__(self, csv_path: str = "data/songs.csv"):
        self.songs = load_songs(csv_path)
        self.agent = Agent(self.songs)

    def _run_once(self, profile: Dict) -> List[Tuple]:
        prefs = {k: v for k, v in profile.items() if k != "name"}
        return self.agent.recommend(prefs, k=5)

    def mean_relevance_score(self, results: List[Tuple]) -> float:
        scores = [r[1] for r in results]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def diversity_index(self, results: List[Tuple]) -> float:
        genres = {r[0].get("genre", "") for r in results}
        return round(len(genres) / len(results), 4) if results else 0.0

    def mean_confidence_score(self, results: List[Tuple]) -> float:
        confidences = [r[3] for r in results if len(r) > 3]
        return round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    def consistency_score(self, profile: Dict) -> float:
        """Returns 1.0 if the top result is the same across two independent runs."""
        r1 = self._run_once(profile)
        r2 = self._run_once(profile)
        if not r1 or not r2:
            return 0.0
        return 1.0 if r1[0][0].get("id") == r2[0][0].get("id") else 0.0

    def run_all(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profiles_evaluated": len(EVAL_PROFILES),
            "results": [],
        }

        for profile in EVAL_PROFILES:
            logger.info("Evaluating profile: %s", profile["name"])
            results = self._run_once(profile)
            consistency = self.consistency_score(profile)

            entry = {
                "profile_name": profile["name"],
                "mean_relevance_score": self.mean_relevance_score(results),
                "diversity_index": self.diversity_index(results),
                "consistency_score": consistency,
                "mean_confidence_score": self.mean_confidence_score(results),
                "top_5_songs": [
                    {
                        "title": r[0].get("title"),
                        "artist": r[0].get("artist"),
                        "genre": r[0].get("genre"),
                        "score": round(r[1], 4),
                        "confidence": r[3] if len(r) > 3 else None,
                    }
                    for r in results
                ],
            }
            report["results"].append(entry)
            logger.info(
                "Profile '%s': relevance=%.3f diversity=%.3f consistency=%.1f",
                profile["name"],
                entry["mean_relevance_score"],
                entry["diversity_index"],
                consistency,
            )

        all_relevance = [e["mean_relevance_score"] for e in report["results"]]
        all_diversity = [e["diversity_index"] for e in report["results"]]
        all_consistency = [e["consistency_score"] for e in report["results"]]
        all_confidence = [e["mean_confidence_score"] for e in report["results"]]

        report["summary"] = {
            "avg_relevance": round(sum(all_relevance) / len(all_relevance), 4),
            "avg_diversity": round(sum(all_diversity) / len(all_diversity), 4),
            "avg_consistency": round(sum(all_consistency) / len(all_consistency), 4),
            "avg_confidence": round(sum(all_confidence) / len(all_confidence), 4),
        }

        logger.info("Evaluation complete. Summary: %s", report["summary"])
        return report

    def save_report(self, report: Dict[str, Any]) -> str:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = LOGS_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", path)
        return str(path)

    def run_and_save(self) -> str:
        report = self.run_all()
        return self.save_report(report)
