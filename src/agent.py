import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from src.logger import get_logger, validate_user_prefs
from src.retrieval import retrieve_similar_songs, generate_explanation
from src.recommender import confidence_score

logger = get_logger("agent")

_DEFAULT_WEIGHTS = {
    "genre": 0.35,
    "mood": 0.25,
    "energy": 0.20,
    "acousticness": 0.10,
    "valence": 0.10,
}

_HIGH_ENERGY_GENRES = {"rock", "synthwave", "pop"}


# ---------------------------------------------------------------------------
# Stretch Feature: Agentic Enhancement — Observable Reasoning Trace
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single observable step in the agent's decision-making chain."""
    tool: str
    reasoning: str
    decision: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))

    def __str__(self) -> str:
        return f"[{self.timestamp}] TOOL:{self.tool} | {self.reasoning} -> {self.decision}"


@dataclass
class AgentState:
    iteration: int = 0
    weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    last_diversity: float = 0.0
    last_relevance: float = 0.0
    converged: bool = False
    history: List[str] = field(default_factory=list)
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)

    def add_step(self, tool: str, reasoning: str, decision: str) -> None:
        step = ReasoningStep(tool=tool, reasoning=reasoning, decision=decision)
        self.reasoning_trace.append(step)
        logger.debug("TRACE %s", step)


class Agent:
    MAX_ITERATIONS = 3
    DIVERSITY_THRESHOLD = 0.5
    RELEVANCE_THRESHOLD = 0.6

    def __init__(self, songs: List[Dict]):
        self.songs = songs
        self.state = AgentState()

    def plan(self, user_prefs: Dict) -> Dict[str, float]:
        """Tool: analyze user preferences and set initial feature weights."""
        weights = dict(_DEFAULT_WEIGHTS)
        adjustments = []

        if user_prefs.get("likes_acoustic", False):
            weights["acousticness"] += 0.05
            weights["energy"] -= 0.05
            adjustments.append("boosted acousticness weight (+5%)")

        genre = user_prefs.get("genre", "")
        if genre in _HIGH_ENERGY_GENRES:
            weights["energy"] += 0.05
            weights["valence"] -= 0.05
            adjustments.append(f"boosted energy for high-energy genre '{genre}' (+5%)")

        weights = _normalize(weights)
        self.state.weights = weights

        reasoning = (
            f"User wants {genre}/{user_prefs.get('mood')} at energy={user_prefs.get('energy', 0.5):.2f}"
            + (f", acoustic={'yes' if user_prefs.get('likes_acoustic') else 'no'}")
        )
        decision = (
            "; ".join(adjustments) if adjustments
            else "using default weights (no profile-specific adjustments needed)"
        )
        self.state.add_step("tool_plan", reasoning, decision)

        logger.info("Plan: weights=%s", {k: f"{v:.3f}" for k, v in weights.items()})
        return weights

    def act(self, user_prefs: Dict, k: int = 5) -> List[Tuple[Dict, float]]:
        """Tool: score and rank all songs using the current weights."""
        weights = self.state.weights
        likes_acoustic = user_prefs.get("likes_acoustic", False)
        user_energy = float(user_prefs.get("energy", 0.5))

        scored = []
        for song in self.songs:
            genre_match = 1.0 if song.get("genre") == user_prefs.get("genre") else 0.0
            mood_match = 1.0 if song.get("mood") == user_prefs.get("mood") else 0.0
            energy_sim = 1.0 - abs(float(song.get("energy", 0.5)) - user_energy)
            acoustic = (
                float(song.get("acousticness", 0.5))
                if likes_acoustic
                else 1.0 - float(song.get("acousticness", 0.5))
            )
            valence = float(song.get("valence", 0.5))

            score = (
                weights["genre"] * genre_match
                + weights["mood"] * mood_match
                + weights["energy"] * energy_sim
                + weights["acousticness"] * acoustic
                + weights["valence"] * valence
            )
            scored.append((song, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        top_song = top[0][0].get("title", "?") if top else "?"
        top_score = top[0][1] if top else 0.0
        genre_hits = sum(1 for s, _ in top if s.get("genre") == user_prefs.get("genre"))

        self.state.add_step(
            "tool_act",
            f"iter={self.state.iteration}: scored {len(self.songs)} songs with "
            f"weights genre={weights['genre']:.2f}/energy={weights['energy']:.2f}",
            f"top='{top_song}' (score={top_score:.3f}), "
            f"{genre_hits}/{min(k, len(top))} songs match target genre",
        )

        logger.info("Act iter=%d: top score=%.3f song='%s'",
                    self.state.iteration, top_score, top_song)
        return top

    def evaluate(self, results: List[Tuple[Dict, float]], k: int) -> Tuple[float, float]:
        """Tool: compute diversity and relevance metrics for current results."""
        top = results[:k]
        genres = {song.get("genre", "") for song, _ in top}
        diversity = len(genres) / k if top else 0.0
        scores = [score for _, score in top]
        relevance = sum(scores) / len(scores) if scores else 0.0

        self.state.last_diversity = round(diversity, 4)
        self.state.last_relevance = round(relevance, 4)

        diversity_ok = diversity >= self.DIVERSITY_THRESHOLD
        relevance_ok = relevance >= self.RELEVANCE_THRESHOLD
        verdict = []
        if diversity_ok:
            verdict.append(f"diversity {diversity:.2f} [OK]")
        else:
            verdict.append(f"diversity {diversity:.2f} [FAIL] (need >={self.DIVERSITY_THRESHOLD})")
        if relevance_ok:
            verdict.append(f"relevance {relevance:.2f} [OK]")
        else:
            verdict.append(f"relevance {relevance:.2f} [FAIL] (need >={self.RELEVANCE_THRESHOLD})")

        self.state.add_step(
            "tool_evaluate",
            f"measured {k} results — genres present: {sorted(genres)}",
            " | ".join(verdict),
        )

        logger.info("Evaluate iter=%d: diversity=%.2f relevance=%.2f",
                    self.state.iteration, diversity, relevance)
        return diversity, relevance

    def refine(self, results: List[Tuple[Dict, float]], k: int) -> List[Tuple[Dict, float]]:
        """Tool: re-rank results to satisfy the diversity threshold."""
        if not results:
            return results

        old_div = self.state.last_diversity
        refined = [results[0]]
        seen_genres = {results[0][0].get("genre", "")}
        remaining = results[1:]

        for song, score in remaining:
            genre = song.get("genre", "")
            if genre not in seen_genres:
                refined.append((song, score))
                seen_genres.add(genre)
            if len(refined) >= k:
                break

        for song, score in remaining:
            if (song, score) not in refined:
                refined.append((song, score))
            if len(refined) >= k:
                break

        new_genres = {s.get("genre", "") for s, _ in refined[:k]}
        new_div = len(new_genres) / k
        self.state.last_diversity = round(new_div, 4)

        swapped_in = sorted(new_genres - {results[0][0].get("genre", "")})
        self.state.add_step(
            "tool_refine",
            f"diversity was {old_div:.2f} < threshold {self.DIVERSITY_THRESHOLD} "
            f"— applying greedy one-per-genre rerank",
            f"diversity {old_div:.2f} -> {new_div:.2f}; "
            f"new genres added: {swapped_in if swapped_in else 'none'}",
        )

        logger.info("Refine: diversity %.2f -> %.2f", old_div, new_div)
        return refined[:k]

    def _adjust_weights(self, diversity: float, relevance: float) -> None:
        """Nudge weights to address low diversity or low relevance."""
        weights = dict(self.state.weights)
        delta = 0.05
        changes = []

        if diversity < self.DIVERSITY_THRESHOLD:
            weights["genre"] = max(0.05, weights["genre"] - delta)
            weights["energy"] = min(0.60, weights["energy"] + delta)
            changes.append("genre↓ energy↑ (diversity too low)")
        if relevance < self.RELEVANCE_THRESHOLD:
            weights["energy"] = min(0.60, weights["energy"] + delta)
            weights["genre"] = max(0.05, weights["genre"] - delta)
            changes.append("energy↑ genre↓ (relevance too low)")

        self.state.weights = _normalize(weights)
        self.state.add_step(
            "tool_adjust_weights",
            f"quality not met after iter={self.state.iteration}",
            "; ".join(changes) if changes else "no adjustment needed",
        )
        logger.debug("Adjust weights: %s",
                     {k: f"{v:.3f}" for k, v in self.state.weights.items()})

    def recommend(self, user_prefs: Dict, k: int = 5) -> List[Tuple]:
        """
        Full agentic loop: plan -> [act -> evaluate -> refine -> adjust] × MAX_ITERATIONS.
        Returns top-k as (song_dict, score, rag_explanation, confidence).
        Every step is recorded in state.reasoning_trace for full observability.
        """
        user_prefs = validate_user_prefs(user_prefs)
        self.state = AgentState()
        self.state.add_step(
            "tool_start",
            f"received request: genre={user_prefs.get('genre')} "
            f"mood={user_prefs.get('mood')} energy={user_prefs.get('energy', 0.5):.2f}",
            f"catalog size={len(self.songs)}, requesting top-{k}",
        )

        self.plan(user_prefs)

        results = []
        for i in range(self.MAX_ITERATIONS):
            self.state.iteration = i + 1
            results = self.act(user_prefs, k=max(k * 2, len(self.songs)))
            diversity, relevance = self.evaluate(results, k)

            self.state.history.append(
                f"iter={i + 1} diversity={diversity:.2f} relevance={relevance:.2f}"
            )

            if diversity < self.DIVERSITY_THRESHOLD:
                results = self.refine(results, k)
                diversity, _ = self.evaluate(results, k)

            if diversity >= self.DIVERSITY_THRESHOLD and relevance >= self.RELEVANCE_THRESHOLD:
                self.state.converged = True
                self.state.add_step(
                    "tool_converge",
                    f"all thresholds met at iteration {i + 1}",
                    f"diversity={diversity:.2f} [OK]  relevance={relevance:.2f} [OK] — stopping loop",
                )
                logger.info("Agent converged at iteration %d", i + 1)
                break

            if not self.state.converged:
                self._adjust_weights(diversity, relevance)

        if not self.state.converged:
            self.state.add_step(
                "tool_exhaust",
                f"reached MAX_ITERATIONS={self.MAX_ITERATIONS} without converging",
                f"returning best available: diversity={self.state.last_diversity:.2f} "
                f"relevance={self.state.last_relevance:.2f}",
            )

        final = results[:k]
        retrieved = retrieve_similar_songs(user_prefs, self.songs, k=len(self.songs))
        id_to_retrieval = {r[0].get("id"): (r[1], r[2]) for r in retrieved}

        output = []
        for song, score in final:
            sim, matched = id_to_retrieval.get(song.get("id"), (score, []))
            explanation = generate_explanation(user_prefs, song, matched, sim)
            conf = confidence_score(user_prefs, song)
            output.append((song, score, explanation, conf))

        self.state.add_step(
            "tool_explain",
            f"attached RAG explanations to {len(output)} results",
            f"pipeline complete — converged={self.state.converged} "
            f"iterations={self.state.iteration}",
        )

        logger.info("Agent done: %d results, converged=%s, iterations=%d",
                    len(output), self.state.converged, self.state.iteration)
        return output


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    return {k: round(v / total, 6) for k, v in weights.items()}
