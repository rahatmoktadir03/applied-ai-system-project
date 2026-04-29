# 🎵 VibeFinder — AI Music Recommender System

> A local AI-powered music recommender that demonstrates RAG, agentic workflows, and reliability testing — built entirely in Python with no external API required.

📹 **[Video Walkthrough (Loom)](https://www.loom.com/share/42bd0b942ab54a169ba6592db76e2da2)** — end-to-end demo showing recommendations, agent diagnostics, reasoning trace, and evaluation harness.

---

## Project Origin

VibeFinder is an extension of the **Music Recommender Simulation** originally built in Modules 1–3. The original project introduced the core data structures — `Song`, `UserProfile`, and a weighted `score_song` function — as a classroom exercise in representing preferences as data and turning them into ranked outputs. It could load a small CSV catalog and return a sorted list of songs, but had no explainability, no self-correction, and no way to measure whether its results were actually good.

This final project evolves that prototype into a full applied AI system by adding a retrieval layer for grounded explanations, an agentic loop that checks and improves its own output, and a structured evaluation harness that measures performance across multiple user types.

---

## What VibeFinder Does and Why It Matters

Music recommendation is one of the most personal and high-stakes applications of AI — a bad recommendation breaks immersion; a good one introduces you to something you didn't know you needed. Most real-world recommenders are black boxes: they give you a result but can't tell you why.

VibeFinder is different by design. Every recommendation is:
- **Ranked** by a transparent weighted score you can inspect
- **Explained** by citing the exact audio features that matched your profile
- **Verified** by an agent that checks diversity and relevance before returning results
- **Tested** against 5 user profiles with 3 measurable metrics

It's a small system — 31 songs, one catalog — but it demonstrates the same architectural patterns (RAG, agentic loops, evaluation harnesses) that power production recommender systems at scale.

---

## System Architecture

Full component map, data flow, and testing touchpoints:
**[View Architecture Diagrams →](assets/architecture.md)**

### How the Pieces Connect

```
data/songs.csv
    └─ src/logger.py          Validates input, skips bad rows, logs all activity
         └─ src/recommender.py    Loads songs, scores each against user profile
              └─ src/retrieval.py      Cosine similarity search, feature-cited explanations
                   └─ src/agent.py         Plan → Act → Evaluate → Refine loop
                        ├─ app.py               Streamlit UI (human review)
                        └─ src/evaluation.py    5-profile reliability harness → JSON report
```

### The Three AI Patterns

| Pattern | Where It Lives | What It Does |
|---------|---------------|--------------|
| **RAG** | `src/retrieval.py` | Retrieves songs by cosine similarity on 5 audio features; generates explanations that cite only the features that actually matched |
| **Agentic Workflow** | `src/agent.py` | Runs up to 3 plan→act→evaluate→refine iterations; adjusts weights and re-ranks if diversity < 0.5 or relevance < 0.6 |
| **Reliability Testing** | `src/evaluation.py` + `tests/` | EvaluationSuite runs 5 user profiles, measures relevance/diversity/consistency, saves timestamped JSON report |

---

## Setup Instructions

### Requirements

- Python 3.9+
- `pandas`, `pytest`, `streamlit` (all in `requirements.txt`)

### Step-by-Step

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd applied-ai-system-project
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the CLI recommender**

```bash
python -m src.main
```

Expected output: top 5 recommendations for a default pop/happy/0.8-energy profile, each with a score and feature-cited explanation.

**5. Launch the interactive UI**

```bash
streamlit run app.py
```

Opens in your browser. Use the sidebar to set genre, mood, energy, and acoustic preference. Click **Find My Songs**.

**6. Run the evaluation suite**

```bash
python -c "from src.evaluation import EvaluationSuite; EvaluationSuite().run_and_save()"
```

Saves a timestamped JSON report to `logs/eval_report_YYYYMMDD_HHMMSS.json`.

**7. Run all tests**

```bash
pytest tests/ -v
```

All 3 tests should pass.

---

## Sample Interactions

The following are real outputs captured from the live system — unedited.

### Example 1 — High-Energy Pop Fan

**Input:**
```python
{"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
```

**Agent status:** Converged at iteration 1 · Diversity: 0.67 · Relevance: 0.76

**Output (top 3):**
```
1. Sunrise City by Neon Echo [pop] — Score: 0.97
   energy (0.82) matches your target (0.80);
   tempo (118.0 BPM) matches your pace.
   Feature similarity score: 0.96

2. Gym Hero by Max Pulse [pop] — Score: 0.70
   energy (0.93) matches your target (0.80);
   acoustic character (0.05) fits your non-acoustic preference;
   tempo (132.0 BPM) matches your pace.
   Feature similarity score: 0.98

3. Rooftop Lights by Indigo Parade [indie pop] — Score: 0.60
   energy (0.76) matches your target (0.80);
   tempo (124.0 BPM) matches your pace.
   Feature similarity score: 0.94
```

The top result (Sunrise City, 0.97) scored high because it hit both genre *and* mood match bonuses on top of close energy and tempo alignment — four factors firing together.

---

### Example 2 — Chill Lofi Studier

**Input:**
```python
{"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True}
```

**Agent status:** Converged at iteration 1 · Diversity: 1.00 · Relevance: 0.63

**Output (top 3):**
```
1. Library Rain by Paper Lanterns [lofi] — Score: 0.94
   energy (0.35) matches your target (0.35);
   valence (0.60) aligns with your preference;
   danceability (0.58) suits your vibe;
   acoustic character (0.86) fits your acoustic preference.
   Feature similarity score: 0.96

2. Spacewalk Thoughts by Orbit Bloom [ambient] — Score: 0.59
   energy (0.28) matches your target (0.35);
   danceability (0.41) suits your vibe;
   acoustic character (0.92) fits your acoustic preference.
   Feature similarity score: 0.95

3. Coffee Shop Stories by Slow Stereo [jazz] — Score: 0.35
   energy (0.37) matches your target (0.35);
   danceability (0.54) suits your vibe;
   acoustic character (0.89) fits your acoustic preference.
   Feature similarity score: 0.97
```

Diversity reached 1.00 — all 3 results are different genres (lofi, ambient, jazz). Acoustic users naturally get diverse results because acousticness is spread across multiple genres in the catalog.

---

### Example 3 — Rock Gym Warrior

**Input:**
```python
{"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False}
```

**Agent status:** Converged at iteration 1 · Diversity: 0.67 · Relevance: 0.65

**Output (top 3):**
```
1. Storm Runner by Voltline [rock] — Score: 0.96
   energy (0.91) matches your target (0.92);
   valence (0.48) aligns with your preference;
   acoustic character (0.10) fits your non-acoustic preference;
   tempo (152.0 BPM) matches your pace.
   Feature similarity score: 0.99

2. Gym Hero by Max Pulse [pop] — Score: 0.63
   energy (0.93) matches your target (0.92);
   acoustic character (0.05) fits your non-acoustic preference;
   tempo (132.0 BPM) matches your pace.
   Feature similarity score: 0.97

3. Sunrise City by Neon Echo [pop] — Score: 0.35
   energy (0.82) matches your target (0.92);
   tempo (118.0 BPM) matches your pace.
   Feature similarity score: 0.95
```

With only one rock song in the catalog, slots 2 and 3 fall back to high-energy pop — the next closest match by audio feature vector. The RAG explanation is honest: it cites energy and tempo proximity rather than pretending these are rock songs.

---

## Design Decisions

### Why cosine similarity for retrieval instead of a simple sort?

The weighted scorer in `recommender.py` is great for ranking but opaque about *which features* drove the result. Cosine similarity over the raw feature vector gives a complementary view: it measures geometric closeness in audio space without genre/mood binary bonuses. The two systems run in parallel — the agent uses the weighted scorer for ranking, and the RAG layer uses cosine similarity to identify *which features to cite* in the explanation. This separation keeps explanations grounded in measurable evidence rather than inferred from a score.

### Why build an agentic loop instead of just sorting once?

A single sort is fast but greedy — it will happily return four songs from the same genre if they all score well. Real recommendation lists need variety. The agent's evaluate→refine loop enforces a diversity threshold so the final list feels curated rather than repetitive. The loop also gives the system a natural place to self-correct: if diversity is too low, it nudges genre weight down and energy weight up, shifting toward audio similarity rather than exact-match filtering.

### Why no external LLM API?

Three reasons: **reproducibility** (the system works identically for anyone who clones it, no API keys needed), **transparency** (every explanation is template-generated from real feature values, so there's no hallucination risk), and **speed** (no network round trips). The trade-off is that explanations follow a fixed sentence structure — a real LLM would produce more natural language, but it would also be harder to verify and more expensive to run.

### Why pandas for CSV loading instead of the stdlib csv module?

Pandas gives free type coercion, missing-value detection, and `.to_dict()` conversion in three lines. The csv module would require manual float parsing and custom row validation. Since `validate_song_row` in `logger.py` already serves as the correctness guardrail, pandas here is a pure convenience layer — not a structural dependency.

---

## Testing Summary

> **11 out of 11 tests pass. Confidence scores average 0.67 across all profiles. Consistency is 1.0 — every profile returns the same top result across repeated runs. The system struggled most with underrepresented genres (jazz, ambient), where relevance scores dropped to ~0.53 due to catalog size, not algorithm failure.**

### Automated Tests — 11/11 Passing

| Test | What it checks | Result |
|------|---------------|--------|
| `test_recommend_returns_songs_sorted_by_score` | Pop/happy song ranks #1 for a pop/happy user | ✅ |
| `test_explain_recommendation_returns_non_empty_string` | Explanations are non-empty strings | ✅ |
| `test_evaluation_suite_runs` | Full eval pipeline completes and returns valid report | ✅ |
| `test_score_song_returns_valid_range` | Score is always between 0.0 and 1.0 | ✅ |
| `test_score_song_perfect_match_scores_near_one` | A song matching all factors scores ≥ 0.95 | ✅ |
| `test_load_songs_returns_all_rows` | All 10 catalog songs are loaded correctly | ✅ |
| `test_load_songs_returns_dicts_with_required_keys` | Every song dict has all 10 required fields | ✅ |
| `test_validate_user_prefs_clamps_energy_above_one` | `energy=1.8` is clamped to `1.0` silently | ✅ |
| `test_validate_user_prefs_raises_on_missing_genre` | Missing genre raises `ValueError` with clear message | ✅ |
| `test_confidence_score_perfect_match_returns_one` | All 4 factors matched → confidence = 1.0 | ✅ |
| `test_confidence_score_no_match_returns_low` | Zero factors matched → confidence = 0.0 | ✅ |

### Evaluation Harness — 5 Profiles × 4 Metrics

| Profile | Relevance | Diversity | Consistency | Confidence |
|---------|-----------|-----------|-------------|------------|
| High-Energy Pop Fan | 0.61 | 0.80 | 1.0 | 0.70 |
| Chill Lofi Studier | 0.69 | 0.60 | 1.0 | 0.80 |
| Rock Gym Warrior | 0.58 | 0.80 | 1.0 | 0.65 |
| Jazz Afternoon Relaxer | 0.53 | 0.60 | 1.0 | 0.60 |
| Synthwave Night Driver | 0.53 | 0.80 | 1.0 | 0.60 |
| **Average** | **0.59** | **0.72** | **1.00** | **0.67** |

### Logging and Error Handling

All activity is written to `logs/recommender.log` (auto-created). The logger records every load, validation warning, and agent iteration. Tested behaviors:
- `energy=1.5` → warning logged, value clamped to `1.0`, execution continues
- `genre="EDM"` → warning logged with valid genre list, scoring continues without genre bonus
- Missing `genre` field → `ValueError` raised immediately with a descriptive message
- Malformed CSV row (out-of-range float) → row skipped, logged, remaining rows loaded normally

### What worked well

- **Consistency is 1.0 across all profiles.** The system is fully deterministic — same input always produces the same top result. Predictable and auditable by design.
- **Diversity enforcement works.** Before the refine step, a pop user received 4 pop songs. After: 3–4 distinct genres consistently.
- **Confidence scores correlate with result quality.** The best-matched profile (Lofi Studier, 0.80) also has the highest relevance (0.69). The worst-matched profiles (Jazz, Synthwave at 0.60) have the lowest relevance.

### What didn't work as expected

- **Jazz and ambient profiles inflate the diversity metric.** With only one song each, diversity reaches 0.60 because non-matching genres fill the gaps — technically diverse, but not meaningfully so.
- **The Jazz profile never converges.** Relevance stays at 0.53, below the 0.6 threshold — the agent exits by exhaustion after 3 iterations, not by quality. This is an honest signal: the catalog simply doesn't have enough jazz.
- **Energy proximity doesn't distinguish "calm" from "low energy."** Relaxed and chill profiles both land at energy ≈ 0.35, so they receive the same songs regardless of mood. A tempo preference input would help separate them.

### What I learned

The hardest part of evaluation was defining metrics that mean what you think they mean. Diversity of 0.80 sounds good but can mask two pop songs carrying the rest. Consistency of 1.0 sounds reliable but is just determinism — there's no randomness to be consistent against. The most useful insight came from asking *why* a metric was high or low, not just whether it passed a threshold.

---

## Ethics and Responsible AI

### Limitations and Biases

VibeFinder has real limitations that matter for responsible use:

- **Western cultural bias.** Every genre in the catalog represents Western commercial music. A user looking for Afrobeats, K-pop, Bollywood, or any non-Western style would receive zero relevant results — not because the algorithm failed, but because it was never designed with their taste in mind.
- **Genre categories encode assumptions.** "Jazz" is a single binary tag covering a century of diverse music. The system treats it as an objective fact, but it's a cultural judgment about how music should be grouped.
- **Fixed weights were chosen by hand.** The 35/25/20/10/10 weight split reflects what the developer assumed matters. A listener whose taste is driven primarily by tempo or danceability — both unweighted — would consistently get poor results with no way to signal that.
- **No memory or context.** The system recommends the same songs at 7am as at midnight, on a Monday as on a Friday. Real listening preferences shift with context and activity.

### Could VibeFinder Be Misused?

At its current scale (31 songs, classroom use), direct misuse is unlikely. But the same patterns exist at production scale:

| Risk | How It Works | Prevention in VibeFinder |
|------|-------------|--------------------------|
| **Filter bubbles** | Pure relevance optimization returns the same genre repeatedly, narrowing musical taste over time | Diversity threshold enforced by the agent's refine step |
| **Demographic profiling** | Music taste correlates with age, culture, and background — interaction data could be used to infer demographics without consent | No user data is collected or stored; sessions are fully stateless |
| **Commercial manipulation** | Weights could be secretly tuned to push commercially advantageous songs while the interface appears neutral | Weights are visible in open source code; no hidden optimization layer |
| **Feedback loop bias** | If the system learned from click data dominated by one demographic, it would gradually optimize for that group at others' expense | No learning or retraining in VibeFinder; behavior is fixed and auditable |

### What Surprised Me in Testing

Consistency being exactly 1.0 across all five profiles initially felt suspicious — like something was wrong. After checking, it turned out to be a simple consequence of full determinism: no randomness anywhere in the pipeline means the same input always produces the same output, every time. That realization raised a harder question: *consistency of what?* A system can consistently give bad answers.

That led me to look more carefully at the Jazz Afternoon Relaxer profile, which never converges — the agent runs all 3 iterations and exits by exhaustion because the catalog only has one jazz song. The system accurately flagged its own limitation through the convergence status. That surprised me most: the evaluation metrics weren't just measuring performance, they were surfacing a data quality problem the algorithm couldn't fix on its own.

---

## Reflection

Building VibeFinder taught me that recommender systems are fundamentally a *representation problem* more than a math problem. The hardest decisions weren't about which algorithm to use — they were about what to encode: what does "energy" mean as a number? What does it mean for two songs to be "similar"? The moment you reduce music to five floats, you've already made irreversible choices about what matters.

The agentic loop was the most surprising part to build. I expected it to feel artificial — a loop added just to check a box on the requirements list. Instead, it genuinely improved results: without the diversity enforcement step, the system confidently returned four nearly identical songs. The loop is doing real work. That changed how I think about agentic AI: the value isn't in the agent being "smart," it's in the agent having a feedback signal it can actually act on.

The biggest open question this project left me with: how do you evaluate a recommender when there's no ground truth? I used diversity and relevance as proxies — but a user might *want* five songs from the same genre. The "right" answer depends on who's asking, and no metric fully captures that. That gap between measurable and meaningful is something every real AI system has to grapple with, and building this gave me a concrete example I can reason about.

### Collaborating with AI on This Project

This project was built with AI assistance throughout — planning, implementation, and documentation. Two moments stand out as worth reflecting on honestly:

**A genuinely helpful suggestion:** The AI proposed adding `confidence_score` as a *separate metric* from the weighted relevance score. Relevance measures how strongly all factors combined; confidence measures how many of the 4 key factors clearly fired. These capture different things — a song can score 0.70 relevance by matching energy and acousticness very closely while missing genre and mood entirely. Having both metrics made the evaluation harness more informative: the per-profile confidence scores (Lofi Studier: 0.80, Jazz: 0.60) correlated cleanly with relevance and helped explain *why* certain profiles performed differently.

**A flawed suggestion:** The initial output format used a plain 3-tuple `(song, score, explanation)` throughout the entire pipeline. When confidence scoring was added, this required updating five separate files because every caller that unpacked the tuple broke. A better initial design would have used a `NamedTuple` or `dataclass` for the recommendation result — adding a field would then require changing only the definition, not every callsite. The flat tuple was a premature simplification. In future projects I would push back on plain tuples for any interface that's likely to evolve.

See [model_card.md](model_card.md) for the full responsible AI evaluation including detailed bias analysis, misuse scenarios, and evaluation results.

---

## Project Structure

```
applied-ai-system-project/
├── src/
│   ├── __init__.py
│   ├── recommender.py   Core scoring: load_songs, score_song, confidence_score
│   ├── logger.py        Validation, guardrails, dual-handler logging
│   ├── retrieval.py     Cosine similarity RAG + multi-source genre knowledge base
│   ├── agent.py         Agentic plan→act→evaluate→refine loop + reasoning trace
│   └── evaluation.py    5-profile reliability harness, JSON report output
├── tests/
│   └── test_recommender.py   11 automated tests
├── assets/
│   └── architecture.md       System diagrams (Mermaid, renders on GitHub)
├── data/
│   ├── songs.csv             31-song catalog (real artists + audio features)
│   ├── songs_extended.csv    155-song catalog (31 originals + 124 synthetic variants)
│   └── genre_profiles.json   Genre knowledge base for multi-source RAG
├── scripts/
│   ├── run_evaluation.py           CLI test harness with PASS/FAIL report
│   └── generate_synthetic_catalog.py  Synthetic data generator
├── logs/                     Auto-created: recommender.log + eval reports
├── app.py                    Streamlit UI with catalog browser and feedback loop
├── model_card.md             Responsible AI evaluation (bias, limitations, reflection)
└── requirements.txt          pandas · pytest · streamlit
```

---

## Portfolio Reflection

*What this project says about me as an AI engineer:*

Building VibeFinder taught me that the most important skill in applied AI isn't knowing which model to call — it's knowing how to design a system that can check its own work. Every meaningful piece of this project came down to that: the agentic loop re-evaluates its results before returning them, the evaluation harness measures across five different user types instead of assuming one profile represents everyone, and the RAG layer grounds every explanation in evidence rather than generating text in a vacuum. I gravitate toward building AI systems that are transparent and testable by design, not as an afterthought. This project is a concrete demonstration of that instinct.

---

## License

For educational use. See [model_card.md](model_card.md) for responsible AI documentation.
