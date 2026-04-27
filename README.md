# 🎵 VibeFinder — Music Recommender System

## Project Summary

VibeFinder is a local AI-powered music recommender built in Python. It takes a user's taste profile — preferred genre, mood, energy level, and acoustic preference — and returns ranked song recommendations with transparent, feature-grounded explanations.

The system demonstrates three applied AI patterns without any external API:

- **RAG (Retrieval-Augmented Generation):** Songs are retrieved by cosine similarity over a 5-dimensional audio feature vector. Explanations are generated from the retrieved feature matches, grounding every recommendation in evidence.
- **Agentic Workflow:** A self-correcting agent loop (plan → act → evaluate → refine) adjusts feature weights across up to 3 iterations, enforcing diversity and relevance thresholds before finalizing results.
- **Reliability Testing:** An evaluation harness runs 5 predefined user profiles and measures three metrics — relevance, diversity, and consistency — saving a timestamped JSON report to `logs/`.

---

## System Architecture

Full component map, data flow, and testing touchpoints:
**[View Architecture Diagrams →](assets/architecture.md)**

---

## How The System Works

### Pipeline Overview

```
data/songs.csv
    └─ load_songs()         Loads and validates song data
         └─ Agent.plan()        Sets feature weights from user profile
              └─ Agent.act()        Scores and ranks all songs
                   └─ Agent.evaluate()   Computes diversity + relevance
                        └─ Agent.refine()     Boosts genre variety if needed
                             └─ rag_recommend()   Attaches RAG explanations
```

### Scoring Factors

Each song is scored against the user profile using weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Genre match | 0.35 | Exact match on genre field |
| Mood match | 0.25 | Exact match on mood field |
| Energy proximity | 0.20 | `1 - abs(song_energy - user_energy)` |
| Acoustic preference | 0.10 | Acousticness (or inverse) based on toggle |
| Valence tiebreaker | 0.10 | Song's positivity score |

Weights are re-normalized dynamically by the agent when diversity or relevance fall below thresholds.

### RAG Explanations

Retrieval uses cosine similarity on a 5-dim feature vector `[energy, valence, danceability, acousticness, tempo_normalized]`. Features within 0.15 of the user's target are tagged as "close matches." Explanations cite only these matched features with real values, making every output traceable.

### Agentic Loop

- **Diversity threshold:** 0.5 — at least half the top-k songs must be different genres
- **Relevance threshold:** 0.6 — mean score across top-k
- **Max iterations:** 3 — convergence status is shown in the Streamlit diagnostics panel

---

## Architecture

```
src/
├── __init__.py
├── recommender.py   Core scoring: load_songs, score_song, recommend_songs, Recommender class
├── logger.py        Dual-handler logger, validate_user_prefs, validate_song_row
├── retrieval.py     Cosine similarity retrieval + template-based explanation generation
├── agent.py         Agentic plan→act→evaluate→refine loop with AgentState tracking
└── evaluation.py    EvaluationSuite: 5 profiles × 3 metrics → JSON report

app.py               Streamlit UI (run from project root)
data/songs.csv       10-song catalog with audio features
logs/                Auto-created: recommender.log + evaluation JSON reports
tests/
└── test_recommender.py  3 unit/smoke tests
```

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac / Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the CLI:

   ```bash
   python -m src.main
   ```

4. Run the Streamlit UI:

   ```bash
   streamlit run app.py
   ```

5. Run the evaluation suite and save a JSON report:

   ```bash
   python -c "from src.evaluation import EvaluationSuite; EvaluationSuite().run_and_save()"
   ```

### Running Tests

```bash
pytest tests/ -v
```

---

## Experiments

### Changing diversity threshold

Lowering `DIVERSITY_THRESHOLD` from 0.5 to 0.2 caused the agent to converge in 1 iteration but often returned 3–4 songs from the same genre. Raising it to 0.8 on a 10-song catalog caused the agent to exhaust all 3 iterations for narrow profiles (e.g. jazz, which has only 1 song in the catalog).

### Genre with only one song

Profiles targeting `jazz` or `ambient` always hit a diversity ceiling: after the single matching song is placed, the refine step fills remaining slots from other genres — which is correct behavior, but the top score gap between slot 1 and slot 2 is large.

### Acoustic vs non-acoustic

Switching `likes_acoustic` flips the acousticness component's polarity. Lofi and ambient songs rise sharply for acoustic users; rock and synthwave drop. The change is immediate because the weight modulation happens in `plan()` before any iteration runs.

---

## Limitations and Risks

- **Tiny catalog:** 10 songs makes true diversity hard to achieve for any profile — in production this would need thousands of entries.
- **Exact-match genre and mood:** A typo (`"Pop"` vs `"pop"`) scores zero on those factors. No fuzzy matching.
- **Hand-tuned weights:** The 0.35/0.25/0.20/0.10/0.10 split was chosen manually, not learned from data.
- **No collaborative filtering:** There is no user history or "people like you also liked" signal.
- **Western pop-centric catalog:** The 10 songs skew toward Western genres; a global user would find fewer relevant results.

---

## Reflection

See [model_card.md](model_card.md) for a full responsible-AI evaluation of VibeFinder.
