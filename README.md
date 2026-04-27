# 🎵 VibeFinder — AI Music Recommender System

> A local AI-powered music recommender that demonstrates RAG, agentic workflows, and reliability testing — built entirely in Python with no external API required.

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

It's a small system — 10 songs, one catalog — but it demonstrates the same architectural patterns (RAG, agentic loops, evaluation harnesses) that power production recommender systems at scale.

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
