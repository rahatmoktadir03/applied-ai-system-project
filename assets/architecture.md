# VibeFinder — System Architecture

---

## Diagram 1: System Component Map

Shows every module, what it does, and how they depend on each other.

```mermaid
graph TB
    subgraph INPUT["Input Layer"]
        A["🧑 User Profile\ngenre · mood · energy · acoustic"]
        CSV["📄 data/songs.csv\n10 songs · 10 features each"]
    end

    subgraph CORE["Core Modules"]
        direction TB
        LOG["src/logger.py\nValidation · Guardrails · Logging"]
        REC["src/recommender.py\nload_songs · score_song · Recommender"]
        RET["src/retrieval.py\nCosine Similarity · RAG Explanations"]
        AGT["src/agent.py\nPlan → Act → Evaluate → Refine"]
    end

    subgraph OUTPUT["Output Layer"]
        CLI["src/main.py\nCLI Runner"]
        UI["app.py\nStreamlit UI"]
    end

    subgraph RELIABILITY["Reliability Layer"]
        EVL["src/evaluation.py\n5 Profiles × 3 Metrics"]
        TST["tests/test_recommender.py\nUnit + Smoke Tests"]
    end

    subgraph PERSIST["Persistence"]
        LOG2["logs/recommender.log"]
        RPT["logs/eval_report_*.json"]
    end

    A --> LOG
    CSV --> REC
    LOG --> REC
    LOG --> AGT
    REC --> AGT
    RET --> AGT
    AGT --> CLI
    AGT --> UI
    EVL --> RPT
    LOG --> LOG2
    TST --> REC
    TST --> EVL
```

---

## Diagram 2: Data Flow

Traces a single request from user input all the way to ranked recommendations.

```mermaid
flowchart LR
    U(["🧑 User\ngenre · mood\nenergy · acoustic"])

    subgraph GUARD["Guardrails"]
        VAL["validate_user_prefs\n• check required fields\n• clamp energy to 0–1\n• warn unknown genre/mood"]
    end

    subgraph AGENT["Agentic Loop  ↺ max 3 iterations"]
        PL["plan()\nSet feature weights\nfrom user profile"]
        AC["act()\nScore every song\nwith weighted factors"]
        EV["evaluate()\ndiversity = unique genres / k\nrelevance = mean score"]
        RF["refine()\nGreedy one-per-genre\nre-rank for variety"]
        CHK{"Quality\nthresholds\nmet?"}
    end

    subgraph RAG["RAG Layer"]
        RT["retrieve()\nCosine similarity\non 5-dim feature vector"]
        GN["generate_explanation()\nCite matched features\nwith real values"]
    end

    OUT(["📋 Top-k Songs\n+ score\n+ explanation"])

    CSV[("📄 songs.csv")]

    U --> VAL
    CSV --> AC
    VAL --> PL
    PL --> AC
    AC --> EV
    EV --> CHK
    CHK -- "No: diversity < 0.5\nor relevance < 0.6" --> RF
    RF --> EV
    CHK -- "Yes: converged" --> RT
    RT --> GN
    GN --> OUT
```

---

## Diagram 3: Human & Testing Touchpoints

Shows exactly where automated tests, the evaluation harness, and human review fit in the system.

```mermaid
flowchart TB
    subgraph AUTO["🤖 Automated Tests  —  pytest tests/"]
        T1["test_recommend_returns_songs_sorted_by_score\nVerifies ranking order is correct"]
        T2["test_explain_recommendation_returns_non_empty_string\nVerifies explanations are generated"]
        T3["test_evaluation_suite_runs\nSmoke tests the full eval pipeline"]
    end

    subgraph EVAL["📊 Automated Evaluation  —  EvaluationSuite"]
        P1["High-Energy Pop Fan"]
        P2["Chill Lofi Studier"]
        P3["Rock Gym Warrior"]
        P4["Jazz Afternoon Relaxer"]
        P5["Synthwave Night Driver"]
        M1["mean_relevance_score"]
        M2["diversity_index"]
        M3["consistency_score"]
        RPT["📁 logs/eval_report_*.json"]
    end

    subgraph HUMAN["🧑 Human Review"]
        UI["Streamlit UI\nVisual inspection of results"]
        DX["Agent Diagnostics Panel\niterations · diversity · relevance · convergence"]
        MC["Model Card\nBias · Limitations · Reflection"]
    end

    PASS{{"✅ All 3\ntests pass?"}}

    T1 & T2 & T3 --> PASS
    PASS -- "Yes" --> P1 & P2 & P3 & P4 & P5
    P1 & P2 & P3 & P4 & P5 --> M1 & M2 & M3
    M1 & M2 & M3 --> RPT
    RPT --> UI
    UI --> DX
    DX --> MC
    PASS -- "No: fix code" --> T1
```

---

## Component Responsibilities Summary

| Module | Role | AI Pattern |
|--------|------|-----------|
| `src/logger.py` | Validates input, logs all activity, skips bad data | Guardrails |
| `src/recommender.py` | Loads songs, scores each against user profile | Core scoring |
| `src/retrieval.py` | Cosine similarity search + feature-cited explanations | RAG |
| `src/agent.py` | Self-correcting loop: plan → act → evaluate → refine | Agentic Workflow |
| `src/evaluation.py` | Runs 5 profiles, measures 3 metrics, saves JSON report | Reliability Testing |
| `app.py` | Interactive UI with agent diagnostics panel | Human-in-the-loop |
| `tests/test_recommender.py` | Automated correctness checks | Reliability Testing |
