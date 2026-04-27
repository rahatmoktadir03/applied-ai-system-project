# 🎧 Model Card: VibeFinder 1.0

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

VibeFinder suggests up to 10 songs from a small catalog based on a user's preferred genre, mood, energy level, and acoustic preference. It is designed for classroom exploration of how recommender systems work — not for real production use. It assumes a single user with clearly-defined preferences and a static catalog of 10 songs.

---

## 3. How the Model Works

When you submit your preferences, VibeFinder does three things:

First, it **plans** by reading your profile and adjusting how much each factor matters. For example, if you said you like acoustic sounds, it gives extra weight to that dimension before scoring anything.

Then it **scores** every song in the catalog. Each song earns points based on how closely it matches your genre, mood, energy level, acoustic preference, and general positivity (valence). The scores add up to a number between 0 and 1.

After ranking, the system **checks the results**. If too many of the top songs are the same genre, it reshuffles to include more variety — because a good recommendation list should expose you to slightly different things, not just the closest clone of one song five times over.

Finally, it **explains** each recommendation by identifying which specific audio features were closest to your preferences. The explanation is grounded in the actual numbers, so you can see exactly why a song was chosen.

---

## 4. Data

The catalog contains **10 songs** across 7 genres: pop, lofi, rock, ambient, jazz, synthwave, and indie pop. There are 5 mood tags: happy, chill, intense, relaxed, moody, and focused.

Each song has 10 attributes: id, title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness. The numeric features are on a 0–1 scale (except tempo, which is in BPM).

No songs were added or removed from the starter dataset. The catalog skews toward Western popular music genres and does not include classical, hip-hop, country, or non-English-language music. Artists are fictional.

---

## 5. Strengths

- **High-energy pop and rock profiles** are served well because the catalog has multiple entries with high energy and low acousticness, giving the system real material to rank and diversify.
- **Explanations are transparent.** Every recommendation cites the actual feature values that drove the match — users can verify or dispute the reasoning.
- **The agentic loop actively corrects bias.** Without the refine step, the system would sometimes return 3 pop songs, 1 indie pop, and 1 synthwave. The diversity enforcement catches this automatically.
- **No hallucination risk.** Because there is no generative language model, the system cannot fabricate song details or invent reasons. Every output is traceable to a real data point.
- **Dual-metric output.** Each recommendation carries both a weighted relevance score (how well all factors combined) and a confidence score (how many factors clearly fired). This gives a richer picture than a single number.

---

## 6. Limitations and Bias

### Data Bias

- **Western cultural bias.** Every genre in the catalog — pop, lofi, rock, ambient, jazz, synthwave, indie pop — represents Western commercial music. A user looking for Afrobeats, K-pop, Bollywood, or any non-Western genre would receive zero relevant results. This isn't a bug; it's a reflection of whose taste was assumed when designing the catalog.
- **Genre categories are not neutral.** The label "jazz" describes a century of diverse music but functions here as a single binary tag. Two songs labeled "jazz" might sound completely different, yet both would score equally for a jazz-seeking user. The category system encodes assumptions about how music should be grouped.
- **Mood tags carry cultural assumptions.** Labeling a song "intense" or "relaxed" reflects a particular listener's judgment. The same song might feel energizing to one listener and exhausting to another. The system treats these tags as objective facts.

### Algorithmic Bias

- **Exact string matching on genre and mood.** Genre and mood use `==` comparison. A user typing "Lo-Fi" instead of "lofi" scores zero on that factor. There is no normalization or fuzzy matching — a small data-entry difference produces a large scoring difference.
- **Fixed, hand-tuned weights.** The base weights (genre 35%, mood 25%, energy 20%, acousticness 10%, valence 10%) were chosen manually based on intuition, not learned from user data. A listener who cares primarily about tempo or danceability — both unweighted factors — would consistently receive worse results, with no way to signal this through the current interface.
- **Binary acoustic preference.** `likes_acoustic` is a boolean toggle. Real acoustic preferences exist on a spectrum: someone might enjoy acoustic guitar but not acoustic piano, or prefer acoustic music only for certain moods. Collapsing this to true/false loses important nuance.
- **No temporal or contextual awareness.** The system recommends the same songs at 7am as at midnight, on a Monday as on a Friday. Real listeners' preferences shift with context — time of day, activity, emotional state — none of which VibeFinder can observe.
- **No personalization over time.** The system has no memory. It cannot learn that a user always skips certain recommendations, or that their taste has shifted. Every session starts from scratch.

---

## 7. Evaluation

The system was tested using an `EvaluationSuite` with 5 predefined profiles measuring 4 metrics:

| Profile | Relevance | Diversity | Consistency | Confidence |
|---------|-----------|-----------|-------------|------------|
| High-Energy Pop Fan | 0.61 | 0.80 | 1.0 | 0.70 |
| Chill Lofi Studier | 0.69 | 0.60 | 1.0 | 0.80 |
| Rock Gym Warrior | 0.58 | 0.80 | 1.0 | 0.65 |
| Jazz Afternoon Relaxer | 0.53 | 0.60 | 1.0 | 0.60 |
| Synthwave Night Driver | 0.53 | 0.80 | 1.0 | 0.60 |
| **Average** | **0.59** | **0.72** | **1.00** | **0.67** |

**Consistency is 1.0 across all profiles** — the system is deterministic, so the same input always produces the same top result.

**Relevance is lowest for Jazz and Synthwave** because there is only one song each in those genres. Remaining slots are filled by best-available alternatives, pulling the mean score down — an honest signal about catalog coverage, not algorithm failure.

**What surprised me most during testing:** Consistency being exactly 1.0 felt suspicious at first — it seemed too good. After investigation, it's simply a consequence of full determinism: no randomness, no sampling, no stochastic step anywhere in the pipeline. The system is consistent the same way a calculator is consistent. This raised a harder question: *consistency of what?* If the system consistently gives wrong answers, 1.0 consistency is not a virtue. That led me to look more carefully at the relevance scores — and the Jazz profile's 0.53 is where the real story is. The agent runs all 3 iterations and still can't converge, because the catalog doesn't contain enough material to satisfy the profile. The system accurately diagnosed its own limitation through the convergence flag.

Eleven automated unit tests cover: score range bounds, perfect-match scoring, CSV loading integrity, input guardrail clamping and error raising, and confidence scoring at both extremes.

---

## 8. Misuse and Risk

VibeFinder is a classroom project with a 10-song catalog, so direct misuse is unlikely. But the patterns it demonstrates exist at production scale, and understanding the risks is part of responsible design.

### How this type of system could be misused

**Filter bubble amplification.** A recommender optimizing purely for relevance (without VibeFinder's diversity enforcement) would gradually narrow a user's musical world — always recommending the same genre, reinforcing existing taste rather than expanding it. At scale, this shapes culture: if millions of users only hear music the algorithm already knows they like, new artists and genres struggle to reach audiences. VibeFinder's diversity threshold is a direct response to this risk.

**Demographic profiling from taste data.** Music preferences correlate with age, geography, cultural background, and socioeconomic status. A bad actor operating a music platform could use recommendation interaction data to infer demographics without directly asking — then use those inferences to target advertising, price-discriminate, or sell the data. The user never consented to this use of their taste profile.

**Commercial manipulation behind a neutral interface.** If the catalog were real and populated by a company with label partnerships, the genre weights could be subtly tuned to push commercially advantageous songs while appearing to serve user preferences. The interface looks objective — it shows a score — but the weights that produce the score could be serving the platform's interests rather than the listener's.

**Feedback loop retraining.** If VibeFinder were extended to learn from user clicks, and those clicks were mostly from one demographic group, the model would gradually optimize for that group's taste at the expense of others. The system would become more biased over time while appearing to improve.

### Prevention measures in VibeFinder

- **Diversity enforcement** is built into the agent loop — the system actively resists returning a monoculture of results.
- **Full transparency** — every recommendation explains exactly which features drove it, using real values the user can verify.
- **No user data storage** — VibeFinder collects nothing. Each session is stateless; there is no profile to exploit.
- **Open weights** — the scoring weights are visible in `src/recommender.py` and `src/agent.py`. There is no hidden optimization layer.
- **Model card** (this document) — naming the biases explicitly is itself a form of prevention. A system whose limitations are publicly documented is harder to misrepresent.

---

## 9. Future Work

- **Expand the catalog** to 100+ songs with genuine diversity across cultures, languages, and non-Western genres.
- **Learn weights from feedback** — let users rate recommendations and adjust scoring weights using a simple reinforcement signal.
- **Add collaborative filtering** — "users with similar profiles also liked..." as a second scoring pass.
- **Fuzzy genre and mood matching** — normalize input strings and use partial credit for related genres (e.g. "indie pop" should score partial credit for a "pop" song).
- **Session memory** — track which songs the user has already seen and penalize repeats.
- **Multi-user group recommendations** — compute a consensus profile across multiple users and find songs that satisfy the most people.
- **Fairness audit** — regularly evaluate whether the system serves all demographic groups equally, not just the profiles it was designed around.

---

## 10. Personal Reflection and AI Collaboration

Building VibeFinder made the abstract idea of a recommender system concrete: at its core, it is just a scoring function that converts preferences and item features into a number. What surprised me most was how much the diversity problem matters — without the refine step, the system confidently returned nearly identical songs because it was optimizing purely for closeness to the profile. A real music app that did this would feel like it was stuck in a loop.

This changed how I think about Spotify or YouTube recommendations: the "diversity tax" they sometimes impose — recommending something slightly outside your comfort zone — is probably a deliberate design choice to avoid that exact trap. The hardest part of recommendation is not finding the most similar thing; it is finding a set of things that is both relevant and *interesting*.

### Collaborating with AI on This Project

This project was built with AI assistance throughout — from architecture planning to code implementation to documentation. Reflecting honestly on that collaboration:

**One instance where the AI suggestion was genuinely helpful:**
The suggestion to implement `confidence_score` as a *separate function* from the weighted relevance score was a meaningful design insight. The relevance score measures how strongly all factors combined — it rewards depth of match. Confidence measures how many of the 4 key factors clearly fired — it rewards breadth. These are different things: a song can score 0.70 relevance by matching energy and acousticness very closely while missing genre and mood entirely, which gives the wrong impression of a "good" match. Having both metrics made the evaluation harness substantially more informative, and the per-profile results (Lofi Studier: confidence 0.80, Jazz: confidence 0.60) correlated cleanly with the relevance scores in a way that validated both metrics.

**One instance where the AI suggestion was flawed:**
The initial output format used a plain 3-tuple `(song, score, explanation)` throughout the entire pipeline — in `recommend_songs`, `agent.recommend`, and all callers. When confidence scoring was added in Phase 4, this required updating five separate files (`recommender.py`, `agent.py`, `evaluation.py`, `main.py`, `app.py`) because every caller that unpacked the tuple broke. A better initial design would have used a `dataclass` or `NamedTuple` for the recommendation result — adding a new field would then require changing only the definition, not every callsite. The AI's choice of a flat tuple was a premature simplification that created unnecessary refactoring work. In future projects, I would push back on plain tuples for any output format that might evolve.
