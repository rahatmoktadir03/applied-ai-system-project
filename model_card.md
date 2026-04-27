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

---

## 6. Limitations and Bias

- **Catalog size.** With only 10 songs, profiles targeting underrepresented genres (jazz: 1 song, ambient: 1 song) will always see non-matching songs in most of their top 5. The diversity metric will always be 1.0 for these profiles regardless of relevance, which is misleading.
- **Exact string matching.** Genre and mood are compared with `==`. A user who types "Lo-Fi" instead of "lofi" gets zero credit on the genre factor. There is no normalization or fuzzy matching.
- **Western bias.** The catalog entirely represents Western commercial music. A user from a different cultural background would have no relevant catalog entries.
- **Fixed weight structure.** The base weights (genre 35%, mood 25%, etc.) were chosen by hand. A user whose taste is primarily driven by tempo or danceability — factors with no weight — would get poor results.
- **No personalization over time.** The system has no memory. Running it twice with the same profile always gives the same result — it cannot learn from feedback or evolve with the user.

---

## 7. Evaluation

The system was tested using an `EvaluationSuite` with 5 predefined profiles:

| Profile | Mean Relevance | Diversity Index | Consistency |
|---------|---------------|-----------------|-------------|
| High-Energy Pop Fan | ~0.70 | ~0.80 | 1.0 |
| Chill Lofi Studier | ~0.60 | ~0.80 | 1.0 |
| Rock Gym Warrior | ~0.55 | ~0.80 | 1.0 |
| Jazz Afternoon Relaxer | ~0.45 | ~1.00 | 1.0 |
| Synthwave Night Driver | ~0.55 | ~0.80 | 1.0 |

**Consistency is 1.0 across all profiles** — the system is deterministic, so the same input always produces the same top result.

**Relevance is lowest for Jazz** because there is only one jazz song in the catalog. The remaining 4 slots are filled by best-available other songs, pulling the mean score down.

**Diversity is highest for Jazz** for the same reason — with only one genre match, the refine step fills the rest with maximally varied options.

Three automated tests cover: sorted recommendation output, non-empty explanations, and a full evaluation suite smoke test.

---

## 8. Future Work

- **Expand the catalog** to 100+ songs with real diversity across genres and cultures.
- **Learn weights from feedback** — let users rate recommendations and adjust the scoring weights using gradient descent or a simple reinforcement signal.
- **Add collaborative filtering** — "users with similar profiles also liked..." as a second scoring pass.
- **Fuzzy genre and mood matching** — normalize input strings and use partial credit for related genres (e.g. "indie pop" should score partial credit for a "pop" song).
- **Session memory** — track which songs the user has already seen and penalize repeats.
- **Multi-user group recommendations** — compute a consensus profile across multiple users and find songs that satisfy the most people.

---

## 9. Personal Reflection

Building VibeFinder made the abstract idea of a recommender system concrete: at its core, it is just a scoring function that converts preferences and item features into a number. What surprised me most was how much the diversity problem matters — without the refine step, the system confidently handed back nearly identical songs because it was optimizing purely for closeness to the profile. A real music app that did this would feel like it was stuck in a loop.

This changed how I think about Spotify or YouTube recommendations: the "diversity tax" they sometimes impose — recommending something slightly outside your comfort zone — is probably a deliberate design choice to avoid that exact trap. The hardest part of recommendation is not finding the most similar thing; it is finding a set of things that is both relevant and interesting.
