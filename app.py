import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
from src.recommender import load_songs
from src.agent import Agent
from src.logger import get_logger, VALID_GENRES, VALID_MOODS

logger = get_logger("streamlit_app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.hero-banner {
    background: linear-gradient(135deg, #e94560 0%, #7c3aed 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(233,69,96,0.3);
}
.hero-title  { font-size: 3rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
.hero-sub    { color: rgba(255,255,255,0.75); font-size: 1rem; margin-top: 0.4rem; }
.song-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(10px);
}
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e94560, #7c3aed);
    color: white; font-weight: 700; font-size: 0.8rem;
    border-radius: 50%; width: 28px; height: 28px;
    line-height: 28px; text-align: center; margin-right: 10px;
}
.song-title { font-size: 1.1rem; font-weight: 700; color: white; }
.song-meta  { color: rgba(255,255,255,0.55); font-size: 0.85rem; margin-top: 2px; }
.genre-tag, .mood-tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; margin-right: 6px;
}
.score-pill {
    background: rgba(233,69,96,0.2); border: 1px solid rgba(233,69,96,0.4);
    color: #e94560; border-radius: 20px; padding: 4px 12px;
    font-size: 0.85rem; font-weight: 700;
}
.conf-pill {
    background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa; border-radius: 20px; padding: 4px 12px;
    font-size: 0.85rem; font-weight: 700;
}
.stat-bar-bg   { background: rgba(255,255,255,0.08); border-radius: 6px; height: 6px; margin: 4px 0 10px; overflow: hidden; }
.stat-bar-fill { height: 6px; border-radius: 6px; background: linear-gradient(90deg, #e94560, #7c3aed); }
.feature-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 8px; margin-top: 10px;
}
.feature-box   { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 8px 4px; text-align: center; }
.feature-label { color: rgba(255,255,255,0.45); font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; }
.feature-value { color: white; font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
.breakdown-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.breakdown-label { width: 62px; color: rgba(255,255,255,0.45); font-size: 0.68rem; text-transform: uppercase; }
.breakdown-track { flex: 1; background: rgba(255,255,255,0.07); border-radius: 4px; height: 7px; overflow: hidden; }
.breakdown-val   { width: 32px; color: rgba(255,255,255,0.6); font-size: 0.68rem; text-align: right; }
.diag-card  { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 16px; text-align: center; }
.diag-value { font-size: 1.6rem; font-weight: 700; color: white; }
.diag-label { color: rgba(255,255,255,0.45); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
.trace-step  { background: rgba(255,255,255,0.04); border-left: 3px solid #e94560; border-radius: 0 8px 8px 0; padding: 10px 14px; margin-bottom: 8px; }
.trace-tool  { color: #e94560; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
.trace-ts    { color: rgba(255,255,255,0.35); font-size: 0.75rem; float: right; }
.trace-text  { color: rgba(255,255,255,0.7); font-size: 0.82rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GENRE_COLORS = {
    "pop": "#e94560", "rock": "#f97316", "lofi": "#6366f1",
    "ambient": "#10b981", "jazz": "#f59e0b", "synthwave": "#8b5cf6", "indie pop": "#ec4899",
}
MOOD_COLORS = {
    "happy": "#facc15", "chill": "#22d3ee", "intense": "#ef4444",
    "relaxed": "#4ade80", "moody": "#a78bfa", "focused": "#60a5fa",
}
GENRE_ICONS = {
    "pop": "🎤", "rock": "🎸", "lofi": "☕", "ambient": "🌌",
    "jazz": "🎷", "synthwave": "🌆", "indie pop": "🌿",
}
MOOD_ICONS = {
    "happy": "😊", "chill": "😌", "intense": "🔥",
    "relaxed": "🛋️", "moody": "🌙", "focused": "🎯",
}
BREAKDOWN_COLORS = {
    "Genre": "#e94560", "Mood": "#a78bfa", "Energy": "#f97316",
    "Acoustic": "#10b981", "Valence": "#60a5fa",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def genre_tag(genre: str) -> str:
    c = GENRE_COLORS.get(genre, "#6b7280")
    icon = GENRE_ICONS.get(genre, "🎵")
    return (f'<span class="genre-tag" style="background:{c}22;color:{c};border:1px solid {c}66;">'
            f'{icon} {genre.title()}</span>')

def mood_tag(mood: str) -> str:
    c = MOOD_COLORS.get(mood, "#6b7280")
    icon = MOOD_ICONS.get(mood, "🎵")
    return (f'<span class="mood-tag" style="background:{c}22;color:{c};border:1px solid {c}66;">'
            f'{icon} {mood.title()}</span>')

def score_bar(value: float) -> str:
    pct = int(min(value, 1.0) * 100)
    return (f'<div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct}%"></div></div>')

def compute_breakdown(user_prefs: dict, song: dict, weights: dict) -> dict:
    """Returns each scoring component's contribution as a 0–max_weight value."""
    user_energy   = float(user_prefs.get("energy", 0.5))
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    genre_match   = 1.0 if song.get("genre") == user_prefs.get("genre") else 0.0
    mood_match    = 1.0 if song.get("mood")  == user_prefs.get("mood")  else 0.0
    energy_sim    = 1.0 - abs(float(song.get("energy", 0.5)) - user_energy)
    acoustic      = float(song.get("acousticness", 0.5)) if likes_acoustic else 1.0 - float(song.get("acousticness", 0.5))
    valence       = float(song.get("valence", 0.5))
    return {
        "Genre":   weights.get("genre",       0.35) * genre_match,
        "Mood":    weights.get("mood",        0.25) * mood_match,
        "Energy":  weights.get("energy",      0.20) * energy_sim,
        "Acoustic":weights.get("acousticness",0.10) * acoustic,
        "Valence": weights.get("valence",     0.10) * valence,
    }

def breakdown_bars_html(breakdown: dict) -> str:
    max_possible = {"Genre": 0.35, "Mood": 0.25, "Energy": 0.20, "Acoustic": 0.10, "Valence": 0.10}
    rows = []
    for label, val in breakdown.items():
        pct  = int((val / max_possible.get(label, 0.35)) * 100)
        color = BREAKDOWN_COLORS.get(label, "#e94560")
        rows.append(
            f'<div class="breakdown-row">'
            f'  <span class="breakdown-label">{label}</span>'
            f'  <div class="breakdown-track">'
            f'    <div style="width:{pct}%;background:{color};height:7px;border-radius:4px;"></div>'
            f'  </div>'
            f'  <span class="breakdown-val">{val:.2f}</span>'
            f'</div>'
        )
    return (
        '<div style="margin-top:12px;padding-top:10px;'
        'border-top:1px solid rgba(255,255,255,0.07);">'
        '<div style="color:rgba(255,255,255,0.35);font-size:0.65rem;'
        'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Score Breakdown</div>'
        + "".join(rows) + "</div>"
    )

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("excluded_ids", set()),
    ("results", None),
    ("agent", None),
    ("last_prefs", None),
    ("last_k", 5),
    ("should_run", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🎵 VibeFinder</div>
  <div class="hero-sub">AI-powered music recommendations — no external APIs.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎚️ Your Taste Profile")
    st.markdown("---")
    genre = st.selectbox(
        "🎸 Favorite Genre", sorted(VALID_GENRES),
        format_func=lambda g: f"{GENRE_ICONS.get(g,'🎵')} {g.title()}",
    )
    mood = st.selectbox(
        "✨ Preferred Mood", sorted(VALID_MOODS),
        format_func=lambda m: f"{MOOD_ICONS.get(m,'🎵')} {m.title()}",
    )
    st.markdown("---")
    energy = st.slider("⚡ Energy Level", 0.0, 1.0, 0.5, 0.05,
                       help="0 = mellow & soft   |   1 = high-energy & intense")
    likes_acoustic = st.toggle("🎸 Prefer acoustic / mellow sounds", value=False)
    st.markdown("---")
    k = st.slider("🎯 Number of recommendations", 1, 10, 5)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Find My Songs", type="primary", use_container_width=True)
    st.markdown("---")
    all_songs_count = len(load_songs("data/songs.csv"))
    st.markdown(
        f'<div style="color:rgba(255,255,255,0.3);font-size:0.72rem;text-align:center;">'
        f'{all_songs_count} songs · local AI · no internet needed</div>',
        unsafe_allow_html=True,
    )

# ── Trigger logic ─────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.excluded_ids = set()
    st.session_state.last_prefs   = {"genre": genre, "mood": mood, "energy": energy, "likes_acoustic": likes_acoustic}
    st.session_state.last_k       = k
    st.session_state.should_run   = True

if st.session_state.should_run and st.session_state.last_prefs:
    with st.spinner("Agent is thinking..."):
        all_songs = load_songs("data/songs.csv")
        filtered  = [s for s in all_songs if s.get("id") not in st.session_state.excluded_ids]
        agent     = Agent(filtered)
        results   = agent.recommend(st.session_state.last_prefs, k=st.session_state.last_k)
    st.session_state.results     = results
    st.session_state.agent       = agent
    st.session_state.should_run  = False
    logger.info("Agent ran: %d results, excluded=%s", len(results), st.session_state.excluded_ids)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rec, tab_browse = st.tabs(["🎧 Recommendations", "📚 Browse All Songs"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommendations
# ════════════════════════════════════════════════════════════════════════════
with tab_rec:
    results = st.session_state.results
    agent   = st.session_state.agent
    prefs   = st.session_state.last_prefs

    if results and agent and prefs:
        g_label = prefs.get("genre", "")
        m_label = prefs.get("mood", "")

        # Header row
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            n_excluded = len(st.session_state.excluded_ids)
            excl_note  = f"  _(+{n_excluded} removed)_" if n_excluded else ""
            st.markdown(
                f"### 🎧 Top {len(results)} Recommendations for "
                f"{GENRE_ICONS.get(g_label,'🎵')} **{g_label.title()}** "
                f"/ {MOOD_ICONS.get(m_label,'✨')} **{m_label.title()}**"
                + excl_note
            )
        with col_h2:
            conv_color = "#4ade80" if agent.state.converged else "#facc15"
            st.markdown(
                f'<div style="text-align:right;padding-top:8px;">'
                f'<span style="color:{conv_color};font-size:0.85rem;font-weight:600;">'
                f'{"✅ Converged" if agent.state.converged else "⚠️ Best effort"} '
                f'({agent.state.iteration} iter)</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        weights = agent.state.weights

        # Song cards
        for rank, (song, score, explanation, confidence) in enumerate(results, start=1):
            g = song.get("genre", "")
            m = song.get("mood", "")
            accent = GENRE_COLORS.get(g, "#e94560")
            song_id = song.get("id")

            breakdown = compute_breakdown(prefs, song, weights)

            st.markdown(
                f'<div class="song-card" style="border-left:4px solid {accent};">'
                f'<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
                f'  <div>'
                f'    <span class="rank-badge">{rank}</span>'
                f'    <span class="song-title">{song["title"]}</span><br>'
                f'    <span class="song-meta" style="margin-left:38px;">{song["artist"]}</span>'
                f'  </div>'
                f'  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
                f'    {genre_tag(g)} {mood_tag(m)}'
                f'    <span class="score-pill">Score {score:.2f}</span>'
                f'    <span class="conf-pill">Confidence {confidence:.0%}</span>'
                f'  </div>'
                f'</div>'
                f'{score_bar(score)}'
                f'<div class="feature-grid">'
                f'  <div class="feature-box"><div class="feature-label">Energy</div>'
                f'    <div class="feature-value">{float(song["energy"]):.2f}</div></div>'
                f'  <div class="feature-box"><div class="feature-label">Valence</div>'
                f'    <div class="feature-value">{float(song["valence"]):.2f}</div></div>'
                f'  <div class="feature-box"><div class="feature-label">Dance</div>'
                f'    <div class="feature-value">{float(song["danceability"]):.2f}</div></div>'
                f'  <div class="feature-box"><div class="feature-label">Acoustic</div>'
                f'    <div class="feature-value">{float(song["acousticness"]):.2f}</div></div>'
                f'  <div class="feature-box"><div class="feature-label">BPM</div>'
                f'    <div class="feature-value">{float(song["tempo_bpm"]):.0f}</div></div>'
                f'</div>'
                f'{breakdown_bars_html(breakdown)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_exp, col_btn = st.columns([5, 1])
            with col_exp:
                with st.expander("💬 Why this song?"):
                    st.write(explanation)
            with col_btn:
                if st.button("👎 Not this", key=f"dislike_{song_id}", help="Remove this song and refresh"):
                    st.session_state.excluded_ids.add(song_id)
                    st.session_state.should_run = True
                    st.rerun()

        st.markdown("---")

        # Agent diagnostics
        with st.expander("🤖 Agent Diagnostics"):
            d1, d2, d3, d4 = st.columns(4)
            for col, label, value in [
                (d1, "Iterations", str(agent.state.iteration)),
                (d2, "Diversity",  f"{agent.state.last_diversity:.2f}"),
                (d3, "Relevance",  f"{agent.state.last_relevance:.2f}"),
                (d4, "Converged",  "Yes" if agent.state.converged else "No"),
            ]:
                col.markdown(
                    f'<div class="diag-card"><div class="diag-value">{value}</div>'
                    f'<div class="diag-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")
            st.markdown("**Iteration log:**")
            for line in agent.state.history:
                st.code(line, language=None)

        # Reasoning trace
        with st.expander("🔍 Reasoning Trace (Observable Tool-Call Chain)"):
            st.caption("Every intermediate decision the agent made, in order.")
            for i, step in enumerate(agent.state.reasoning_trace, 1):
                st.markdown(
                    f'<div class="trace-step">'
                    f'<span class="trace-tool">Step {i} · {step.tool}</span>'
                    f'<span class="trace-ts">{step.timestamp}</span>'
                    f'<div class="trace-text"><b>Reasoning:</b> {step.reasoning}</div>'
                    f'<div class="trace-text"><b>Decision:</b> {step.decision}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            '<div style="text-align:center;padding:4rem 2rem;color:rgba(255,255,255,0.35);">'
            '<div style="font-size:4rem;">🎧</div>'
            '<div style="font-size:1.2rem;font-weight:600;margin-top:1rem;color:rgba(255,255,255,0.5);">'
            'Set your taste profile and click <b>Find My Songs</b></div>'
            '<div style="margin-top:0.5rem;font-size:0.9rem;">'
            '31 songs across Pop · Rock · Jazz · Lo-Fi · Synthwave · Ambient · Indie Pop</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Browse All Songs
# ════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.markdown("### 📚 Full Song Catalog")

    all_songs = load_songs("data/songs.csv")
    df = pd.DataFrame(all_songs)

    # Filter controls
    col_search, col_genre_f, col_mood_f = st.columns([2, 1, 1])
    with col_search:
        search = st.text_input("🔍 Search by title or artist", placeholder="e.g. Taylor Swift, Blinding...")
    with col_genre_f:
        genre_filter = st.multiselect("Genre", sorted(VALID_GENRES),
                                      format_func=lambda g: f"{GENRE_ICONS.get(g,'')} {g.title()}")
    with col_mood_f:
        mood_filter = st.multiselect("Mood", sorted(VALID_MOODS),
                                     format_func=lambda m: f"{MOOD_ICONS.get(m,'')} {m.title()}")

    # Apply filters
    view = df.copy()
    if search:
        mask = (view["title"].str.contains(search, case=False, na=False) |
                view["artist"].str.contains(search, case=False, na=False))
        view = view[mask]
    if genre_filter:
        view = view[view["genre"].isin(genre_filter)]
    if mood_filter:
        view = view[view["mood"].isin(mood_filter)]

    st.caption(f"Showing {len(view)} / {len(df)} songs")

    # Display as a clean table
    display_cols = ["title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence", "danceability", "acousticness"]
    display_df = view[display_cols].rename(columns={
        "title": "Title", "artist": "Artist", "genre": "Genre", "mood": "Mood",
        "energy": "Energy", "tempo_bpm": "BPM", "valence": "Valence",
        "danceability": "Dance", "acousticness": "Acoustic",
    }).reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=520,
        column_config={
            "Energy":   st.column_config.ProgressColumn("Energy",   min_value=0, max_value=1, format="%.2f"),
            "Valence":  st.column_config.ProgressColumn("Valence",  min_value=0, max_value=1, format="%.2f"),
            "Dance":    st.column_config.ProgressColumn("Dance",    min_value=0, max_value=1, format="%.2f"),
            "Acoustic": st.column_config.ProgressColumn("Acoustic", min_value=0, max_value=1, format="%.2f"),
            "BPM":      st.column_config.NumberColumn("BPM", format="%d"),
        },
        hide_index=True,
    )

    # Download button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download catalog as CSV",
        data=csv_bytes,
        file_name="vibefinder_catalog.csv",
        mime="text/csv",
    )
