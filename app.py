import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.recommender import load_songs
from src.agent import Agent
from src.logger import get_logger, VALID_GENRES, VALID_MOODS

logger = get_logger("streamlit_app")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #e94560 0%, #7c3aed 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(233,69,96,0.3);
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* ── Song card ── */
.song-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.song-card:hover {
    border-color: rgba(233,69,96,0.5);
}
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e94560, #7c3aed);
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    margin-right: 10px;
}
.song-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
}
.song-meta {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    margin-top: 2px;
}
.genre-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-right: 6px;
}
.mood-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.score-pill {
    background: rgba(233,69,96,0.2);
    border: 1px solid rgba(233,69,96,0.4);
    color: #e94560;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 700;
}
.conf-pill {
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 700;
}
.stat-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 6px;
    margin: 4px 0 10px;
    overflow: hidden;
}
.stat-bar-fill {
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(90deg, #e94560, #7c3aed);
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-top: 10px;
}
.feature-box {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 8px 4px;
    text-align: center;
}
.feature-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.feature-value {
    color: white;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 2px;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.8) !important;
    font-weight: 500;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: white;
}
.sidebar-section {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.sidebar-heading {
    color: rgba(255,255,255,0.5);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

/* ── Diagnostics ── */
.diag-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.diag-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
}
.diag-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.trace-step {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #e94560;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.trace-tool {
    color: #e94560;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.trace-ts {
    color: rgba(255,255,255,0.35);
    font-size: 0.75rem;
    float: right;
}
.trace-text {
    color: rgba(255,255,255,0.7);
    font-size: 0.82rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Genre / mood helpers ──────────────────────────────────────────────────────
GENRE_COLORS = {
    "pop":       ("#e94560", "#fff0f3"),
    "rock":      ("#f97316", "#fff7ed"),
    "lofi":      ("#6366f1", "#eef2ff"),
    "ambient":   ("#10b981", "#ecfdf5"),
    "jazz":      ("#f59e0b", "#fffbeb"),
    "synthwave": ("#8b5cf6", "#f5f3ff"),
    "indie pop": ("#ec4899", "#fdf2f8"),
}
MOOD_COLORS = {
    "happy":    ("#facc15", "#422006"),
    "chill":    ("#22d3ee", "#0c4a6e"),
    "intense":  ("#ef4444", "#450a0a"),
    "relaxed":  ("#4ade80", "#052e16"),
    "moody":    ("#a78bfa", "#2e1065"),
    "focused":  ("#60a5fa", "#1e3a5f"),
}
GENRE_ICONS = {
    "pop": "🎤", "rock": "🎸", "lofi": "☕", "ambient": "🌌",
    "jazz": "🎷", "synthwave": "🌆", "indie pop": "🌿",
}
MOOD_ICONS = {
    "happy": "😊", "chill": "😌", "intense": "🔥",
    "relaxed": "🛋️", "moody": "🌙", "focused": "🎯",
}


def genre_tag(genre: str) -> str:
    bg, fg = GENRE_COLORS.get(genre, ("#6b7280", "#f9fafb"))
    icon = GENRE_ICONS.get(genre, "🎵")
    return (
        f'<span class="genre-tag" style="background:{bg}22;'
        f'color:{bg};border:1px solid {bg}66;">'
        f'{icon} {genre.title()}</span>'
    )


def mood_tag(mood: str) -> str:
    bg, fg = MOOD_COLORS.get(mood, ("#6b7280", "#f9fafb"))
    icon = MOOD_ICONS.get(mood, "🎵")
    return (
        f'<span class="mood-tag" style="background:{bg}22;'
        f'color:{bg};border:1px solid {bg}66;">'
        f'{icon} {mood.title()}</span>'
    )


def score_bar(value: float) -> str:
    pct = int(min(value, 1.0) * 100)
    return (
        f'<div class="stat-bar-bg">'
        f'<div class="stat-bar-fill" style="width:{pct}%"></div>'
        f'</div>'
    )


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

    genre_opts = sorted(VALID_GENRES)
    mood_opts  = sorted(VALID_MOODS)

    genre = st.selectbox(
        "🎸 Favorite Genre",
        genre_opts,
        format_func=lambda g: f"{GENRE_ICONS.get(g, '🎵')} {g.title()}",
    )
    mood = st.selectbox(
        "✨ Preferred Mood",
        mood_opts,
        format_func=lambda m: f"{MOOD_ICONS.get(m, '🎵')} {m.title()}",
    )

    st.markdown("---")
    energy = st.slider(
        "⚡ Energy Level",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = mellow & soft   |   1 = high-energy & intense",
    )
    likes_acoustic = st.toggle("🎸 Prefer acoustic / mellow sounds", value=False)

    st.markdown("---")
    k = st.slider("🎯 Number of recommendations", min_value=1, max_value=10, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Find My Songs", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div style="color:rgba(255,255,255,0.3);font-size:0.72rem;text-align:center;">'
        '30 songs · local AI · no internet needed'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main content ──────────────────────────────────────────────────────────────
if run_btn:
    user_prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }
    logger.info("UI request: %s", user_prefs)

    with st.spinner("Agent is thinking..."):
        songs = load_songs("data/songs.csv")
        agent = Agent(songs)
        results = agent.recommend(user_prefs, k=k)

    # ── Result header ──
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            f"### 🎧 Top {len(results)} Recommendations for "
            f"{GENRE_ICONS.get(genre,'🎵')} **{genre.title()}** "
            f"/ {MOOD_ICONS.get(mood,'✨')} **{mood.title()}**"
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

    # ── Song cards ──
    for rank, (song, score, explanation, confidence) in enumerate(results, start=1):
        g = song.get("genre", "")
        m = song.get("mood", "")
        accent, _ = GENRE_COLORS.get(g, ("#e94560", ""))

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
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("💬 Why this song?"):
            st.write(explanation)

    st.markdown("---")

    # ── Agent diagnostics ──
    with st.expander("🤖 Agent Diagnostics"):
        d1, d2, d3, d4 = st.columns(4)
        for col, label, value in [
            (d1, "Iterations", str(agent.state.iteration)),
            (d2, "Diversity", f"{agent.state.last_diversity:.2f}"),
            (d3, "Relevance", f"{agent.state.last_relevance:.2f}"),
            (d4, "Converged", "Yes" if agent.state.converged else "No"),
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

    # ── Reasoning trace ──
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
    # ── Empty state ──
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;color:rgba(255,255,255,0.35);">'
        '<div style="font-size:4rem;">🎧</div>'
        '<div style="font-size:1.2rem;font-weight:600;margin-top:1rem;color:rgba(255,255,255,0.5);">'
        'Set your taste profile and click <b>Find My Songs</b></div>'
        '<div style="margin-top:0.5rem;font-size:0.9rem;">30 songs across Pop · Rock · Jazz · Lo-Fi · Synthwave · Ambient · Indie Pop</div>'
        '</div>',
        unsafe_allow_html=True,
    )
