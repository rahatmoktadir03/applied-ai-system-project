import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.recommender import load_songs
from src.agent import Agent
from src.logger import get_logger, VALID_GENRES, VALID_MOODS

logger = get_logger("streamlit_app")

st.set_page_config(page_title="VibeFinder", page_icon="🎵", layout="centered")

st.title("🎵 VibeFinder")
st.caption("Local AI-powered music recommendations — no external APIs.")

with st.sidebar:
    st.header("Your Taste Profile")
    genre = st.selectbox("Favorite Genre", sorted(VALID_GENRES))
    mood = st.selectbox("Preferred Mood", sorted(VALID_MOODS))
    energy = st.slider("Energy Level", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    likes_acoustic = st.toggle("I like acoustic / mellow sounds", value=False)
    k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    run_btn = st.button("Find My Songs", type="primary", use_container_width=True)

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

    st.subheader(f"Top {len(results)} Recommendations")

    for rank, (song, score, explanation) in enumerate(results, start=1):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rank}. {song['title']}**")
                st.caption(
                    f"{song['artist']} · {song['genre'].title()} · {song['mood'].title()}"
                )
            with col2:
                st.metric("Score", f"{score:.2f}")
            st.progress(min(float(score), 1.0))
            with st.expander("Why this song?"):
                st.write(explanation)
                st.write(
                    f"**Energy:** {song['energy']:.2f} &nbsp;|&nbsp; "
                    f"**Valence:** {song['valence']:.2f} &nbsp;|&nbsp; "
                    f"**Danceability:** {song['danceability']:.2f} &nbsp;|&nbsp; "
                    f"**Acousticness:** {song['acousticness']:.2f} &nbsp;|&nbsp; "
                    f"**Tempo:** {song['tempo_bpm']:.0f} BPM"
                )

    with st.expander("Agent Diagnostics"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Iterations", agent.state.iteration)
        col2.metric("Diversity", f"{agent.state.last_diversity:.2f}")
        col3.metric("Relevance", f"{agent.state.last_relevance:.2f}")
        st.write(f"**Converged:** {agent.state.converged}")
        st.write("**Iteration log:**")
        for line in agent.state.history:
            st.text(line)
else:
    st.info("Set your taste profile in the sidebar and click **Find My Songs**.")
