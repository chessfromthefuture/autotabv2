"""Streamlit UI: upload a guitar recording, get tablature."""

import autotab.TabPrediction as tp
import streamlit as st

st.set_page_config(page_title="AutoTab tab generator", page_icon="🎸", layout="centered")

BANNER = (
    "https://images.unsplash.com/photo-1535587566541-97121a128dc5?auto=format&fit=crop&w=1200&q=80"
)
st.markdown(
    f"""
    <style>
    .banner {{
        background: linear-gradient(rgba(0,0,0,.45), rgba(0,0,0,.45)),
                    url({BANNER}) center/cover;
        border-radius: 12px; padding: 56px 24px; margin-bottom: 24px;
        text-align: center; color: white;
    }}
    .banner h1 {{ margin: 0; font-size: 52px; }}
    .banner p {{ margin: 8px 0 0; font-size: 22px; opacity: .8; }}
    </style>
    <div class="banner"><h1>AutoTab</h1><p>Learning the guitar the easy way</p></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return tp.load_model_and_weights()


@st.cache_data(show_spinner="Transcribing…")
def transcribe(
    file_bytes: bytes, mode: str, bars: int, bar_len: int, silence_weight: float, isolate: bool
) -> str:
    import io

    return tp.predict_tab(
        io.BytesIO(file_bytes),
        model=get_model(),
        mode=mode,
        num_div=bars,
        len_div=bar_len,
        silence_weight=silence_weight,
        isolate=isolate,
    )


def separation_available() -> bool:
    try:
        import demucs.api  # noqa: F401

        return True
    except ImportError:
        return False


uploaded_file = st.file_uploader(
    "Choose a guitar recording or a full song", type=["wav", "flac", "ogg", "mp3"]
)

if separation_available():
    isolate = st.toggle(
        "Isolate the guitar first (full songs with vocals, drums, bass…)",
        value=False,
        help="Runs Demucs htdemucs_6s and transcribes only the guitar stem. Takes about "
        "a minute for a 4-minute song on Apple Silicon.",
    )
else:
    isolate = False
    st.caption(
        'Install the `separate` extra to transcribe full songs: `uv pip install -e ".[separate]"`'
    )

MODES = {"Ergonomic Simple": "simple", "Ergonomic Rhythm": "rhythm", "All Frames": "frames"}
SENSITIVITY = {"Conservative": 1.0, "Balanced": 0.05, "Sensitive": 0.02}
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    mode_label = st.radio("Tab style", list(MODES), horizontal=True)
with col2:
    sensitivity = st.select_slider(
        "Note sensitivity",
        options=list(SENSITIVITY),
        value="Balanced",
        help="Conservative is the raw 2021 model; Balanced and Sensitive write more notes.",
    )
with col3:
    bars = st.slider("Bars per line", 1, 8, 4)

if uploaded_file is not None:
    st.audio(uploaded_file)
    tab = transcribe(
        uploaded_file.getvalue(), MODES[mode_label], bars, 16, SENSITIVITY[sensitivity]
    )
    st.subheader(f"{mode_label} predicted tab")
    st.code(tab, language=None)
    st.download_button("Download tab (.txt)", tab, file_name=f"{uploaded_file.name}.tab.txt")
else:
    st.info(
        "Upload a mono or stereo guitar recording to get started. The model was trained on "
        "solo acoustic guitar (GuitarSet), so clean single-guitar recordings work best."
    )
