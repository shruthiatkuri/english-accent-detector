import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from audio import download_video, extract_audio, load_audio_for_model, save_uploaded_file
from embeddings import load_wav2vec2, get_embedding
from train import load_model, ACCENT_LABELS

# ── Page setup ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎧",
    layout="centered"
)

# ── Load models once at startup ─────────────────────────────────────────
# @st.cache_resource means: run once, remember result, never run again
# Without this, the 360MB model would reload every time user clicks anything
@st.cache_resource
def get_models():
    processor, wav2vec_model, device = load_wav2vec2()
    accent_classifier = load_model()
    return processor, wav2vec_model, device, accent_classifier

with st.spinner("Loading AI models... (first run takes 1-2 minutes)"):
    processor, wav2vec_model, device, accent_classifier = get_models()


# ── Prediction function ─────────────────────────────────────────────────
def predict_accent(audio_path: str) -> dict:
    """
    Full pipeline: audio file → accent prediction.
    Returns dict with accent name, confidence, and all probabilities.
    """
    # Load and clean audio
    speech_tensor = load_audio_for_model(audio_path)

    

    # Get 768-number fingerprint from Wav2Vec2
    embedding = get_embedding(speech_tensor, processor, wav2vec_model, device)
    embedding_array = np.array(embedding).reshape(1, -1)


    # Read labels DIRECTLY from model — never hardcode them
    # This guarantees label order always matches what model expects
    model_labels = accent_classifier.named_steps['classifier'].classes_

    # Get probabilities for each accent
    proba = accent_classifier.predict_proba(embedding_array)[0]

        
    # Find the top prediction
    top_index = np.argmax(proba)
    top_accent = model_labels[top_index]
    top_confidence = round(proba[top_index] * 100, 1)

    all_probs = {
        label: round(prob * 100, 1)
        for label, prob in zip(model_labels, proba)
    }

    return {
        'accent': top_accent,
        'confidence': top_confidence,
        'all_probs': all_probs
    }


# ── Results display ─────────────────────────────────────────────────────
def display_results(result: dict):
    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Accent", result['accent'].title())
    with col2:
        st.metric("Confidence", f"{result['confidence']}%")

    # Bar chart of all accent probabilities
    accents = list(result['all_probs'].keys())
    probs = list(result['all_probs'].values())
    colors = [
        '#ff7f0e' if a == result['accent'] else '#1f77b4'
        for a in accents
    ]

    fig = go.Figure(go.Bar(
        x=probs,
        y=[a.title() for a in accents],
        orientation='h',
        marker_color=colors,
        text=[f"{p}%" for p in probs],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="Probability (%)"),
        yaxis_title="",
        height=250,
        margin=dict(l=20, r=40, t=20, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Detected **{result['accent'].title()}** accent with "
        f"**{result['confidence']}%** confidence. "
        f"Model trained on the Speech Accent Archive dataset using "
        f"Wav2Vec2 features + logistic regression."
    )


# ── UI ──────────────────────────────────────────────────────────────────
st.title("🎧 English Accent Detector")
st.markdown(
    "Upload a video or audio file to detect the English accent. "
    "Supports: American, British, Australian, Indian."
)
st.divider()

tab_upload, tab_url = st.tabs(["Upload File", "Paste URL"])

# Tab 1: File upload
with tab_upload:
    st.markdown("**Supported formats:** MP4, WAV, MP3, M4A")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["mp4", "mov", "wav", "mp3", "m4a"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Analyze Accent", key="btn_upload", type="primary"):

            with st.spinner("Saving file..."):
                file_path = save_uploaded_file(uploaded_file)

            # Extract audio if video format
            audio_path = file_path
            if uploaded_file.name.lower().endswith(('.mp4', '.mov')):
                with st.spinner("Extracting audio from video..."):
                    try:
                        audio_path = extract_audio(file_path)
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Could not extract audio: {e}")
                        st.stop()

            with st.spinner("Analyzing accent..."):
                try:
                    result = predict_accent(audio_path)
                    os.remove(audio_path)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

            display_results(result)

# Tab 2: URL input
with tab_url:
    st.markdown("Paste a direct link to a public MP4 file.")
    st.caption("Note: YouTube links do not work. Use direct MP4 URLs only.")

    video_url = st.text_input("Video URL", placeholder="https://example.com/video.mp4")

    if st.button("Analyze Accent", key="btn_url", type="primary"):
        if not video_url.strip():
            st.warning("Please enter a URL first.")
            st.stop()

        with st.spinner("Downloading video..."):
            try:
                video_path = download_video(video_url.strip())
            except Exception as e:
                st.error(f"Could not download: {e}")
                st.stop()

        with st.spinner("Extracting audio..."):
            try:
                audio_path = extract_audio(video_path)
                os.remove(video_path)
            except Exception as e:
                st.error(f"Could not extract audio: {e}")
                st.stop()

        with st.spinner("Analyzing accent..."):
            try:
                result = predict_accent(audio_path)
                os.remove(audio_path)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        display_results(result)

# About section
with st.expander("How does this work?"):
    st.markdown("""
    **Dataset:** Speech Accent Archive — 2,140 speakers from 177 countries
    reading the same standardized paragraph.

    **Feature extraction:** Facebook's Wav2Vec2 (94M parameters, trained on
    960 hours of speech) converts raw audio into a 768-number fingerprint.

    **Classifier:** Logistic regression trained on 210 labeled audio samples.
    Achieves 75% accuracy on held-out test data.

    **Accents supported:** American · British · Australian · Indian
    """)