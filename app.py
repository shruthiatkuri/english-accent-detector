# app.py

import streamlit as st
import tempfile
import os
import torch
import torchaudio
import urllib.request
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
import numpy as np

st.set_page_config(page_title="English Accent Detector", layout="centered")

st.title("🎙️ English Accent Detection Tool")
st.markdown("Upload or paste a public video link (MP4/Loom), and we'll analyze the speaker’s English accent.")

# ---------------------- Accent Labels & Dummy Model ---------------------- #
accent_labels = ['American', 'British', 'Australian', 'Indian', 'Irish']

# NOTE: In a real use-case, train this model with accent-labeled embeddings.
# Here we simulate a trained model with dummy weights just for demonstration.
model = LogisticRegression()
model.classes_ = np.array(accent_labels)
model.coef_ = np.random.randn(len(accent_labels), 768)  # Simulated
model.intercept_ = np.random.randn(len(accent_labels))
model._fit_intercept = True

# ---------------------- Load Pretrained Wav2Vec2 ---------------------- #
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

device = "cuda" if torch.cuda.is_available() else "cpu"
wav2vec_model.to(device)

# ---------------------- Functions ---------------------- #
def download_video(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name

def extract_audio(video_path):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio track found in the video.")
    
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
    return audio_path

def get_embedding(audio_path):
    speech_array, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if speech_array.shape[0] > 1:
        speech_array = speech_array.mean(dim=0, keepdim=True)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze()

    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


def predict_accent(embedding):
    probs = softmax(model.decision_function(embedding))[0]
    top_idx = np.argmax(probs)
    return accent_labels[top_idx], round(probs[top_idx] * 100, 2), probs

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ---------------------- UI Workflow ---------------------- #
video_url = st.text_input("Enter video URL (public MP4 or Loom):")

if st.button("Analyze") and video_url:
    with st.spinner("Downloading video..."):
        try:
            video_file = download_video(video_url)
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            st.stop()

    with st.spinner("Extracting audio..."):
        audio_path = extract_audio(video_file)

    with st.spinner("Analyzing accent..."):
        embedding = get_embedding(audio_path)
        accent, confidence, probs = predict_accent(embedding)

        st.success(f"**Accent Detected**: {accent}")
        st.metric(label="Confidence Score", value=f"{confidence} %")
        st.markdown(f"🔍 This result is based on speech features extracted using Wav2Vec2 and a sample classifier.")

        st.audio(audio_path, format='audio/wav')

    os.remove(video_file)
    os.remove(audio_path)
