# English Accent Detector

A web app that detects English accents from speech using deep learning.
Upload any audio or video file and the app identifies the speaker's accent.

## Supported Accents
American · British · Australian · Indian

## How It Works

Audio/Video → ffmpeg extracts WAV → Wav2Vec2 creates 768-number speech fingerprint → Logistic Regression predicts accent

### Stage 1 — Feature Extraction
Facebook's Wav2Vec2 model (94M parameters, pretrained on 960 hours of English speech)
converts raw audio into a 768-number embedding that captures phonetic patterns
like vowel sounds, consonant sharpness, and speech rhythm.

### Stage 2 — Classification
A logistic regression classifier trained on 198 labeled samples from the
Speech Accent Archive learns which embedding patterns correspond to which accent.

## Dataset
Speech Accent Archive — George Mason University
2,140 speakers from 177 countries all reading the same standardized paragraph.
Controlled design means the only variable between samples is accent,
not vocabulary or content.

- American: 60 samples
- British: 48 samples (England-only for consistency)
- Indian: 58 samples
- Australian: 32 samples

## Model Performance
- Overall accuracy: 75% on held-out test data
- American: f1 = 0.89
- Indian: f1 = 0.80
- British: f1 = 0.57
- Australian: f1 = 0.57

## Why These Choices

Why Wav2Vec2 over other models?
Wav2Vec2 is pretrained on 960 hours of speech using self-supervised learning.
Transfer learning means it works well with only 198 labeled samples.
Whisper would be overkill — it is designed for transcription and is 10x larger.
MFCCs are older hand-engineered features that miss subtle accent patterns.

Why Logistic Regression over neural networks?
With only 198 training samples, a deep neural network would overfit badly.
Logistic regression generalizes better on small datasets and is fully interpretable.

## Project Structure

accent-detector/
├── app.py                 — Streamlit web UI
├── audio.py               — Audio extraction and processing
├── embeddings.py          — Wav2Vec2 feature extraction
├── train.py               — Model loading and retraining
├── accent_model.pkl       — Trained classifier (75% accuracy)
├── embeddings.npz         — Saved speech embeddings (198 samples)
├── accent_labels.pkl      — Accent label definitions
├── accent_detector.ipynb  — Full training notebook with explanations
├── requirements.txt       — Python dependencies
└── packages.txt           — System dependencies

## Setup and Run

Step 1 — Install ffmpeg

Windows:
winget install ffmpeg

Mac:
brew install ffmpeg

Step 2 — Install Python dependencies
pip install -r requirements.txt

Step 3 — Run the app
streamlit run app.py

Open http://localhost:8501 in your browser.

## Retrain the Model
To retrain using saved embeddings without reprocessing audio:
python train.py

## Tech Stack
- Streamlit — web interface
- Wav2Vec2 by Facebook — speech feature extraction
- scikit-learn — logistic regression classifier
- ffmpeg — audio and video processing
- Plotly — interactive charts

## Key Findings
- Indian accent performs best because it has very distinct phonetic features
- British accuracy improved from 50% to 57% by filtering to England-only speakers
- Main lesson learned: better data quality beats better algorithms
- Model fails on audio longer than 15 seconds — fixed by trimming to match training data