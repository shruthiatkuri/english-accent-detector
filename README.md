# English Accent Detection Tool

This is a simple Streamlit app that detects English accents from public video URLs (MP4 or Loom).

## Features

- Accepts a public video URL
- Extracts audio from the video using ffmpeg
- Uses a pre-trained Wav2Vec2 model to extract speech features
- Uses a sample classifier to predict accents like American, British, Australian, Indian, and Irish
- Shows confidence score and plays back extracted audio

## How to use

1. Open the app link: [https://english-accent-detector-soyxj7sato67p4lpibl3hf.streamlit.app/]
2. Paste a public video URL with clear English speech (e.g., TED Talk mp4 link)
3. Click "Analyze"
4. View predicted accent and confidence

## Limitations

- Accent classification model is a demo with simulated weights (not trained on real accent data)
- Some videos without audio or very noisy audio may fail
- Performance depends on quality and length of speech in the video

## Improvements

- Train a robust classifier on real labeled accent datasets
- Add support for audio-only URLs and local uploads
- Improve UI and error handling
- Deploy with better scalability and logging

---

Thanks for reviewing!

