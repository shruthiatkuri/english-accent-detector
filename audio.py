import os
import tempfile
import urllib.request
import torchaudio
import ffmpeg


def download_video(url: str) -> str:
    """Downloads a video from a URL to a temp file. Returns the file path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name

def trim_audio(speech_tensor, max_seconds=15):
    """
    Trims audio to first max_seconds only.
    
    WHY: Wav2Vec2 was trained on short clips (8-15 seconds).
    Long audio creates too many time steps to average meaningfully.
    The first 15 seconds contains enough speech for accent detection.
    Speech Accent Archive samples are all 8-15 seconds long.
    """
    max_samples = max_seconds * 16000  # 15 seconds × 16000 samples/sec
    
    if len(speech_tensor) > max_samples:
        print(f"Trimming audio from {len(speech_tensor)/16000:.1f}s to {max_seconds}s")
        return speech_tensor[:max_samples]
    
    return speech_tensor

def extract_audio(video_path: str) -> str:
    """
    Extracts audio from a video file.
    Converts to mono WAV at 16kHz — the format Wav2Vec2 needs.
    Returns the path to the audio file.
    """
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                audio_path,
                format='wav',
                acodec='pcm_s16le',  # uncompressed audio
                ac=1,                # mono
                ar='16000'           # 16kHz sample rate
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
    return audio_path


def load_audio_for_model(audio_path: str):
    """
    Loads any audio file and converts to the format Wav2Vec2 needs.
    Uses ffmpeg first to convert to WAV — handles any input format.
    Returns a 1D PyTorch tensor ready for Wav2Vec2.
    """
    wav_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav"
    ).name

    try:
        (
            ffmpeg
            .input(audio_path)
            .output(
                wav_path,
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar='16000'
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Could not convert audio: {e.stderr.decode()}")

    speech_array, sample_rate = torchaudio.load(wav_path)
    os.remove(wav_path)

    if speech_array.shape[0] > 1:
        speech_array = speech_array.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        speech_array = resampler(speech_array)

    speech_tensor = speech_array.squeeze()
    
    # ── Trim to first 15 seconds ─────────────────────────────────────
    speech_tensor = trim_audio(speech_tensor, max_seconds=15)
    
    return speech_tensor


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name
