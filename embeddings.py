import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_wav2vec2():
    """
    Downloads and loads Wav2Vec2 processor and model.
    First run downloads 360MB. After that loads from cache instantly.
    Returns processor, model, device.
    """
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # prediction mode, not training mode

    return processor, model, device


def get_embedding(speech_tensor, processor, model, device) -> np.ndarray:
    """
    Converts a 1D audio tensor into a 768-number embedding.

    Steps:
    1. Processor formats audio for the model
    2. Neural network processes it through 12 transformer layers
    3. We average across time to get one fixed-size vector
    4. Returns numpy array of shape (768,)
    """
    inputs = processor(
        speech_tensor.numpy(),
        return_tensors="pt",
        sampling_rate=16000,
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():  # no gradient tracking needed for inference
        outputs = model(input_values)

    # Average across time steps: (1, time, 768) → (1, 768)
    embedding = outputs.last_hidden_state.mean(dim=1)

    # Return flat array: (1, 768) → (768,)
    return embedding.cpu().numpy()[0]