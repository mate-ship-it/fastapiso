import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from pydub import AudioSegment
import numpy as np

def load_model():
    print("Loading Somali ASR model...")

    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")  # Fetch from environment variables

    processor = Wav2Vec2Processor.from_pretrained("Mustafaa4a/ASR-Somali", token=hf_token)
    model = Wav2Vec2ForCTC.from_pretrained("Mustafaa4a/ASR-Somali", token=hf_token)

    model.eval()
    return processor, model

def transcribe_audio(path, processor, model):
    # Load audio and resample
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # normalize to [-1, 1]

    # Process input for model
    input_values = processor(samples, sampling_rate=16000, return_tensors="pt").input_values

    # Inference without gradients
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.strip()
