from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

def load_model():
    print("Loading Somali ASR model...")
    processor = Wav2Vec2Processor.from_pretrained("Mustafaa4a/ASR-Somali")
    model = Wav2Vec2ForCTC.from_pretrained("Mustafaa4a/ASR-Somali")
    model.eval()
    return processor, model

def transcribe_audio(path, processor, model):
    speech_array, sampling_rate = torchaudio.load(path)

    # Resample if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)

    input_values = processor(speech_array[0], sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.strip()
