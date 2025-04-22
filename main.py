from fastapi import FastAPI, File, UploadFile
from model_loader import load_model, transcribe_audio
from utils import convert_ogg_to_wav
import tempfile

app = FastAPI()

# Load Hugging Face model at startup (once)
processor, model = load_model()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded OGG/MP3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert OGG to WAV
    wav_path = convert_ogg_to_wav(tmp_path)

    # Transcribe the audio
    transcription = transcribe_audio(wav_path, processor, model)

    return {"transcription": transcription}
