from fastapi import FastAPI, File, UploadFile
from model_loader import load_model, transcribe_audio
from utils import convert_ogg_to_wav
from fastapi.responses import JSONResponse
import tempfile
import os

app = FastAPI()

# Load Hugging Face model at startup (once)
processor, model = load_model()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save the uploaded OGG file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert OGG to WAV
        wav_path = convert_ogg_to_wav(tmp_path)

        # Transcribe the audio
        transcription = transcribe_audio(wav_path, processor, model)

        # Cleanup temp files
        os.remove(tmp_path)
        os.remove(wav_path)

        return {"transcription": transcription}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
