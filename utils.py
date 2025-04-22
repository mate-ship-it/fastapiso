from pydub import AudioSegment

def convert_ogg_to_wav(ogg_path):
    sound = AudioSegment.from_ogg(ogg_path)
    wav_path = ogg_path.replace(".ogg", ".wav")
    sound.export(wav_path, format="wav")
    return wav_path
