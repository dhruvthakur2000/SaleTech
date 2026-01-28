import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
from experiments.stt_eval.models.whisper import WhisperSTT
#from experiments.stt_eval.models.vakyansh import VakyanshSTT
#from experiments.stt_eval.models.nemo import NemoSTT

DURATION = 10  # seconds
SAMPLE_RATE = 16000

MODELS = [
    WhisperSTT("small"),
    # VakyanshSTT("PATH_TO_NEMO_MODEL"),
    # NemoSTT("PATH_TO_NEMO_MODEL"),
]

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio.squeeze()

def save_temp_wav(audio, sample_rate):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wav.write(temp.name, sample_rate, audio)
    return temp.name

def main():
    audio = record_audio()
    wav_path = save_temp_wav(audio, SAMPLE_RATE)
    print(f"Saved temp audio: {wav_path}")
    for model in MODELS:
        print(f"\nModel: {model.name}")
        text, latency = model.transcribe(wav_path)
        print(f"Transcription: {text}")
        print(f"Latency: {latency:.2f}s")

if __name__ == "__main__":
    main()
