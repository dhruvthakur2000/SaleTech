import librosa
import soundfile as sf
import os

AUDIO_DIR = "experiments/stt_eval/data/audio"

for file in os.listdir(AUDIO_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_DIR, file)
    audio, _ = librosa.load(path, sr=16000, mono=True)
    sf.write(path, audio, 16000, subtype="PCM_16")

print("✅ Audio standardized to 16kHz mono")
