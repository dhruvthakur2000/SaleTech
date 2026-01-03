
from pathlib import Path

AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("transcripts")

def load_dataset():
    audio_files = list(AUDIO_DIR.rglob("*.wav"))
    return audio_files

def main():
    audio_files = load_dataset()
    print(f"Found {len(audio_files)} audio samples")

if __name__ == "__main__":
    main()
