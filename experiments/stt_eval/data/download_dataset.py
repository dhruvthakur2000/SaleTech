from datasets import load_dataset
import soundfile as sf
import os

DATASET_NAME = "KAI-KratosAI/e-commerce-customersupport-hinglish-audio"
OUTPUT_DIR = "experiments/stt_eval/data"

os.makedirs(f"{OUTPUT_DIR}/audio", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/transcripts", exist_ok=True)

dataset = load_dataset(DATASET_NAME, split="train")

# Take only first 10 samples (FAST)
dataset = dataset.select(range(10))


for idx, item in enumerate(dataset):
    audio_path = f"{OUTPUT_DIR}/audio/sample_{idx}.wav"
    sf.write(audio_path, item["audio"]["array"], item["audio"]["sampling_rate"])

print("✅ Audio files saved. No transcript available in this dataset.")
