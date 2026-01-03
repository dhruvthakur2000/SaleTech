
import os
from experiments.stt_eval.models.whisper import WhisperSTT
# from experiments.stt_eval.models.vakyansh import VakyanshSTT
# from experiments.stt_eval.models.nemo import NemoSTT
from experiments.stt_eval.eval.schema import STTEvalResult, STTEvalBatchResult
import json
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"

MODELS = [
    WhisperSTT("small"),
    # VakyanshSTT("PATH_TO_NEMO_MODEL"),
    # NemoSTT("PATH_TO_NEMO_MODEL"),
]

def run_batch_eval():
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    results = []
    for model in MODELS:
        for audio_path in audio_files:
            pred, latency = model.transcribe(str(audio_path))
            result = STTEvalResult(
                model_name=model.name,
                audio_file=str(audio_path),
                reference_text="",  # No reference, just transcription
                predicted_text=pred,
                wer=0.0,
                cer=0.0,
                latency_ms=latency * 1000,
                gpu_memory_mb=None,
                cpu_used=None,
                notes=None
            )
            results.append(result)
    batch = STTEvalBatchResult(results=results)
    with open("stt_eval_results.json", "w", encoding="utf-8") as f:
        f.write(batch.model_dump_json(indent=2))
    print("Evaluation complete. Results saved to stt_eval_results.json")

if __name__ == "__main__":
    run_batch_eval()
