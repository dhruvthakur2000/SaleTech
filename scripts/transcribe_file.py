import argparse
import asyncio
import os
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import AppSettings
from saletech.transcriber.pipeline import TranscriptionPipeline


def read_pcm_wav(path: Path, target_sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sample_rate:
        duration = len(audio) / sample_rate
        source_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        target_len = int(duration * target_sample_rate)
        target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
        audio = np.interp(target_times, source_times, audio).astype(np.float32)

    return audio.astype(np.float32, copy=False)


async def transcribe_file(path: Path, session_id: str) -> Path:
    settings = AppSettings()
    audio = read_pcm_wav(path, settings.sample_rate)

    pipeline = TranscriptionPipeline(session_id=session_id)
    await pipeline.initialize()

    frame_size = settings.vad_frame_size_samples
    frame_duration = frame_size / settings.sample_rate
    start_time = time.time()

    try:
        for start in range(0, len(audio), frame_size):
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))

            await pipeline.process_frame(frame)

            # Keep timestamps realistic for VAD end-of-turn logic without
            # forcing the test to run in wall-clock real time.
            target_elapsed = ((start // frame_size) + 1) * frame_duration
            drift = target_elapsed - (time.time() - start_time)
            if drift > 0:
                await asyncio.sleep(min(drift, 0.005))

        silence_samples = int(settings.sample_rate * (settings.eot_silence_duration_ms + 200) / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)
        for start in range(0, len(silence), frame_size):
            frame = silence[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            await pipeline.process_frame(frame)

        await pipeline.flush()
        return Path(pipeline.writer.file_path)
    finally:
        await pipeline.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local VAD -> buffer -> ASR transcription pipeline on a WAV file.")
    parser.add_argument("audio_file", type=Path, help="Path to a PCM WAV file.")
    parser.add_argument("--session-id", default=f"local_{uuid.uuid4().hex[:8]}")
    parser.add_argument("--model", help="Faster Whisper model name/path, for example tiny, base, small, or large-v3.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device for Faster Whisper.")
    parser.add_argument("--compute-type", help="Faster Whisper compute type, for example int8, float16, or float32.")
    args = parser.parse_args()

    if args.model:
        os.environ["SALETECH_WHISPER_MODEL_PATH"] = args.model
    if args.device:
        os.environ["SALETECH_WHISPER_DEVICE"] = args.device
    if args.compute_type:
        os.environ["SALETECH_WHISPER_COMPUTE_TYPE"] = args.compute_type

    output = asyncio.run(transcribe_file(args.audio_file, args.session_id))
    print(f"Transcript written to: {output}")


if __name__ == "__main__":
    main()
