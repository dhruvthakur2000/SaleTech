import time
import whisper
from .base import STTModel

class WhisperSTT(STTModel):
    def __init__(self, model_size: str = "small"):
        self._model_size = model_size
        self.model = whisper.load_model(model_size)

    @property
    def name(self) -> str:
        return f"whisper-{self._model_size}"

    def transcribe(self, audio_path: str, **kwargs) -> str:
        start = time.time()
        result = self.model.transcribe(audio_path, language="hi")
        latency = time.time() - start
        return result["text"], latency
