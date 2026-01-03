import time
import nemo.collections.asr as nemo_asr
from .base import STTModel

class VakyanshSTT(STTModel):
    def __init__(self, model_path: str):
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)

    @property
    def name(self) -> str:
        return "vakyansh"

    def transcribe(self, audio_path: str, **kwargs) -> str:
        start = time.time()
        result = self.model.transcribe([audio_path])[0]
        latency = time.time() - start
        return str(result).strip(), latency
