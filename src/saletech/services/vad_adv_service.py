from typing import Optional
import time
import numpy as np
from src.saletech.core.vad_state import VADSessionState
from src.saletech.media.vad_model import AdvancedVadModel


class VADService:
    """
    Production VAD interface.

    - Shares one model globally
    - Keeps state per session
    - Provides stable API to audio buffers
    """

    _vad_model: Optional[AdvancedVadModel] = None

    def __init__(self):
        self.state =VADSessionState()

    async def initialize(self):
        if VADService._vad_model is None:
            model = AdvancedVadModel()
            await model.initialize()
            VADService._vad_model= model

    def detect_speech(self,audio: np.ndarray):
        """
        Full VAD pipeline.

        Returns:
            is_speech
            end_of_turn
            confidence
            metadata
        """
        now=time.time()

        #energy estimation
        energy= float(np.sqrt(np.mean(audio**2)))
        self.state.update_energy(energy)

        background_noise=self.state.background_noise

        #raw VAD inference
        is_speech, confidence, meta = (
            VADService._vad_model.detect_speech(audio,background_noise)
        )

        #end of turn logic
        eot, eot_meta = self.state.detect_end_of_turn(is_speech, now)
        meta.update(eot_meta)

        return is_speech,eot,confidence,meta
