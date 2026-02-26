from typing import Optional
import time
import numpy as np
from saletech.media.vad.vad_state import VADSessionState
from saletech.media.vad.vad_model import AdvancedVadModel
from src.saletech.utils.errors import SaleTechException, ValidationError
from src.saletech.utils.logger import get_logger


logger = get_logger("saletech.vad.service")

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
        try:
            if VADService._vad_model is None:
                logger.info("initializing_global_vad_model")

                model = AdvancedVadModel()
                await model.initialize()

                VADService._vad_model= model

        except Exception as e:
            logger.error("Vad_service_init_failed", error= str(e))
            raise SaleTechException(
                message="VADervice initialization failed",
                error_code="VAD_SERVICE_INIT_FAILED",
                original_exception=e
            )

    def detect_speech(self, audio: np.ndarray):
        """
        Full VAD pipeline.

        Returns:
            is_speech
            end_of_turn
            confidence
            metadata
        """
        if VADService._vad_model is None:
            raise ValidationError("VAD model not initialized")
        
        if not isinstance(audio, np.ndarray):
            raise ValidationError("Audio must be numpy array")
        
        try:


            now=time.time()

            background_noise=self.state.background_noise

            #raw VAD inference
            is_speech, confidence, meta = (
                VADService._vad_model.detect_speech(audio,background_noise)
            )
            
            self.state.update_energy(meta["energy"])

            #end of turn logic
            is_eot, eot_meta = self.state.detect_end_of_turn(is_speech, now)
            meta.update(eot_meta)

            return is_speech,confidence,is_eot,meta
        
        except SaleTechException:
            raise

        except Exception as e:
            logger.error("vad_service_detect_failed", error=str(e))
            raise SaleTechException(
                message="VADService detection failure",
                error_code="VAD_SERVICE_DETECT_FAILED",
                original_exception=e
            )
            

