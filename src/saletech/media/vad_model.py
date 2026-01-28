import torch
import numpy as np
import webrtcvad
from typing import Tuple , Optional
import time

from src.saletech.utils.logger import get_logger, setup_logging
from config.settings import AppSettings

setup_logging()
logger = get_logger("saletech.vad.model")

class AdvancedVadMOdel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.settings = AppSettings()
        self.sample_rate = self.settings.sample_rate

        self.silero_model: Optional[torch.nn.Module]= None
        self.webrtc_vad= webrtcvad.Vad(self.settings.vad_aggressiveness)

        self.initialized=False

    async def initialize(self):
        """Load VAD Model"""
        if self.initialized:
            return
        
        try:
            logger.info("Loading VAD models")
            start= time.time()

            # load selero VAD V5

            self.silero_model, _ = torch.hub.load(
                repo_or_dir=self.settings.vad_repo,
                model=self.settings.vad_model_name,
                force_reload=False,
                trust_repo=True
            )

            self.silero_model.to(self.device).eval()
            load_time = (time.time() - start) * 1000

            #warmup
            dummy=torch.randn(1,self.sample_rate).to(self.device)
            with torch.no_grad():
                _=self.silero_model(dummy,self.sample_rate)

            self.initialized=True
            logger.info("vad_models_loaded",load_time_ms=load_time
            )

        except Exception as e:
            logger.error("VAD_load_failed", error=str(e), exc_info=True)

@torch.no_grad()
def detect_speech(self,
        audio:np.ndarray,
        background_noise:float
    )->tuple[bool, float, dict]:
        
    """
        Detect speech in audio frame.
        
        Args:
            audio: Audio samples (float32, [-1, 1])
        
        Returns:
            (is_speech, confidence, metadata)
        """
    
    start_time= time.time()

    # calculating audio energy

    energy=float(np.sqrt(np.mean(audio**2)))
    snr= energy/ (background_noise+1e-6)

    # Dual VAD: Silero + WebRTC
    webrtc_result = self._webrtc_detect(audio)
    silero_prob = self._silero_detect(audio)

    adaptive_threshold = self._get_adaptive_threshold()


    is_speech = (
            silero_prob >= adaptive_threshold or
            (silero_prob >= 0.3 and webrtc_result)
        ) and snr >= 2.0
    
    latency_ms = (time.time() - start_time) * 1000

    metadata={
            "silero_prob": silero_prob,
            "webrtc_result": webrtc_result,
            "energy": energy,
            "snr": snr,
            "adaptive_threshold": adaptive_threshold,
            "latency_ms": latency_ms
    }

    return is_speech, silero_prob, metadata
