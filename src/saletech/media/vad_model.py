import torch
import numpy as np
import webrtcvad
from typing import Tuple , Optional
import time

from src.saletech.utils.logger import get_logger, setup_logging
from config.settings import AppSettings

setup_logging()
logger = get_logger("saletech.vad.model")

class AdvancedVadModel:
    """
    REVERSE ENGINEERING: Dual VAD with Adaptive Thresholding
    
    COMPONENTS:
    1. Silero VAD v5 (neural, GPU-accelerated)
    2. WebRTC VAD (traditional, CPU)
    3. Energy-based filtering
    4. Adaptive threshold calculation
    5. End-of-turn detection state machine
    
    WHY NOT SINGLE VAD:
    - Silero alone: Might miss rapid speech starts
    - WebRTC alone: Too many false positives on noise
    - Combined: Robust to both noise and rapid changes
    """
    def __init__(self):
        # GPU: 8ms inference, CPU: 20ms inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.settings = AppSettings()

        self.sample_rate = self.settings.sample_rate #16000Hz

        self.silero_model: Optional[torch.nn.Module]= None
        self.webrtc_vad = webrtcvad.Vad(self.settings.vad_aggressiveness)

        self.initialized = False

        logger.info("VAD_init", device=self.device)

    async def initialize(self):
        """Load VAD Model once.
        
        FLOW:
        1. Load Silero VAD from torch.hub
        2. Move to GPU and set to eval mode
        3. Initialize WebRTC VAD with aggressiveness #already done
        4. Warmup with dummy audio
        
        WHY ASYNC:
        - Model loading can take 1-2 seconds
        - Doesn't block event loop during startup"""

        if self.initialized:
            return
        
        try:
            logger.info("Loading VAD models")
            start= time.time()

            # load selero VAD V5

            self.silero_model, _ = torch.hub.load(
                repo_or_dir=self.settings.vad_repo, #"snakers/silero-vad"
                model=self.settings.vad_model_name, #silero-vad
                force_reload=False, # use cached if available
                onnx=False, # use pytorch (faster on gpu)
                trust_repo=True
            )

            self.silero_model.to(self.device).eval() # model weights to gpu memory
                                                     # and set model to evaluation mode 
            
            load_time = (time.time() - start) * 1000

            # GPU warmup 
            dummy=torch.randn(1,self.sample_rate).to(self.device) #creates 1 sec of random audio
            
            with torch.no_grad(): #disable gradient computation
                _=self.silero_model(dummy,self.sample_rate) #one inference to warmup GPU

            self.initialized=True
             
            logger.info("vad_models_loaded",load_time_ms=load_time)

        except Exception as e:
            logger.error("VAD_load_failed", error=str(e), exc_info=True)
            raise

    @torch.no_grad()
    def detect_speech(self,
            audio: np.ndarray,
            background_noise: float
        )->tuple[bool, float, dict]:    
        """
            Detect speech in audio frame.
            Args:
            audio: Audio samples (float32, [-1, 1])
                20ms chunks @ 16KHz
            Returns:
                is_speech: bool - Final decision,
                confidence: float - silero prob,
                metadata: dict - debug info
            Latency: 8-12ms
            """
        if not self.initialized:
            raise RuntimeError("VAD model not initialized")


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


    def _silero_detect(self, audio: np.ndarray) -> float:
        """Run silero neural VAD"""
        try:
            if isinstance(audio, np.ndarray):
                audio_tensor=torch.from_numpy(audio).float()
            else:
                audio_tensor=audio.float()

            if audio_tensor.dim() == 1:
                audio_tensor=audio_tensor.unsqueeze(0)

            audio_tensor= audio_tensor.to(self.device)
            prob = self.silero_model(audio_tensor, self.sample_rate).item()
            return prob
        except Exception as e:
            logger.error("silero_detect_failed", error=str(e))
            return 0.5 # Neutral default (uncertain)
        

    def _webrtc_detect(self, audio: np.ndarray) -> bool:
        """
        REVERSE ENGINEERING: WebRTC VAD Inference
        
        ALGORITHM (Simplified):
        1. Convert float32 → int16 PCM
        2. Calculate energy in frequency bands
        3. Compute zero-crossing rate
        4. Apply decision tree based on aggressiveness
        
        WHY INT16:
        - WebRTC is C library from phone systems
        - Phone systems use 16-bit PCM
        - Library doesn't accept float32
        
        FRAME SIZE REQUIREMENT:
        - Must be 10ms, 20ms, or 30ms at 8kHz, 16kHz, or 32kHz
        - We use 30ms at 16kHz = 480 samples
        """
        try:
            if audio.dtype != np.int16:
                audio_int16 = (audio * 32767).astype(np.int16)
                #multiply by max int16 value
            else:
                audio_int16=audio

            frame_duration_ms = self.settings.vad_frame_duration_ms #30ms

            expected_length= int(self.sample_rate*frame_duration_ms/1000)
            # CALCULATES: 16000 * 0.030 = 480 samples

            if len(audio_int16) != expected_length:
                if len(audio_int16) < expected_length:
                    #PAD: add zeroes to end
                    audio_int16=np.pad(audio_int16,
                                       (0,expected_length-len(audio_int16))
                    )
                else:
                    #take only first n samples
                    audio_int16 = audio_int16[:expected_length]
                    
            return self.webrtc_vad.is_speech(
                audio_int16.tobytes(), # convert array to bytes
                self.sample_rate
            )
            
        except Exception as e:
            logger.error("webrtc_detect_failed", error=str(e))
            return False
         

    def _get_adaptive_threshold(self, background_noise: float) -> float:
        base = self.settings.speech_onset_threshold
        noise_factor = min(background_noise * 10, 0.3)
        threshold=base+noise_factor
        return min(threshold, 0.8)

    def cleanup(self):
        """Release GPU memory if needed."""
        if self.silero_model is not None:
            del self.silero_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialized = False
        logger.info("vad_model_cleaned")