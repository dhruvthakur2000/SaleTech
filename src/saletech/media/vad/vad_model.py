import torch
import numpy as np
import webrtcvad
import time
from typing import Tuple, Optional

from src.saletech.utils.errors import SaleTechException, ValidationError
from src.saletech.utils.logger import get_logger, setup_logging
from config.settings import AppSettings

setup_logging()
logger = get_logger("saletech.vad.model")


class AdvancedVadModel:
    """
    Dual VAD:
    - silero (Neural)
    - WebRTC (Classical)
    - Adaptive thresholding
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.settings = AppSettings()
        self.sample_rate = self.settings.sample_rate
        self.vad_frame_duration_ms= self.settings.vad_frame_duration_ms
        self.silero_model: Optional[torch.nn.Module] = None
        self.webrtc_vad = webrtcvad.Vad(self.settings.vad_aggressiveness)

        self.initialized = False

        logger.info("vad_init", device=self.device)

    async def initialize(self):
        if self.initialized:
            return

        try:
            start = time.time()
            logger.info("vad_loading_models")

            self.silero_model, _ = torch.hub.load(
                repo_or_dir=self.settings.vad_repo,
                model=self.settings.vad_model_name,
                force_reload=False,
                onnx=False,
                trust_repo=True
            )

            self.silero_model.to(self.device).eval()
            frame_sample=max(
                512,int((self.sample_rate*self.vad_frame_duration_ms)/1000)
            )

            # GPU warmup
            dummy = torch.randn(1, frame_sample).to(self.device)
            with torch.no_grad():
                _ = self.silero_model(dummy, self.sample_rate)

            load_time = (time.time() - start) * 1000

            self.initialized = True

            logger.info("vad_models_loaded", load_time_ms=load_time)

        except Exception as e:
            logger.error("vad_load_failed", error=str(e), exc_info=True)
            raise SaleTechException(
                message="Failed to load VAD models",
                error_code="VAD_LOAD_FAILED",
                context={"class": "AdvancedVadModel"},
                original_exception=e
            )

    @torch.no_grad()
    def detect_speech(
        self,
        audio: np.ndarray,
        background_noise: float
    ) -> Tuple[bool, float, dict]:

        if not self.initialized:
            raise ValidationError(
                message="VAD model not initialized",
                context={"class": "AdvancedVadModel"}
            )

        if not isinstance(audio, np.ndarray):
            raise ValidationError(
                message="Audio must be numpy.ndarray",
                context={"received_type": str(type(audio))}
            )
        
        if audio.size == 0:
            raise ValidationError (
                message="Audio array empty",
                context={"audio array size":"0"}
              )
        
        if np.isnan(audio).any() or np.isinf(audio).any():
            raise ValidationError(
                message="Audio contains NaN/Inf"
                )

        if not isinstance(background_noise, (int, float)):
            raise ValidationError("Invalid background noise value")

        start_time = time.time()

        try:
            if audio.ndim > 1:
                audio = audio.flatten()
                
            start = time.time()

            audio= audio.astype(np.float32, copy=False)

            # Energy + SNR
            energy = float(np.sqrt(np.mean(audio ** 2)))
            snr = energy / (background_noise + 1e-6)

            # Model inference
            silero_prob = self._silero_detect(audio)
            webrtc_result = self._webrtc_detect(audio)

            adaptive_threshold = self._get_adaptive_threshold(background_noise)

            is_speech = (
                silero_prob >= adaptive_threshold or
                (silero_prob >= 0.3 and webrtc_result)
            ) and snr >= 2.0

            latency_ms = (time.time() - start_time) * 1000

            metadata = {
                "silero_prob": silero_prob,
                "webrtc_result": webrtc_result,
                "energy": energy,
                "snr": snr,
                "adaptive_threshold": adaptive_threshold,
                "latency_ms": latency_ms
            }

            return is_speech, silero_prob, metadata

        except SaleTechException:
            raise
        except Exception as e:
            logger.error("vad_detect_failed", error=str(e), exc_info=True)
            raise SaleTechException(
                message="Speech detection failed",
                error_code="VAD_DETECT_FAILED",
                context={"class": "AdvancedVadModel"},
                original_exception=e
            )

    def _silero_detect(self, audio: np.ndarray) -> float:
        try:
            if audio.shape[-1]<512:
                audio=np.pad(audio,(0,512 -audio.shape[-1]))

            audio_tensor = torch.from_numpy(audio).float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self.device)

            prob = self.silero_model(audio_tensor, self.sample_rate).item()
            return float(prob)

        except Exception as e:
            logger.error("silero_detect_failed", error=str(e), exc_info=True)
            raise SaleTechException(
                message="silero VAD detection failed",
                error_code="SILERO_DETECT_FAILED",
                context={"class": "AdvancedVadModel"},
                original_exception=e
            )

    def _webrtc_detect(self, audio: np.ndarray) -> bool:
        try:
            if audio.dtype != np.int16:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            frame_duration_ms = self.settings.vad_frame_duration_ms
            expected_length = int(self.sample_rate * frame_duration_ms / 1000)

            if len(audio_int16) != expected_length:
                if len(audio_int16) < expected_length:
                    audio_int16 = np.pad(
                        audio_int16,
                        (0, expected_length - len(audio_int16))
                    )
                else:
                    audio_int16 = audio_int16[:expected_length]

            return bool(
                self.webrtc_vad.is_speech(
                    audio_int16.tobytes(),
                    self.sample_rate
                )
            )

        except Exception as e:
            logger.error("webrtc_detect_failed", error=str(e), exc_info=True)
            raise SaleTechException(
                message="WebRTC VAD detection failed",
                error_code="WEBRTC_DETECT_FAILED",
                context={"class": "AdvancedVadModel"},
                original_exception=e
            )

    def _get_adaptive_threshold(self, background_noise: float) -> float:
        base = self.settings.speech_onset_threshold
        noise_factor = min(background_noise * 10, 0.3)
        threshold = base + noise_factor
        return min(threshold, 0.8)

    def cleanup(self):
        try:
            if self.silero_model is not None:
                del self.silero_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.initialized = False
            logger.info("vad_cleanup_complete")

        except Exception as e:
            logger.error("vad_cleanup_failed", error=str(e), exc_info=True)
            raise SaleTechException(
                message="VAD cleanup failed",
                error_code="VAD_CLEANUP_FAILED",
                context={"class": "AdvancedVadModel"},
                original_exception=e
            )