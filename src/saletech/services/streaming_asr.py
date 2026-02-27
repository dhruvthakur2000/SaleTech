import numpy as np
import asyncio
import time
from typing import Optional 
from concurrent.futures import ThreadPoolExecutor
from ..utils.errors import AudioProcessingError
from faster_whisper import WhisperModel
from ..utils.logger import get_logger
from config.settings import AppSettings
from ..models.schemas import TranscriptionResult


logger= get_logger("saletech.asr")

_asr_instance: Optional["StreamingASR"] = None
_asr_init_lock = asyncio.Lock()


class StreamingASR:
    """
    Production-grade ASR service for SaleTech.

    Design:
    - Singleton model
    - GPU concurrency control
    - Strict exception discipline
    - No internal unbounded accumulation
    """
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.settings=AppSettings()
        self.device = self.settings.whisper_device
        self.compute_type = self.settings.whisper_compute_type
        self.model_path = self.settings.whisper_model_path

        self._initialized= False #Flag indicating model is loaded and ready 1 if initialize() complete

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=self.settings.asr_workers)
        self._semaphore = asyncio.Semaphore(self.settings.asr_max_concurrent_jobs)

        logger.info(
            "streaming asr initialized",
            device= self.device,
            model= self.model_path
        )

    async def initialize(self):
        """
        Load whisper midel
        """
        if self._initialized:
            return
        
        try:
            logger.info("loading_whisper_model", model=self.model_path)
            
            start_time = time.time()

            loop = asyncio.get_running_loop()

            # Load model in executor (non-blocking)
            self.model = await loop.run_in_executor(
                self._executor,
                self._load_model_blocking
            )

            #warmup
            dummy= np.zeros(self.settings.sample_rate,dtype=np.float32)

            await loop.run_in_executor(
                self._executor,
                lambda: self.model.transcribe(dummy)
            )

            self._initialized= True

            logger.info(
                "asr_model_loaded",
                load_time_ms= (time.time()-start_time)*1000
            )     

        except Exception as e :
            logger.error("asr_model loading failed",exc_info=True)
            raise AudioProcessingError(
                "ASR model initialization failed",
                original_exception=e
            )
        

    
    def _load_model_blocking(self) -> WhisperModel:
        return WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1
        )

#---------------------------------------------------------------------------------
    async def transcribe(
            self,
            audio:np.ndarray,
            language: Optional[str] =  None,
            session_id: Optional[str] = None
    )-> TranscriptionResult:
        
        if not self._initialized:
            raise AudioProcessingError("ASR service not initialized")
        
        if audio is None or len(audio)==0:
            return TranscriptionResult(text="",confidence=0.0,language=language)
        
        async with self._semaphore:
            start= time.time()

            try:
                #ensure float 32
                if audio.dtype != np.float32:
                    audio= audio.astype(np.float32)
                
                #normalize safely
                # WHY NECESSARY?
                # - Some audio sources produce out-of-range values
                # - Whisper expects [-1, 1]
                # - Normalization prevents model errors

                max_val= np.max(np.abs(audio))
                if max_val> 1.0:
                    audio = audio/max_val

                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    beam_size=self.settings.asr_beam_size,
                    best_of = self.settings.asr_best_of,
                    temperature=0.0,
                    vad_filter=False,
                    task= "transcribe",
                    word_timestamps=False
                  )
                
                segments= list(segments)

                text=" ".join(
                    s.text.strip() for s in segments
                ).strip()

                confidence = self._compute_confidence(segments)

                latency_ms= (time.time()- start) * 1000

                logger.info(
                    "asr_transcription_complete",
                    session_id=session_id,
                    latency_ms=latency_ms,
                    audio_duration_ms=len(audio) /
                    self.settings.sample_rate * 1000
                )

                return TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language=language
                )

            except Exception as e:
                logger.error("asr_transcription_failed",
                            session_id=session_id,
                            exc_info=True
                            )
                
                raise AudioProcessingError(
                    message="ASR transcription failed",
                    original_exception=e
                )


    def _compute_confidence(self, segments):

        if not segments:
            return 0.0

        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)

        # Safer mapping
        confidence = 1.0 / (1.0 + np.exp(-avg_logprob))

        return float(max(0.0, min(1.0, confidence)))


    async def cleanup(self):

        logger.info("asr_cleanup_started")

        if self.model:
            # Explicitly clear GPU memory
            try:
                del self.model
            except Exception:
                pass

        self._executor.shutdown(wait=False)

        self.model = None
        self._initialized = False

        logger.info("asr_service_cleaned_up")


        
async def get_asr_service() -> StreamingASR:

    global _asr_instance

    async with _asr_init_lock:
        if _asr_instance is None:
            instance = StreamingASR()
            await _asr_instance.initialize()
            _asr_instance = instance

    return _asr_instance