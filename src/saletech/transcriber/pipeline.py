import time
import numpy as np
from typing import Optional

from src.saletech.services.vad_adv_service import VADService
from src.saletech.services.streaming_asr import get_asr_service
from saletech.media.buffer.StreamingVadBuffer import StreamingBuffer
from saletech.transcriber.writer import TranscriptWriter
from saletech.utils.logger import get_logger
from saletech.utils.errors import AudioProcessingError

logger=get_logger("saletech.transcriber.pipeline")

class TranscriptionPipeline:
    """
    Orchestrates full VAD → Buffer → ASR → JSON writer pipeline.

    Per WebSocket session instance.

    Responsibilities:
    - Initialize VAD + ASR
    - Process incoming PCM frames
    - Detect utterance boundaries
    - Run ASR on finalized utterances
    - Write structured metadata to JSON file
    - Clean shutdown
    """
    def __init__(self,session_id:str):
        self.session_id = session_id

        #per session VAD state
        self.vad_service: Optional[VADService] = None

        #speech segmentation buffer
        self.speech_buffer = StreamingBuffer()
        
        #singleton ASR 
        self.asr_service = None

        #JSON writter for this session
        self.writer = TranscriptWriter(session_id=session_id)

        #track VAD speech window timing
        self._current_vad_start: Optional[float] = None

        self.initialized = False

        #------------------------------------------------------------------

    async def initialize(self):
        """
        Initialize VAD + ASR services

        Must be called before processing frames

        """

        try:
            self.vad_service = VADService()
            await self.vad_service.initialize()

            self.asr_service = await get_asr_service()

            self._initialized = True

            logger.info(
                "transcription_pipeline_initialized",
                session_id = self.session_id
            )        

        except Exception as e:
            logger.error(
                "transcription_pipeline_init_failed",
                session_id=self.session_id,
                exc_info = True
            )
            raise AudioProcessingError(
                message="Pipeline initialization failed",
                original_exception = e 
            )
        
#----------------------------------------------------------------------------
    async def process_frame(self,pcm:np.ndarray):
        """
        Main entry point for each incoming audio frame.

        Args: 
            pcm: float32 numpy array[-1,1]

        """
        
        if not self._initialized:
            raise AudioProcessingError(
                "Pipeline not initialized"
            )
        
        try:
            now = time.time()

            # step 1: VAD detection
            is_speech,confidence,is_eot,meta = self.vad_service.detect_speech(pcm)

            #track start of speech
            if is_speech and self._current_vad_start is None:
                self._current_vad_start = now

            #step 2: push into speech window buffer
            result = self.speech_buffer.add_frame(
                pcm=pcm,
                is_speech=is_speech,
                is_eot=is_eot,
                timestamp=now
            )

            #step 3: if utterance finalized -> run ASR
            if result is not None:
                audio, buffer_meta = result

                vad_start = self._current_vad_start
                vad_end = buffer_meta["end_ts"]
                duration_ms = buffer_meta["duration_ms"]

                #reset start marker
                self._current_vad_start = None

                await self._handle_finalized_utterance(
                    audio=audio,
                    vad_start=vad_start,
                    vad_end=vad_end,
                    duration_ms=duration_ms
                )

        except Exception as e:
            logger.error(
                "transcription_frame_processing_failed",
                session_id = self.session_id,
                exc_info = True
            ) 

            raise AudioProcessingError(
                "Frame processing failed",
                original_exception = e
            )
        
    #--------------------------------------------------------------------------------


    async def _handle_finalized_utterance(
        self,
        audio: np.ndarray,
        vad_start:float,
        vad_end: float,
        duration_ms: float
    ):
        """
        called when speech buffer finalizes an utterance.

        Runs ASR and writes metadata
        """
        try:
            asr_start= time.time()

            result = await self.asr_service.transcribe(
                audio=audio,
                session_id = self.session_id
            )

            asr_latency_ms = (time.time()-asr_start)*1000

            payload= {
                "session_id": self.session_id,
                "vad_start_ts": vad_start,
                "vad_end_ts": vad_end,
                "speech_duration_ms": duration_ms,
                "asr_latency_ms": asr_latency_ms,
                "text": result.text,
                "confidence": result.confidence,
                "timestamp": time.time()
            } 
            self.writer.write(payload)

            logger.info(
                "utterance_transcribed",
                session_id = self.session_id,
                duration_ms = duration_ms,
                asr_latency_ms = asr_latency_ms
            )
        except Exception as e:
            logger.error(
                "utterance_processing_failed",
                session_id=self.session_id,
                exc_info=True
            )
            raise AudioProcessingError(
                "Utterance transciption failed",
                original_exception=e
            )
    #-------------------------------------------------------------------------

    async def shutdown(self):
        """
        clean shutdown of pipeline
        """

        try:
            self.writter.close()

            logger.info("transciption pipeline is shut down",
                        session_id=self.session_id
            )
        except Exception as e:
            logger.error(
                "pipeline shutdown failed",
                session_id=self.session_id,
                exc_info= True
            )



