from collections import deque
import numpy as np 
from typing import Optional
import time
from config.settings import AppSettings
from src.saletech.utils.logger import get_logger
from src.saletech.utils.errors import AudioProcessingError

logger = get_logger("Saletech.audio.speech_window")

class StreamingBuffer:
    """
    Conversational speech segmentation buffer.

    Responsibilities:
    - Accumulate frames classified as speech
    - Detect utterance completion using VAD EOT
    - Provide finalized audio chunks for ASR
    - Mantain padding context around speech boundaries
    """

    def __init__(self):
        self.settings= AppSettings()
        #configuration
        self.sample_rate=self.settings.sample_rate
        self.min_speech_samples = self.settings.min_speech_samples
        self.max_speech_samples = self.settings.max_speech_samples
        self.speech_pad_samples = int(self.settings.sample_rate * self.settings.speech_pad_ms / 1000)

        #state
        self._frame_buffer:deque = deque(maxlen=100) #recent frames for padding
        self._current_utterance: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_speech = False


        # Timestamps
        self._utterance_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None

        logger.info("Streaming_buffer_initialized")
    

    def add_frame(
            self, 
            audio: np.ndarray,
            is_speech:bool,
            is_eot: bool,
            timestamp:float
            ) -> Optional[tuple[np.ndarray, dict]]:
        """
        Add audio frame with VAD and EOT results.
        
        Args:
            audio: Audio frame
            is_speech: VAD speech detection
            is_eot: End-of-turn detection
            timestamp: Frame timestamp
        
        Returns:
            (utterance_audio, metadata) if complete utterance, else None
            """
        if audio is None or len(audio)==0:
            return None
        
        if not isinstance(audio,np.ndarray):
            raise AudioProcessingError("Invalid audio frames")
        
        if audio.ndim>1:
            audio=audio.reshape(-1)
        #1. add to circular frame buffer
        self._frame_buffer.append(audio)

        #2.handle speech frame
        if is_speech:
            self._speech_samples += len(audio)
            self._silence_samples=0

            if not self._in_speech:
                self._in_speech= True
                self._utterance_start_time = timestamp

                #include padding frames
                pad_frames= min(
                    len(self._frame_buffer),
                    self.speech_pad_samples//len(audio)
                                )
                # get last N frames from buffer for padding

                self.current_utterance = list (self._frame_buffer)[-pad_frames:]

            else:
                self._current_utterance.append(audio)

            self._last_speech_time = timestamp

            #3.handle silence frame

        else:
            self._silence_samples+=len(audio)
            if self._in_speech:
                self._current_utterance.append(audio)

        #eot finalize

        if is_eot and self._in_speech:
            if self._speech_samples >= self.min_speech_samples:
                utterance = self._finalize_utterance(timestamp)
                return utterance
            else:
                self._reset

        # check max length (safety valve)
        if self._speech_samples >= self.max_speech_samples:
            logger.warning("streaming_firce_finalize")
            utterance=self._finalize_utterance(timestamp)
            return utterance
        return None 
        # - Most frames return None (not complete yet)
        # - Only return utterance when:
        #   1. EOT detected + min length met
        #   2. Max length exceeded
        # - Caller continues calling add_frame() for each frame

    def _finalize_utterance(
            self,
            end_time: float
            )->tuple[np.ndarray,dict]:
        """
        Finalize and return complete utterance.
        
        WHAT HAPPENS:
        1. Concatenate all frames into single numpy array
        2. Create metadata dictionary
        3. Reset state for next utterance
        4. Return (audio, metadata)  
              
        Args:
            end_time: Timestamp when utterance ended
        
        Returns:
            (utterance_audio, metadata)
        """
        if not self._current_utterance:
            self._reset()
            return None
        
        try:
            utterance_audio= np.concatenate(self._current_utterance)

            meta={
                "start_ts": self._utterance_start_time,
                "end_ts": end_time,
                "duration_ms": end_time-self._utterance_start_time* 1000,
                "audio_length_ms": len(utterance_audio)/self.sample_rate *1000,
                "speech_sample": self._speech_samples               
            }

            logger.info(
                "utterance_finalized",
                duration_ms=meta["duration_ms"]
            )

            self.reset()

            return utterance_audio, meta
        
        except Exception as e:
            self._reset()
            raise AudioProcessingError(
                "Utterance finalize failed",
                original_exception = e
            )


    def _reset(self):
        """Clear all state variables to prepare for next utterance."""
        self._current_utterance = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_speech = False
        self._utterance_start_time = None
        self._last_speech_time = None


    def force_finalize(self, ts):

        if self._in_speech and self._speech_samples >= self.min_speech_samples:
            return self._finalize_utterance(timestamp)

        return None

    @property
    def metrics(self):

        return {
            "in_speech": self._in_speech,
            "speech_samples": self._speech_samples,
            "silence_samples": self._silence_samples,
            "frames_buffered": len(self._frames),
        }