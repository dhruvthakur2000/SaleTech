import time
import numpy as np
from collections import deque
from typing import Tuple, Optional
from src.saletech.utils.errors import SaleTechException, ValidationError
from config.settings import AppSettings
from src.saletech.utils.logger import get_logger

logger= get_logger("Saletech.vad.state")

class VADSessionState:
    """
    Per-session conversational VAD state.

    Handles:
    - speech start/end tracking
    - silence duration measurement
    - adaptive end-of-turn detection
    - background noise estimation
    """
    def __init__(self):
        self.settings= AppSettings()
        #speech state
        self.speech_active = False #in speech segment or not
        self.speech_start_time: Optional[float] = None #when current speech segment started
        self.last_speech_time: Optional[float]= None #Last frame with speech detected
        
        self.silence_start_time: Optional[float]= None 
        # When silence started after speech
        # USAGE: Detect end-of-turn (silence > threshold)
        
        #noise tracking
        self.energy_history = deque(maxlen=100) 
        # CIRCULAR BUFFER: Last 100 energy values (~2 seconds at 20ms/frame)
        # WHY DEQUE: O(1) append, automatic size limit
        # USAGE: Calculate running average for noise estimation

        #speaking rate adaption
        self.speaking_rate_history = deque(maxlen=10) 
        # STORES: Duration of last 10 utterances
        # USAGE: Calculate average speaking rate
        # FAST SPEAKER: Short utterances (500-1000ms)
        # SLOW SPEAKER: Long utterances (3000-5000ms)

    def update_energy(self, energy: float):
        """
        Update rolling energy window.

        Raises:
            ValidationError if energy invalid
        """
        if energy is None:
            raise ValidationError("Energy cannot be none")
        
        if not isinstance(energy,(int,float)):
            raise ValidationError(
                "Energy must be numeric",
                context={"received": type(energy).__name__}
            )
        if np.isnan(energy) or np.isinf(energy):
            raise ValidationError("Energy contains Nan or Inf")
        
        self.energy_history.append(float(energy))

    @property
    def background_noise(self)-> float:
        """
        Estimate background noise level
        """
        try:
            if len(self.energy_history)<10:
                return 0.01
        
            return float(np.mean(list(self.energy_history)[-20:]))
        
        except Exception as e:
            logger.error("background_noise_calc_failed", error=str(e))
            raise SaleTechException(
                "Background noise estimation failed",
                error_code="VAD_NOISE_ESTIMATE_FAILED",
                original_exception=e
            )

    #End of Turn detection

    def detect_end_of_turn(
            self, 
            is_speech: bool,
            current_time: float
    )-> Tuple[bool, dict]:
        """
        Detect if user has finished their turn.
        
        Args:
            is_speech: Current frame has speech
            current_time: Current timestamp
        
        Returns:
            is_eot:True if end of turn detected
            metadata: Debug Info
        """

        if current_time is None:
            raise ValidationError("TImeStamp cannot be none")
        
        if not isinstance(current_time, (int, float)):
            raise ValidationError(
                "Timestamp must be numeric",
                context={"recieved": type(current_time).__name__}
            )

        try:
            meta={}

            #state transition---speech detected 
            if is_speech:
                #current frame: speech detected
                if not self.speech_active:
                    #speech started 
                    self.speech_active= True
                    self.speech_start_time=current_time
                    self.silence_start_time=None
                    #resets silence tracking

                self.last_speech_time=current_time
                return False, meta
            
            #silence detected while speaking
            if self.speech_active:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time

                silence_ms=( current_time -self.silence_start_time) * 1000

                speech_ms=0
                if self.last_speech_time and self.speech_start_time:
                    speech_ms=(self.last_speech_time-self.speech_start_time)*1000

                eot_threshold= self._adaptive_eot_threshold(speech_ms)

                meta.update({
                    "silence_ms": silence_ms,
                    "speech_ms": speech_ms,
                    "eot_threshold_ms": eot_threshold
                })

                if silence_ms>=eot_threshold:
                # EOT DETECTED: Silence exceeded threshold
                    #reset state
                    self._reset_state()
                    self.speaking_rate_history.append(speech_ms)
                    return True, meta# RETURNS: EOT detected

            return False, meta# RETURNS: No EOT (still in utterance or idle)
        except SaleTechException:
            raise
        except Exception as e:
            logger.error("eot_detection_failed", error= str(e), exc_info=True)
            raise SaleTechException(
                "End=of-turn detection failed",
                error_code="VAD_EOT_FAILED",
                original_exception=e
            )
    
    def _adaptive_eot_threshold(self,speech_ms: float) -> float:
        """REVERSE ENGINEERING: Adaptive EOT Threshold
        
        INNOVATION: Adjust EOT based on speaking rate
        
        OBSERVATION:
        - Fast speakers: Short utterances, quick pauses
        - Slow speakers: Long utterances, longer pauses
        
        ADAPTATION:
        - Track average utterance duration
        - Short average → Reduce EOT threshold (respond faster)
        - Long average → Increase EOT threshold (don't cut off)
        
        EXAMPLE:
        User says quick "yes" responses (500ms avg):
        → EOT threshold = 600ms * 0.7 = 420ms
        
        User gives thoughtful responses (4000ms avg):
        → EOT threshold = 600ms * 1.2 = 720ms"""
        try:
            if not self.settings.eot_adaptive_enabled or len(self.speaking_rate_history)<3:
                # NOT ENOUGH DATA: Use default
                return self.settings.eot_silence_duration_ms
            
            # Calculate average speaking rate
            avg_rate = np.mean(list(self.speaking_rate_history))
            # AVERAGES: Last 10 utterance durations
            
            # Apply adaptive scaling
            if avg_rate < 1000:  # Very fast responses
                return self.settings.eot_silence_duration_ms * 0.7
                # REDUCE: 600ms → 420ms
                
            elif avg_rate > 3000:  # Longer, thoughtful responses
                return  self.settings.eot_silence_duration_ms * 1.2
                # INCREASE: 600ms → 720ms
                
            else:  # Normal speaking rate
                return self.settings.eot_silence_duration_ms
                # DEFAULT: 600ms
        
        except Exception as e:
            logger.error("adaptive_threshold_failed", error=str(e))
            raise SaleTechException(
                "Adaptive EOT threshold calculation failed",
                error_code="VAD_THRESHOLD_FAILED",
                original_exception=e
            )


    def _reset_state(self):
        """Reset VAD state (for new session or after barge-in)"""
        self.speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None            

    def reset(self):
        """
        pubglic reset for session restart/ barge in
        """
        try:
            self._reset_state()
            self.energy_history.cler()
            self.speaking_rate_history.clear()

        except Exception as e:
            logger.error("vad_state_reset_failed", error=str(e))
            raise SaleTechException(
                "VAD session reset failed",
                error_code="VAD_RESET_FAILED",
                original_exception=e
            )           