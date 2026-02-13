import time
import numpy as np
from collections import deque
from typing import Tuple

from config.settings import AppSettings

class VADSessionState:
    def __init__(self):
        self.speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        self.settings= AppSettings()

        self.energy_history = deque(maxlen=100)
        self.speaking_rate_history = deque(maxlen=10)

    def update_energy(self,energy:float):
        self.energy_history.append(energy)

    @property
    def background_noise(self)-> float:
        if len(self.energy_history)<10:
            return 0.01
        return float(np.mean(list(self.energy_history)[-20:]))


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
            (is_eot, metadata)
        """

        meta={}
        #state transition
        if is_speech:
            #current frame: speech detected
            if not self.speech_active:
                #speech started 
                self.speech_active= True
                self.speech_start_time=current_time
                self.silence_start_time=None

            self.last_speech_time=current_time
            return False, meta
        

        if self.speech_active:
            if self.silence_start_time is None:
                self.silence_start_time = current_time

            silence_ms=( current_time -self.silence_start_time) * 1000
            speech_ms=(self.last_speech_time-self.speech_start_time)*1000

            eot_threshold= self._adaptive_eot_threshold(speech_ms)

            meta.update({
                "silence_ms": silence_ms,
                "speech_ms": speech_ms,
                "eot_threshold_ms": eot_threshold
            })

            if silence_ms>=eot_threshold:
                self._reset_state()
                self.speaking_rate_history.append(speech_ms)
                return True, meta# RETURNS: EOT detected

            return False, meta# RETURNS: No EOT (still in utterance or idle)
    
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

        if not self.settings.eot_adaptive_enabled or len(self.speaking_rate_history)<3:
            # NOT ENOUGH DATA: Use default
            return self.settings.eot_silence_duration_ms
        
        # Calculate average speaking rate
        avg_rate = np.mean(list(self._speaking_rate_history))
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
        



    def _reset_state(self):
        """Reset VAD state (for new session or after barge-in)"""
        self._speech_active = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None            
