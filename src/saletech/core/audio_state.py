from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Optional

from saletech.media.audio_buffer import VADAudioBuffer


@dataclass
class AudioRuntimeState:
    buffer: VADAudioBuffer
    speech_active: bool = False
    vad_confidence: float = 0.0

    last_speech_start: Optional[datetime] = None
    last_speech_end: Optional[datetime] = None

    # Used for hangover / EoT logic
    silence_frames: int = 0

    # Lock protects VAD + buffer together
    lock: Lock = Lock()
