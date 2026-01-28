import time
from saletech.media.audio_buffer import VADAudioBuffer


class AudioState:
    """
    Per-session audio runtime state.
    """
    def __init__(self, buffer: VADAudioBuffer, speech_active: bool = False, last_speech_ts: float = 0.0):
        self.buffer = buffer
        self.speech_active = speech_active
        self.last_speech_ts = last_speech_ts

    def mark_speech_start(self):
        self.speech_active = True
        self.last_speech_ts = time.time()
        self.buffer.start_utterance()

    def mark_speech_end(self) -> bytes:
        self.speech_active = False
        self.last_speech_ts = time.time()
        return self.buffer.end_utterance()
