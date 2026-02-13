from collections import deque
from typing import Optional
import time
from config.settings import AppSettings
from src.saletech.services import vad_adv


class SpeechWindowBuffer:
    """
        Builds utterances from raw audio frames using VAD + silence.

    """

    def __init__(
            self
    ):
        self.vad= vad_adv.get_vad_model
        self.settings= AppSettings
        self.frame_samples=