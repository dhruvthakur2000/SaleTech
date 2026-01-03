from collections import deque
from typing import Deque, List
import threading


class VADAudioBuffer:
    """
    Ultra-low-latency audio buffer with pre-roll support.
    """

    def __init__(
        self,
        frame_bytes: int,
        pre_roll_frames: int = 10,
        max_utterance_frames: int = 2000,
    ) -> None:
        self.frame_bytes = frame_bytes

        self._pre_roll: Deque[bytes] = deque(maxlen=pre_roll_frames)
        self._active: Deque[bytes] = deque(maxlen=max_utterance_frames)

        self._speech_active: bool = False
        self._partial = bytearray()
        self._lock = threading.Lock()

    def push_samples(self, data: bytes) -> None:
        if not data:
            return

        with self._lock:
            self._partial.extend(data)

            while len(self._partial) >= self.frame_bytes:
                frame = bytes(self._partial[: self.frame_bytes])
                del self._partial[: self.frame_bytes]

                self._pre_roll.append(frame)

                if self._speech_active:
                    self._active.append(frame)

    def start_utterance(self) -> None:
        with self._lock:
            if self._speech_active:
                return

            self._active.clear()
            for frame in self._pre_roll:
                self._active.append(frame)

            self._speech_active = True

    def end_utterance(self) -> bytes:
        with self._lock:
            self._speech_active = False
            utterance = b"".join(self._active)
            self._active.clear()
            return utterance

    def cancel(self) -> None:
        with self._lock:
            self._speech_active = False
            self._active.clear()

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active
