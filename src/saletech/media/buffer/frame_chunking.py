import asyncio
import threading
import time
from typing import Optional, Tuple
from src.saletech.utils.logger import get_logger
from src.saletech.utils.errors import AudioProcessingError

logger=get_logger("saletech.audio.frame_buffer")

class AudioIngressBuffer:
    """
    Thread-safe raw audio ingestion buffer.

    Responsibility:
    - Accept PCM frames from WebSocket callback threads
    - Buffer them safely
    - Deliver to async consumers
    """
    SENTINEL = (b"", -1.0)

    def __init__(self, max_size: int = 1000):
        self._loop = asyncio.get_running_loop()

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._closed = False  # Indicates buffer is closed: False->accepting audio, True->reject audio (shutdown)
        self._lock = threading.Lock()
        # Performance metrics
        self._frames_received = 0
        self._frames_dropped = 0
        self._last_frame_time = 0.0
        self._frame_attempted = 0

        self._shutdown_event = asyncio.Event()

        logger.info("audio_buffer_initialized",maxsize=max_size)
        
        #INGESTION THREAD SAFE 
    def put_nowait(self, pcm: bytes, timestamp: Optional[float] = None) -> None:
        """
        Put audio chunk (called from callback thread).
        Args:
            pcm: Audio data bytes
            timestamp: Optional timestamp
        """
        if not isinstance(pcm, (bytes, bytearray)):
            logger.error("invalid_audio_frame_type")
            raise AudioProcessingError(
                "Audio frame must be bytes-like."
            )
        
        if not pcm:
            logger.warning("empty_audio_frame_ignored")
            return
        
        if timestamp is None:
                timestamp = time.time()

        with self._lock:
            if self._closed:
                logger.warning("Buffer closed, rejecting frame.")
                return
            
            self._frame_attempted += 1
            
        self._loop.call_soon_threadsafe(
            self._enqueue_frame,
            pcm,
            timestamp
        )

        def _enqueue_frame(self,pm:bytes, timestamp:float):

            try:
                self._queue.put_nowait((pcm, timestamp))
                self._frames_received += 1
                self._last_frame_time = timestamp

            except asyncio.QueueFull:

                # Drop oldest frame if full
                try:
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait((pcm, timestamp))
                    self._frames_dropped += 1
                    self._last_frame_time = timestamp

                    logger.warning("Queue full, dropped oldest frame.",
                                   queue_size=self._queue.qsize(),
                                   )

                except Exception :
                    self._frames_dropped += 1
                    logger.error("Failed to add frame after dropping.")

    #async consumer API

    async def get(self, timeout: float = 0.05) -> Optional[Tuple[bytes, float]]:
        """
        Async consumer side for ASR/inference.
        Returns:
        - frame tuple
        - None if timeout
        - None if shutdown sentinel received

        """
        if self._shutdown_event.is_set():
            logger.info("Shutdown event set, returning None.")
            return None
        
        try:
            item = await asyncio.wait_for(
                self._queue.get(),
                timeout=timeout,
                )
        except asyncio.TimeoutError:
            return None
        
        if item == self.SENTINEL:
            return None
        
        return item

    #SHUTDOWN HANDLING

    def close(self) -> None:
        """
        Signal shutdown and prevent further ingestion.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            self._shutdown_event.set()

        # Insert sentinel safely
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait,
            self.SENTINEL
        )
        logger.info("Buffer closed and queue cleared.")

    @property
    def metrics(self) -> dict[str, float]:

        return {
            "frames_attempted": self._frames_attempted,
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "drop_rate": (
                self._frames_dropped / self._frames_attempted
                if self._frames_attempted else 0.0
            ),
            "queue_size": self._queue.qsize(),
            "last_frame_ts": self._last_frame_ts,
            "closed": self._closed,
        }