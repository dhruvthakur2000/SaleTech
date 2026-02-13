import asyncio
import threading
import time
from typing import Optional, Tuple

class FrameIngressBuffer:
    """
    Thread-safe raw audio ingestion buffer.

    Responsibility:
    - Accept PCM frames from WebSocket threads
    - Buffer them safely
    - Deliver to async consumers
    """
    def __init__(self,max_frames:int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_frames)
        self._closed= False

        self._lock = threading.Lock()

        #metrics
        self.frames_recieved=0
        self.frames_dropped=0
        self._last_frame_time = 0.0


        def put_nowait(self, pcm:bytes, timestamp: Optional[float]= None)-> None:
            """
            runs in Websocket callback thread.
            
            :param self: Description
            :param pcm: Description
            :type pcm: bytes
            :param timestamp: Description
            :type timestamp: Optional[float]
            """
            if timestamp is None:
                timestamp = time.time()

            with self._lock:
                if self._closed:
                    return

            try:
                self._queue.put_nowait((pcm, timestamp))
                self.frames_received += 1

            except asyncio.QueueFull:
                try:
                    # drop oldest
                    self._queue.get_nowait()
                    self._queue.put_nowait((pcm, timestamp))
                    self.frames_dropped += 1
                except Exception:
                    self.frames_dropped += 1


    async def get(self) -> Optional[Tuple[bytes, float]]:
        """
        Async consumer side.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True

