import json
import os
import threading
from datetime import datetime
from typing import Dict, Any

from saletech.utils.logger import get_logger
from saletech.utils.errors import AudioProcessingError


logger = get_logger("saletech.transcriber.writer")


class TranscriptWriter:
    """
    Writes transcription results to a per-session JSONL file.

    Design goals:
    - One file per WebSocket session
    - Thread-safe append operations
    - One JSON object per line (JSONL format)
    - Graceful shutdown
    """

    def __init__(self, session_id: str, output_dir: str = "transcripts"):
        """
        Initialize a writer for one transcription session.

        Args:
            session_id: Unique session identifier
            output_dir: Directory where transcript files will be stored
        """

        self.session_id = session_id
        self.output_dir = output_dir

        # Ensure transcript directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Construct output filename (changed to .jsonl extension)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(
            self.output_dir,
            f"session_{session_id}_{timestamp}.jsonl"
        )

        # Internal state
        self._file = None
        self._lock = threading.Lock()
        self._closed = False

        try:
            # Open file in append mode for JSONL
            self._file = open(self.file_path, "a", encoding="utf-8")
            
            logger.info(
                "transcript_writer_initialized",
                session_id=session_id,
                file_path=self.file_path
            )

        except Exception as e:
            logger.error(
                "transcript_writer_init_failed",
                session_id=session_id,
                exc_info=True
            )
            raise AudioProcessingError(
                message="Failed to initialize transcript writer",
                original_exception=e
            )

    # -------------------------------------------------------------

    def write(self, record: Dict[str, Any]) -> None:
        """
        Append a transcription record as a JSON line to the file.

        Args:
            record: Dictionary containing utterance metadata
        """

        if self._closed:
            raise AudioProcessingError(
                message="Attempted to write after writer closed"
            )

        try:
            with self._lock:
                # Write JSON object followed by newline (JSONL format)
                json.dump(record, self._file, ensure_ascii=False)
                self._file.write("\n")
                
                # Ensure disk write
                self._file.flush()

        except Exception as e:
            logger.error(
                "transcript_write_failed",
                session_id=self.session_id,
                exc_info=True
            )
            raise AudioProcessingError(
                message="Failed to write transcript record",
                original_exception=e
            )

    # -------------------------------------------------------------

    def close(self) -> None:
        """
        Close the file handle.
        """

        if self._closed:
            return

        try:
            with self._lock:
                if self._file:
                    self._file.close()

            self._closed = True

            logger.info(
                "transcript_writer_closed",
                session_id=self.session_id,
                file_path=self.file_path
            )

        except Exception as e:
            logger.error(
                "transcript_writer_close_failed",
                session_id=self.session_id,
                exc_info=True
            )
            raise AudioProcessingError(
                message="Failed to close transcript writer",
                original_exception=e
            )
