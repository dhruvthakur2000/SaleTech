# src/saletech/transcriber/session.py
import asyncio
import time
import numpy as np
from typing import Optional

from src.saletech.media.vad.vad_model import AdvancedVadModel
from src.saletech.services.vad_adv_service import VADService
from src.saletech.media.vad.vad_state import VADSessionState
from src.saletech.media.buffer.StreamingVadBuffer import StreamingBuffer
from src.saletech.media.buffer.frame_chunking import AudioIngressBuffer
from src.saletech.services.streaming_asr import StreamingASR
from src.saletech.transcriber.writer import TranscriptWriter
from src.saletech.utils.logger import get_logger
from src.saletech.utils.errors import AudioProcessingError

class transSession:
    """
    """

