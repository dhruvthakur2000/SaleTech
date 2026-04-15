import numpy as np
import time
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.saletech.transcriber.pipeline import TranscriptionPipeline
from src.saletech.transcriber.session import sess


from src.saletech.utils.logger import get_logger

router = APIRouter()

logger = get_logger("saletech.transcriber.ws")

@router.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """
    websocket endpoint for real time speech transcription
    """
    await ws.accept()

    session_id = str(uuid.uuid4())

    logger.info("Transcriber session started", session_id=session_id)

    pipeline = TranscriptionPipeline(session_id=session_id)

    await pipeline.initialize()

    try:
        while True:
            audio_bytes = await ws.receive_bytes()

            audio_np= np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

            audio/=32768.0
