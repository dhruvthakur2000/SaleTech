import time
from fastapi import APIRouter, WebSocket

router=APIRouter()

SILENCE_THRESHOLD=100
SILENCE_TIMEOUT=0.5

@router.websocket("/ws/audio_debug")
async def audio_debug_ws(websocket: WebSocket):
    await websocket.accept()
    print("Audio debug connection opened")