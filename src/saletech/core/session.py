from typing import List, Dict, Any, Literal, Optional
import time
import asyncio
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from src.saletech.services.streaming_asr import StreamingASR


from saletech.core.audio_state import AudioState
from saletech.media.audio_buffer import VADAudioBuffer



class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    session_id: str

    # High-level lifecycle
    state: SessionState = SessionState.IDLE

    # Conversation memory
    history: List[Message] = Field(default_factory=list)

    # Runtime audio state (NOT serialized)
    # audio_state: AudioState = Field(
    #     default_factory=lambda: AudioState(
    #         buffer=VADAudioBuffer(
    #             frame_bytes=320,  # 20ms @ 16KHz, 16 bit mono
    #             pre_roll_frames=10
    #         )
    #     )
    # )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)

    # Extra data
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now()
        self.last_active = datetime.now()


    class VoiceSession:

        async def start(self):
            self.asr = await StreamingASR()