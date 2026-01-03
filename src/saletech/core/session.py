from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from saletech.core.session_state import SessionState
from saletech.media.audio_state import AudioRuntimeState


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    session_id: str

    # High-level lifecycle
    state: SessionState = SessionState.IDLE

    # Conversation memory
    history: List[Message] = Field(default_factory=list)

    # Runtime audio state (NOT serialized)
    audio: Optional[AudioRuntimeState] = Field(default=None, exclude=True)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)

    # Extra data
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now()
        self.last_active = datetime.now()
