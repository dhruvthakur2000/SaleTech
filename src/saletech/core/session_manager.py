import asyncio
import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta

from saletech.core.session import Session
from saletech.core.session_state import SessionState
from config.settings import settings
from saletech.utils.logger import setup_logger

logger = setup_logger()

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, metadata: Optional[dict] = None) -> Session:
        async with self._lock:
            if len(self._sessions) >= settings.max_sessions:
                raise RuntimeError("Max session limit reached")

            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                metadata=metadata or {}
            )
            session.audio_state = AudioState(
                buffer=VADAudioBuffer(
                    frame_bytes=320,
                    pre_roll_frames=10
                )
            )
            self._sessions[session_id] = session
            # logger.info(f"Session created: {session_id}")
            # logger.debug(f"Active sessions: {list(self._sessions.keys())}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_state(self, session_id: str, new_state: SessionState):
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            session.state = new_state
            session.last_active = datetime.now()

    async def close_session(self, session_id: str):
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.state = SessionState.CLOSED
                logger.info(f"Session closed: {session_id}")

    async def cleanup_inactive_sessions(self, timeout_seconds: int = 300):
        async with self._lock:
            now = datetime.now()
            to_remove = [
                sid for sid, s in self._sessions.items()
                if (now - s.last_active) > timedelta(seconds=timeout_seconds)
            ]

            for sid in to_remove:
                self._sessions.pop(sid)
                logger.info(f"Session expired: {sid}")