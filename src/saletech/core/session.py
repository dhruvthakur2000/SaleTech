import asyncio
import uuid
import time
from typing import Optional
from datetime import datetime
from ..models.schemas import (
    SessionState,ConversationMessage,SessionMetrics
)
from ..services.vad_adv_service import VADService
from ..services.streaming_asr import StreamingASR
from saletech.core import session_manager
from ..utils.logger import SessionLogger

from ..utils.logger import get_logger

logger= get_logger("Saletech.session")


class VoiceSession:
    """
    Production-grade Voice Session for SaleTech.
    Owns full conversational pipeline for one user.
    """

    def __init__(
            self,
            session_id:str,
            customer_name: Optional[str] = None
            ):
        
        self.session_id = session_id
        self.customer_name = customer_name 

        #atate
        self.state = SessionState.IDLE
        self.created_at = datetime.now
        self.last_activity = datetime.now

        #conversation
        self.conversation: list[ConversationMessage]=[]
        self.metrics =  SessionMetrics(session_id=session_id)

        #services
        self._vad_service = None
        self._asr_service = None

        #control
        self._running = False
        self_state_lock = asyncio.Lock()
        self._main_task: Optional[asyncio.Task] = None

        self.logger = SessionLogger(session_id)

        self.logger.info (
            "session created ",
            customer_name=customer_name
            )
        
    async def start(self):
        """
        Start session with all advanced features.
        """
        if self._running:
            return
        
        # loading services (singleton)
        self._vad_service = await VADService()
        self._asr_service= await StreamingASR()

        self._running = True
        self._main_task = asyncio.create_task(self._audio_loop())
        logger.info("session_started.", session_id=self.session_id)
        logger.info("session_started.",session_id=self.session_id)



    async def _audio_loop(self):
        # ...existing code or placeholder...
        pass

