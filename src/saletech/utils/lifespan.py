from contextlib import asynccontextmanager
from saletech.utils.logger import setup_logging
from saletech.core.session_manager import SessionManager


logger = setup_logging()


@asynccontextmanager
async def lifespan(app):
    app.state.session_manager = SessionManager()
    # logger.info("SaleTech started:")
    
    # TO D0 later:
    # - connect Redis
    # - load ASR
    # - load LLM
    # - warm GPU

    yield

    # logger.info(" SaleTech shutting down")

    # TODO:
    # - close Redis
    # - release GPU