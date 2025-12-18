from contextlib import asynccontextmanager
from src.saletech.utils.logger import setup_logger

logger=setup_logger()


@asynccontextmanager
async def lifespan(app):
    logger.info("SaleTech started:")
    
    # TO D0 later:
    # - connect Redis
    # - load ASR
    # - load LLM
    # - warm GPU

    yield

    logger.info("🛑 SaleTech shutting down")

    # TODO:
    # - close Redis
    # - release GPU