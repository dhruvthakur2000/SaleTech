from fastapi import FastAPI
from saletech.utils.lifespan import lifespan
from saletech.utils.logger import setup_logging
from saletech.api.health import router as health_router
from saletech.api.sessions import router as session_router 
from config.settings import AppSettings
from fastapi import WebSocket
from src.saletech.api.exception_handler import (
    saletech_exception_handler
)
from src.saletech.utils.errors import SaleTechException

settings = AppSettings()
logger = setup_logging()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan
    )

    #routers
    app.include_router(health_router)
    app.include_router(session_router)

    app.add_exception_handler(
        SaleTechException,
        saletech_exception_handler,
    )

    return app

app = create_app()

# Expose session_manager globally for import in other modules
session_manager = app.state.session_manager

@app.websocket("/ws/audio/{session_id}")
async def audio_endpoint(websocket: WebSocket, session_id: str):
    await audio_ws(websocket, session_id)