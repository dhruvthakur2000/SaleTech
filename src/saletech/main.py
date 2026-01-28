from fastapi import FastAPI
from saletech.utils.lifespan import lifespan
from saletech.utils.logger import setup_logger
from saletech.api.health import router as health_router
from saletech.api.sessions import router as session_router 
from config.settings import AppSettings
from fastapi import WebSocket
from saletech.api.audio_ws import audio_ws

settings = AppSettings()
logger = setup_logger()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan
    )

    #routers
    app.include_router(health_router)
    app.include_router(session_router)

    # @app.websocket("/ws/test")
    # async def test_ws(websocket: WebSocket):
    #     await websocket.accept()
    #     await websocket.send_text("Hello from WebSocket!")
    #     await websocket.close()

    return app

app = create_app()

# Expose session_manager globally for import in other modules
session_manager = app.state.session_manager

@app.websocket("/ws/audio/{session_id}")
async def audio_endpoint(websocket: WebSocket, session_id: str):
    await audio_ws(websocket, session_id)