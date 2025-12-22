from fastapi import FastAPI
from saletech.utils.lifespan import lifespan
from saletech.utils.logger import setup_logger
from saletech.api.health import router as health_router
from saletech.api.sessions import router as session_router 
from config.settings import AppSettings

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
    return app

app = create_app()


