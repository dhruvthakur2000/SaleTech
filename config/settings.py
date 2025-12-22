from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """
    Application settings.
    """
    #app configuration 
    app_name: str = 'SaleTech'
    app_version: str ="1.0.0"
    log_level: str ="INFO"
    environment: Environment =Environment.DEVELOPMENT
    debug: bool =False

    host: str = '0.0.0.0'
    port: int = 8000

    #concurrency and performance
    max_sessions: int = 3
    session_timeout_seconds: int = 300
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SALETECH_",
        case_sensitive=False
    
    )



settings=AppSettings()


