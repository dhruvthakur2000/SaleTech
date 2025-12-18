from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Application settings.
    """
    #app configuration 
    app_name: str = 'SaleTech'
    app_version: str ="1.0.0"
    log_level: str ="INFO"
    environment: str ='development'
    debug: bool =False

    host: str = '0.0.0.0'
    port: int = 8000

    #concurrency and performance
    max_sessions: int =50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SALETECH_",
        case_sensitive=False
    
    )



settings=AppSettings()


