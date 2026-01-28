from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from pydantic import Field, field_validator
from typing import Optional
import os




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

    log_level: str = Field(default= "INFO", env="LOG_LEVEL")


    #Environment Configuration
    environment: Environment =Environment.DEVELOPMENT
    debug: bool =False

    #Server Configuration
    host: str = Field(default ="0.0.0.0", env= "HOST")
    port: int = Field(default=8000, env= "PORT")

    #concurrency and performance
    max_sessions: int = Field(default=20, env="MAX_SESSIONS")
    session_timeout_seconds: int = 300
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SALETECH_",
        case_sensitive=False
    
    )

    #GPU Configuration
    cuda_visible_device: str= Field(default="0", env="CUDA_VISIBLE_DEVICE")
    gpu_memory_fraction: float = Field(default=0.9, env="GPU_MEMORY_FRACTION")

    #Model Paths(Free, top-tier)
    vad_model_name:str = "Silero_Vad"
    vad_repo:str = "snakers4/silero-vad"
    # HARDCODED: These don't change, no env var needed
    # LOADED VIA: torch.hub.load(repo, model_name)

    #Audio Configuration
    sample_rate:int =Field(default=16000, env= "SAMPLE_RATE")
    chunk_duration_ms: int = Field(default=20, env="CHUNK_DURATION_MS")


    # ========================================
    # Streaming ASR Configuration
    # ========================================
    # REVERSE ENG: Controls real-time transcription behavior
    
    asr_streaming_enabled: bool = True
    # WHY: Core v2 feature - show transcription as user speaks
    # IMPACT: Users see text appearing in real-time
    
    asr_min_audio_ms: int = 300
    # PREVENTS: Transcribing very short sounds (coughs, noise)
    # WHY 300ms: Minimum for a short word like "yes"
    
    asr_max_chunk_ms: int = 3000
    # WHY: Break long speech into chunks for streaming
    # PREVENTS: Waiting 10 seconds before showing any text
    
    asr_partial_update_interval_ms: int = 500
    # HOW OFTEN: Send partial transcription updates to UI
    # WHY 500ms: Balance between smoothness and overhead
    # FASTER (250ms): More updates but more CPU
    # SLOWER (1000ms): Feels laggy
    
    asr_partial_confidence_threshold: float = 0.5
    # FILTERS: Don't show partial text if confidence < 0.5
    # WHY: Avoid showing wrong words that change immediately
    
    asr_final_confidence_threshold: float = 0.7
 
    # VAD Configuration (Advanced)
    vad_threshold: float = Field(default=0.5, env="VAD_THRESHOLD")
    vad_aggressiveness: int = 3
    vad_frame_duration_ms: int = 30
    speech_onset_threshold: float = 0.5
    speech_offset_threshold: float = 0.3
    min_speech_duration_ms: int = 200  # REDUCED from 300ms to 200 ms
    max_speech_duration_ms: int = 15000  # 15 seconds
    speech_pad_ms: int = 200  # REDUCED from 300ms

    # End-of-Turn Detection
    
    eot_silence_duration_ms: int = 600  # REDUCED from 800ms

    
    eot_max_pause_ms: int = 400

    
    eot_adaptive_enabled: bool = True


    # Barge-In Handling
    barge_in_enabled: bool = True
    barge_in_energy_threshold: float = 0.6
    barge_in_delay_ms: int = 300
    barge_in_grace_period_ms: int = 500


settings=AppSettings()


