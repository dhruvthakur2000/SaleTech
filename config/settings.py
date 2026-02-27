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
    # ========================================
    # Model Paths (Free, Top-Tier Models)
    # ========================================
    # LLM: Qwen-2.5-7B (Best open-source LLM)
    qwen_model_path: str = Field(default="Qwen/Qwen2.5-7B-Instruct", env="QWEN_MODEL_PATH")
    
    # ASR: Faster-Whisper Large-v3 (Best free ASR, streaming capable)
    whisper_model_path: str = Field(default="large-v3", env="WHISPER_MODEL_PATH")
    whisper_device: str = Field(default="cuda", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", env="WHISPER_COMPUTE_TYPE")
    
    #Model Paths(Free, top-tier)
    vad_model_name:str = "silero_vad"
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
    asr_beam_size:int=5
    asr_best_of:int=5
    asr_max_concurrent_jobs: int = 2
    asr_cpu_threads: int = 4

 
    # VAD Configuration (Advanced)
    vad_threshold: float = Field(default=0.5, env="VAD_THRESHOLD")
    vad_aggressiveness: int = 3
    vad_frame_duration_ms: int = 32
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
    #
    #performance tuning 
    # Audio buffer sizes
    audio_input_buffer_size: int = 100  # Frames
    audio_output_buffer_size: int = 50  # Frames
    
    # Worker pool sizes
    vad_workers: int = 2
    asr_workers: int = 4
    tts_workers: int = 2
    
    # Batch processing
    vad_batch_size: int = 16  # Process multiple frames together
    
    # ========================================
    # Latency Budgets (Aggressive)
    # ========================================
    vad_timeout_ms: int = 50
    asr_timeout_ms: int = 300


    @property
    def chunk_size_samples(self) -> int:
        """Audio chunk size in samples"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    @property
    def vad_frame_size_samples(self) -> int:
        """VAD frame size in samples"""
        return int(self.sample_rate * self.vad_frame_duration_ms / 1000)

    @property
    def min_speech_samples(self) -> int:
        return int(self.sample_rate * self.min_speech_duration_ms / 1000)
    
    @property
    def max_speech_samples(self) -> int:
        return int(self.sample_rate * self.max_speech_duration_ms / 1000)
    
    @property
    def eot_silence_samples(self) -> int:
        return int(self.sample_rate * self.eot_silence_duration_ms / 1000)


settings=AppSettings()


