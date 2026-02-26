import pytest
import pytest_asyncio
import numpy as np
from src.saletech.services.vad_adv_service import VADService

@pytest.mark.asyncio
async def test_vad_service_smoke():
    vad_service = VADService()

    await vad_service.initialize()

    # Silero VAD expects input shape (batch, 512) and sample_rate=16000
    dummy_audio = np.zeros((1, 512), dtype=np.float32)
    # If your detect_speech expects (audio, sample_rate), pass both
    try:
        result = vad_service.detect_speech(dummy_audio)
    except TypeError:
        # Try with sample_rate if required by your wrapper
        result = vad_service.detect_speech(dummy_audio, 16000)

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) >= 2
