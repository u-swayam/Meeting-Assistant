import os
from common.errors import STTConfigError
from .base import STTProvider


def get_stt_provider(name: str | None = None, **kwargs) -> STTProvider:
    name = (name or os.environ.get("STT_PROVIDER", "deepgram")).lower()
    if name == "deepgram":
        from .deepgram_nova import DeepgramNovaSTT
        return DeepgramNovaSTT(**kwargs)
    if name == "groq":
        from .whisper_cloud import GroqWhisperSTT
        return GroqWhisperSTT(**kwargs)
    raise STTConfigError(f"Unknown STT provider: {name}")
