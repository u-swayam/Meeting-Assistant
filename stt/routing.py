"""Provider routing. English -> Deepgram Nova-3 (default, unchanged). Supported Indian language -> Sarvam.

    resolve_route(choice, audio)  ->  LanguageRoute        (Auto Detect uses Sarvam on a short clip)
    build_stt(route, **kw)        ->  STTProvider
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from common.errors import STTConfigError
from .base import STTProvider
from .languages import LanguageRoute, PROVIDER_SARVAM, is_auto, route_for_language


def resolve_route(choice: Optional[str], audio_path: str | Path | None = None, *, sarvam=None) -> LanguageRoute:
    """Explicit choice -> table lookup (no network). Auto Detect -> Sarvam language detection on a short clip;
    no key / no detection result is an ERROR, never a silent fallback to another language."""
    if not is_auto(choice):
        return route_for_language(choice)
    if audio_path is None:
        raise STTConfigError("Auto Detect needs an audio file.")
    if sarvam is None:
        from .sarvam import SarvamSTT
        sarvam = SarvamSTT(language_code="unknown")          # raises STTConfigError if SARVAM_API_KEY is unset
    code = sarvam.detect_language(audio_path)
    if not code:
        raise STTConfigError("Could not detect the language automatically. Choose the language explicitly.")
    return route_for_language(code, detected=True)           # unsupported detected language -> clear error


def build_stt(route: LanguageRoute, **kwargs) -> STTProvider:
    if route.provider == PROVIDER_SARVAM:
        from .sarvam import SarvamSTT
        return SarvamSTT(language_code=route.language_code, **{k: v for k, v in kwargs.items() if k != "timeout"})
    from . import get_stt_provider                            # English: exactly the pre-existing path
    return get_stt_provider(None, language="en", **kwargs)
