"""Sarvam speech-to-text (Indian languages) via the official `sarvamai` SDK.

Long meetings use Sarvam's Batch API (up to 2 h per file): create job -> upload -> start -> wait ->
download JSON. The model is Saaras (v3 by default), `mode="transcribe"`, so the output is the ORIGINAL
language transcript (no translation here; translation happens later, in llm1.translator).
Batch timestamps are chunk (sentence/phrase) level, not word level; speakers come from the existing
pyannote stage, so Sarvam's own diarization is NOT requested.

Docs: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api
The API key is read from SARVAM_API_KEY on the server only and is never logged or sent to the browser.
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from typing import Any, Optional

from common.audio import validate_audio, to_16k_mono
from common.errors import (STTConfigError, STTError, STTProviderError, STTRateLimitError,
                           UnsupportedLanguageError)
from .base import STTProvider
from .languages import SARVAM_LANGUAGES
from .schemas import STTResult, STTSegment

MAX_DURATION_S = 2 * 3600          # Sarvam batch limit: 2 hours per file
DETECT_CLIP_S = 25.0               # REST endpoint accepts <= 30 s; leave headroom
DEFAULT_MODEL = "saaras:v3"


def _scrub(text: Any, key: str) -> str:
    t = str(text or "").replace("\n", " ")[:300]
    return t.replace(key, "***") if key else t


def classify_sarvam_error(exc: Exception, key: str = "") -> STTError:
    """Turn an SDK/network exception into our error types (clear message, never the API key)."""
    st = getattr(exc, "status_code", None)
    if not isinstance(st, int):
        st = getattr(getattr(exc, "response", None), "status_code", None)
    msg = _scrub(getattr(exc, "body", None) or exc, key)
    if isinstance(exc, TimeoutError) or type(exc).__name__ in ("Timeout", "ReadTimeout", "ConnectTimeout"):
        return STTProviderError(f"Sarvam request timed out: {msg}")
    if st in (401, 403):
        return STTConfigError(f"Sarvam rejected the API key (HTTP {st}). Check SARVAM_API_KEY.")
    if st == 429:
        return STTRateLimitError("Sarvam rate limit / quota hit (HTTP 429).")
    if st in (400, 422):
        return STTProviderError(f"Sarvam rejected the request (HTTP {st}): {msg}")
    if st is not None:
        return STTProviderError(f"Sarvam HTTP {st}: {msg}")
    return STTProviderError(f"Sarvam error ({type(exc).__name__}): {msg}")


def parse_sarvam_output(data: dict[str, Any], *, model: str, audio_path: Optional[str], duration: float,
                        requested_language: Optional[str]) -> STTResult:
    """Pure function (no network) so it is unit-testable offline. `data` is one batch output JSON:
    {"transcript": str, "timestamps": {"chunks": [...], "start_time_seconds": [...],
     "end_time_seconds": [...]}, "language_code": "hi-IN", ...}"""
    if not isinstance(data, dict) or "transcript" not in data:
        raise STTProviderError("Unexpected Sarvam response shape (no 'transcript').")
    text = str(data.get("transcript") or "").strip()
    ts = data.get("timestamps") or {}
    chunks, starts, ends = (ts.get("chunks") or []), (ts.get("start_time_seconds") or []), (ts.get("end_time_seconds") or [])
    warnings: list[str] = []
    if not (len(chunks) == len(starts) == len(ends)):
        raise STTProviderError("Malformed Sarvam timestamps (chunks/start/end lengths differ).")
    rows = [(float(a), float(b), str(c).strip()) for c, a, b in zip(chunks, starts, ends) if str(c).strip()]
    rows.sort(key=lambda r: r[0])
    segments = [STTSegment(id=i, start=a, end=max(b, a), text=c) for i, (a, b, c) in enumerate(rows)]
    if not segments and text:
        warnings.append("Sarvam returned no timestamps; the transcript is one segment spanning the audio.")
        segments = [STTSegment(id=0, start=0.0, end=float(duration), text=text)]
    if not segments:
        warnings.append("No speech detected in audio.")
    lang = data.get("language_code") or (requested_language if requested_language not in (None, "unknown") else None)
    return STTResult(text=text, language=lang, duration=float(duration), segments=tuple(segments),
                     words=(), provider="sarvam", model=model, audio_path=audio_path,
                     warnings=tuple(warnings), raw_response=[{"response": data}])


class SarvamSTT(STTProvider):
    name = "sarvam"

    def __init__(self, api_key: Optional[str] = None, language_code: str = "hi-IN", model: Optional[str] = None,
                 client=None, poll_interval: int = 5, timeout_s: Optional[int] = None):
        """language_code: one of SARVAM_LANGUAGES, or 'unknown' (Sarvam auto-detects). `client` is injectable
        so tests never touch the network."""
        if language_code != "unknown" and language_code not in SARVAM_LANGUAGES:
            raise UnsupportedLanguageError(f"{language_code!r} is not a language supported by Sarvam STT.")
        key = api_key or os.environ.get("SARVAM_API_KEY")
        if not key:
            raise STTConfigError("SARVAM_API_KEY is not set (put it in .env or the environment).")
        self._key = key
        self.language_code = language_code
        self.model = model or os.environ.get("SARVAM_STT_MODEL", DEFAULT_MODEL)
        self.poll_interval = poll_interval
        self.timeout_s = int(timeout_s or os.environ.get("SARVAM_TIMEOUT_S") or 1800)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from sarvamai import SarvamAI
            except ImportError as e:
                raise STTConfigError("Package 'sarvamai' missing: pip install sarvamai") from e
            self._client = SarvamAI(api_subscription_key=self._key)
        return self._client

    # -- language auto-detection (short clip, REST endpoint) ----------------------------------
    def detect_language(self, audio_path: str | Path) -> Optional[str]:
        """Return Sarvam's detected BCP-47 code (e.g. 'hi-IN', 'en-IN') for the first ~25 s, or None."""
        info = validate_audio(audio_path)
        with tempfile.TemporaryDirectory(prefix="sarvam_detect_") as td:
            clip = Path(td) / "clip.wav"
            to_16k_mono(info.path, clip, "pcm_s16le")
            clip = _trim(clip, Path(td) / "clip25.flac", DETECT_CLIP_S)
            try:
                with clip.open("rb") as fh:
                    resp = self.client.speech_to_text.transcribe(
                        file=fh, model=self.model, language_code="unknown", mode="transcribe")
            except STTError:
                raise
            except Exception as e:  # noqa: BLE001
                raise classify_sarvam_error(e, self._key) from None
        return getattr(resp, "language_code", None) or (resp.get("language_code") if isinstance(resp, dict) else None)

    # -- full transcription (batch) -------------------------------------------------------------
    def transcribe(self, audio_path: str | Path) -> STTResult:
        info = validate_audio(audio_path)
        if info.duration > MAX_DURATION_S:
            raise STTError("Audio is longer than Sarvam's 2-hour limit; split the recording.")
        with tempfile.TemporaryDirectory(prefix="sarvam_") as td:
            wav = Path(td) / "audio.wav"
            to_16k_mono(info.path, wav, "pcm_s16le")
            outdir = Path(td) / "out"
            outdir.mkdir()
            try:
                job = self.client.speech_to_text_job.create_job(
                    model=self.model, mode="transcribe", language_code=self.language_code,
                    with_diarization=False)
                job.upload_files(file_paths=[str(wav)])
                job.start()
                job.wait_until_complete(poll_interval=self.poll_interval, timeout=self.timeout_s)
                results = job.get_file_results()
                failed = (results or {}).get("failed") or []
                if failed or not (results or {}).get("successful"):
                    why = _scrub((failed[0] or {}).get("error_message") if failed else "no output produced", self._key)
                    raise STTProviderError(f"Sarvam failed to transcribe the audio: {why}")
                job.download_outputs(output_dir=str(outdir))
            except STTError:
                raise
            except Exception as e:  # noqa: BLE001
                raise classify_sarvam_error(e, self._key) from None
            files = sorted(outdir.rglob("*.json"))
            if not files:
                raise STTProviderError("Sarvam job finished but no output file was downloaded.")
            try:
                data = json.loads(files[0].read_text(encoding="utf-8"))
            except ValueError as e:
                raise STTProviderError("Sarvam returned invalid JSON.") from e
        return parse_sarvam_output(data, model=self.model, audio_path=str(info.path),
                                   duration=info.duration, requested_language=self.language_code)


def _trim(src: Path, dst: Path, seconds: float) -> Path:
    from common.audio import extract_chunk
    extract_chunk(src, dst, 0.0, seconds)   # flac content; Sarvam auto-detects codec
    return dst
