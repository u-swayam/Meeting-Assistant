"""Deepgram Nova-3 pre-recorded transcription via plain REST (no SDK, so no SDK-version drift).

Endpoint: POST https://api.deepgram.com/v1/listen  (raw audio body, Authorization: Token <key>)
Segments come from Deepgram `utterances`; word timings from channels[0].alternatives[0].words.
"""
from __future__ import annotations
import os, tempfile, time
from pathlib import Path
from typing import Any, Optional

from common.audio import validate_audio, to_16k_mono
from common.errors import STTConfigError, STTProviderError, STTRateLimitError, STTError
from .base import STTProvider
from .schemas import STTResult, STTSegment, Word

API_URL = "https://api.deepgram.com/v1/listen"
MAX_BYTES = 2 * 1024 ** 3  # Deepgram documents a 2 GB max file size for pre-recorded audio


def parse_deepgram_response(data: dict[str, Any], *, model: str, audio_path: Optional[str],
                            duration: float, requested_language: Optional[str]) -> STTResult:
    """Pure function (no network) so it can be unit-tested offline."""
    try:
        channel = data["results"]["channels"][0]
        alt = channel["alternatives"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise STTProviderError("Unexpected Deepgram response shape "
                               "(no results.channels[0].alternatives[0]).") from e
    text = (alt.get("transcript") or "").strip()

    words = []
    for w in alt.get("words") or []:
        t = str(w.get("punctuated_word") or w.get("word") or "").strip()
        if t:
            words.append(Word(start=float(w["start"]), end=float(w["end"]), text=t,
                              confidence=w.get("confidence")))

    utts = data["results"].get("utterances")
    if utts is None:
        if text:
            raise STTProviderError("Response has no utterances; the request must include utterances=true.")
        utts = []
    segments = [STTSegment(id=i, start=float(u["start"]), end=float(u["end"]),
                           text=str(u.get("transcript", "")).strip(), confidence=u.get("confidence"))
                for i, u in enumerate(sorted(utts, key=lambda u: float(u["start"])))]

    warnings = []
    if not segments:
        warnings.append("No speech detected in audio.")
    lang = channel.get("detected_language")
    if not lang and requested_language not in (None, "detect", "multi"):
        lang = requested_language
    meta_dur = (data.get("metadata") or {}).get("duration")
    return STTResult(text=text, language=lang, duration=float(meta_dur or duration),
                     segments=tuple(segments), words=tuple(words), provider="deepgram",
                     model=model, audio_path=audio_path, warnings=tuple(warnings),
                     raw_response=[{"response": data}])


class DeepgramNovaSTT(STTProvider):
    name = "deepgram"
    MODEL_ID = "nova-3"

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_ID,
                 language: Optional[str] = None, keyterms: Optional[list[str]] = None,
                 diarize: bool = False, max_retries: int = 3,
                 timeout: tuple[float, float] = (10.0, 900.0)):
        """language: 'en' (default, env DEEPGRAM_LANGUAGE), another code, 'multi', or 'detect'.
        diarize: ask Deepgram for its own speaker labels (kept only in raw_response; the pipeline
        uses pyannote for speakers)."""
        key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise STTConfigError("DEEPGRAM_API_KEY is not set (put it in .env or the environment).")
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise STTConfigError("Package 'requests' missing: pip install requests") from e
        self._key = key
        self.model = model
        self.language = language or os.environ.get("DEEPGRAM_LANGUAGE", "en")
        self.keyterms = keyterms or []
        self.diarize = diarize
        self.max_retries = max_retries
        self.timeout = timeout

    def _params(self) -> list[tuple[str, str]]:
        p = [("model", self.model), ("smart_format", "true"), ("utterances", "true")]
        if self.language == "detect":
            p.append(("detect_language", "true"))
        else:
            p.append(("language", self.language))
        if self.diarize:
            p.append(("diarize", "true"))
        p += [("keyterm", k) for k in self.keyterms]   # Nova-3 keyterm prompting (opt-in)
        return p

    def _post(self, payload: bytes) -> dict[str, Any]:
        import requests
        headers = {"Authorization": f"Token {self._key}", "Content-Type": "audio/flac"}
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                r = requests.post(API_URL, params=self._params(), headers=headers,
                                  data=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if last:
                    raise STTProviderError(f"Network error talking to Deepgram: {e}") from e
                time.sleep(delay); delay *= 2
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    raise STTProviderError("Deepgram returned invalid JSON.") from e
            if r.status_code in (401, 403):
                raise STTConfigError(f"Deepgram rejected the API key (HTTP {r.status_code}).")
            if r.status_code in (429, 500, 502, 503, 504) and not last:
                try:
                    wait = float(r.headers.get("Retry-After", delay))
                except ValueError:
                    wait = delay
                time.sleep(min(wait, 30.0)); delay *= 2
                continue
            if r.status_code == 429:
                raise STTRateLimitError("Deepgram rate/concurrency limit hit (HTTP 429).")
            raise STTProviderError(f"Deepgram HTTP {r.status_code}: {r.text[:300]}")
        raise STTProviderError("Deepgram request failed.")  # unreachable

    def transcribe(self, audio_path: str | Path) -> STTResult:
        info = validate_audio(audio_path)
        with tempfile.TemporaryDirectory(prefix="stt_") as td:
            flac = Path(td) / "audio.flac"
            to_16k_mono(info.path, flac, "flac")      # 16 kHz mono FLAC: small, lossless
            if flac.stat().st_size > MAX_BYTES:
                raise STTError("Audio exceeds Deepgram's 2 GB limit even after compression.")
            payload = flac.read_bytes()
        data = self._post(payload)
        return parse_deepgram_response(data, model=self.model, audio_path=str(info.path),
                                       duration=info.duration, requested_language=self.language)
