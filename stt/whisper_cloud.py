"""Optional alternate provider: Whisper large-v3-turbo on Groq (kept so providers stay swappable)."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from typing import Any, Optional

from common.audio import validate_audio, to_16k_mono, extract_chunk, plan_chunks
from common.errors import STTConfigError, STTProviderError, STTRateLimitError, STTError
from .base import STTProvider
from .schemas import STTResult, STTSegment, Word

MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # Groq free tier: 25 MB


class GroqWhisperSTT(STTProvider):
    name = "groq"
    MODEL_ID = "whisper-large-v3-turbo"

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_ID,
                 language: Optional[str] = None, prompt: Optional[str] = None,
                 chunk_seconds: float = 600.0, word_timestamps: bool = True,
                 max_retries: int = 3, timeout: float = 120.0):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise STTConfigError("GROQ_API_KEY is not set.")
        try:
            from groq import Groq
        except ImportError as e:
            raise STTConfigError("Package 'groq' missing: pip install groq") from e
        self._client = Groq(api_key=key, max_retries=max_retries, timeout=timeout)
        self.model, self.language, self.prompt = model, language, prompt
        self.chunk_seconds, self.word_timestamps = chunk_seconds, word_timestamps

    def _call(self, chunk: Path) -> dict[str, Any]:
        import groq
        kw: dict[str, Any] = dict(
            model=self.model, response_format="verbose_json", temperature=0.0,
            timestamp_granularities=["word", "segment"] if self.word_timestamps else ["segment"])
        if self.language: kw["language"] = self.language
        if self.prompt: kw["prompt"] = self.prompt
        try:
            resp = self._client.audio.transcriptions.create(file=(chunk.name, chunk.read_bytes()), **kw)
        except groq.RateLimitError as e:
            raise STTRateLimitError(f"Groq rate limit hit: {e}") from e
        except groq.APIConnectionError as e:
            raise STTProviderError(f"Network error talking to Groq: {e}") from e
        except groq.APIStatusError as e:
            raise STTProviderError(f"Groq API error {e.status_code}: {e.message}") from e
        data = resp if isinstance(resp, dict) else resp.model_dump()
        return json.loads(json.dumps(data, default=str))

    def transcribe(self, audio_path: str | Path) -> STTResult:
        info = validate_audio(audio_path)
        segments, words, texts, langs, raw, warnings = [], [], [], [], [], []
        with tempfile.TemporaryDirectory(prefix="stt_") as td:
            tmp = Path(td); flac = tmp / "full.flac"
            to_16k_mono(info.path, flac, "flac")
            plan = plan_chunks(info.duration, self.chunk_seconds)
            if len(plan) > 1:
                warnings.append(f"Audio split into {len(plan)} chunks; words at chunk boundaries may be cut.")
            for ci, (start, length) in enumerate(plan):
                chunk = flac if len(plan) == 1 else tmp / f"chunk_{ci:04d}.flac"
                if len(plan) > 1:
                    extract_chunk(flac, chunk, start, length)
                if chunk.stat().st_size > MAX_UPLOAD_BYTES:
                    raise STTError(f"Chunk {ci} too large; lower chunk_seconds.")
                data = self._call(chunk)
                raw.append({"chunk_index": ci, "offset": start, "response": data})
                text, segs = (data.get("text") or "").strip(), data.get("segments") or []
                if text and not segs:
                    raise STTProviderError("Provider returned text without segment timestamps.")
                for s in segs:
                    segments.append(STTSegment(
                        id=len(segments), start=float(s["start"]) + start, end=float(s["end"]) + start,
                        text=str(s.get("text", "")).strip(), avg_logprob=s.get("avg_logprob"),
                        no_speech_prob=s.get("no_speech_prob"),
                        compression_ratio=s.get("compression_ratio"), chunk_index=ci))
                for w in data.get("words") or []:
                    words.append(Word(start=float(w["start"]) + start, end=float(w["end"]) + start,
                                      text=str(w.get("word", w.get("text", ""))).strip()))
                if text: texts.append(text)
                if data.get("language"): langs.append(str(data["language"]))
        if not segments: warnings.append("No speech detected in audio.")
        language = max(set(langs), key=langs.count) if langs else None
        return STTResult(text=" ".join(texts), language=language, duration=info.duration,
                         segments=tuple(segments), words=tuple(words), provider=self.name,
                         model=self.model, audio_path=str(info.path), warnings=tuple(warnings),
                         raw_response=raw)
