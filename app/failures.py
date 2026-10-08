"""Turns pipeline exceptions into short, human-readable failure explanations (stdlib only).

The raw exception is never shown as the main message; callers keep it as `technical_error` and log it."""
from __future__ import annotations
import re

STAGE_KEYS = ["upload", "transcription", "diarization", "alignment", "refinement",
              "transformation", "documentation", "output"]
STAGE_NAMES = {"upload": "Upload", "transcription": "Transcription", "diarization": "Diarization",
               "alignment": "Alignment", "refinement": "Refinement", "transformation": "Transformation",
               "documentation": "Documentation", "output": "Output generation"}
_PROVIDER = {"transcription": "Deepgram", "diarization": "the diarization model",
             "refinement": "the language-model provider", "transformation": "the language-model provider",
             "documentation": "the language-model provider"}
_RETRY = "Retry the meeting."


class OutputError(Exception):
    """A result file could not be written or an expected artifact is missing."""


def _has(low, *ks):
    return any(k in low for k in ks)


def classify(stage: str, exc: BaseException) -> dict:
    """-> {"message", "hint", "technical"}"""
    chain = [exc] + [e for e in (exc.__cause__, exc.__context__) if e is not None]
    tech = f"{type(exc).__name__}: {exc}"
    low = " ".join(f"{type(e).__name__} {e}" for e in chain).lower()
    names = " ".join(type(e).__name__ for e in chain).lower()
    prov = _PROVIDER.get(stage, "the service")
    P = prov[0].upper() + prov[1:]
    P = "Sarvam" if stage == "transcription" and "sarvam" in " ".join(str(e).lower() for e in chain) else P
    llm = stage in ("refinement", "transformation", "documentation")

    def r(msg, hint):
        return {"message": msg, "hint": hint, "technical": tech}

    if _has(names, "unsupportedlanguage"):
        return r(str(exc)[:240], "Pick a supported language (or Auto Detect) and upload again.")
    if _has(names, "translationerror"):
        return r(f"Translation failed. The original transcript has been preserved. ({str(exc)[:160]})", _RETRY)
    if stage == "transcription" and _has(low, "is not set"):
        return r(f"{('Sarvam' if 'sarvam' in low else 'Deepgram')} API key is not configured.",
                 "Add the key to the .env file (SARVAM_API_KEY / DEEPGRAM_API_KEY), restart the server, then retry.")
    if stage == "transcription" and "sarvam" in low:
        P = "Sarvam"
    if isinstance(exc, OutputError) or _has(names, "permissionerror") or _has(low, "no space left", "read-only file system"):
        return r(f"A result file could not be written or is missing: {str(exc)[:200]}",
                 "Check disk space and write permission on the outputs folder, then retry.")
    if _has(names, "modulenotfounderror", "importerror", "dependencyerror") or _has(low, "not found on path", "package '"):
        return r(f"A required dependency is missing: {str(exc)[:160]}",
                 "Install it (see requirements.txt / README), restart the server, then retry.")
    if stage != "upload" and _has(names, "emptyaudio", "unreadableaudio", "audionotfound", "audioerror") or _has(low, "ffmpeg failed", "invalid audio"):
        return r(f"The audio could not be read or is invalid: {str(exc)[:160]}",
                 "Upload a different or re-encoded audio file (wav/mp3/m4a).")
    if stage in ("transcription", "refinement", "transformation", "documentation") or (stage == "diarization" and _has(low, "hf_token", "huggingface", "gated")):
        if _has(low, "timed out", "timeout"):
            return r(f"{P} request timed out.", f"Check your internet connection and {_RETRY[0].lower()+_RETRY[1:]}")
        if re.search(r"\b429\b", low) or _has(low, "rate limit", "ratelimit", "quota", "rate/concurrency"):
            return r(f"{P} rate or quota limit was reached.", "Wait a minute or check your plan/quota, then retry.")
        if re.search(r"\b(401|403)\b", low) or _has(low, "api key", "api_key", "unauthor", "authentication", "hf_token", "gated"):
            return r(f"{P} rejected the credentials (authentication failure).",
                     "Check the API key / token in the .env file, then retry.")
        if _has(low, "network error", "connectionerror", "connection refused", "connection aborted",
                "connection reset", "getaddrinfo", "max retries", "name resolution"):
            return r(f"Could not reach {prov} (network problem).", "Check your internet connection and retry.")
    if stage == "diarization":
        if _has(low, "cuda", "out of memory", "no kernel image", "cudnn", "gpu"):
            return r("The diarization model failed on the compute device (CUDA/GPU).",
                     "Free GPU memory or switch to CPU, then retry.")
        if _has(low, "pyannote", "checkpoint", "pipeline.from_pretrained", "could not be loaded", "model"):
            return r("The diarization model is unavailable or could not be loaded.",
                     "Check the pyannote install and Hugging Face token/model access, then retry.")
    if llm:
        if _has(names, "jsondecodeerror", "validationerror", "outputerror", "transformationerror") or \
           _has(low, "malformed", "not valid json", "truncated", "unusable", "validation", "schema", "unexpected"):
            return r("The model returned output that was malformed or failed validation (after retries).",
                     "Retry; if it keeps happening, check the technical details in the server log.")
        if _has(names, "filenotfound") or _has(low, "no such file", "missing"):
            prev = {"refinement": "aligned transcript", "transformation": "refined transcript",
                    "documentation": "refined transcript / transformation output"}[stage]
            return r(f"A required input is missing: the {prev}.", "Retry the meeting (a full rerun regenerates it).")
    if stage == "alignment":
        if _has(names, "filenotfound") or _has(low, "missing", "no such file"):
            return r("The transcript or diarization needed for alignment is missing.", _RETRY)
        return r("Transcript and diarization could not be aligned (missing, malformed or incompatible timestamps/data).",
                 "Retry; if it repeats, the audio may be unusual — see the server log.")
    if stage == "diarization":
        return r("Speaker diarization failed.", "Retry; if it repeats, check the server log.")
    if stage == "transcription":
        return r(f"Transcription failed: {str(exc)[:200]}", "Check the audio file and your Deepgram settings, then retry.")
    return r(f"{STAGE_NAMES.get(stage, stage)} failed: {type(exc).__name__}: {str(exc)[:200]}",
             "Retry; if it keeps failing, check the server log.")
