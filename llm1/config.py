from __future__ import annotations
import os
from dataclasses import dataclass

from common.errors import PipelineError


class LLM1Error(PipelineError): ...
class LLM1ConfigError(LLM1Error): ...
class LLM1ProviderError(LLM1Error): ...
class LLM1UnavailableError(LLM1ProviderError):
    """Provider unreachable/overloaded/timed out after retries (eligible for fallback)."""


_MODELS = {"cerebras": "gpt-oss-120b", "groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash",
           "deepseek": "deepseek-flash"}


@dataclass(frozen=True)
class LLM1Config:
    provider: str = "cerebras"
    model: str = ""                # empty -> provider default (_MODELS)
    api_key: str = ""
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 3
    chunk_turns: int = 40          # turns per API call
    min_confidence: float = 0.70   # per-change acceptance threshold (our own guard)
    fallback_provider: str = "none"          # none | groq | cerebras | gemini
    fallback_model: str = ""   # empty -> provider default (_MODELS)
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_max_requests_per_run: int = 10   # safety guard counting every HTTP attempt
    openrouter_max_wait_s: float = 60.0         # if Retry-After exceeds this, fail instead of waiting
    openrouter_503_retries: int = 2
    openrouter_response_format: str = "auto"    # auto | json_schema | json_object | none
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_max_tokens: int = 8192             # must hold a full 40-turn chunk's JSON
    deepseek_retries: int = 3                   # bounded; capped by max_retries (0 when a fallback is set)
    deepseek_max_wait_s: float = 120.0          # Retry-After above this -> fail instead of waiting
    deepseek_thinking: str = "disabled"         # disabled | enabled | omit
    debug_raw_response: bool = False            # LLM1_DEBUG_RAW_RESPONSE: save per-chunk request/raw response

    @classmethod
    def from_env(cls) -> "LLM1Config":
        g = os.environ.get
        prov = (g("LLM1_PROVIDER") or "openrouter").lower()
        fb = (g("LLM1_FALLBACK_PROVIDER") or "none").lower()
        key = g("GEMINI_API_KEY") or g("GOOGLE_API_KEY") or ""
        return cls(
            provider=prov,
            model=g("LLM1_MODEL") or _MODELS.get(prov, ""),
            api_key=key,
            temperature=float(g("LLM1_TEMPERATURE") or 0.0),
            timeout_s=float(g("LLM1_TIMEOUT_S") or 120),
            max_retries=int(g("LLM1_MAX_RETRIES") or 3),
            chunk_turns=int(g("LLM1_CHUNK_TURNS") or 40),
            min_confidence=float(g("LLM1_MIN_CONFIDENCE") or 0.70),
            fallback_provider=fb,
            fallback_model=g("LLM1_FALLBACK_MODEL") or _MODELS.get(fb, ""),
            groq_api_key=g("GROQ_API_KEY") or "",
            cerebras_api_key=g("CEREBRAS_API_KEY") or "",
            openrouter_api_key=g("OPENROUTER_API_KEY") or "",
            openrouter_max_requests_per_run=int(g("OPENROUTER_MAX_REQUESTS_PER_RUN") or 10),
            openrouter_max_wait_s=float(g("OPENROUTER_MAX_RETRY_AFTER_S") or 60),
            openrouter_503_retries=int(g("OPENROUTER_503_RETRIES") or 2),
            openrouter_response_format=(g("OPENROUTER_RESPONSE_FORMAT") or "auto").lower(),
            deepseek_api_key=g("DEEPSEEK_API_KEY") or "",
            deepseek_base_url=g("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            deepseek_max_tokens=int(g("DEEPSEEK_MAX_TOKENS") or 8192),
            deepseek_retries=int(g("DEEPSEEK_MAX_RETRIES") or 3),
            deepseek_max_wait_s=float(g("DEEPSEEK_MAX_RETRY_AFTER_S") or 120),
            deepseek_thinking=(g("DEEPSEEK_THINKING") or "disabled").lower(),
            debug_raw_response=(g("LLM1_DEBUG_RAW_RESPONSE") or "").strip().lower() in ("1", "true", "yes", "on"),
        )