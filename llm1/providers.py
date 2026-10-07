from __future__ import annotations

import dataclasses

from llm1.config import _MODELS, LLM1Config, LLM1ConfigError, LLM1UnavailableError
from llm1.deepseek_client import DeepSeekClient


class FallbackClient:
    """Try the primary provider; if it is unavailable (503/429/timeout/network), use the fallback
    for the rest of the run. Auth/config/bad-request errors on the primary are NOT masked.
    `events` collects notes that the refiner adds to the output warnings."""

    def __init__(self, primary, fallback, primary_name: str, fallback_name: str):
        self.primary, self.fallback = primary, fallback
        self.primary_name, self.fallback_name = primary_name, fallback_name
        self.events: list[str] = []
        self.primary_down = False   # once True, primary is skipped for the rest of the run
        self._used = primary        # which client answered last (diagnostics only)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.primary_down:
            try:
                self._used = self.primary
                return self.primary.generate_json(system_prompt, user_prompt)
            except LLM1UnavailableError as e:
                self.primary_down = True
                self.events.append(f"{self.primary_name} unavailable ({str(e)[:120]}); "
                                   f"using fallback {self.fallback_name} for the rest of this run")
        self._used = self.fallback
        return self.fallback.generate_json(system_prompt, user_prompt)

    @property
    def last_raw(self):
        return getattr(self._used, "last_raw", None)

    @property
    def last_request_params(self):
        return getattr(self._used, "last_request_params", None)


def _make(provider: str, cfg: LLM1Config, model: str):
    model = model or _MODELS.get(provider, "")
    if provider == "deepseek":
        return DeepSeekClient(cfg, model)

    raise LLM1ConfigError(f"Unsupported provider '{provider}' (use deepseek, openrouter, cerebras, groq or gemini).")


def build_client(cfg: LLM1Config):
    fb = cfg.fallback_provider
    if fb in ("", "none", "off") or fb == cfg.provider:
        return _make(cfg.provider, cfg, cfg.model)
    primary = _make(cfg.provider, dataclasses.replace(cfg, max_retries=0), cfg.model)
    fallback = _make(fb, cfg, cfg.fallback_model)
    return FallbackClient(primary, fallback, f"{cfg.provider}/{primary.model if hasattr(primary,'model') else cfg.model}",
                          f"{fb}/{fallback.model if hasattr(fallback,'model') else cfg.fallback_model}")