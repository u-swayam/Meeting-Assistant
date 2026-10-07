"""Official DeepSeek API client (OpenAI-compatible SDK, NOT via OpenRouter).

Same interface as the other providers: generate_json(system_prompt, user_prompt) -> dict.
Never logs/echoes the API key. Requests are sequential. Retries are bounded.
Malformed/truncated JSON is NOT raised: it is returned as a dict that fails ModelResponse validation,
so the refiner's existing safe path (retry smaller, then keep original + warning) handles it.
"""
from __future__ import annotations
import json
import random
import threading
import time

from llm1.config import LLM1Config, LLM1ConfigError, LLM1ProviderError, LLM1UnavailableError

_RETRYABLE = {408, 429, 500, 502, 503, 504, 599}
_HINTS = {
    400: "bad request (check model name / parameters)",
    401: "authentication failed (check DEEPSEEK_API_KEY)",
    402: "insufficient balance (top up the DeepSeek account)",
    403: "forbidden",
}


def _new_sdk(cfg: LLM1Config):
    try:
        from openai import OpenAI
    except ImportError as e:   # lazy so other providers work without the package
        raise LLM1ConfigError("The 'openai' package is required for DeepSeek: pip install openai") from e
    # max_retries=0: the SDK's hidden retries would bypass our bounded retry/backoff policy.
    return OpenAI(api_key=cfg.deepseek_api_key, base_url=cfg.deepseek_base_url,
                  timeout=cfg.timeout_s, max_retries=0)


class DeepSeekClient:
    name = "deepseek"

    def __init__(self, cfg: LLM1Config, model: str, sdk=None):
        if not cfg.deepseek_api_key:
            raise LLM1ConfigError("DEEPSEEK_API_KEY is not set (put it in .env).")
        if not model:
            raise LLM1ConfigError("LLM1_MODEL is not set.")
        self.cfg, self.model = cfg, model
        self.sdk = sdk if sdk is not None else _new_sdk(cfg)
        self.events: list[str] = []
        self.actual_model: str | None = None
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.requests_made = 0
        self.last_raw: dict | None = None            # diagnostics only: last raw content/finish_reason/model/usage
        self.last_request_params: dict | None = None  # diagnostics only: request minus messages (no secrets)
        self._lock = threading.Lock()
        self._sleep, self._rand = time.sleep, random.random      # injectable for tests
        # With a configured fallback the caller sets max_retries=0 => fail over immediately.
        self._retries = min(cfg.deepseek_retries, cfg.max_retries)

    # -- helpers -----------------------------------------------------------
    def _scrub(self, text) -> str:
        t = str(text or "").replace("\n", " ")[:300]
        key = self.cfg.deepseek_api_key
        return t.replace(key, "***") if key else t

    @staticmethod
    def _status(exc: Exception) -> int | None:
        st = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(st, int):
            return st
        if isinstance(exc, (TimeoutError, ConnectionError)) or \
                type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "Timeout", "ConnectError"):
            return 599
        return None

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        try:
            h = exc.response.headers  # type: ignore[attr-defined]
            return max(float(h.get("retry-after")), 0.0)
        except (AttributeError, TypeError, ValueError):
            return None

    def _backoff(self, n: int) -> float:
        base = min(4.0 * (2 ** n), 60.0)
        return base / 2 + self._rand() * base / 2        # equal jitter, never ~0

    def _kwargs(self, system_prompt: str, user_prompt: str) -> dict:
        kw = {"model": self.model,
              "messages": [{"role": "system", "content": system_prompt},
                           {"role": "user", "content": user_prompt}],
              "response_format": {"type": "json_object"},
              "temperature": self.cfg.temperature,
              "max_tokens": self.cfg.deepseek_max_tokens}
        if self.cfg.deepseek_thinking != "omit":
            kw["extra_body"] = {"thinking": {"type": self.cfg.deepseek_thinking}}   # "disabled" by default
        return kw

    def _invalid(self, why: str) -> dict:
        self.events.append(f"deepseek: {why}")
        return {"__invalid_provider_output__": why}       # fails ModelResponse validation by design

    def _parse(self, resp) -> dict:
        try:
            choice = resp.choices[0]
            content, finish = choice.message.content, getattr(choice, "finish_reason", None)
            self.last_raw = {"content": content, "finish_reason": finish, "model": getattr(resp, "model", None),
                             "usage": {k: getattr(getattr(resp, "usage", None), k, None) for k in self.usage_totals}}
        except (AttributeError, IndexError, TypeError):
            return self._invalid("response had no choices/content")
        m = getattr(resp, "model", None)
        if isinstance(m, str):
            self.actual_model = m
        u = getattr(resp, "usage", None)
        for k in self.usage_totals:
            v = getattr(u, k, None)
            if isinstance(v, int):
                self.usage_totals[k] += v
        if finish == "length":
            return self._invalid(f"output truncated at max_tokens={self.cfg.deepseek_max_tokens}; "
                                 f"raise DEEPSEEK_MAX_TOKENS or lower LLM1_CHUNK_TURNS")
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
        except ValueError:
            return self._invalid(f"output was not valid JSON (finish_reason={finish})")
        return data if isinstance(data, dict) else self._invalid("JSON output was not an object")

    # -- public ------------------------------------------------------------
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        with self._lock:
            return self._generate(system_prompt, user_prompt)

    def _generate(self, system_prompt: str, user_prompt: str) -> dict:
        attempt = 0
        while True:
            try:
                self.requests_made += 1
                self.last_raw = None
                kw = self._kwargs(system_prompt, user_prompt)
                self.last_request_params = {k: v for k, v in kw.items() if k != "messages"}
                resp = self.sdk.chat.completions.create(**kw)
                return self._parse(resp)
            except Exception as e:  # noqa: BLE001 - classified below
                status = self._status(e)
                safe = self._scrub(e)
                if status is None:
                    raise LLM1ProviderError(f"deepseek/{self.model} unexpected error "
                                            f"({type(e).__name__}): {safe}") from None
                if status not in _RETRYABLE:
                    hint = _HINTS.get(status, "request rejected")
                    raise LLM1ProviderError(f"deepseek/{self.model} HTTP {status}: {hint}: {safe}") from None
                if attempt >= self._retries:
                    self.events.append(f"deepseek HTTP {status}: giving up after {attempt} retr{'y' if attempt == 1 else 'ies'}")
                    raise LLM1UnavailableError(
                        f"deepseek/{self.model} unavailable (HTTP {status}) after {attempt} retries: {safe}") from None
                ra = self._retry_after(e)
                if ra is not None:
                    if ra > self.cfg.deepseek_max_wait_s:
                        self.events.append(f"deepseek HTTP {status}: server asked to wait {ra:.0f}s; not retrying")
                        raise LLM1UnavailableError(
                            f"deepseek/{self.model} HTTP {status}: retry after {ra:.0f}s exceeds limit") from None
                    wait = ra                                   # exactly what the server asked
                else:
                    wait = self._backoff(attempt)
                attempt += 1
                self.events.append(f"deepseek HTTP {status}: retry {attempt}/{self._retries} in {wait:.1f}s")
                self._sleep(wait)