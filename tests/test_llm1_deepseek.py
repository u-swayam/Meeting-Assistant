import json
from types import SimpleNamespace as NS

import pytest

from llm1 import deepseek_client as dc
from llm1.config import LLM1Config, LLM1ConfigError, LLM1ProviderError, LLM1UnavailableError
from llm1.deepseek_client import DeepSeekClient
from llm1.prompts import SYSTEM_PROMPT
from llm1.providers import build_client
from llm1.schemas import ModelResponse

KEY = "sk-ds-SECRET-KEY-999"
GOOD = {"segments": [{"index": 0, "refined_text": "hi", "changes": []}]}


def resp(content=None, finish="stop", model="deepseek-flash-0001"):
    content = json.dumps(GOOD) if content is None else content
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish)], model=model,
              usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15))


class APIErr(Exception):
    def __init__(self, status, headers=None, msg="err"):
        super().__init__(msg)
        self.status_code = status
        self.response = NS(headers=headers or {}, status_code=status)


class FakeSDK:
    def __init__(self, seq):
        self.seq, self.calls = list(seq), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        x = self.seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


def mk(seq, **cfg):
    sdk = FakeSDK(seq)
    c = DeepSeekClient(LLM1Config(provider="deepseek", deepseek_api_key=KEY, **cfg), "deepseek-flash", sdk=sdk)
    sleeps = []
    c._sleep, c._rand = sleeps.append, lambda: 0.5
    return c, sdk, sleeps


def test_sdk_built_with_official_base_url_and_key_from_config(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw): seen.update(kw)

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    DeepSeekClient(LLM1Config(deepseek_api_key=KEY), "deepseek-flash")
    assert seen["base_url"] == "https://api.deepseek.com" and seen["api_key"] == KEY
    assert seen["max_retries"] == 0                      # SDK hidden retries disabled


def test_request_shape_model_json_mode_messages_thinking_disabled():
    c, sdk, _ = mk([resp()])
    out = c.generate_json(SYSTEM_PROMPT, "USER PROMPT")
    kw = sdk.calls[0]
    assert kw["model"] == "deepseek-flash"
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["messages"] == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "USER PROMPT"}]
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kw["temperature"] == 0.0 and kw["max_tokens"] == 8192
    assert "provider" not in kw and KEY not in json.dumps(kw, default=str)
    ModelResponse.model_validate(out)
    assert c.actual_model == "deepseek-flash-0001" and c.usage_totals["total_tokens"] == 15


def test_thinking_can_be_omitted():
    c, sdk, _ = mk([resp()], deepseek_thinking="omit")
    c.generate_json("s", "u")
    assert "extra_body" not in sdk.calls[0]


@pytest.mark.parametrize("bad", ["not json at all", "[1,2]", ""])
def test_malformed_json_returns_invalid_dict_not_exception(bad):
    c, sdk, _ = mk([resp(bad)])
    out = c.generate_json("s", "u")
    with pytest.raises(Exception):
        ModelResponse.model_validate(out)               # refiner's schema-invalid path
    assert len(sdk.calls) == 1 and c.events


def test_truncated_output_flagged():
    c, _, _ = mk([resp('{"segments": [', finish="length")])
    out = c.generate_json("s", "u")
    assert "__invalid_provider_output__" in out and any("truncated" in e for e in c.events)


def test_malformed_json_keeps_original_via_refiner():
    from llm1.refiner import refine_transcript
    c, _, _ = mk([resp("garbage")] * 2)
    r = refine_transcript({"turns": [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "hello"}]}, LLM1Config(), c)
    assert r.segments[0].refined_text == "hello" and r.total_changes == 0
    assert any("schema-invalid" in w or "kept original" in w for w in r.warnings)


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_non_retryable_errors_not_retried(status):
    c, sdk, sleeps = mk([APIErr(status), resp()])
    with pytest.raises(LLM1ProviderError, match=f"HTTP {status}"):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 1 and sleeps == []


def test_429_bounded_and_respects_retry_after():
    c, sdk, sleeps = mk([APIErr(429, {"retry-after": "7"}), APIErr(429, {"retry-after": "7"}), resp()])
    assert c.generate_json("s", "u") == GOOD
    assert sleeps == [7.0, 7.0] and len(sdk.calls) == 3
    c, sdk, sleeps = mk([APIErr(429, {"retry-after": "1"})] * 10)
    with pytest.raises(LLM1UnavailableError):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 4                           # 1 + 3 retries, never infinite


def test_429_retry_after_over_limit_fails_without_waiting():
    c, sdk, sleeps = mk([APIErr(429, {"retry-after": "9999"}), resp()])
    with pytest.raises(LLM1UnavailableError, match="exceeds limit"):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 1 and sleeps == []


def test_429_without_retry_after_uses_backoff_not_fixed_60():
    c, sdk, sleeps = mk([APIErr(429), resp()])
    c.generate_json("s", "u")
    assert sleeps == [3.0]                               # base 4 equal-jitter, rand=.5


def test_5xx_exponential_backoff_bounded():
    c, sdk, sleeps = mk([APIErr(503), APIErr(502), APIErr(500), resp()])
    assert c.generate_json("s", "u") == GOOD
    assert sleeps == [3.0, 6.0, 12.0]
    c, sdk, sleeps = mk([APIErr(500)] * 10)
    with pytest.raises(LLM1UnavailableError):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 4


def test_timeout_and_network_errors_retried_then_unavailable():
    class APITimeoutError(Exception): pass
    c, sdk, _ = mk([APITimeoutError("timed out"), TimeoutError("t"), ConnectionError("c"), APIErr(408)])
    with pytest.raises(LLM1UnavailableError):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 4


def test_no_retries_when_fallback_configured():
    c, sdk, _ = mk([APIErr(503), resp()], max_retries=0)
    with pytest.raises(LLM1UnavailableError):
        c.generate_json("s", "u")
    assert len(sdk.calls) == 1


def test_api_key_never_in_errors_or_events():
    leaky = f"Incorrect API key provided: {KEY}"
    for status in (401, 503):
        c, sdk, _ = mk([APIErr(status, msg=leaky)] * 5)
        with pytest.raises((LLM1ProviderError, LLM1UnavailableError)) as e:
            c.generate_json("s", "u")
        assert KEY not in str(e.value) and KEY not in " ".join(c.events)


def test_unexpected_exception_not_retried_and_scrubbed():
    c, sdk, _ = mk([ValueError(f"boom {KEY}"), resp()])
    with pytest.raises(LLM1ProviderError) as e:
        c.generate_json("s", "u")
    assert KEY not in str(e.value) and len(sdk.calls) == 1


def test_config_and_provider_selection_from_env(monkeypatch):
    for k in ("LLM1_PROVIDER", "LLM1_MODEL", "DEEPSEEK_API_KEY", "LLM1_FALLBACK_PROVIDER", "DEEPSEEK_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=lambda **kw: object()))
    monkeypatch.setenv("LLM1_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    cfg = LLM1Config.from_env()
    assert cfg.model == "deepseek-flash" and cfg.deepseek_base_url == "https://api.deepseek.com"
    c = build_client(cfg)
    assert isinstance(c, DeepSeekClient) and c.model == "deepseek-flash"
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "12000")
    assert LLM1Config.from_env().deepseek_max_tokens == 12000
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(LLM1ConfigError, match="DEEPSEEK_API_KEY"):
        build_client(LLM1Config.from_env())


def test_65_segments_two_sequential_requests():
    import re
    from llm1.refiner import refine_transcript
    turns = [{"start": float(i), "end": i + .5, "speaker": "A", "text": f"t{i}"} for i in range(65)]
    seen = []

    class SDK(FakeSDK):
        def _create(self, **kw):
            idx = [int(x) for x in re.findall(r"index=(\d+)", kw["messages"][1]["content"])]
            seen.append((idx[0], idx[-1]))
            return resp(json.dumps({"segments": [{"index": i, "refined_text": f"t{i}", "changes": []} for i in idx]}))

    c = DeepSeekClient(LLM1Config(deepseek_api_key=KEY, chunk_turns=40), "deepseek-flash", sdk=SDK([]))
    r = refine_transcript({"turns": turns}, LLM1Config(chunk_turns=40), c)
    assert seen == [(0, 39), (40, 64)] and r.total_changes == 0