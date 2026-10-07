import json
from types import SimpleNamespace as NS

from llm1 import refine_file, refine_transcript
from llm1.config import LLM1Config
from llm1.deepseek_client import DeepSeekClient
from llm1.providers import FallbackClient

KEY = "sk-ds-SECRET-KEY-999"
TURNS = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "we deploy on cube netties today"},
         {"speaker": "SPEAKER_01", "start": 5.0, "end": 9.0, "text": "ok sounds good"}]
DATA = {"turns": TURNS, "speakers": ["SPEAKER_00", "SPEAKER_01"], "duration": 9.0}
GOOD = {"segments": [
    {"index": 0, "refined_text": "we deploy on Kubernetes today",
     "changes": [{"original": "cube netties", "refined": "Kubernetes", "reason": "ctx", "confidence": 0.95}]},
    {"index": 1, "refined_text": "ok sounds good", "changes": []}]}


def cfg(debug, **kw):
    return LLM1Config(provider="deepseek", model="m", deepseek_api_key=KEY, debug_raw_response=debug, **kw)


class FakeClient:
    """Stands in for a provider; mimics DeepSeekClient's diagnostics attributes."""
    def __init__(self, responses):
        self.responses, self.events = list(responses), []
        self.last_raw = self.last_request_params = None

    def generate_json(self, system_prompt, user_prompt):
        r = self.responses.pop(0)
        self.last_raw = {"content": json.dumps(r), "finish_reason": "stop", "model": "m", "usage": {}}
        self.last_request_params = {"model": "m", "temperature": 0.0}
        return r


def alltext(d):
    return "".join(p.read_text(encoding="utf-8") for p in d.rglob("*") if p.is_file())


def test_disabled_writes_no_debug_files_and_result_is_identical(tmp_path):
    d = tmp_path / "debug"
    off = refine_transcript(DATA, cfg(False), FakeClient([GOOD]), debug_dir=d)
    assert not d.exists()
    on = refine_transcript(DATA, cfg(True), FakeClient([GOOD]), debug_dir=tmp_path / "dbg2")
    assert off.model_dump() == on.model_dump()            # debug never alters the refinement
    assert off.changed_segments == 1 and off.segments[0].refined_text == "we deploy on Kubernetes today"


def test_enabled_writes_request_and_raw_with_analysis(tmp_path):
    d = tmp_path / "debug"
    refine_transcript(DATA, cfg(True), FakeClient([GOOD]), debug_dir=d)
    req = json.loads((d / "chunk_00_request.json").read_text(encoding="utf-8"))
    raw = json.loads((d / "chunk_00_raw.json").read_text(encoding="utf-8"))
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert "cube netties" in req["messages"][1]["content"] and req["requested_indices"] == [0, 1]
    assert raw["chunk_number"] == 0 and raw["requested_indices"] == [0, 1]
    assert json.loads(raw["raw_response"]["content"]) == GOOD and raw["parsed_json"] == GOOD
    assert raw["analysis"]["segments_where_model_text_differs"] == 1
    assert raw["analysis"]["total_itemised_changes"] == 1


def test_analysis_shows_model_returned_nothing_changed(tmp_path):
    same = {"segments": [{"index": i, "refined_text": t["text"], "changes": []} for i, t in enumerate(TURNS)]}
    refine_transcript(DATA, cfg(True), FakeClient([same]), debug_dir=tmp_path)
    a = json.loads((tmp_path / "chunk_00_raw.json").read_text(encoding="utf-8"))["analysis"]
    assert a["segments_where_model_text_differs"] == 0 and a["total_itemised_changes"] == 0


def test_retry_chunks_get_their_own_files(tmp_path):
    bad = {"segments": [{"index": 0, "refined_text": "x", "changes": []}]}      # index 1 missing
    ok0 = {"segments": [GOOD["segments"][0]]}
    ok1 = {"segments": [GOOD["segments"][1]]}
    refine_transcript(DATA, cfg(True), FakeClient([bad, ok0, ok1]), debug_dir=tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "chunk_00_raw.json" in names and "chunk_00_retry_0_raw.json" in names and "chunk_00_retry_1_request.json" in names


def test_api_key_never_written_even_if_model_echoes_it(tmp_path):
    echoed = {"segments": [{"index": 0, "refined_text": f"leak {KEY}", "changes": []},
                           {"index": 1, "refined_text": "ok sounds good", "changes": []}]}
    refine_transcript(DATA, cfg(True), FakeClient([echoed]), debug_dir=tmp_path)
    text = alltext(tmp_path)
    assert KEY not in text and "***" in text and "Authorization" not in text


def test_failed_call_is_still_recorded_and_error_propagates(tmp_path):
    class Boom(FakeClient):
        def generate_json(self, s, u):
            raise RuntimeError("network down")
    try:
        refine_transcript(DATA, cfg(True), Boom([]), debug_dir=tmp_path)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert (tmp_path / "chunk_00_request.json").exists()
    assert "network down" in (tmp_path / "chunk_00_raw.json").read_text(encoding="utf-8")


def test_refine_file_debug_dir_is_outdir_debug(tmp_path):
    src = tmp_path / "final_output.json"
    src.write_text(json.dumps(DATA), encoding="utf-8")
    refine_file(src, cfg=cfg(False), client=FakeClient([GOOD]))
    assert not (tmp_path / "debug").exists()
    refine_file(src, cfg=cfg(True), client=FakeClient([GOOD]))
    assert (tmp_path / "debug" / "chunk_00_raw.json").exists()


def test_env_flag(monkeypatch):
    monkeypatch.delenv("LLM1_DEBUG_RAW_RESPONSE", raising=False)
    assert LLM1Config.from_env().debug_raw_response is False
    monkeypatch.setenv("LLM1_DEBUG_RAW_RESPONSE", "false")
    assert LLM1Config.from_env().debug_raw_response is False
    monkeypatch.setenv("LLM1_DEBUG_RAW_RESPONSE", "true")
    assert LLM1Config.from_env().debug_raw_response is True


class _SDK:
    def __init__(self, content):
        resp = NS(choices=[NS(message=NS(content=content), finish_reason="stop")], model="ds-1",
                  usage=NS(prompt_tokens=1, completion_tokens=2, total_tokens=3))
        self.chat = NS(completions=NS(create=lambda **kw: resp))


def test_deepseek_client_behaviour_unchanged_and_exposes_raw_without_secrets():
    c = DeepSeekClient(cfg(False), "deepseek-flash", sdk=_SDK(json.dumps(GOOD)))
    assert c.generate_json("s", "u") == GOOD                       # same return value as before
    assert json.loads(c.last_raw["content"]) == GOOD and c.last_raw["finish_reason"] == "stop"
    assert "messages" not in c.last_request_params and KEY not in json.dumps(c.last_request_params)
    assert c.last_request_params["response_format"] == {"type": "json_object"}
    bad = DeepSeekClient(cfg(False), "deepseek-flash", sdk=_SDK("not json"))
    assert "__invalid_provider_output__" in bad.generate_json("s", "u")
    assert bad.last_raw["content"] == "not json"                    # raw text kept for diagnosis


def test_fallback_client_exposes_answering_clients_raw():
    prim = DeepSeekClient(cfg(False), "deepseek-flash", sdk=_SDK(json.dumps(GOOD)))
    fb = FallbackClient(prim, object(), "p", "f")
    fb.generate_json("s", "u")
    assert json.loads(fb.last_raw["content"]) == GOOD and fb.last_request_params["model"] == "deepseek-flash"