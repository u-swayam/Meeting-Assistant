"""Multilingual (Sarvam) support: routing, Sarvam parsing/errors (mocked), translation+refinement,
segment-id/metadata preservation, LLM2 input, evidence traceability, server bundle, pipeline wiring.
No test touches the network: Sarvam, DeepSeek and Deepgram are all faked."""
import json
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from common.errors import (STTConfigError, STTProviderError, STTRateLimitError, TranslationError,
                           UnsupportedLanguageError)
from stt import languages as L
from stt import sarvam as sv
from stt.routing import build_stt, resolve_route

HI = [
    {"start": 751.4, "end": 756.8, "speaker": "SPEAKER_01", "text": "हमें शुक्रवार तक deploy करना चाहिए।"},
    {"start": 757.0, "end": 760.0, "speaker": "SPEAKER_00", "text": "नहीं, हम 20 नवंबर से पहले release नहीं करेंगे।"},
    {"start": 761.0, "end": 764.0, "speaker": "SPEAKER_01", "text": "शायद हमें Kubernetes cluster बदलना पड़ेगा।"},
]
EN = ["We should deploy by Friday.", "No, we will not release before 20 November.",
      "We might have to change the Kubernetes cluster."]
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


class FakeLLM:
    """Stands in for the DeepSeek client: answers translation requests by index."""
    def __init__(self, texts=EN, mutate=None):
        self.texts, self.calls, self.events, self.mutate = texts, [], [], mutate

    def generate_json(self, system, user):
        self.calls.append((system, user))
        idx = [int(m) for m in re.findall(r"### index=(\d+)", user)]
        segs = [{"index": i, "refined_text": self.texts[i], "changes": []} for i in idx]
        return self.mutate(segs) if self.mutate else {"segments": segs}


def hindi_data():
    return {"speakers": ["SPEAKER_00", "SPEAKER_01"], "duration": 800.0, "warnings": [],
            "language": {"code": "hi-IN", "name": "Hindi", "stt_provider": "sarvam", "detected": False, "translated_to": "en"},
            "turns": [dict(t, confidence=1.0, has_overlap=False, flags=[], source_segment_ids=[i]) for i, t in enumerate(HI)]}


def cfg():
    from llm1.config import LLM1Config
    return LLM1Config(provider="deepseek", model="deepseek-flash", deepseek_api_key="k")


# ------------------------------------------------------------------ 1-3 routing
def test_english_routes_to_deepgram(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    for choice in ("en", "English", "en-US", None):
        r = resolve_route(choice)
        assert (r.provider, r.language_code, r.is_english) == ("deepgram", "en", True)
    from stt.deepgram_nova import DeepgramNovaSTT
    assert isinstance(build_stt(L.route_for_language("en")), DeepgramNovaSTT)


def test_hindi_routes_to_sarvam(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sv")
    for choice in ("hi-IN", "hi", "Hindi"):
        r = resolve_route(choice)
        assert (r.provider, r.language_code, r.is_english) == ("sarvam", "hi-IN", False)
    assert isinstance(build_stt(L.route_for_language("hi-IN")), sv.SarvamSTT)


def test_all_listed_indian_languages_route_to_sarvam():
    for code in L.SARVAM_LANGUAGES:
        assert L.route_for_language(code).provider == "sarvam"
    names = {x["name"] for x in L.supported_languages()}
    assert {"Auto Detect", "English", "Hindi", "Bengali", "Gujarati", "Kannada", "Malayalam", "Marathi",
            "Odia", "Punjabi", "Tamil", "Telugu", "Urdu"} <= names


def test_unsupported_language_fails_clearly(monkeypatch):
    with pytest.raises(UnsupportedLanguageError, match="Unsupported language 'klingon'"):
        L.route_for_language("klingon")
    monkeypatch.setenv("SARVAM_API_KEY", "sv")
    with pytest.raises(UnsupportedLanguageError):
        sv.SarvamSTT(language_code="fr-FR")


def test_missing_sarvam_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(STTConfigError, match="SARVAM_API_KEY"):
        sv.SarvamSTT(language_code="hi-IN")


class FakeDetector:
    def __init__(self, code): self.code = code
    def detect_language(self, p): return self.code


def test_auto_detect_routes_by_detected_language(tmp_path):
    a = tmp_path / "a.wav"
    assert resolve_route("auto", a, sarvam=FakeDetector("hi-IN")).provider == "sarvam"
    assert resolve_route("auto", a, sarvam=FakeDetector("hi-IN")).detected is True
    assert resolve_route("auto", a, sarvam=FakeDetector("en-IN")).provider == "deepgram"
    with pytest.raises(UnsupportedLanguageError):
        resolve_route("auto", a, sarvam=FakeDetector("fr-FR"))
    with pytest.raises(STTConfigError, match="Could not detect"):
        resolve_route("auto", a, sarvam=FakeDetector(None))


# ------------------------------------------------------------------ 4 Sarvam parsing / provider (mocked)
SARVAM_JSON = {"request_id": "r1", "transcript": " ".join(t["text"] for t in HI), "language_code": "hi-IN",
               "timestamps": {"chunks": [t["text"] for t in HI],
                              "start_time_seconds": [t["start"] for t in HI],
                              "end_time_seconds": [t["end"] for t in HI]}}


def test_parse_keeps_original_language_text_and_timestamps():
    r = sv.parse_sarvam_output(SARVAM_JSON, model="saaras:v3", audio_path="a.wav", duration=800.0, requested_language="hi-IN")
    assert r.provider == "sarvam" and r.language == "hi-IN"
    assert [s.text for s in r.segments] == [t["text"] for t in HI]
    assert [(s.start, s.end) for s in r.segments] == [(t["start"], t["end"]) for t in HI]
    assert all(DEVANAGARI.search(s.text) for s in r.segments) and r.raw_response[0]["response"] == SARVAM_JSON


def test_parse_malformed_and_empty():
    with pytest.raises(STTProviderError):
        sv.parse_sarvam_output({"nope": 1}, model="m", audio_path=None, duration=1.0, requested_language="hi-IN")
    bad = dict(SARVAM_JSON, timestamps={"chunks": ["a"], "start_time_seconds": [], "end_time_seconds": []})
    with pytest.raises(STTProviderError, match="Malformed"):
        sv.parse_sarvam_output(bad, model="m", audio_path=None, duration=1.0, requested_language="hi-IN")
    e = sv.parse_sarvam_output({"transcript": "", "timestamps": {}}, model="m", audio_path=None, duration=3.0, requested_language="hi-IN")
    assert e.segments == () and "No speech detected in audio." in e.warnings


class FakeJob:
    def __init__(self, data, fail=False, wait_exc=None):
        self.data, self.fail, self.wait_exc, self.uploaded = data, fail, wait_exc, None
    def upload_files(self, file_paths): self.uploaded = file_paths
    def start(self): pass
    def wait_until_complete(self, poll_interval, timeout):
        if self.wait_exc: raise self.wait_exc
    def get_file_results(self):
        return ({"successful": [], "failed": [{"file_name": "audio.wav", "error_message": "bad audio"}]} if self.fail
                else {"successful": [{"file_name": "audio.wav"}], "failed": []})
    def download_outputs(self, output_dir): (Path(output_dir) / "0.json").write_text(json.dumps(self.data), encoding="utf-8")


def fake_client(job, seen=None):
    def create_job(**kw):
        if seen is not None: seen.update(kw)
        return job
    return NS(speech_to_text_job=NS(create_job=create_job))


@pytest.fixture
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(sv, "validate_audio", lambda p: NS(path=Path(p), duration=800.0, size_bytes=1))
    monkeypatch.setattr(sv, "to_16k_mono", lambda src, dst, codec="flac": Path(dst).write_bytes(b"RIFF"))


def test_sarvam_transcribe_uses_batch_transcribe_mode_without_translation(no_ffmpeg):
    seen = {}
    s = sv.SarvamSTT(api_key="sv", language_code="hi-IN", client=fake_client(FakeJob(SARVAM_JSON), seen))
    r = s.transcribe("meeting.wav")
    assert seen["mode"] == "transcribe" and seen["language_code"] == "hi-IN" and seen["with_diarization"] is False
    assert r.language == "hi-IN" and len(r.segments) == 3 and DEVANAGARI.search(r.text)


def test_sarvam_api_failures_are_classified(no_ffmpeg):
    class Err(Exception):
        def __init__(self, st): super().__init__("boom sv-secret"); self.status_code = st
    for exc, kind in [(Err(401), STTConfigError), (Err(429), STTRateLimitError), (Err(500), STTProviderError),
                      (TimeoutError("slow"), STTProviderError)]:
        s = sv.SarvamSTT(api_key="sv-secret", language_code="hi-IN", client=fake_client(FakeJob({}, wait_exc=exc)))
        with pytest.raises(kind) as ei:
            s.transcribe("m.wav")
        assert "sv-secret" not in str(ei.value)
    s = sv.SarvamSTT(api_key="sv", language_code="hi-IN", client=fake_client(FakeJob({}, fail=True)))
    with pytest.raises(STTProviderError, match="bad audio"):
        s.transcribe("m.wav")


def test_sarvam_empty_and_unreadable_audio_are_rejected(tmp_path):
    s = sv.SarvamSTT(api_key="sv", language_code="hi-IN", client=fake_client(FakeJob({})))
    empty = tmp_path / "e.wav"; empty.write_bytes(b"")
    from common.errors import AudioError
    with pytest.raises(AudioError):
        s.transcribe(empty)
    junk = tmp_path / "j.wav"; junk.write_bytes(b"not audio at all")
    with pytest.raises(AudioError):
        s.transcribe(junk)


# ------------------------------------------------------------------ 5-8, 15 translation + refinement
def test_final_transcript_is_english_and_ids_speakers_times_unchanged():
    from llm1.translation import translate_transcript
    data = hindi_data()
    res = translate_transcript(data, "hi-IN", "Hindi", cfg(), FakeLLM())
    out = json.loads(res.model_dump_json())
    assert [s["index"] for s in out["segments"]] == [0, 1, 2]
    for i, s in enumerate(out["segments"]):
        t = HI[i]
        assert (s["speaker"], s["start"], s["end"]) == (t["speaker"], t["start"], t["end"])
        assert s["original_text"] == t["text"] and s["source_language"] == "hi-IN"       # original preserved
        assert s["refined_text"] == EN[i] and not DEVANAGARI.search(s["refined_text"])  # final is English
    assert out["segments"][0]["refined_text"] == "We should deploy by Friday."
    assert out["source_language"] == "hi-IN" and out["target_language"] == "en"
    assert json.dumps(data, sort_keys=True) == json.dumps(hindi_data(), sort_keys=True)  # input not mutated


def test_prompt_demands_negation_uncertainty_numbers_terms_and_no_invention():
    from llm1.translation import system_prompt, build_user_prompt
    p = system_prompt("Hindi").lower()
    for needle in ("negation", "uncertainty", "numbers", "technical terminology", "commitments", "must not",
                   "summarise", "invent", "same index", "we should probably deploy by friday"):
        assert needle in p, needle
    u = build_user_prompt([{"index": 7, "speaker": "S", "start": 1.0, "end": 2.0, "text": "x"}], 10, "Hindi")
    assert "index=7" in u


def test_guards_flag_lost_negation_uncertainty_numbers_and_script():
    from llm1.translation import check_translation as c
    assert "possible_negation_loss" in c(HI[1]["text"], "We will release before 20 November.")
    assert c(HI[1]["text"], EN[1]) == []
    assert "possible_uncertainty_loss" in c("शायद हमें शुक्रवार तक deploy करना चाहिए।", "We deploy Friday.")
    assert c("हमें शायद शुक्रवार तक deploy कर देना चाहिए।", "We should probably deploy by Friday.") == []
    assert any(f.startswith("number_not_preserved") for f in c("बजट २० लाख है", "The budget is large."))
    assert c("बजट २० लाख है", "The budget is 20 lakh.") == []                           # Indic digits normalised
    assert "not_english" in c("x", "हमें शुक्रवार तक")
    assert c("हमें Kubernetes cluster बदलना है", "We need to change the Kubernetes cluster.") == []


def test_flags_surface_as_warnings_not_silent_edits():
    from llm1.translation import translate_transcript
    bad = list(EN); bad[1] = "We will release before 20 November."
    res = translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(), FakeLLM(texts=bad))
    assert res.segments[1].refined_text == bad[1]                                      # text is never rewritten by us
    assert "possible_negation_loss" in res.segments[1].flags
    assert any("segment 1" in w and "possible_negation_loss" in w for w in res.warnings)


def test_malformed_response_ids_mismatch_and_failures_raise_translation_error():
    from llm1.translation import translate_transcript
    with pytest.raises(TranslationError, match="original transcript has been preserved"):
        translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(), FakeLLM(mutate=lambda s: {"oops": 1}))
    with pytest.raises(TranslationError, match="ids mismatch"):
        translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(),
                             FakeLLM(mutate=lambda s: {"segments": [dict(x, index=x["index"] + 100) for x in s]}))
    with pytest.raises(TranslationError, match="empty translation"):
        translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(),
                             FakeLLM(mutate=lambda s: {"segments": [dict(x, refined_text="") for x in s]}))

    class Boom:
        events = []
        def generate_json(self, s, u): raise TimeoutError("timed out")
    with pytest.raises(TranslationError, match="Translation failed: TimeoutError"):
        translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(), Boom())
    d = hindi_data(); d["turns"] = []
    with pytest.raises(TranslationError):
        translate_transcript(d, "hi-IN", "Hindi", cfg(), FakeLLM())


def test_retry_with_smaller_chunks_recovers():
    from llm1.translation import translate_transcript
    state = {"n": 0}
    def flaky(segs):
        state["n"] += 1
        return {"oops": 1} if state["n"] == 1 else {"segments": segs}
    res = translate_transcript(hindi_data(), "hi-IN", "Hindi", cfg(), FakeLLM(mutate=flaky))
    assert [s.refined_text for s in res.segments] == EN and any("retrying" in w for w in res.warnings)


def test_translate_file_writes_english_refined_output_and_keeps_original(tmp_path):
    from llm1.translation import translate_file
    fo = tmp_path / "final_output.json"
    fo.write_text(json.dumps(hindi_data(), ensure_ascii=False), encoding="utf-8")
    before = fo.read_text(encoding="utf-8")
    res, jp, tp = translate_file(fo, cfg=cfg(), client=FakeLLM())
    assert fo.read_text(encoding="utf-8") == before                                     # original transcript preserved
    out = json.loads(jp.read_text(encoding="utf-8"))
    assert out["segments"][0]["refined_text"] == EN[0] and out["segments"][0]["original_text"] == HI[0]["text"]
    assert not DEVANAGARI.search(tp.read_text(encoding="utf-8"))                       # user-facing txt is English only
    assert not any("translated_raw" in k or "raw_translation" in k for k in out)       # no separate translated-raw output
    assert not (tmp_path / "translated_raw_transcript.txt").exists()
    # LLM2 reads these exact keys
    assert {"speaker", "start", "end", "refined_text"} <= set(out["segments"][0])


def test_translate_file_refuses_english_transcripts(tmp_path):
    from llm1.translation import translate_file
    d = hindi_data(); d.pop("language")
    p = tmp_path / "final_output.json"; p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(TranslationError):
        translate_file(p, cfg=cfg(), client=FakeLLM())


# ------------------------------------------------------------------ 13 LLM2 receives English
def llm2_good():
    return {"summary": "The team discussed deploying by Friday.",
            "minutes": [{"topic": "Deployment", "discussion": "A deployment by Friday was suggested.", "segment_ids": [0]}],
            "decisions": [], "action_items": []}


def refined_dict(tmp_path):
    from llm1.translation import translate_file
    fo = tmp_path / "final_output.json"
    fo.write_text(json.dumps(hindi_data(), ensure_ascii=False), encoding="utf-8")
    _, jp, _ = translate_file(fo, cfg=cfg(), client=FakeLLM())
    return jp, json.loads(jp.read_text(encoding="utf-8"))


def test_llm2_receives_final_english_transcript_only(tmp_path):
    from llm2 import document_transcript
    _, refined = refined_dict(tmp_path)

    class Fake2:
        events, calls = [], []
        def generate_json(self, s, u): self.calls.append((s, u)); return llm2_good()
    c = Fake2()
    res = document_transcript(refined, cfg(), c)
    user = c.calls[0][1]
    assert "We should deploy by Friday." in user and not DEVANAGARI.search(user)
    assert res.total_segments == 3 and res.minutes[0].segment_ids == [0]


# ------------------------------------------------------------------ 14 evidence traceability
def test_evidence_traces_english_back_to_original_segment(tmp_path):
    from app.evidence import build_evidence
    _, refined = refined_dict(tmp_path)
    doc = {"minutes": [{"topic": "Deployment", "discussion": "Deploy by Friday.", "segment_ids": [0]}],
           "decisions": [{"decision": "Deploy Friday", "segment_ids": [0, 1]}], "action_items": [], "warnings": []}
    ev = build_evidence(refined, doc)
    seg = ev["items"][0]["segments"][0]
    assert (seg["id"], seg["speaker"], seg["start"], seg["end"]) == (0, "SPEAKER_01", 751.4, 756.8)
    assert seg["original_text"] == "हमें शुक्रवार तक deploy करना चाहिए।"
    assert seg["refined_text"] == "We should deploy by Friday."
    assert seg["translated"] is True and seg["source_language"] == "hi-IN"


def test_english_evidence_is_unchanged():
    from app.evidence import build_evidence
    refined = {"segments": [{"speaker": "A", "start": 0.0, "end": 1.0, "original_text": "hi", "refined_text": "hi",
                             "changed": False, "changes": []}]}
    doc = {"minutes": [{"topic": "t", "discussion": "d", "segment_ids": [0]}], "decisions": [], "action_items": []}
    seg = build_evidence(refined, doc)["items"][0]["segments"][0]
    assert "translated" not in seg and "source_language" not in seg


# ------------------------------------------------------------------ 6, 17 server / UI
def test_server_bundle_exposes_only_final_english_transcript(tmp_path, monkeypatch):
    from app import server
    d = tmp_path / "hi_meeting"; d.mkdir()
    (d / "final_output.json").write_text(json.dumps(hindi_data(), ensure_ascii=False), encoding="utf-8")
    from llm1.translation import translate_file
    translate_file(d / "final_output.json", cfg=cfg(), client=FakeLLM())
    monkeypatch.setattr(server, "OUTPUTS", tmp_path)
    b = server.bundle("hi_meeting")
    assert b["translation"]["source_language_name"] == "Hindi" and b["language"]["code"] == "hi-IN"
    assert [s["refined_text"] for s in b["refined"]["segments"]] == EN
    assert not any("translated_raw" in k for k in b)
    assert b["raw_aligned_with_refined"] is True                                       # original_text == raw turn text


def test_server_language_validation_and_list():
    from stt.languages import supported_languages
    assert supported_languages()[0]["code"] == "auto" and supported_languages()[1]["code"] == "en"


def test_ui_labels_translated_transcript_not_translated_raw():
    html = (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Refined (English)" in html and "['raw','Raw ('+L+')']" in html and "langSel" in html and "X-Language" in html
    assert "Translated Raw" not in html and "Raw English" not in html


# ------------------------------------------------------------------ pipeline wiring (all external calls faked)
def _install_fakes(monkeypatch, tmp_path, calls, stt_text_lang):
    from alignment import align as _a  # noqa: F401  (real alignment is used)
    from diarization.schemas import DiarSegment, DiarizationResult
    from stt.schemas import STTResult, STTSegment

    class FakeDiar:
        def diarize(self, path, num_speakers=None):
            segs = tuple(DiarSegment(start=t["start"], end=t["end"], speaker=t["speaker"]) for t in HI)
            return DiarizationResult(segments=segs, exclusive_segments=segs, speakers=("SPEAKER_00", "SPEAKER_01"))
    monkeypatch.setitem(sys.modules, "diarization.diarizer", types.SimpleNamespace(PyannoteDiarizer=FakeDiar))

    texts = [t["text"] for t in HI] if stt_text_lang == "hi" else ["We should deploy by Friday.", "OK.", "Fine."]

    class FakeSTT:
        def __init__(self, name): self.name = name
        def transcribe(self, path):
            calls.append(("stt", self.name))
            segs = tuple(STTSegment(id=i, start=HI[i]["start"], end=HI[i]["end"], text=texts[i]) for i in range(3))
            return STTResult(text=" ".join(texts), language=stt_text_lang, duration=800.0, segments=segs, provider=self.name)
    import stt.routing as routing
    monkeypatch.setattr(routing, "build_stt", lambda route, **kw: FakeSTT(route.provider))
    import llm1, llm1.translation as tr, transformation, llm2
    monkeypatch.setattr(tr, "build_client", lambda c: FakeLLM())
    monkeypatch.setattr(tr.LLM1Config, "from_env", classmethod(lambda cls: cfg()))
    real_translate = tr.translate_file
    monkeypatch.setattr(llm1, "translate_file", lambda p: calls.append(("translate",)) or real_translate(p))
    monkeypatch.setattr(llm1, "refine_file", lambda p: calls.append(("refine",)) or
                        (Path(p).parent / "refined_output.json").write_text(json.dumps({"segments": []}), encoding="utf-8"))
    monkeypatch.setattr(transformation, "transform_file", lambda p: calls.append(("transform", Path(p).read_text(encoding="utf-8"))))
    monkeypatch.setattr(llm2, "document_file", lambda p: (calls.append(("llm2", Path(p).read_text(encoding="utf-8"))),
                        (Path(p).parent / "documentation_output.json").write_text(json.dumps(
                            {"summary": "s", "minutes": [], "decisions": [], "action_items": []}), encoding="utf-8")))


def _run(monkeypatch, tmp_path, language, stt_lang):
    from app import pipeline_runner as pr
    calls = []
    _install_fakes(monkeypatch, tmp_path, calls, stt_lang)
    job = pr.new_job("m")
    outdir = tmp_path / "out"
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    pr.run_pipeline(job, tmp_path / "a.wav", outdir, language=language)
    for _ in range(300):
        if job["done"]: break
        time.sleep(0.02)
    return job, calls, outdir


def test_pipeline_hindi_goes_sarvam_translate_then_llm2_with_english(tmp_path, monkeypatch):
    job, calls, out = _run(monkeypatch, tmp_path, "hi-IN", "hi")
    assert job["error"] is None, job
    assert ("stt", "sarvam") in calls and ("refine",) not in calls and ("translate",) in calls
    fo = json.loads((out / "final_output.json").read_text(encoding="utf-8"))
    assert fo["language"]["code"] == "hi-IN" and DEVANAGARI.search(fo["turns"][0]["text"])  # original preserved
    ref = json.loads((out / "refined_output.json").read_text(encoding="utf-8"))
    assert [s["index"] for s in ref["segments"]] == list(range(len(ref["segments"])))
    assert ref["segments"][0]["refined_text"] == EN[0] and ref["segments"][0]["original_text"] == HI[0]["text"]
    llm2_in = next(c for c in calls if c[0] == "llm2")[1]
    assert "We should deploy by Friday." in llm2_in and job["stages"][3]["name"] == "Translation & Refinement"


def test_pipeline_english_goes_deepgram_and_existing_refine(tmp_path, monkeypatch):
    job, calls, out = _run(monkeypatch, tmp_path, "en", "en")
    assert job["error"] is None, job
    assert ("stt", "deepgram") in calls and ("refine",) in calls and ("translate",) not in calls
    assert "language" not in json.loads((out / "final_output.json").read_text(encoding="utf-8"))   # schema unchanged
    assert job["stages"][3]["name"] == "LLM1 Refinement"


def test_pipeline_unsupported_language_shows_clear_error(tmp_path, monkeypatch):
    job, calls, _ = _run(monkeypatch, tmp_path, "klingon", "hi")
    assert "Unsupported language" in (job["error"] or "") and not calls


def test_failures_are_friendly_for_multilingual_errors():
    from app.failures import classify
    f = classify("refinement", TranslationError("segment ids mismatch"))
    assert f["message"].startswith("Translation failed. The original transcript has been preserved.")
    assert "Unsupported" in classify("transcription", UnsupportedLanguageError("Unsupported language 'x'."))["message"]
    assert "Sarvam" in classify("transcription", STTConfigError("SARVAM_API_KEY is not set"))["message"]


def test_pipeline_writes_standalone_downloads_and_serves_them(tmp_path, monkeypatch):
    job, calls, out = _run(monkeypatch, tmp_path, "hi-IN", "hi")
    assert job["error"] is None, job
    from app.downloads import read_artifact
    body, ctype, name = read_artifact(out, "refined_transcript")
    assert "We should deploy by Friday." in body.decode() and not DEVANAGARI.search(body.decode())
    assert read_artifact(out, "action_items")[2] == "action_items.txt"
    assert read_artifact(out, "raw_transcript")[0].decode()                   # original-language audit text


# ---- aggregate metadata is derived from the segments, never left at []/0.0 ----
def test_translation_metadata_derived_from_segments():
    from llm1.schemas import derive_meta
    turns = [{"speaker": "Speaker 01", "start": 0.0, "end": 7.1, "text": "a"}, {"speaker": "Speaker 02", "start": 7.1, "end": 20.5, "text": "b"},
             {"speaker": "", "start": 20.5, "end": 9.0, "text": "c"}, {"speaker": "Speaker 01", "start": 21.0, "end": 1224.0, "text": "d"}]
    assert derive_meta(turns, [], 0.0) == (["Speaker 01", "Speaker 02"], 1224.0)
    assert derive_meta(turns, ["X"], 5.0) == (["X"], 5.0)            # meaningful supplied values win
    assert derive_meta([], None, None) == ([], 0.0)


def test_translate_transcript_fills_speakers_and_duration_when_final_output_has_none():
    from llm1.translation import translate_transcript
    from llm1.config import LLM1Config

    class C:
        events = []
        def generate_json(self, s, u):
            return {"segments": [{"index": i, "refined_text": f"English {i}", "changes": []} for i in (0, 1)]}
    data = {"turns": [{"speaker": "Speaker 01", "start": 0.0, "end": 7.1, "text": "नमस्ते"},
                      {"speaker": "Speaker 02", "start": 7.1, "end": 15.0, "text": "ठीक है"}], "language": {"code": "hi-IN"}}
    r = translate_transcript(data, "hi-IN", "Hindi", LLM1Config(provider="deepseek", model="deepseek-flash"), C())
    assert r.speakers == ["Speaker 01", "Speaker 02"] and r.duration == 15.0
    assert [s.index for s in r.segments] == [0, 1] and r.segments[0].original_text == "नमस्ते" and r.segments[0].source_language == "hi-IN"
    assert "segments" in r.model_dump() and "turns" not in r.model_dump()


def test_ui_derives_metadata_and_does_not_hide_translated_segments():
    from pathlib import Path
    h = (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "function meta()" in h and "r?.duration??raw?.duration" not in h
    assert "S.onlyChanged&&!isTr()&&!s.changed" in h        # sticky 'changed only' must not blank a translated transcript
    assert "B.refined?.segments" in h


def test_documentation_detected_from_file_not_name(tmp_path, monkeypatch):
    import app.server as srv
    d = tmp_path / "anything"; d.mkdir(); (d / "refined_output.json").write_text("{}"); (d / "documentation_output.json").write_text("{}")
    monkeypatch.setattr(srv, "OUTPUTS", tmp_path)
    assert srv.list_meetings()[0]["has_documentation"] is True


# ------------------------------------------------------------------ regression: Hindi "मत" must be a whole word
def test_hindi_sahmat_is_not_negation_but_standalone_mat_is():
    from llm1.translation import check_translation as c
    assert "possible_negation_loss" not in c("मैं सहमत हूं।", "I agree.")                 # "सहमत" (agree) merely contains "मत"
    assert "possible_negation_loss" not in c("हम सब सहमत हैं।", "We all agree.")
    assert "possible_negation_loss" in c("मत करो", "Do it.")                                # standalone negation still detected
    assert "possible_negation_loss" in c("secrets मत इस्तेमाल करें।", "Use the secrets.")
    assert "possible_negation_loss" in c("यह मत करो।", "Do this.")                          # danda right after the word
    assert "possible_negation_loss" in c("हम इसे इस्तेमाल नहीं करेंगे।", "We will use it.")  # real negative statement
    assert c("मत करो", "Do not do it.") == []


# ------------------------------------------------------------------ regression: transcript fixtures never report an STT provider
def test_transcript_fixture_does_not_report_sarvam_as_stt_provider(tmp_path):
    from llm1.translation import translate_file
    d = hindi_data()
    d["language"] = {"code": "hi-IN", "name": "Hindi", "detected": False, "translated_to": "en"}   # what run_from_transcript.py writes: no STT step
    fo = tmp_path / "final_output.json"
    fo.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    res, jp, _ = translate_file(fo, cfg=cfg(), client=FakeLLM())
    assert res.stt_provider != "sarvam" and not res.stt_provider
    assert json.loads(jp.read_text(encoding="utf-8")).get("stt_provider") != "sarvam"


def test_real_audio_language_block_still_records_its_stt_provider(tmp_path):
    from llm1.translation import translate_file
    fo = tmp_path / "final_output.json"
    fo.write_text(json.dumps(hindi_data(), ensure_ascii=False), encoding="utf-8")                 # real Hindi audio route: sarvam
    assert translate_file(fo, cfg=cfg(), client=FakeLLM())[0].stt_provider == "sarvam"


def test_run_from_transcript_script_does_not_stamp_sarvam():
    src = (Path(__file__).parent / "run_from_transcript.py").read_text(encoding="utf-8")
    assert '"stt_provider"' not in src
