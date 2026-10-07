import json
import subprocess
import sys
from pathlib import Path

from common.errors import PipelineError
from llm1.config import LLM1Config
from llm1.refiner import refine_file as real_refine
from tests import run_llm1

DATA = {"speakers": ["A"], "duration": 5.0, "warnings": [],
        "turns": [{"start": 0.0, "end": 2.0, "speaker": "A", "text": "the wider em are rate"}]}


class Fake:
    def generate_json(self, s, u):
        return {"segments": [{"index": 0, "refined_text": "the wider MR rate",
                              "changes": [{"original": "em are", "refined": "MR", "reason": "term", "confidence": 0.95}]}]}


def make_dir(tmp_path):
    d = tmp_path / "out"; d.mkdir()
    (d / "final_output.json").write_text(json.dumps(DATA))
    (d / "raw_transcript.txt").write_text("raw")
    return d


def test_usage_error(capsys):
    assert run_llm1.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_missing_dir_and_missing_input(tmp_path, capsys):
    assert run_llm1.main([str(tmp_path / "nope")]) == 2
    assert "directory not found" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert run_llm1.main([str(tmp_path / "empty")]) == 2
    assert "input file not found" in capsys.readouterr().err


def test_success_prints_path_and_stats_and_writes_file(tmp_path, monkeypatch, capsys):
    d = make_dir(tmp_path)
    monkeypatch.setattr(run_llm1, "refine_file", lambda p: real_refine(p, cfg=LLM1Config(provider="openrouter", model="m/x"), client=Fake()))
    assert run_llm1.main([str(d)]) == 0
    o = capsys.readouterr().out
    assert "refined_output.json" in o and "total_changes=1" in o and "'em are' -> 'MR'" in o
    assert (d / "refined_output.json").is_file()
    assert (d / "raw_transcript.txt").read_text() == "raw"


def test_pipeline_error_exit_nonzero(tmp_path, monkeypatch, capsys):
    d = make_dir(tmp_path)
    def boom(p): raise PipelineError("provider down")
    monkeypatch.setattr(run_llm1, "refine_file", boom)
    assert run_llm1.main([str(d)]) == 1
    assert "provider down" in capsys.readouterr().err


def test_unexpected_exception_is_loud(tmp_path, monkeypatch, capsys):
    d = make_dir(tmp_path)
    def boom(p): raise ValueError("kaboom")
    monkeypatch.setattr(run_llm1, "refine_file", boom)
    assert run_llm1.main([str(d)]) == 1
    e = capsys.readouterr().err
    assert "UNEXPECTED ERROR (ValueError)" in e and "Traceback" in e


def test_stale_output_detected(tmp_path, monkeypatch, capsys):
    d = make_dir(tmp_path)
    stale = d / "refined_output.json"; stale.write_text("{}")
    import os; os.utime(stale, (1_000_000_000, 1_000_000_000))        # old mtime
    class R: provider = "p"; model = "m"; total_segments = changed_segments = total_changes = rejected_changes = 0; segments = []; warnings = []
    monkeypatch.setattr(run_llm1, "refine_file", lambda p: (R(), stale))   # claims success, wrote nothing
    assert run_llm1.main([str(d)]) == 1
    assert "not written/updated" in capsys.readouterr().err


def test_module_invocation_is_never_silent(tmp_path):
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "tests.run_llm1", str(tmp_path / "nope")],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 2 and "directory not found" in r.stderr


def test_three_value_refine_file_signature_supported(tmp_path, monkeypatch, capsys):
    d = make_dir(tmp_path)

    def three(p):
        ret = real_refine(p, cfg=LLM1Config(provider="openrouter", model="m/x"), client=Fake())
        res, jp = ret[0], ret[1]          # works for both 2- and 3-value refine_file signatures
        return res, jp, jp.with_name("refined_transcript.txt")

    monkeypatch.setattr(run_llm1, "refine_file", three)
    assert run_llm1.main([str(d)]) == 0
    o = capsys.readouterr().out
    assert "refined_output.json" in o and "total_changes=1" in o and "refined_transcript.txt" in o