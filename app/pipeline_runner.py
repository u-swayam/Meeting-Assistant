"""Runs the EXISTING pipeline stages (same calls as tests/run_pipeline.py, run_llm1.py, run_llm2.py).
No model logic lives here; stages are imported lazily so the server starts without ML deps."""
from __future__ import annotations
import json, threading, time, traceback, uuid
from pathlib import Path

STAGES = ["STT", "Diarization", "Alignment", "LLM1 Refinement", "Transformation", "LLM2 Documentation"]
JOBS: dict[str, dict] = {}


def new_job(meeting_id: str) -> dict:
    job = {"id": uuid.uuid4().hex[:10], "meeting_id": meeting_id, "done": False, "error": None,
           "stages": [{"name": n, "status": "pending", "seconds": None, "detail": None} for n in STAGES]}
    JOBS[job["id"]] = job
    return job


def _run_stage(job, idx, fn):
    st = job["stages"][idx]
    st["status"] = "running"
    t = time.time()
    try:
        out = fn()
    except Exception as e:                       # surface the real error, never hide it
        st["status"], st["detail"] = "failed", f"{type(e).__name__}: {e}"
        job["error"] = f"{STAGES[idx]} failed: {st['detail']}"
        traceback.print_exc()
        raise
    st["status"], st["seconds"] = "done", round(time.time() - t, 1)
    return out


def run_pipeline(job: dict, audio: Path, outdir: Path, language: str = "en", num_speakers=None):
    def work():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            outdir.mkdir(parents=True, exist_ok=True)
            from stt import get_stt_provider
            from diarization.diarizer import PyannoteDiarizer
            from alignment.align import align_transcript
            from llm1 import refine_file
            from transformation import transform_file
            from llm2 import document_file

            stt = _run_stage(job, 0, lambda: get_stt_provider(None, language=language, timeout=(120.0, 900.0)).transcribe(str(audio)))
            raw = str(getattr(stt, "text", "") or "").strip()
            (outdir / "raw_transcript.txt").write_text(raw, encoding="utf-8")
            diar = _run_stage(job, 1, lambda: PyannoteDiarizer().diarize(str(audio), num_speakers=num_speakers))

            def align():
                res = align_transcript(stt, diar)
                fo = res.model_dump(mode="json")
                fo["raw_transcript"] = raw
                (outdir / "final_output.json").write_text(json.dumps(fo, indent=2, ensure_ascii=False), encoding="utf-8")
            _run_stage(job, 2, align)
            _run_stage(job, 3, lambda: refine_file(outdir / "final_output.json"))
            _run_stage(job, 4, lambda: transform_file(outdir / "refined_output.json"))
            _run_stage(job, 5, lambda: document_file(outdir / "refined_output.json"))
        except Exception as e:
            if not job["error"]:                 # e.g. missing dependency / bad .env before any stage ran
                job["error"] = f"Pipeline could not start: {type(e).__name__}: {e}"
                traceback.print_exc()
        finally:
            job["done"] = True
    threading.Thread(target=work, daemon=True).start()
