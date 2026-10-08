"""Runs the EXISTING pipeline stages (same calls as tests/run_pipeline.py, run_llm1.py, run_llm2.py).
No model logic lives here; stages are imported lazily so the server starts without ML deps."""
from __future__ import annotations
import json, threading, time, traceback, uuid
from pathlib import Path

from app.failures import STAGE_KEYS, classify

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
        f = classify(STAGE_KEYS[idx + 1], e)            # STAGES[0..5] -> transcription..documentation
        st["status"], st["detail"] = "failed", f["message"]
        job["error"] = f"{f['message']} {f['hint']}"
        job["technical_error"] = f["technical"]         # raw exception kept for logs/debugging only
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
            from stt.routing import resolve_route, build_stt
            from diarization.diarizer import PyannoteDiarizer
            from alignment.align import align_transcript
            from llm1 import refine_file, translate_file
            from transformation import transform_file
            from llm2 import document_file

            (outdir / "pipeline_meta.json").write_text(json.dumps({"language_choice": language}), encoding="utf-8")

            # ---- language routing: English -> Deepgram Nova-3 (unchanged); Indian language -> Sarvam ----
            def stt_stage():
                route = resolve_route(language, audio)        # Auto Detect uses Sarvam on a short clip
                job["route"] = {"provider": route.provider, "language_code": route.language_code,
                                "language_name": route.language_name, "detected": route.detected}
                return route, build_stt(route, timeout=(120.0, 900.0)).transcribe(str(audio))
            route, stt = _run_stage(job, 0, stt_stage)
            raw = str(getattr(stt, "text", "") or "").strip()
            (outdir / "raw_transcript.txt").write_text(raw, encoding="utf-8")
            diar = _run_stage(job, 1, lambda: PyannoteDiarizer().diarize(str(audio), num_speakers=num_speakers))

            def align():
                res = align_transcript(stt, diar)
                fo = res.model_dump(mode="json")
                fo["raw_transcript"] = raw
                if not route.is_english:                      # English output stays byte-for-byte as before
                    fo["language"] = {"code": route.language_code, "name": route.language_name,
                                      "stt_provider": route.provider, "detected": route.detected,
                                      "translated_to": "en"}
                (outdir / "final_output.json").write_text(json.dumps(fo, indent=2, ensure_ascii=False), encoding="utf-8")
            _run_stage(job, 2, align)
            if route.is_english:
                _run_stage(job, 3, lambda: refine_file(outdir / "final_output.json"))
            else:
                job["stages"][3]["name"] = "Translation & Refinement"
                _run_stage(job, 3, lambda: translate_file(outdir / "final_output.json"))
            _run_stage(job, 4, lambda: transform_file(outdir / "refined_output.json"))
            _run_stage(job, 5, lambda: document_file(outdir / "refined_output.json"))
            from app.downloads import write_standalone       # summary/minutes/decisions/action_items .txt (no LLM)
            doc = json.loads((outdir / "documentation_output.json").read_text(encoding="utf-8"))
            write_standalone(doc, outdir)
        except Exception as e:
            if not job["error"]:                 # e.g. missing dependency / bad .env before any stage ran
                job["error"] = f"Pipeline could not start: {classify('upload', e)['message']}"
                job["technical_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        finally:
            job["done"] = True
    threading.Thread(target=work, daemon=True).start()
