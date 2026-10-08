"""Run the pipeline from an EXISTING aligned transcript (skips STT, diarization and alignment).

    python -m tests.run_from_transcript outputs/test3 --language hi-IN     # Hindi: translate+refine -> transform -> LLM2
    python -m tests.run_from_transcript outputs/test3                      # English: normal LLM1 refine -> ...

Input: <dir>/final_output.json with a "turns" list (speaker/start/end/text), i.e. the aligned transcript.
For a non-English language a `language` block is added to final_output.json if it is missing.
"""
import argparse, json, sys, time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--language", default="en", help="en, or e.g. hi-IN / Hindi / bn-IN ... (language of the transcript text)")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()
    from stt.languages import route_for_language
    from common.errors import PipelineError

    d = Path(a.outdir)
    fo_path = d / "final_output.json"
    if not fo_path.is_file():
        print(f"error: {fo_path} not found (need the aligned transcript JSON with a 'turns' list)", file=sys.stderr); return 2
    data = json.loads(fo_path.read_text(encoding="utf-8"))
    if not data.get("turns"):
        print("error: final_output.json has no 'turns'", file=sys.stderr); return 2
    try:
        route = route_for_language(a.language)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr); return 2

    if not route.is_english and (data.get("language") or {}).get("code") != route.language_code:
        data["language"] = {"code": route.language_code, "name": route.language_name,   # no stt_provider: fixture run, no STT step
                            "detected": False, "translated_to": "en"}
        fo_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    from llm1 import refine_file, translate_file
    from transformation import transform_file
    from llm2 import document_file
    from app.downloads import write_standalone

    def step(name, fn):
        t = time.time(); print(f"[{name}] ...", flush=True)
        out = fn(); print(f"[{name}] done in {time.time() - t:.1f}s", flush=True); return out
    try:
        if route.is_english:
            step("LLM1 refinement", lambda: refine_file(fo_path))
        else:
            step(f"Translate {route.language_name} -> English + refine", lambda: translate_file(fo_path))
        step("Transformation", lambda: transform_file(d / "refined_output.json"))
        step("LLM2 documentation", lambda: document_file(d / "refined_output.json"))
        doc = json.loads((d / "documentation_output.json").read_text(encoding="utf-8"))
        write_standalone(doc, d)
    except PipelineError as e:
        print(f"FAILED: {e}", file=sys.stderr); return 1
    print(f"\nDone. Outputs in {d}: refined_output.json, refined_transcript.txt, documentation_output.json, "
          "summary.txt, minutes.txt, decisions.txt, action_items.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
