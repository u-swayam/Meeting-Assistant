"""Run LLM2 (meeting documentation) on an existing output folder.

Usage:
    python -m tests.run_llm2 outputs/test5
Reads  <dir>/refined_output.json   (never modified)
Writes <dir>/documentation_output.json and <dir>/documentation_output.txt
"""
import hashlib
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from common.errors import PipelineError
from llm2 import document_file

INPUT_NAME = "refined_output.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def out(*a):
    print(*a, flush=True)


def err(*a):
    print(*a, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        err("usage: python -m tests.run_llm2 <output_dir>")
        return 2
    load_dotenv()
    d = Path(argv[0])
    src = d / INPUT_NAME
    if not d.is_dir():
        err(f"ERROR: output directory not found: {d.resolve()}")
        return 2
    if not src.is_file():
        err(f"ERROR: input file not found: {src.resolve()} (run LLM1 first)")
        return 2

    before = sha(src)
    out(f"LLM2: documenting {src.resolve()}")
    started = time.time()
    try:
        res, jp, tp = document_file(src)
    except PipelineError as e:
        err(f"ERROR ({type(e).__name__}): {e}")
        return 1
    except Exception as e:
        err(f"UNEXPECTED ERROR ({type(e).__name__}): {e}")
        traceback.print_exc()
        return 1

    for p in (Path(jp), Path(tp)):
        if not p.is_file() or p.stat().st_mtime < started - 1:
            err(f"ERROR: {p} was not written by this run.")
            return 1
    if sha(src) != before:
        err("ERROR: refined_output.json was modified!")
        return 1

    out(f"Documentation JSON : {Path(jp).resolve()}")
    out(f"Documentation TXT  : {Path(tp).resolve()}")
    out(f"Provider           : {res.provider} / {res.model}")
    out(f"Guidance           : {res.transformation_source or 'none (refined transcript only)'}")
    out(f"minutes={len(res.minutes)} decisions={len(res.decisions)} action_items={len(res.action_items)} "
        f"warnings={len(res.warnings)}")
    for w in res.warnings:
        out("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
