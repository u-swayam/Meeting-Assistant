"""Run LLM1 (transcript refinement) on an existing output folder.

Usage:
    python -m tests.run_llm1 outputs/test2.wav
Reads  <dir>/final_output.json   (never modified)
Writes <dir>/refined_output.json (JSON only)

Always prints something and exits non-zero on any failure.
"""
import hashlib
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from common.errors import PipelineError
from llm1 import refine_file

INPUT_NAME = "final_output.json"
OUTPUT_NAME = "refined_output.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def out(*a):
    print(*a, flush=True)


def err(*a):
    print(*a, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        err("usage: python -m tests.run_llm1 <output_dir>")
        return 2
    load_dotenv()

    d = Path(argv[0])
    src = d / INPUT_NAME
    if not d.is_dir():
        err(f"ERROR: output directory not found: {d.resolve()}")
        return 2
    if not src.is_file():
        err(f"ERROR: input file not found: {src.resolve()}")
        return 2

    watched = [p for p in (src, d / "raw_transcript.txt") if p.exists()]
    before = [sha(p) for p in watched]
    out(f"LLM1: refining {src.resolve()}")
    started = time.time()
    try:
        ret = refine_file(src)
        # tolerate both signatures: (result, json_path) and (result, json_path, txt_path)
        res, jp = ret[0], ret[1]
        extra = [Path(x) for x in ret[2:]]
    except PipelineError as e:
        err(f"ERROR ({type(e).__name__}): {e}")
        return 1
    except Exception as e:  # anything unexpected must be loud, not silent
        err(f"UNEXPECTED ERROR ({type(e).__name__}): {e}")
        traceback.print_exc()
        return 1

    jp = Path(jp)
    if not jp.is_file() or jp.stat().st_mtime < started - 1:
        err(f"ERROR: {jp} was not written/updated by this run.")
        return 1
    if before != [sha(p) for p in watched]:
        err("ERROR: original input files were modified!")
        return 1

    out(f"Refined JSON : {jp.resolve()}")
    for x in extra:
        out(f"Extra output  : {x.resolve()}")
    out(f"Provider     : {res.provider} / {res.model}")
    out(f"segments={res.total_segments} changed_segments={res.changed_segments} "
        f"total_changes={res.total_changes} rejected={res.rejected_changes}")
    for s in res.segments:
        for c in s.changes:
            out(f"  [{s.start:.1f}s {s.speaker}] {c.original!r} -> {c.refined!r} ({c.confidence}) {c.reason}")
    for w in res.warnings:
        out("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())