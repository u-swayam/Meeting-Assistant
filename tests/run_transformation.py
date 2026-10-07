"""Run the Transformation layer on an existing output folder.

Usage:  python -m tests.run_transformation outputs/test5
Reads  <dir>/refined_output.json (never modified); writes <dir>/transformation_output.json
Then run LLM2 as usual: python -m tests.run_llm2 outputs/test5   (it picks the file up automatically)
"""
import hashlib
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from common.errors import PipelineError
from transformation import transform_file


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m tests.run_transformation <output_dir>", file=sys.stderr)
        return 2
    load_dotenv()
    d = Path(argv[0])
    src = d / "refined_output.json"
    if not src.is_file():
        print(f"ERROR: input file not found: {src.resolve()} (run LLM1 first)", file=sys.stderr)
        return 2
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"Transformation: {src.resolve()}", flush=True)
    try:
        res, jp = transform_file(src)
    except PipelineError as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    if hashlib.sha256(src.read_bytes()).hexdigest() != before:
        print("ERROR: refined_output.json was modified!", file=sys.stderr)
        return 1
    print(f"Transformation JSON: {Path(jp).resolve()}")
    print(f"topics={len(res.topics)} units={len(res.discussion_units)} decision_candidates="
          f"{len(res.decision_candidates)} action_candidates={len(res.action_candidates)} warnings={len(res.warnings)}")
    for w in res.warnings:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
