"""Usage: python -m tests.run_diarization meeting.mp3 [--num-speakers 3] [--out diar.json]"""
import argparse, sys
from dotenv import load_dotenv
from common.errors import PipelineError
from diarization.diarizer import PyannoteDiarizer

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--num-speakers", type=int, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
try:
    res = PyannoteDiarizer().diarize(a.audio, num_speakers=a.num_speakers)
except PipelineError as e:
    sys.exit(f"ERROR ({type(e).__name__}): {e}")
print(f"device={res.device} speakers={list(res.speakers)} segments={len(res.segments)}")
for w in res.warnings:
    print("WARNING:", w)
for s in res.segments:
    print(f"{s.start:8.2f} - {s.end:8.2f}  {s.speaker}")
if a.out:
    open(a.out, "w", encoding="utf-8").write(res.model_dump_json(indent=2))
