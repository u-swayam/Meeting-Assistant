"""Usage: python -m tests.run_stt meeting.mp3 [--language en|multi|detect] [--out raw_stt.json]"""
import argparse, sys
from dotenv import load_dotenv
from common.errors import PipelineError
from stt import get_stt_provider
from alignment.align import fmt_ts

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--provider", default=None, help="deepgram (default) | groq")
ap.add_argument("--language", default=None, help="deepgram: en, hi, multi, detect ...")
ap.add_argument("--out", default=None, help="save the full STT result (incl. raw provider JSON)")
a = ap.parse_args()
kw = {"language": a.language} if a.language else {}
try:
    res = get_stt_provider(a.provider, **kw).transcribe(a.audio)
except PipelineError as e:
    sys.exit(f"ERROR ({type(e).__name__}): {e}")
print(f"provider={res.provider} model={res.model} language={res.language} duration={res.duration:.1f}s "
      f"segments={len(res.segments)} words={len(res.words)}")
for w in res.warnings:
    print("WARNING:", w)
for s in res.segments:
    print(f"[{fmt_ts(s.start)} -> {fmt_ts(s.end)}] {s.text}")
if a.out:
    open(a.out, "w", encoding="utf-8").write(res.model_dump_json(indent=2))
