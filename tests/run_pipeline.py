"""
Integration pipeline:

Audio
  -> Deepgram Nova-3
  -> PyAnnote diarization
  -> Transcript alignment
  -> final_output.json + raw_transcript.txt

The raw transcript is kept untouched for display/auditing.

final_output.json contains the structured speaker-labelled transcript
with timestamps, confidence and other metadata produced by the pipeline.

Usage:
    python -m tests.run_pipeline meeting.mp3 [--language en] [--num-speakers 3] [--outdir outputs]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from common.errors import PipelineError
from stt import get_stt_provider
from diarization.diarizer import PyannoteDiarizer
from alignment.align import align_transcript


load_dotenv()

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument(
    "audio",
    help="Path to meeting audio file"
)

ap.add_argument(
    "--provider",
    default=None,
    help="STT provider"
)

ap.add_argument(
    "--language",
    default=None,
    help="Language, e.g. en"
)

ap.add_argument(
    "--num-speakers",
    type=int,
    default=None,
    help="Optional known number of speakers"
)

ap.add_argument(
    "--outdir",
    default="outputs",
    help="Output directory"
)

ap.add_argument(
    "--refine",
    action="store_true",
    help="Also run LLM1 transcript refinement (needs GEMINI_API_KEY)"
)

a = ap.parse_args()


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

audio_path = Path(a.audio)

out = Path(a.outdir) / audio_path.stem
out.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# STT
# ------------------------------------------------------------

kw = {"language": a.language} if a.language else {}

try:
    print("\n[1/3] Running Deepgram transcription...")

    stt = get_stt_provider(
        a.provider,
        **kw
    ).transcribe(str(audio_path))

except PipelineError as e:
    sys.exit(
        f"ERROR ({type(e).__name__}): {e}"
    )


# ------------------------------------------------------------
# Raw transcript
# ------------------------------------------------------------

raw_transcript = getattr(stt, "text", None)

if not raw_transcript:
    raw_transcript = getattr(stt, "transcript", None)

if not raw_transcript:
    sys.exit(
        "ERROR: STT provider returned no transcript text."
    )

raw_transcript = str(raw_transcript).strip()

print("[2/3] Saving raw transcript...")

(out / "raw_transcript.txt").write_text(
    raw_transcript,
    encoding="utf-8"
)


# ------------------------------------------------------------
# Diarization
# ------------------------------------------------------------

try:
    print("[3/3] Running diarization and transcript alignment...")

    diarizer = PyannoteDiarizer()

    diarization = diarizer.diarize(
        str(audio_path),
        num_speakers=a.num_speakers
    )

    # Align Deepgram transcript with diarization.
    result = align_transcript(
        stt,
        diarization
    )

except PipelineError as e:
    sys.exit(
        f"ERROR ({type(e).__name__}): {e}"
    )


# ------------------------------------------------------------
# Build the single structured JSON passed to LLM1
# ------------------------------------------------------------

final_output = result.model_dump(
    mode="json"
)

# Add the raw transcript as metadata as well.
#
# This means LLM1 has both:
#   1. the original raw transcript
#   2. the timestamp/speaker-aligned representation
#
# The original aligned structure produced by the project is
# preserved rather than manually rebuilding its schema.

if isinstance(final_output, dict):
    final_output["raw_transcript"] = raw_transcript


# ------------------------------------------------------------
# Save ONLY one structured JSON
# ------------------------------------------------------------

(out / "final_output.json").write_text(
    json.dumps(
        final_output,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RAW TRANSCRIPT")
print("=" * 70)
print(raw_transcript)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Raw transcript : {out / 'raw_transcript.txt'}")
print(f"Structured JSON: {out / 'final_output.json'}")

print("=" * 70)

for warning in getattr(result, "warnings", []):
    print("WARNING:", warning)

# ------------------------------------------------------------
# Optional LLM1 refinement (implementation lives in llm1/)
# ------------------------------------------------------------

if a.refine:
    from llm1 import refine_file

    try:
        _res, _jp, _tp = refine_file(out / "final_output.json")
    except PipelineError as e:
        sys.exit(f"ERROR ({type(e).__name__}): {e}")

    print(f"Refined JSON   : {_jp}")
    print(f"Refined TXT    : {_tp}")
    print(f"LLM1 changes   : {_res.total_changes}")
