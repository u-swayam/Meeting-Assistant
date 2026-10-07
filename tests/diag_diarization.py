"""Diagnose speaker-clustering problems without knowing the speaker count.

  python -m tests.diag_diarization meeting.mp3
  python -m tests.diag_diarization meeting.mp3 --at 40.9 --at 55.2
  python -m tests.diag_diarization meeting.mp3 --ref ref.txt
  python -m tests.diag_diarization meeting.mp3 --sweep 0.5,0.6,0.7,0.8 --ref ref.txt

ref.txt = what you hear, one line per turn:  start end speaker   (speaker names are arbitrary, e.g. A/B/C)
"""
import argparse, sys
from dotenv import load_dotenv
from common.errors import PipelineError
from diarization.diarizer import PyannoteDiarizer
from diarization.postprocess import PostConfig, cosine, durations, postprocess

load_dotenv()


def summarize(title, segs):
    dur = durations(segs)
    total = sum(dur.values()) or 1.0
    print(f"  {title}: {len(dur)} speakers")
    for k in sorted(dur):
        n = sum(1 for *_, s in segs if s == k)
        print(f"    {k}: {dur[k]:6.1f}s ({dur[k] / total:5.1%}) in {n} segments")


def der_text(ref_path, segs):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hyp = Annotation(), Annotation()
    for line in open(ref_path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 3 and not line.lstrip().startswith("#"):
            ref[Segment(float(p[0]), float(p[1]))] = p[2]
    for i, (s, e, k) in enumerate(segs):
        hyp[Segment(s, e), i] = k
    r = DiarizationErrorRate(collar=0.25, skip_overlap=False)(ref, hyp, detailed=True)
    t = r["total"] or 1.0
    return (f"DER {r['diarization error rate']:.1%} (confusion {r['confusion'] / t:.1%}, "
            f"missed {r['missed detection'] / t:.1%}, false alarm {r['false alarm'] / t:.1%})")


def at_time(label, segs, t):
    hits = [f"{s:.2f}-{e:.2f} {k}" for s, e, k in segs if s <= t <= e]
    print(f"    {label:<22} {', '.join(hits) if hits else '(no speaker)'}")


def analyse(d, audio, a, header):
    raw = d.infer_raw(audio)
    print(f"\n===== {header} =====")
    for w in raw.warnings:
        print("  WARNING:", w)
    print(f"  output fields: {raw.out_fields}")
    print(f"  exclusive diarization: {'PRESENT' if raw.exclusive else 'ABSENT (alignment falls back to overlapping view)'}")
    print(f"  speaker centroids:     {'PRESENT' if raw.centroids else 'ABSENT (post-processing uses temporal rules only)'}")
    basis = raw.exclusive or raw.full
    summarize("raw", basis)
    if raw.centroids and len(raw.centroids) > 1:
        ks = sorted(raw.centroids)
        print("  centroid cosine similarity (higher = more alike; same-voice clusters score high):")
        print("          " + "  ".join(f"{k[-2:]:>5}" for k in ks))
        for x in ks:
            print(f"    {x[-8:]:>6} " + "  ".join(f"{cosine(raw.centroids[x], raw.centroids[y]):5.2f}" for y in ks))
    pr = postprocess(raw.full, raw.exclusive, raw.centroids, PostConfig.from_env())
    print("  post-processing notes:", *(pr.notes or ["(none - nothing to change)"]), sep="\n    ")
    summarize("after post-processing", pr.exclusive or pr.full)
    for t in a.at:
        print(f"  --- speakers at t={t}s")
        at_time("raw regular", raw.full, t); at_time("raw exclusive", raw.exclusive, t)
        at_time("post regular", pr.full, t); at_time("post exclusive", pr.exclusive, t)
    if a.ref:
        print("  " + "raw regular      : " + der_text(a.ref, raw.full))
        print("  " + "post regular     : " + der_text(a.ref, pr.full))
    return raw, pr


ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--at", type=float, action="append", default=[])
ap.add_argument("--ref", default=None)
ap.add_argument("--sweep", default=None, help="comma-separated clustering.threshold values")
a = ap.parse_args()
try:
    d = PyannoteDiarizer()
    print("pipeline parameters:", d.pipeline_params())
    analyse(d, a.audio, a, "default settings")
    for thr in [float(x) for x in a.sweep.split(",")] if a.sweep else []:
        d.set_cluster_threshold(thr)
        analyse(d, a.audio, a, f"clustering.threshold = {thr}")
except PipelineError as e:
    sys.exit(f"ERROR ({type(e).__name__}): {e}")
