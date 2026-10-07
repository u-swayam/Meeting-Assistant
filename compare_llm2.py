"""Compare two documentation_output.json files (old vs new). No API calls. Not part of the project.
Usage: python compare_llm2.py old.json new.json [refined_output.json]"""
import json, re, sys

META = ["STT", "transcription", "final record", "machine-readable", "human-readable", "Swayam", "unchanged",
        "item potency", "should remain exactly", "turned into a confirmed decision", "not be assigned",
        "documentation system", "final transcript"]


def load(p): return json.load(open(p, encoding="utf-8"))
def words(t): return len(t.split())
def mwords(d): return sum(words(m["topic"]) + words(m["discussion"]) for m in d["minutes"])
def blob(d): return " ".join([d["summary"]] + [m["topic"] + " " + m["discussion"] for m in d["minutes"]] +
                             [x["decision"] for x in d["decisions"]] + [x["task"] for x in d["action_items"]])


old, new = load(sys.argv[1]), load(sys.argv[2])
tw = None
if len(sys.argv) > 3:
    segs = load(sys.argv[3])["segments"]; tw = sum(words(s["refined_text"]) for s in segs)
    full = " ".join(s["refined_text"] for s in segs)
print(f"{'':22}{'old':>8}{'new':>8}")
for k, f in (("summary words", lambda d: words(d["summary"])), ("minutes words", mwords),
             ("minutes topics", lambda d: len(d["minutes"])), ("decisions", lambda d: len(d["decisions"])),
             ("action items", lambda d: len(d["action_items"])), ("warnings", lambda d: len(d["warnings"]))):
    print(f"{k:22}{f(old):>8}{f(new):>8}")
if tw:
    print(f"minutes / transcript   {mwords(old)/tw:>8.0%}{mwords(new)/tw:>8.0%}   (transcript {tw} words)")
print("\nMeta-commentary phrases found (old -> new):")
for m in META:
    o, n = len(re.findall(re.escape(m), blob(old), re.I)), len(re.findall(re.escape(m), blob(new), re.I))
    if o or n: print(f"  {m!r}: {o} -> {n}")
print("\nDecisions (new):"); [print("  -", x["decision"]) for x in new["decisions"]]
print("\nAction items (new):")
for a in new["action_items"]: print(f"  - {a['task']}\n      owner={a['owner']}  deadline={a['deadline']}  ids={a['segment_ids']}")
print("\nWarnings (new):"); [print("  -", w) for w in new["warnings"]]
ident = lambda t: set(re.findall(r"[A-Za-z0-9_.\-/]*[_/][A-Za-z0-9_./\-]+|v\d+\.\d+\.\d+|HTTP \d+|\d[\d,.]*\d%?|\d+ percent", t))
lost = sorted(i for i in ident(blob(old)) if i.strip(".,") not in blob(new))
print(f"\nIdentifiers/numbers in OLD docs but absent from NEW ({len(lost)}) - review whether they matter:")
print("  ", lost)
nums_bad = []
if tw:
    nums_bad = sorted(n for n in set(re.findall(r"\d[\d,.]*\d|\d", blob(new))) if n.strip(".,") not in full)
    print("Numbers in NEW docs not found in transcript:", nums_bad)
print("\nContradiction check (owner=null but text says owned/responsible):")
for a in new["action_items"]:
    if a["owner"] is None:
        key = " ".join(a["task"].split()[-3:]).lower().rstrip(".")
        for s in re.split(r"(?<=[.;])\s", blob(new)):
            if key in s.lower() and re.search(r"\b(owned by|owns|responsible|assigned to)\b", s, re.I):
                print("  ?", s.strip()[:200])