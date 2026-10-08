"""LLM1 for non-English meetings: ONE step that translates the original-language transcript into English
and refines it (same philosophy as llm1.prompts: faithful, no invention). Reuses the existing DeepSeek
client/config (llm1.providers.build_client) and the existing ModelResponse schema.

Segment identity: the model must return the same `index` for every input segment; indices are
validated exactly like llm1.refiner does. Speaker/start/end are copied from the input, never from the model.
The original-language text is kept in `original_text` (provenance for the evidence view).
"""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from common.errors import TranslationError
from llm1.config import LLM1Config
from llm1.outputs import write_outputs
from llm1.providers import build_client
from llm1.schemas import ModelResponse, RefinedSegment, RefinementResult, derive_meta

SYSTEM_PROMPT_TEMPLATE = """You are a domain-aware meeting transcript translation and refinement system. You receive numbered, speaker-labelled segments of an automatic speech-recognition (ASR) transcript in {language} (it may contain English technical words mixed in). Produce, for each segment, the final readable ENGLISH transcript text in ONE step: translate faithfully, and fix only plausible ASR errors using the whole transcript as context.

PRESERVE EXACTLY (never alter or lose):
- negation (नहीं / not / no / never ... must stay negated), questions, conditions
- uncertainty and hedging ("शायद" -> "probably/maybe", "I think", "might", "could be")
- commitments vs proposals vs suggestions: "should" stays "should"; "will" only if the speaker committed; never turn a suggestion or proposal into a decision or commitment
- names of people, organisations and places (transliterate names into Latin letters; do not translate them)
- numbers, dates, days, times, quantities, units, versions, identifiers (write numbers as digits exactly as spoken; do not round or convert)
- technical terminology, acronyms, product/tool names (keep English technical words as they are; correct obvious ASR misspellings of them only when the context strongly supports it)
- the speaker's register and the meaning of every sentence

YOU MUST NOT:
- summarise, shorten, omit meaningful content, add commentary, or explain
- add, invent or infer information, owners, deadlines, tasks or context
- merge, split, drop, reorder or add segments; change speakers or timing
- leave text in the original script (everything in refined_text must be English, Latin script)

Example: "हमें शायद शुक्रवार तक deploy कर देना चाहिए।" -> "We should probably deploy by Friday."  (NOT "We will deploy by Friday.")

OUTPUT FORMAT (strict JSON): {{"segments":[{{"index":<int, same as input>,"refined_text":"<final English text for that segment>","changes":[]}}]}}
- Return exactly one entry per input segment with the same index. `changes` must be [].
- If a segment is unintelligible, translate what is intelligible and keep uncertain words in brackets like [unclear]; never guess.
"""


def system_prompt(language_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(language=language_name)


def build_user_prompt(turns: list[dict], total: int, language_name: str) -> str:
    lines = [f"Translate and refine the following {len(turns)} {language_name} segments (of {total} in the meeting) into English. "
             "Speaker and timing are shown for context only.", ""]
    for t in turns:
        lines.append(f"### index={t['index']} speaker={t['speaker']} [{t['start']:.1f}s-{t['end']:.1f}s]")
        lines.append(t["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- deterministic guards (warnings only)
_NEG_SRC = re.compile(r"(नहीं|नही|(?<![\u0900-\u0963\u0966-\u097F\w])मत(?![\u0900-\u0963\u0966-\u097F\w])|कभी नहीं|না\b|নেই|নয়|नाही|नको|இல்லை|లేదు|ഇല്ല|ಇಲ್ಲ|નથી|નહીં|نہیں|نہ\b|ਨਹੀਂ|ନାହିଁ)")
_HEDGE_SRC = re.compile(r"(शायद|संभवतः|हो सकता है|पता नहीं|शायद कि|कदाचित|बहुतेक|হয়তো|સંભવ|ಬಹುಶಃ|ஒருவேளை|బహుశా|ഒരുപക്ഷേ|شاید|ਸ਼ਾਇਦ|ବୋଧହୁଏ)")
_NEG_OUT = re.compile(r"\b(not|no|never|none|nothing|nobody|neither|nor|cannot|without|n't|isn't|don't|won't|can't|didn't|doesn't|haven't|hasn't|aren't|wasn't)\b|n't\b", re.I)
_HEDGE_OUT = re.compile(r"\b(maybe|perhaps|probably|possibly|might|may|could|think|guess|suppose|not sure|don't know|likely|unsure|unclear|seems?)\b", re.I)


def _ascii_numbers(text: str) -> list[str]:
    """Digit runs, with Indic/Arabic-Indic digits normalised to ASCII (so '२०' == '20')."""
    norm = "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in text)
    return re.findall(r"\d+(?:[.,]\d+)?", norm)


def _latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return 1.0 if not letters else sum(1 for c in letters if c.isascii()) / len(letters)


def check_translation(src: str, out: str) -> list[str]:
    """Cheap, language-agnostic-ish checks. Return flags; they become warnings + metadata (never edits text)."""
    flags: list[str] = []
    if src.strip() and not out.strip():
        flags.append("empty_translation")
        return flags
    if _latin_ratio(out) < 0.7:
        flags.append("not_english")
    missing = [n for n in _ascii_numbers(src) if n.replace(",", "") not in out.replace(",", "")]
    if missing:
        flags.append("number_not_preserved:" + ",".join(missing[:5]))
    if _NEG_SRC.search(src) and not _NEG_OUT.search(out):
        flags.append("possible_negation_loss")
    if _HEDGE_SRC.search(src) and not _HEDGE_OUT.search(out) and not re.search(r"\b(should|if)\b", out, re.I):
        flags.append("possible_uncertainty_loss")
    return flags


# ---------------------------------------------------------------- result schema (extends, never replaces)
class TranslatedSegment(RefinedSegment):
    index: int                         # == position in final_output.json turns == LLM2 segment_id
    source_language: str               # BCP-47 code of original_text
    flags: list[str] = []


class TranslationResult(RefinementResult):
    """Same fields LLM2/evidence/UI already read, plus language metadata."""
    stage: str = "llm1_translation_refinement"
    source_language: str
    source_language_name: str
    target_language: str = "en"
    stt_provider: str = ""
    segments: list[TranslatedSegment]


def translate_transcript(data: dict, language_code: str, language_name: str, cfg: LLM1Config | None = None,
                         client=None, source_file: str = "", stt_provider: str = "sarvam") -> TranslationResult:
    """Translate+refine `data['turns']` (original language) into English. Input is never mutated.
    Raises TranslationError (original transcript is untouched on disk) if the model output is unusable."""
    cfg = cfg or LLM1Config.from_env()
    client = client or build_client(cfg)
    turns = data.get("turns") or []
    if not turns:
        raise TranslationError("Translation failed: the original transcript has no segments. "
                               "The original transcript has been preserved.")
    sysp, n = system_prompt(language_name), max(cfg.chunk_turns, 1)
    out: dict[int, str] = {}
    warnings: list[str] = []

    def request(chunk):
        try:
            raw = client.generate_json(sysp, build_user_prompt(chunk, len(turns), language_name))
        except Exception as e:  # noqa: BLE001
            raise TranslationError(f"Translation failed: {type(e).__name__}: {str(e)[:200]}. "
                                   "The original transcript has been preserved.") from e
        try:
            resp = ModelResponse.model_validate(raw)
        except ValidationError:
            return None, "malformed translation response (schema-invalid)"
        got, want = [m.index for m in resp.segments], [c["index"] for c in chunk]
        if sorted(got) != want:
            return None, f"segment ids mismatch (missing={sorted(set(want) - set(got))[:8]}, extra={sorted(set(got) - set(want))[:8]})"
        by = {m.index: m.refined_text for m in resp.segments}
        bad = [i for i, c in zip(want, chunk) if c["text"].strip() and not str(by[i]).strip()]
        if bad:
            return None, f"empty translation for segments {bad[:8]}"
        return by, ""

    for s in range(0, len(turns), n):
        chunk = [{"index": i, "speaker": t["speaker"], "start": t["start"], "end": t["end"], "text": t["text"]}
                 for i, t in enumerate(turns[s:s + n], start=s)]
        got, why = request(chunk)
        if got is None:                       # one retry with smaller requests, then fail clearly (never invent)
            mid = len(chunk) // 2
            parts = [chunk[:mid], chunk[mid:]] if mid else [chunk]
            warnings.append(f"segments {chunk[0]['index']}-{chunk[-1]['index']}: {why}; retrying in {len(parts)} request(s)")
            for part in parts:
                got, why = request(part)
                if got is None:
                    raise TranslationError(f"Translation failed: {why}. The original transcript has been preserved.")
                out.update(got)
        else:
            out.update(got)

    segs = []
    for i, t in enumerate(turns):
        text = str(out[i]).strip()
        flags = check_translation(t["text"], text)
        if flags:
            warnings.append(f"segment {i}: " + "; ".join(flags))
        meta = {k: t[k] for k in ("confidence", "has_overlap", "flags", "source_segment_ids") if k in t}
        segs.append(TranslatedSegment(index=i, start=t["start"], end=t["end"], speaker=t["speaker"],
                                      original_text=t["text"], refined_text=text, changed=False, changes=[],
                                      metadata=meta, source_language=language_code, flags=flags))
    warnings += list(getattr(client, "events", []))
    return TranslationResult(
        provider=cfg.provider, model=cfg.model, source_file=source_file, total_segments=len(segs),
        changed_segments=0, total_changes=0,
        speakers=derive_meta(segs, data.get("speakers"), data.get("duration"))[0],
        duration=derive_meta(segs, data.get("speakers"), data.get("duration"))[1], warnings=warnings + list(data.get("warnings", [])),
        segments=segs, source_language=language_code, source_language_name=language_name,
        stt_provider=stt_provider)


def translate_file(structured_json: str | Path, outdir: str | Path | None = None, cfg: LLM1Config | None = None,
                   client=None) -> tuple[TranslationResult, Path, Path]:
    """Read final_output.json (its `language` block says what was spoken), write refined_output.json/.txt
    (English). final_output.json is never modified, so the original transcript is always preserved."""
    p = Path(structured_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    lang = data.get("language") or {}
    if not lang.get("code") or lang.get("code") == "en":
        raise TranslationError("translate_file called on a transcript without a non-English `language` block.")
    out = Path(outdir) if outdir else p.parent
    res = translate_transcript(data, lang["code"], lang.get("name") or lang["code"], cfg, client,
                               source_file=p.name, stt_provider=lang.get("stt_provider") or "")
    jp, tp = write_outputs(res, out)
    return res, jp, tp
