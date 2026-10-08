"""Language selection and provider routing (the ONE place that decides Deepgram vs Sarvam).

Sarvam languages are taken from Sarvam's published Saaras v3/v4 list (BCP-47 codes); nothing outside
that table is ever sent to Sarvam. English always routes to Deepgram Nova-3 (the default pipeline).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from common.errors import UnsupportedLanguageError

PROVIDER_DEEPGRAM = "deepgram"
PROVIDER_SARVAM = "sarvam"
AUTO = "auto"

# code -> display name. Source: docs.sarvam.ai "How to specify language codes" (Saaras v3/v4).
SARVAM_LANGUAGES: dict[str, str] = {
    "hi-IN": "Hindi", "bn-IN": "Bengali", "gu-IN": "Gujarati", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia", "pa-IN": "Punjabi",
    "ta-IN": "Tamil", "te-IN": "Telugu", "ur-IN": "Urdu", "as-IN": "Assamese",
    "ne-IN": "Nepali", "kok-IN": "Konkani", "ks-IN": "Kashmiri", "sd-IN": "Sindhi",
    "sa-IN": "Sanskrit", "sat-IN": "Santali", "mni-IN": "Manipuri", "brx-IN": "Bodo",
    "mai-IN": "Maithili", "doi-IN": "Dogri",
}
_NAME_TO_CODE = {v.lower(): k for k, v in SARVAM_LANGUAGES.items()}
_SHORT_TO_CODE = {k.split("-")[0].lower(): k for k in SARVAM_LANGUAGES}
_SHORT_TO_CODE["or"] = "od-IN"            # ISO 639-1 for Odia; Sarvam spells it od-IN
_ENGLISH = {"en", "en-in", "en-us", "en-gb", "english", "eng"}


@dataclass(frozen=True)
class LanguageRoute:
    provider: str                 # "deepgram" | "sarvam"
    language_code: str            # "en" for Deepgram, BCP-47 (e.g. "hi-IN") for Sarvam
    language_name: str
    detected: bool = False        # True when Auto Detect chose it

    @property
    def is_english(self) -> bool:
        return self.provider == PROVIDER_DEEPGRAM


ENGLISH_ROUTE = LanguageRoute(PROVIDER_DEEPGRAM, "en", "English")


def is_auto(choice: Optional[str]) -> bool:
    return (choice or "").strip().lower() in (AUTO, "detect", "unknown", "auto detect", "auto-detect")


def route_for_language(choice: Optional[str], *, detected: bool = False) -> LanguageRoute:
    """Map a user choice or detected code to a route. Raises UnsupportedLanguageError (never guesses)."""
    c = (choice or "en").strip()
    low = c.lower()
    if low in _ENGLISH:
        return LanguageRoute(PROVIDER_DEEPGRAM, "en", "English", detected)
    for code in SARVAM_LANGUAGES:
        if low == code.lower():
            return LanguageRoute(PROVIDER_SARVAM, code, SARVAM_LANGUAGES[code], detected)
    code = _NAME_TO_CODE.get(low) or _SHORT_TO_CODE.get(low)
    if code:
        return LanguageRoute(PROVIDER_SARVAM, code, SARVAM_LANGUAGES[code], detected)
    raise UnsupportedLanguageError(
        f"Unsupported language {c!r}. Supported: English and "
        + ", ".join(sorted(SARVAM_LANGUAGES.values())) + ".")


def supported_languages() -> list[dict]:
    """For the UI dropdown: Auto Detect, English, then Sarvam languages (primary Indian languages first)."""
    first = ["hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "ur-IN"]
    rest = sorted(c for c in SARVAM_LANGUAGES if c not in first)
    out = [{"code": AUTO, "name": "Auto Detect", "provider": None},
           {"code": "en", "name": "English", "provider": PROVIDER_DEEPGRAM}]
    out += [{"code": c, "name": SARVAM_LANGUAGES[c], "provider": PROVIDER_SARVAM} for c in first + rest]
    return out
