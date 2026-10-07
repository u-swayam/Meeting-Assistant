import pytest
from common.errors import STTProviderError
from stt.deepgram_nova import parse_deepgram_response

FAKE = {
    "metadata": {"duration": 12.5, "request_id": "abc"},
    "results": {
        "channels": [{"alternatives": [{
            "transcript": "We should increase the battery capacity. I agree.",
            "confidence": 0.99,
            "words": [
                {"word": "we", "punctuated_word": "We", "start": 0.3, "end": 0.5, "confidence": 0.99},
                {"word": "should", "punctuated_word": "should", "start": 0.5, "end": 0.9, "confidence": 0.98},
                {"word": "i", "punctuated_word": "I", "start": 5.1, "end": 5.2, "confidence": 0.97},
                {"word": "agree", "punctuated_word": "agree.", "start": 5.2, "end": 5.7, "confidence": 0.97},
            ]}]}],
        "utterances": [
            {"start": 5.1, "end": 5.7, "confidence": 0.97, "transcript": "I agree.", "channel": 0},
            {"start": 0.3, "end": 4.7, "confidence": 0.99, "transcript": "We should increase the battery capacity.", "channel": 0},
        ],
    },
}


def _parse(d, lang="en"):
    return parse_deepgram_response(d, model="nova-3", audio_path="x.wav", duration=12.0, requested_language=lang)


def test_parse_segments_words_and_order():
    r = _parse(FAKE)
    assert r.provider == "deepgram" and r.model == "nova-3" and r.language == "en"
    assert [s.text for s in r.segments] == ["We should increase the battery capacity.", "I agree."]
    assert r.segments[0].start == 0.3 and r.duration == 12.5
    assert r.words[3].text == "agree."
    assert r.raw_response[0]["response"] is FAKE      # raw output preserved


def test_requested_multi_is_not_reported_as_language():
    assert _parse(FAKE, "multi").language is None


def test_silence_gives_warning_not_error():
    d = {"results": {"channels": [{"alternatives": [{"transcript": "", "words": []}]}], "utterances": []}}
    r = _parse(d)
    assert r.segments == () and r.warnings


def test_bad_shape_raises():
    with pytest.raises(STTProviderError):
        _parse({"results": {}})
    with pytest.raises(STTProviderError):   # text but no utterances
        _parse({"results": {"channels": [{"alternatives": [{"transcript": "hi", "words": []}]}]}})
