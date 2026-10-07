from stt.schemas import STTResult, STTSegment, Word
from diarization.schemas import DiarizationResult, DiarSegment
from alignment.align import align_transcript


def _stt(segs, words=()):
    return STTResult(text=" ".join(s[2] for s in segs), duration=30.0,
                     segments=tuple(STTSegment(id=i, start=s[0], end=s[1], text=s[2]) for i, s in enumerate(segs)),
                     words=tuple(Word(start=w[0], end=w[1], text=w[2]) for w in words))


def _diar(segs):
    return DiarizationResult(segments=tuple(DiarSegment(start=a, end=b, speaker=s) for a, b, s in segs))


def test_prompt_example():
    stt = _stt([(0.3, 4.7, "We should increase the battery capacity."),
                (5.1, 8.9, "I agree, but that will increase the weight."),
                (9.2, 12.0, "Okay, let's evaluate both options.")])
    diar = _diar([(0, 5, "SPEAKER_00"), (5, 9, "SPEAKER_01"), (9, 12.5, "SPEAKER_00")])
    r = align_transcript(stt, diar)
    assert [t.speaker for t in r.turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert r.turns[0].text == "We should increase the battery capacity."
    assert r.to_text().startswith("[00:00:00.3] SPEAKER_00\nWe should")


def test_speaker_change_inside_segment_is_split():
    words = [(i, i + 1, w) for i, w in enumerate("one two three four five six".split())]
    stt = _stt([(0, 6, "one two three four five six")], words)
    diar = _diar([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")])
    r = align_transcript(stt, diar)
    assert [(t.speaker, t.text) for t in r.turns] == [("SPEAKER_00", "one two three"),
                                                       ("SPEAKER_01", "four five six")]
    assert "split_segment" in r.turns[0].flags


def test_word_straddling_boundary_goes_to_max_overlap_speaker():
    words = [(0, 1, "one"), (1, 2, "two"), (2, 3.4, "three"), (3.4, 4.4, "four"), (4.4, 5.4, "five")]
    stt = _stt([(0, 5.4, "one two three four five")], words)
    diar = _diar([(0, 3.1, "SPEAKER_00"), (3.1, 6, "SPEAKER_01")])
    r = align_transcript(stt, diar)
    assert r.turns[0].speaker == "SPEAKER_00" and r.turns[0].text.endswith("three")


def test_tiny_unknown_island_is_smoothed_into_surrounding_speaker():
    words = [(0, 1, "a"), (1, 2, "b"), (3.9, 4.1, "x"), (5, 6, "c"), (6, 7, "d")]
    stt = _stt([(0, 7, "a b x c d")], words)
    diar = _diar([(0, 2, "SPEAKER_00"), (5, 8, "SPEAKER_00")])
    r = align_transcript(stt, diar)
    assert len(r.turns) == 1 and r.turns[0].speaker == "SPEAKER_00"
    assert "smoothed" in r.turns[0].flags


def test_silence_gives_unknown_not_a_guess():
    r = align_transcript(_stt([(20, 22, "Hello?")]), _diar([(0, 5, "SPEAKER_00")]))
    assert r.turns[0].speaker == "UNKNOWN"


def test_overlap_flagged():
    r = align_transcript(_stt([(0, 4, "we both talk")]),
                         _diar([(0, 4, "SPEAKER_00"), (0.5, 3.5, "SPEAKER_01")]))
    assert r.turns[0].speaker == "SPEAKER_00" and r.turns[0].has_overlap


def test_empty_stt_and_empty_diarization_do_not_crash():
    assert align_transcript(_stt([]), _diar([])).turns == ()
    r = align_transcript(_stt([(0, 1, "hi")]), _diar([]))
    assert r.turns[0].speaker == "UNKNOWN" and r.warnings
