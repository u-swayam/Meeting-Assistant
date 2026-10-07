from stt.schemas import STTResult, STTSegment, Word
from diarization.schemas import DiarizationResult, DiarSegment
from alignment.align import align_transcript


def test_exclusive_drives_assignment_and_regular_flags_overlap():
    words = tuple(Word(start=i, end=i + 1, text=f"w{i + 1}") for i in range(4))
    stt = STTResult(text="w1 w2 w3 w4", duration=4.0, words=words,
                    segments=(STTSegment(id=0, start=0, end=4, text="w1 w2 w3 w4"),))
    d = lambda xs: tuple(DiarSegment(start=a, end=b, speaker=s) for a, b, s in xs)
    diar = DiarizationResult(
        segments=d([(0, 4, "SPEAKER_00"), (1, 3, "SPEAKER_01")]),                      # overlapping view
        exclusive_segments=d([(0, 1, "SPEAKER_00"), (1, 3, "SPEAKER_01"), (3, 4, "SPEAKER_00")]))
    r = align_transcript(stt, diar)
    assert [(t.speaker, t.text) for t in r.turns] == [
        ("SPEAKER_00", "w1"), ("SPEAKER_01", "w2 w3"), ("SPEAKER_00", "w4")]
    assert r.turns[1].has_overlap and not r.turns[0].has_overlap
