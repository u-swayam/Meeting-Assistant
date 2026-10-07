from diarization.postprocess import PostConfig, postprocess

A, B, C, D = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"


def unit(i, n=8):  # unit vectors
    v = [0.0] * n; v[i] = 1.0; return v


def mix(i, j, w=0.9, n=8):
    v = [0.0] * n; v[i] = w; v[j] = (1 - w ** 2) ** 0.5; return v


def segs():
    # A and B real; C real but quiet; D = 2.5 s fragment
    return [(0, 30, A), (30, 55, B), (55, 70, C), (70, 72.5, D), (72.5, 100, A)]


def test_spurious_cluster_absorbed_when_voice_similar():
    cent = {A: unit(0), B: unit(1), C: unit(2), D: mix(1, 3, 0.8)}   # D resembles B (cos 0.8)
    r = postprocess(segs(), [], cent, PostConfig(merge_sim=0.95))
    assert r.mapping[D] == B
    assert {k for *_, k in r.full} == {A, B, C}
    assert any("absorbed" in n for n in r.notes)


def test_tiny_but_distinct_speaker_is_kept():
    cent = {A: unit(0), B: unit(1), C: unit(2), D: unit(3)}          # D unlike everyone
    r = postprocess(segs(), [], cent, PostConfig())
    assert r.mapping[D] == D and D in {k for *_, k in r.full}
    assert any("kept tiny" in n for n in r.notes)


def test_near_identical_speakers_merged_and_relabelled():
    cent = {A: unit(0), B: unit(1), C: unit(2), D: unit(3)}
    cent[C] = mix(0, 2, 0.99)                                         # C ~ A
    r = postprocess(segs(), [], cent, PostConfig(merge_sim=0.9))
    assert r.mapping[C] == r.mapping[A]
    labels = sorted({k for *_, k in r.full})
    assert labels == [f"SPEAKER_{i:02d}" for i in range(len(labels))]  # contiguous


def test_without_centroids_short_fragment_joins_temporal_neighbour():
    s = [(0, 40, A), (40, 41.5, B), (41.5, 80, A), (80, 120, C)]
    r = postprocess(s, [], None, PostConfig())
    assert r.mapping[B] == A


def test_without_centroids_two_second_rule_is_conservative():
    s = [(0, 40, A), (40, 43, B), (43, 80, A)]                        # B has 3 s > 2 s limit
    r = postprocess(s, [], None, PostConfig())
    assert r.mapping[B] == B


def test_long_meeting_real_speaker_is_never_absorbed():
    s = [(0, 1800, A), (1800, 1860, B)]                               # 60 s >> 4 s cap
    r = postprocess(s, [], {A: unit(0), B: mix(0, 1, 0.9)}, PostConfig(merge_sim=0.99))
    assert r.mapping[B] == B


def test_single_speaker_untouched_and_empty_ok():
    assert postprocess([(0, 1, A)], [], None).full == [(0, 1, A)]
    assert postprocess([], [], None).full == []


def test_overlap_preserved_between_different_speakers():
    full = [(0, 10, A), (4, 6, B), (10, 40, B), (40, 70, A)]
    r = postprocess(full, [], None, PostConfig())
    assert (0, 10, A) in r.full and (4, 6, B) in r.full
