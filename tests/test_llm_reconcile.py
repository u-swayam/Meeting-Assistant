from llm1.schemas import ModelChange, ModelSegment
from llm1.validation import reconcile

TH = 0.70


def mc(o, r, conf=0.95, reason="term"):
    return ModelChange(original=o, refined=r, reason=reason, confidence=conf)


def run(original, refined, *changes):
    return reconcile(original, ModelSegment(index=0, refined_text=refined, changes=list(changes)), TH)


def test_larger_span_kubernetes_accepted_with_model_confidence():
    text, ch, rej, _ = run("we deployed the cube net ease namespace", "we deployed the Kubernetes namespace",
                           mc("the cube net ease namespace", "the Kubernetes namespace"))
    assert text == "we deployed the Kubernetes namespace" and rej == 0
    assert [(c.original, c.refined, c.confidence) for c in ch] == [("cube net ease", "Kubernetes", 0.95)]


def test_larger_span_openid_connect_accepted():
    text, ch, rej, _ = run("we use open I D Connect here", "we use OpenID Connect here",
                           mc("open I D Connect", "OpenID Connect"))
    assert text == "we use OpenID Connect here" and rej == 0
    assert (ch[0].original, ch[0].refined, ch[0].confidence) == ("open I D", "OpenID", 0.95)


def test_larger_span_grafana_accepted():
    text, ch, rej, _ = run("the graf on a panels load", "the Grafana panels load",
                           mc("graf on a panels", "Grafana panels"))
    assert text == "the Grafana panels load" and rej == 0
    assert (ch[0].original, ch[0].refined, ch[0].confidence) == ("graf on a", "Grafana", 0.95)


def test_exact_span_still_accepted():
    text, ch, rej, _ = run("we scrape promise us metrics", "we scrape Prometheus metrics", mc("promise us", "Prometheus"))
    assert text == "we scrape Prometheus metrics" and ch[0].confidence == 0.95 and rej == 0


def test_mismatched_or_inconsistent_span_rejected():
    orig, ref = "we deployed the cube net ease namespace", "we deployed the Kubernetes namespace"
    for bad in (mc("the cube net ease cluster", "the Kubernetes cluster"),      # context not in the text
                mc("the cube net ease namespace", "the Kubernetes cluster"),    # refined context inconsistent
                mc("foo bar", "Baz")):                                          # unrelated
        text, ch, rej, _ = run(orig, ref, bad)
        assert text == orig and ch == [] and rej == 1


def test_ambiguous_two_different_model_changes_rejected():
    orig, ref = "we deployed the cube net ease namespace", "we deployed the Kubernetes namespace"
    text, ch, rej, _ = run(orig, ref, mc("the cube net ease namespace", "the Kubernetes namespace", 0.95),
                           mc("deployed the cube net ease", "deployed the Kubernetes", 0.80))
    assert text == orig and ch == [] and rej == 1


def test_duplicate_identical_model_change_is_not_ambiguous():
    c = mc("the cube net ease namespace", "the Kubernetes namespace")
    text, ch, rej, _ = run("we deployed the cube net ease namespace", "we deployed the Kubernetes namespace", c, c)
    assert text == "we deployed the Kubernetes namespace" and len(ch) == 1


def test_unitemised_change_rejected():
    text, ch, rej, _ = run("we scrape promise us metrics", "we scrape Prometheus metrics")
    assert text == "we scrape promise us metrics" and ch == [] and rej == 1
    text, ch, rej, _ = run("we scrape promise us metrics", "we scrape Prometheus metrics", mc("metrics", "stats"))
    assert text == "we scrape promise us metrics" and rej == 1


def test_numeric_change_rejected_even_inside_larger_span():
    for o, r, m in (("we have 5 items now", "we have 6 items now", mc("have 5 items", "have 6 items")),
                    ("we have 5 items now", "we have 6 items now", mc("5", "6")),
                    ("use v2 here", "use v3 here", mc("use v2 here", "use v3 here"))):
        text, ch, rej, _ = run(o, r, m)
        assert text == o and ch == [] and rej == 1


def test_negation_change_rejected_even_inside_larger_span():
    for m in (mc("are not ready", "are ready"), mc("not", "")):
        text, ch, rej, _ = run("we are not ready yet", "we are ready yet", m)
        assert text == "we are not ready yet" and ch == [] and rej == 1


def test_hedge_or_commitment_change_rejected_itemised_or_silent_or_larger_span():
    o = "I think we should probably change this"
    for m in ([], [mc("I think", "")], [mc("probably change", "change")]):
        text, ch, rej, _ = run(o, "we should change this" if not m else
                               ("we should probably change this" if m[0].original == "I think" else "I think we should change this"), *m)
        assert text == o and ch == [] and rej >= 1
    for m in ([], [mc("might", "will")], [mc("we might ship", "we will ship")]):
        text, ch, rej, _ = run("we might ship it", "we will ship it", *m)
        assert text == "we might ship it" and ch == [] and rej == 1


def test_low_confidence_rejected_including_larger_span_and_boundary():
    orig, ref = "the graf on a panels load", "the Grafana panels load"
    text, ch, rej, _ = run(orig, ref, mc("graf on a panels", "Grafana panels", 0.5))
    assert text == orig and ch == [] and rej == 1
    text, ch, rej, _ = run(orig, ref, mc("graf on a panels", "Grafana panels", 0.69))
    assert text == orig and rej == 1
    text, ch, rej, _ = run(orig, ref, mc("graf on a panels", "Grafana panels", 0.70))
    assert text == ref and rej == 0


def test_multi_hunk_final_text_is_exactly_intended_and_unsafe_hunk_reverted():
    orig = "the cube net ease and open I D Connect handle 5 users"
    ref = "the Kubernetes and OpenID Connect handle 6 users"
    text, ch, rej, w = run(orig, ref, mc("the cube net ease and", "the Kubernetes and"),
                           mc("open I D Connect handle", "OpenID Connect handle"), mc("5", "6"))
    assert text == "the Kubernetes and OpenID Connect handle 5 users"     # numeric hunk reverted, others applied
    assert [c.refined for c in ch] == ["Kubernetes", "OpenID"] and rej == 1
    assert any("numeric" in x for x in w)