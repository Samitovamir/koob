"""the memory spec §6.1: hot-path commit p95 ≤ 50 ms.
Spec marker: `[T perf/test_hot::<case>]`."""

from __future__ import annotations

import time

from koob.memory import api, labeler, policy
from koob.memory.recall.vectors import HashEmbedder

from ..hotpath._world import sasha_world, turn_of
from ..races._kit import handle

EMBED_CALL_S = 0.017  # a warm local embedder: ~15-20 ms per call
EMBED_TEXT_S = 0.001  # plus a little per extra window


class _WarmEmbedder(HashEmbedder):
    """The hash embedder with the latency of a warm ONNX model (sleep releases the GIL too)."""

    def __init__(self) -> None:
        super().__init__(id="warm@1", dim=384)
        self.calls = 0
        self.warm = False

    def embed_docs(self, texts):
        if self.warm:
            self.calls += 1
            time.sleep(EMBED_CALL_S + EMBED_TEXT_S * max(0, len(texts) - 1))
        return super().embed_docs(texts)


def _warm_labeler(tmp_path) -> _WarmEmbedder:
    """The real `Labeler` installed and warm (prototypes built before the clock runs)."""
    emb = _WarmEmbedder()
    lab = labeler.Labeler(embedder=lambda: emb, cache_root=tmp_path / "labeler", background=False)
    policy.set_classifier(lab)
    assert lab.ready()
    emb.warm = True
    return emb


def test_commit_p95(mem, tmp_path):
    """§6.1: hot-path commit p95 ≤ 50 ms (remember and fix, on a file-backed memory.db), with
    the real labeler installed over a warm embedder."""
    emb = _warm_labeler(tmp_path)
    _sasha, bday = sasha_world(mem)
    times: list[float] = []
    for i in range(30):
        q = f"запомни: встреча номер {i} в пятницу"
        text = f"У Дениса встреча номер {i} в пятницу"  # the brain's text, not the user's quote
        t0 = time.perf_counter()
        r = api.remember(text, turn=turn_of(q, n=10 + i), because_quote=q, store=mem)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.outcome == "committed"
    for i in range(10):
        q = f"у Саши день рождения {i + 1} мая"
        t0 = time.perf_counter()
        r = api.fix(handle(bday.id), q, turn=turn_of(q, n=50 + i), because_quote=q, store=mem)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.outcome == "committed"
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"hot p95 {p95:.1f} ms with a warm labeler")
    assert p95 <= 50.0, f"p95 {p95:.1f} ms"
    from koob.memory.writers import hot

    hot.drain_labels(timeout=30)  # the classifier passes ran off the hot path
    assert emb.calls >= 40  # every remember and fix was scored (the labeler caches repeats)
    # and each pass ended with final labels (the cloud path opens for them)
    finals = mem.execute(
        "SELECT SUM(labels_final), COUNT(*) FROM records WHERE status = 'active'"
        " AND origin_quote LIKE 'запомни: встреча номер %'"
    ).fetchone()
    assert tuple(finals) == (30, 30)


def test_classifier_labels_arrive_after_the_commit(mem):
    """The classifier runs off the hot path; its label still lands on the record right
    after the commit (labels only grow), and then the labels are final (§5.3)."""
    from koob.memory.model import Label
    from koob.memory.writers import hot

    policy.set_classifier(lambda t: {Label.HEALTH: 0.9} if "зюзя" in t else {})
    q = "запомни: у Саши зюзя"
    r = api.remember(q, turn=turn_of(q, n=99), because_quote=q, store=mem)
    hot.drain_labels()
    labels, final = mem.execute(
        "SELECT labels, labels_final FROM records WHERE id = ?", (r.record_ids[0],)
    ).fetchone()
    assert Label(labels) & Label.HEALTH and final == 1


def test_code_labels_match_policy_without_a_classifier():
    """The hot path's lexicon labels are `policy.label` minus the classifier."""
    from koob.memory.writers import hot

    policy.set_classifier(None)
    for t in (
        "у Саши диабет",
        "я должен Диме 5000 руб",
        "живу на ул. Ленина, д. 5, кв. 12",
        "встреча в пятницу",
        "у меня кредит и долг",
        "",
    ):
        assert hot.code_labels(t) == policy.label(t), t
