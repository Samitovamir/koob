"""`scheduler.finalize_labels` (§5.3): the code labels are OR-ed into the raw's labels in the
write itself, so a label another writer added between the read and the write survives;
labels only grow."""

from __future__ import annotations

from koob.memory import policy
from koob.memory.model import Label
from koob.memory.workers import scheduler

from .. import _ledger as L
from ._kit import raw


def _labels(conn, rid: str) -> tuple[Label, int]:
    row = conn.execute(
        "SELECT labels, labels_final FROM raw_chunks WHERE id = ?", (rid,)
    ).fetchone()
    return Label(row[0]), row[1]


def test_a_label_added_mid_batch_is_not_overwritten(tmp_path, monkeypatch):
    conn = L.open_seeded(tmp_path)
    try:
        raw(conn, "r1", "у меня диабет, плачу ипотеку", labels=int(Label.LEGAL))

        def label(text: str) -> Label:
            # another writer labels the raw while the classifier runs (outside the write)
            conn.execute(
                "UPDATE raw_chunks SET labels = labels | ? WHERE id = 'r1'", (int(Label.MINOR),)
            )
            conn.commit()
            return Label.HEALTH | Label.MONEY

        monkeypatch.setattr(policy, "classifier_ready", lambda: True)
        monkeypatch.setattr(policy, "label", label)
        assert scheduler.finalize_labels(conn, ["r1"]) == 1
        got, final = _labels(conn, "r1")
        assert final == 1
        assert got == Label.LEGAL | Label.MINOR | Label.HEALTH | Label.MONEY
    finally:
        conn.close()


def test_labels_only_grow(tmp_path, monkeypatch):
    """A classifier that proposes nothing never clears a label the raw already has."""
    conn = L.open_seeded(tmp_path)
    try:
        raw(conn, "r1", "просто текст", labels=int(Label.THIRD_PARTY_PRIVATE))
        monkeypatch.setattr(policy, "classifier_ready", lambda: True)
        monkeypatch.setattr(policy, "label", lambda text: Label.NONE)
        assert scheduler.finalize_labels(conn, ["r1"]) == 1
        assert _labels(conn, "r1") == (Label.THIRD_PARTY_PRIVATE, 1)
    finally:
        conn.close()
