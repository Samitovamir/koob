"""Pre-send check: the memory spec §4.2, §7.4, §7.6 (store).

`store.pre_send_check` is what `api.pre_send_check` returns as `PreSend` (same field names).
Spec marker: `[T consistency/test_presend::<case>]`."""

from __future__ import annotations

import pytest

from koob.memory import store
from koob.memory.model import ReadRef, ReadSrc, Status

from .._store_seed import add_derived, add_fact, add_raw, ref, version


@pytest.mark.parametrize(
    ("st", "usable"),
    [(s, s in (Status.ACTIVE, Status.ARCHIVED)) for s in Status],
)
def test_only_usable_sent(mem, turn, st, usable):
    """§4.2, §7.4: everything that reaches the model is usable at send time."""
    add_fact(mem, "r1", "Sasha's birthday is 12 May", version=4, status=st.value)
    result = store.pre_send_check(
        mem, thread_id=turn.thread_id, read_set=[ReadRef(id="r1", version=4)]
    )
    assert result.ok is usable
    assert result.stale == ([] if usable else [ReadRef(id="r1", version=4)])


def test_only_usable_sent_covers_versions_raw_and_derived(mem, turn):
    """§7.4: a changed version, a missing id, a forgotten or revised raw chunk and an invalid
    derived item are all refused; the rest of the read set passes."""
    add_raw(mem, "w1", "у Саши день рождения 12 мая")
    add_raw(mem, "w2", "old text", status="forgotten")
    add_fact(mem, "r1", "a", raw="w1", version=2)
    add_fact(mem, "r2", "b", version=5)
    add_derived(mem, "d1", [ref(mem, "r1")], "summary")
    add_derived(mem, "d2", [ref(mem, "r2")], "core")
    mem.execute("UPDATE derived SET valid = 0 WHERE id = 'd2'")
    good = [
        ReadRef(id="r1", version=2),
        ReadRef(id="w1", version=1, src=ReadSrc.RAW),
        ReadRef(id="d1", version=0, src=ReadSrc.DERIVED),
    ]
    bad = [
        ReadRef(id="r2", version=4),  # changed since read
        ReadRef(id="gone", version=1),  # deleted
        ReadRef(id="w2", version=1, src=ReadSrc.RAW),  # forgotten
        ReadRef(id="w1", version=0, src=ReadSrc.RAW),  # redacted since (rev moved)
        ReadRef(id="d2", version=0, src=ReadSrc.DERIVED),  # invalid
    ]
    result = store.pre_send_check(mem, thread_id=turn.thread_id, read_set=good + bad)
    assert not result.ok
    assert sorted(result.stale, key=repr) == sorted(bad, key=repr)
    assert store.pre_send_check(mem, thread_id=turn.thread_id, read_set=good).ok


def test_only_usable_sent_after_a_correction(mem, turn):
    """G2: a basket read before a correction is refused after it."""
    add_fact(mem, "r1", "12 May", version=1)
    basket = [ref(mem, "r1")]
    mem.execute("UPDATE records SET version = 2 WHERE id = 'r1'")
    assert version(mem, "r1") == 2
    assert not store.pre_send_check(mem, thread_id=turn.thread_id, read_set=basket).ok


def test_refuses_undelivered_updates(mem, turn):
    """§7.6: the pre-send check refuses a prompt for a thread with undelivered updates."""
    add_fact(mem, "r1", "12 May", version=3)
    mem.execute(
        "INSERT INTO pending_updates (thread_id, item_id, old_version, new_version, created_at)"
        " VALUES (?, 'r1', 2, 3, 't')",
        (turn.thread_id,),
    )
    refs = [ReadRef(id="r1", version=3)]
    blocked = store.pre_send_check(mem, thread_id=turn.thread_id, read_set=refs)
    assert blocked.ok is False
    assert blocked.stale == []
    assert blocked.undelivered_updates == 1
    # another thread is not held back by this thread's update
    assert store.pre_send_check(mem, thread_id="thread-other", read_set=refs).ok
    # the update block consumes the row; then the prompt may go
    mem.execute("DELETE FROM pending_updates WHERE thread_id = ?", (turn.thread_id,))
    assert store.pre_send_check(mem, thread_id=turn.thread_id, read_set=refs).ok


def test_presend_result_matches_api_shape(mem, turn):
    """The api facade wraps it as `PreSend(**asdict(result))`."""
    from dataclasses import asdict

    from koob.memory.api import PreSend

    add_fact(mem, "r1", "x", version=1)
    result = store.pre_send_check(
        mem, thread_id=turn.thread_id, read_set=[ReadRef(id="r1", version=0)]
    )
    shaped = PreSend(**asdict(result))
    assert shaped.ok is False and shaped.stale == [ReadRef(id="r1", version=0)]
