"""Policy fixtures: rows go into a migrated memory.db with plain SQL, so these tests do not
depend on the ledger. Offline, tmp_path only."""

from __future__ import annotations

import hashlib
import json

import pytest

from koob.memory import policy
from koob.memory.model import Label, TurnCtx, TurnOrigin

TS = "2026-10-12T09:00:00.000000Z"
DEFAULT_ROUTING = {
    "unlabeled": ["cloud"],
    **{name: ["local"] for name in policy.LABEL_NAMES},
}
DEFAULT_SENSITIVE = [n for n in policy.LABEL_NAMES if n != "location_precise"]


@pytest.fixture(autouse=True)
def _tables():
    """The contract's default tables, independent of the local settings file."""
    policy.configure(routing=DEFAULT_ROUTING, sensitive_labels=DEFAULT_SENSITIVE)
    policy.set_classifier(None)
    yield
    policy.configure()
    policy.set_classifier(None)


def owner(words: str, *, turn_id: str = "turn-1", **kw) -> TurnCtx:
    return TurnCtx(
        turn_id=turn_id,
        thread_id="thread-1",
        origin=TurnOrigin.OWNER,
        owner_words=words,
        clock=kw.pop("clock", TS),
        **kw,
    )


class Mem:
    """Plain-SQL writer for the tables policy reads."""

    def __init__(self, conn):
        self.conn = conn

    def _record(
        self,
        rid,
        kind,
        *,
        tier,
        labels,
        final,
        status,
        author=None,
        pinned=None,
        quote=None,
        version=1,
    ):
        self.conn.execute(
            "INSERT INTO records (id, kind, version, status, labels, labels_final, author_tier,"
            " author_entity_id, recorded_at, pinned_category, origin_quote, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                kind,
                version,
                status,
                int(labels),
                int(final),
                tier,
                author,
                TS,
                pinned,
                quote,
                TS,
                TS,
            ),
        )

    def entity(self, eid, name, *aliases, etype="person"):
        self._record(eid, "entity", tier="user", labels=Label.NONE, final=True, status="active")
        self.conn.execute(
            "INSERT INTO entities (id, etype, display_name) VALUES (?, ?, ?)", (eid, etype, name)
        )
        for i, alias in enumerate((name, *aliases)):
            self.conn.execute(
                "INSERT INTO aliases (id, entity_id, alias, alias_norm, script, kind, tier,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, 'cyrl', 'name', 'user', ?, ?)",
                (f"{eid}-a{i}", eid, alias, policy.norm(alias), TS, TS),
            )
        return eid

    def fact(
        self,
        fid,
        *,
        subject,
        text_en,
        quote=None,
        labels=Label.NONE,
        tier="user",
        final=True,
        status="active",
        applicability=None,
        pinned=None,
        keywords_ru=None,
        source_raw_id=None,
        author=None,
        kind="fact",
        relation="has",
    ):
        self._record(
            fid,
            kind,
            tier=tier,
            labels=labels,
            final=final,
            status=status,
            author=author,
            pinned=pinned,
        )
        self.conn.execute(
            "INSERT INTO facts (id, subject_id, relation, text_en, keywords_ru, quote_orig,"
            " fkind, applicability, source_raw_id) VALUES (?, ?, ?, ?, ?, ?, 'state', ?, ?)",
            (
                fid,
                subject,
                relation,
                text_en,
                keywords_ru,
                quote,
                json.dumps(applicability) if applicability else None,
                source_raw_id,
            ),
        )
        return fid

    def raw(
        self,
        rid,
        *,
        text,
        chat=None,
        turn=None,
        tier="other",
        ts=TS,
        labels=Label.NONE,
        final=True,
        author=None,
        status="active",
        source_kind="telegram",
    ):
        self.conn.execute(
            "INSERT INTO raw_chunks (id, status, source_kind, chat_id, turn_id, author_tier,"
            " author_entity_id, ts, text, content_hmac, labels, labels_final, policy_version,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                rid,
                status,
                source_kind,
                chat,
                turn,
                tier,
                author,
                ts,
                text,
                hashlib.sha256(text.encode()).hexdigest(),
                int(labels),
                int(final),
                TS,
                TS,
            ),
        )
        return rid

    def derived(self, did, read_set, *, labels, content="summary", dkind="summary"):
        self.conn.execute(
            "INSERT INTO derived (id, dkind, dkey, content, read_set, labels, built_at,"
            " created_at, updated_at) VALUES (?, ?, '2026-10-12', ?, ?, ?, ?, ?, ?)",
            (
                did,
                dkind,
                content,
                json.dumps([[r.id, r.version] for r in read_set]),
                int(labels),
                TS,
                TS,
                TS,
            ),
        )
        return did

    def count(self, table, where="1", params=()):
        return self.conn.execute(f"SELECT count(*) FROM {table} WHERE {where}", params).fetchone()[
            0
        ]

    def table_counts(self):
        tables = [
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                " AND name NOT LIKE 'fts_%'"
            )
        ]
        return {t: self.count(t) for t in tables}


@pytest.fixture
def db(mem):
    return Mem(mem)
