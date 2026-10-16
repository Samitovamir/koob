-- Cube memory ledger, core schema (§3.2).
-- Applied by koob.memory.store.migrate inside BEGIN IMMEDIATE ... COMMIT.
-- Pragmas (WAL, foreign_keys, busy_timeout, synchronous, secure_delete, auto_vacuum) are set
-- per connection by store.open, never here: a PRAGMA inside a transaction is ignored.
--
-- Conventions
--   * id TEXT PRIMARY KEY is a ULID; times are UTC ISO-8601 text.
--   * Every kind row (entities, facts, ...) shares its id with its `records` header.
--   * `fts_rowid` (records, aliases, raw_chunks) is an explicit integer key for the FTS rowid:
--     plain rowids of a TEXT-keyed table may change on VACUUM, FTS rowids must not.
--     A trigger fills it on insert when the writer leaves it NULL.
--   * JSON columns hold JSON text; `labels` is the §5.3 bitmask.

-- ---------------------------------------------------------------- meta and migrations

CREATE TABLE schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- One row. memory_version increments on every commit (§7.1); a restore continues from
-- max(backup, journal, hot log) + 1.
CREATE TABLE meta (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    memory_version   INTEGER NOT NULL DEFAULT 0,
    policy_version   INTEGER NOT NULL DEFAULT 0,
    schema_version   INTEGER NOT NULL DEFAULT 0,
    embedder_id      TEXT,
    sweep_watermark  INTEGER NOT NULL DEFAULT 0,   -- last koob_messages.id ingested (§6.2)
    updated_at       TEXT NOT NULL
);
INSERT INTO meta (id, memory_version, policy_version, schema_version, updated_at)
VALUES (1, 0, 0, 1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));

-- ---------------------------------------------------------------- records (§3.2, §4)

CREATE TABLE records (
    id                TEXT PRIMARY KEY,
    fts_rowid         INTEGER UNIQUE,
    kind              TEXT NOT NULL CHECK (kind IN (
                          'entity', 'fact', 'preference', 'impression', 'decision',
                          'work_state', 'intention', 'loop', 'lesson',
                          'episode', 'criterion')),
    version           INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                          'active', 'archived', 'superseded', 'retracted', 'needs_review',
                          'forgotten', 'purged')),
    review            INTEGER NOT NULL DEFAULT 0 CHECK (review IN (0, 1)),
    labels            INTEGER NOT NULL DEFAULT 0,
    labels_final      INTEGER NOT NULL DEFAULT 0 CHECK (labels_final IN (0, 1)),
    author_tier       TEXT NOT NULL CHECK (author_tier IN (
                          'user', 'user_confirmed', 'user_room', 'other', 'tool', 'system')),
    author_entity_id  TEXT,
    importance        REAL,
    confidence        REAL,
    recorded_at       TEXT NOT NULL,
    retired_at        TEXT,
    scope_key         TEXT,
    pinned_category   TEXT,
    origin_quote      TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX records_kind_status ON records (kind, status);
CREATE INDEX records_scope ON records (scope_key) WHERE scope_key IS NOT NULL;
CREATE INDEX records_review ON records (id) WHERE review = 1;

CREATE TRIGGER records_fts_rowid AFTER INSERT ON records WHEN NEW.fts_rowid IS NULL
BEGIN
    UPDATE records SET fts_rowid = (SELECT coalesce(max(fts_rowid), 0) + 1 FROM records)
    WHERE id = NEW.id;
END;

-- Never version-bearing.
CREATE TABLE record_stats (
    record_id     TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    uses          INTEGER NOT NULL DEFAULT 0,
    last_used_at  TEXT,
    heat          REAL NOT NULL DEFAULT 0
);

-- Closed relation vocabulary, loaded from config/memory/relations.yaml (§4.1).
CREATE TABLE relations (
    relation     TEXT PRIMARY KEY,
    family       TEXT NOT NULL,
    cardinality  TEXT NOT NULL CHECK (cardinality IN ('single', 'multi')),
    value_type   TEXT NOT NULL
);

CREATE TABLE entities (
    id            TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    etype         TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    stable_ids    TEXT NOT NULL DEFAULT '[]'   -- JSON [{kind, value, tier, source}] (§13.3)
);

CREATE TABLE aliases (
    id          TEXT PRIMARY KEY,
    fts_rowid   INTEGER UNIQUE,
    entity_id   TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    alias_norm  TEXT NOT NULL,                  -- NFKC + casefold, for exact match
    script      TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('name', 'diminutive', 'handle', 'nickname', 'translit')),
    tier        TEXT NOT NULL CHECK (tier IN (
                    'user', 'user_confirmed', 'user_room', 'other', 'tool', 'system')),
    confirmed   INTEGER NOT NULL DEFAULT 0 CHECK (confirmed IN (0, 1)),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX aliases_norm ON aliases (alias_norm);
CREATE INDEX aliases_entity ON aliases (entity_id);

CREATE TRIGGER aliases_fts_rowid AFTER INSERT ON aliases WHEN NEW.fts_rowid IS NULL
BEGIN
    UPDATE aliases SET fts_rowid = (SELECT coalesce(max(fts_rowid), 0) + 1 FROM aliases)
    WHERE id = NEW.id;
END;

-- Facts and preferences (records.kind = 'preference', relation = 'prefers'); G4 dims (§4.1).
CREATE TABLE facts (
    id                 TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    subject_id         TEXT NOT NULL,
    relation           TEXT NOT NULL,
    value_text         TEXT,
    value_norm         TEXT,
    value_entity_id    TEXT,
    text_en            TEXT NOT NULL,
    keywords_ru        TEXT,
    keywords_en        TEXT,
    quote_orig         TEXT,
    lang_orig          TEXT,
    fkind              TEXT NOT NULL CHECK (fkind IN ('state', 'event', 'plan', 'condition', 'opinion')),
    polarity           TEXT NOT NULL DEFAULT 'affirmed' CHECK (polarity IN ('affirmed', 'negated')),
    valid_from         TEXT,
    valid_to           TEXT,
    stability          TEXT CHECK (stability IS NULL OR stability IN ('lasting', 'one_off', 'habit')),
    applicability      TEXT,                    -- JSON {for_whom, task, audience, situation, period, channel, general, unstated}
    speaker_entity_id  TEXT,
    source_raw_id      TEXT
);
CREATE INDEX facts_slot ON facts (subject_id, relation);
CREATE INDEX facts_value ON facts (value_norm) WHERE value_norm IS NOT NULL;
CREATE INDEX facts_source ON facts (source_raw_id) WHERE source_raw_id IS NOT NULL;

CREATE TABLE impressions (
    id                    TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    about_entity_id       TEXT NOT NULL,
    text_en               TEXT NOT NULL,
    quote_orig            TEXT,
    observed_at           TEXT NOT NULL,
    decay_half_life_days  REAL NOT NULL DEFAULT 90
);
CREATE INDEX impressions_about ON impressions (about_entity_id);

CREATE TABLE decisions (
    id                   TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    matter_id            TEXT,
    domain               TEXT CHECK (domain IS NULL OR domain IN (
                             'purchase', 'travel', 'study', 'work', 'career', 'home', 'health',
                             'social', 'money', 'tools')),
    chosen_text          TEXT NOT NULL,
    alternatives         TEXT NOT NULL DEFAULT '[]',   -- JSON [{text, reasons}]
    revision_conditions  TEXT NOT NULL DEFAULT '[]',   -- JSON, each backed by an intention
    decided_at           TEXT
);
CREATE INDEX decisions_matter ON decisions (matter_id);

-- AND within group_no, OR across groups (§8.1). One row per member link.
CREATE TABLE justifications (
    id                TEXT PRIMARY KEY,
    target_record_id  TEXT NOT NULL REFERENCES records (id) ON DELETE CASCADE,
    group_no          INTEGER NOT NULL,
    member_record_id  TEXT NOT NULL,           -- a record id or a raw_chunks id (a raw root)
    provenance        TEXT NOT NULL CHECK (provenance IN ('stated', 'rule', 'inferred')),
    created_at        TEXT NOT NULL,
    UNIQUE (target_record_id, group_no, member_record_id)
);
CREATE INDEX justifications_member ON justifications (member_record_id);

CREATE TABLE work_states (
    id               TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    matter_id        TEXT NOT NULL,
    state            TEXT NOT NULL,            -- JSON (§14)
    last_resumed_at  TEXT
);
CREATE INDEX work_states_matter ON work_states (matter_id);

CREATE TABLE intentions (
    id          TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    trigger     TEXT NOT NULL,                 -- JSON, slot-bound (§12)
    completion  TEXT,
    action      TEXT,
    state       TEXT NOT NULL DEFAULT 'armed' CHECK (state IN (
                    'armed', 'fired', 'done', 'cancelled', 'needs_attention')),
    fired_at    TEXT,
    outcome     TEXT
);
CREATE INDEX intentions_state ON intentions (state);

-- Processed set: a redelivered event collides.
CREATE TABLE intention_events (
    intention_id  TEXT NOT NULL REFERENCES intentions (id) ON DELETE CASCADE,
    event_key     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (intention_id, event_key)
);

CREATE TABLE loops (
    id       TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    owner    TEXT NOT NULL,                    -- 'user', 'cube' or an entity id
    text_en  TEXT NOT NULL,
    due      TEXT,
    state    TEXT NOT NULL DEFAULT 'open' CHECK (state IN (
                 'open', 'done', 'overdue', 'cancelled', 'unclear'))
);
CREATE INDEX loops_due ON loops (state, due);

CREATE TABLE lessons (
    id             TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    rule_en        TEXT NOT NULL,
    applicability  TEXT,                       -- JSON
    helpful        INTEGER NOT NULL DEFAULT 0,
    harmful        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE criteria (
    id             TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    text_en        TEXT NOT NULL,
    applicability  TEXT,                       -- JSON
    evidence       TEXT NOT NULL DEFAULT '[]', -- JSON [record ids]
    exceptions     TEXT NOT NULL DEFAULT '[]', -- JSON
    status         TEXT NOT NULL DEFAULT 'hypothesis' CHECK (status IN (
                       'hypothesis', 'confirmed', 'retired'))
);

CREATE TABLE episodes (
    id          TEXT PRIMARY KEY REFERENCES records (id) ON DELETE CASCADE,
    event_type  TEXT,
    span_start  TEXT,
    span_end    TEXT,
    place       TEXT,
    outcome     TEXT,
    summary_en  TEXT NOT NULL
);
CREATE INDEX episodes_span ON episodes (span_start);

CREATE TABLE episode_links (
    episode_id  TEXT NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('participant', 'place', 'cause', 'outcome', 'about')),
    target_id   TEXT NOT NULL,
    PRIMARY KEY (episode_id, role, target_id)
);
CREATE INDEX episode_links_target ON episode_links (target_id);

-- ---------------------------------------------------------------- raw archive (§6.2)

CREATE TABLE raw_chunks (
    id                TEXT PRIMARY KEY,
    fts_rowid         INTEGER UNIQUE,
    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'forgotten', 'purged')),
    rev               INTEGER NOT NULL DEFAULT 1,
    source_kind       TEXT NOT NULL,
    source_ref        TEXT,
    chat_id           TEXT,
    turn_id           TEXT,
    author_tier       TEXT NOT NULL CHECK (author_tier IN (
                          'user', 'user_confirmed', 'user_room', 'other', 'tool', 'system')),
    author_entity_id  TEXT,
    ts                TEXT NOT NULL,
    text              TEXT,                    -- NULL once purged
    content_hmac      TEXT NOT NULL,
    labels            INTEGER NOT NULL DEFAULT 0,
    labels_final      INTEGER NOT NULL DEFAULT 0 CHECK (labels_final IN (0, 1)),
    cloud_seen        INTEGER NOT NULL DEFAULT 0 CHECK (cloud_seen IN (0, 1)),
    policy_version    INTEGER NOT NULL,
    quarantined       INTEGER NOT NULL DEFAULT 0 CHECK (quarantined IN (0, 1)),
    junk              INTEGER NOT NULL DEFAULT 0 CHECK (junk IN (0, 1)),
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX raw_turn ON raw_chunks (turn_id);
CREATE INDEX raw_chat_ts ON raw_chunks (chat_id, ts);
CREATE INDEX raw_hmac ON raw_chunks (content_hmac);

CREATE TRIGGER raw_fts_rowid AFTER INSERT ON raw_chunks WHEN NEW.fts_rowid IS NULL
BEGIN
    UPDATE raw_chunks SET fts_rowid = (SELECT coalesce(max(fts_rowid), 0) + 1 FROM raw_chunks)
    WHERE id = NEW.id;
END;

-- Per Cube turn: chats and paths its steps read (§5.1). chat_id holds a path for files.
CREATE TABLE turn_sources (
    turn_id      TEXT NOT NULL,
    source_kind  TEXT NOT NULL,
    chat_id      TEXT NOT NULL,
    requested    INTEGER NOT NULL DEFAULT 0 CHECK (requested IN (0, 1)),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (turn_id, source_kind, chat_id)
);
CREATE INDEX turn_sources_chat ON turn_sources (chat_id);

-- ---------------------------------------------------------------- lineage and derived

CREATE TABLE edges (
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    etype       TEXT NOT NULL CHECK (etype IN ('derived_from', 'supports', 'supersedes', 'mentions', 'about')),
    created_at  TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, etype)
);
CREATE INDEX edges_to ON edges (to_id, etype);

CREATE TABLE derived (
    id          TEXT PRIMARY KEY,
    dkind       TEXT NOT NULL CHECK (dkind IN ('core_section', 'summary', 'assoc_link')),
    dkey        TEXT,                          -- core section name, summary day, goal id
    content     TEXT NOT NULL,
    read_set    TEXT NOT NULL,                 -- JSON [[id, version], ...]
    labels      INTEGER NOT NULL DEFAULT 0,
    built_at    TEXT NOT NULL,
    valid       INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0, 1)),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX derived_kind_key ON derived (dkind, dkey);
CREATE INDEX derived_invalid ON derived (id) WHERE valid = 0;

-- Reverse index of read sets: who read item_id at which version. Invalidation (§7.3) and
-- the open-proposal check look up holders by item without parsing JSON.
CREATE TABLE read_deps (
    holder_kind  TEXT NOT NULL CHECK (holder_kind IN ('derived', 'action', 'job', 'outbox', 'token')),
    holder_id    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    version      INTEGER NOT NULL,
    PRIMARY KEY (holder_kind, holder_id, item_id)
);
CREATE INDEX read_deps_item ON read_deps (item_id);

-- ---------------------------------------------------------------- served set (§7.6)

CREATE TABLE served (
    id          INTEGER PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    session_id  TEXT,
    item_id     TEXT NOT NULL,
    item_kind   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    turn_id     TEXT,
    via         TEXT NOT NULL CHECK (via IN ('core', 'basket', 'tool', 'write', 'update', 'session')),
    created_at  TEXT NOT NULL
);
CREATE INDEX served_thread_item ON served (thread_id, item_id);
CREATE INDEX served_item ON served (item_id);
CREATE INDEX served_session ON served (session_id) WHERE session_id IS NOT NULL;

CREATE TABLE pending_updates (
    thread_id    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    old_version  INTEGER NOT NULL,
    new_version  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (thread_id, item_id)
);

-- ---------------------------------------------------------------- forgetting (§9)

CREATE TABLE suppressions (
    hmac        TEXT PRIMARY KEY,              -- hmac(subject_id, family, value_norm)
    forget_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE raw_retractions (
    raw_id                TEXT NOT NULL,
    record_id             TEXT NOT NULL,
    retracted_value_norm  TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    PRIMARY KEY (raw_id, record_id, retracted_value_norm)
);
CREATE INDEX raw_retractions_record ON raw_retractions (record_id);

CREATE TABLE retired_values (
    hmac        TEXT PRIMARY KEY,              -- hmac(value_norm) of a contact-like value
    record_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE trash (
    id          TEXT PRIMARY KEY,
    forget_id   TEXT NOT NULL,
    store       TEXT NOT NULL,                 -- 'memory', 'career', 'vectors', ...
    row_ref     TEXT NOT NULL,
    payload     TEXT,                          -- NULL once purged
    until       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX trash_forget ON trash (forget_id);
CREATE INDEX trash_until ON trash (until);

-- ---------------------------------------------------------------- work queue and hands

CREATE TABLE jobs (
    id               TEXT PRIMARY KEY,
    jkind            TEXT NOT NULL,
    payload          TEXT NOT NULL,            -- JSON
    state            TEXT NOT NULL DEFAULT 'queued' CHECK (state IN (
                         'queued', 'running', 'done', 'failed', 'parked', 'needs_attention')),
    attempts         INTEGER NOT NULL DEFAULT 0,
    not_before       TEXT,
    idempotency_key  TEXT NOT NULL UNIQUE,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX jobs_ready ON jobs (state, not_before);

-- Outbox-first actions (G14, §16.2).
CREATE TABLE actions (
    id                TEXT PRIMARY KEY,
    action_key        TEXT NOT NULL UNIQUE,
    origin_key        TEXT NOT NULL,
    seq               INTEGER NOT NULL,
    tool              TEXT NOT NULL,
    target_entity_id  TEXT,
    stable_target     TEXT NOT NULL,
    summary_en        TEXT,
    state             TEXT NOT NULL DEFAULT 'pending' CHECK (state IN (
                          'pending', 'ok', 'failed', 'unknown', 'undone')),
    read_set          TEXT NOT NULL DEFAULT '[]',
    turn_id           TEXT,
    intention_id      TEXT,
    reserved_at       TEXT NOT NULL,
    closed_at         TEXT
);
CREATE INDEX actions_origin_target ON actions (origin_key, tool, stable_target);
CREATE INDEX actions_open ON actions (state) WHERE state IN ('pending', 'unknown');

-- ---------------------------------------------------------------- policy (§5.2, §5.6)

CREATE TABLE chat_policy (
    chat_id         TEXT PRIMARY KEY,
    mode            TEXT NOT NULL DEFAULT 'normal' CHECK (mode IN ('normal', 'no_dig', 'banned')),
    policy_version  INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE admit_windows (
    id            TEXT PRIMARY KEY,
    chat_id       TEXT NOT NULL,
    sender_id     TEXT,
    after_ts      TEXT NOT NULL,
    until_ts      TEXT,
    reason        TEXT NOT NULL,
    action_key    TEXT,
    intention_id  TEXT,
    created_at    TEXT NOT NULL,
    CHECK (action_key IS NOT NULL OR intention_id IS NOT NULL)
);
CREATE INDEX admit_windows_chat ON admit_windows (chat_id, after_ts);

-- ---------------------------------------------------------------- feeds (content-free)

CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('write', 'fix', 'forget', 'mode', 'restore', 'undo')),
    memory_version  INTEGER NOT NULL,
    item_ids        TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE audit (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,
    record_id  TEXT,
    actor      TEXT NOT NULL
);

-- ---------------------------------------------------------------- full-text search (§3.2, §10.3)
-- Text is stemmed in Python (PyStemmer, per script) before it is written; FTS only tokenizes.
-- rowid = the owning row's fts_rowid. All three keep secure-delete on.

CREATE VIRTUAL TABLE fts_records USING fts5 (
    text, keywords, aliases,
    tokenize = 'unicode61'
);
INSERT INTO fts_records (fts_records, rank) VALUES ('secure-delete', 1);

CREATE VIRTUAL TABLE fts_names USING fts5 (
    name,
    tokenize = 'trigram'
);
INSERT INTO fts_names (fts_names, rank) VALUES ('secure-delete', 1);

CREATE VIRTUAL TABLE fts_raw USING fts5 (
    text,
    content = '', contentless_delete = 1,
    tokenize = 'unicode61'
);
INSERT INTO fts_raw (fts_raw, rank) VALUES ('secure-delete', 1);
