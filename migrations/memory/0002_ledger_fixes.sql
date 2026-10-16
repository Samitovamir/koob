-- Memory ledger v0.4 fixes to the initial schema (§20, row 17; §3.2, §5.6, §7.4, §12,
-- §13.3). Runs inside the migrations runner's BEGIN IMMEDIATE.

-- §3.2, §12 (A7): an intention's action read set is mirrored into the reverse index with
-- holder_kind = 'intention'. SQLite cannot widen a CHECK in place: rebuild the table.
CREATE TABLE read_deps_new (
    holder_kind  TEXT NOT NULL CHECK (holder_kind IN (
                     'derived', 'action', 'job', 'outbox', 'token', 'intention')),
    holder_id    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    version      INTEGER NOT NULL,
    PRIMARY KEY (holder_kind, holder_id, item_id)
);
INSERT INTO read_deps_new (holder_kind, holder_id, item_id, version)
    SELECT holder_kind, holder_id, item_id, version FROM read_deps;
DROP TABLE read_deps;
ALTER TABLE read_deps_new RENAME TO read_deps;
CREATE INDEX read_deps_item ON read_deps (item_id);

-- §5.6 (F4): a direct-reply window names the Cube's sent message by its platform id, written
-- when the send closes `ok`; admission matches `reply_to == cube_message_id`. Rows written
-- before this migration keep `reason = 'reply:<id>'`.
ALTER TABLE admit_windows ADD COLUMN cube_message_id TEXT;
CREATE INDEX admit_windows_cube_message ON admit_windows (chat_id, cube_message_id)
    WHERE cube_message_id IS NOT NULL;

-- §7.4: a derived item's revision, bumped on every rebuild. A derived ref's `version` is this
-- `rev`, so a prompt holding the text built before a forget is stale once the item is rebuilt,
-- though it is `valid` again.
ALTER TABLE derived ADD COLUMN rev INTEGER NOT NULL DEFAULT 0;

-- §13.3 (G9): stable-id lookup without scanning `entities.stable_ids` JSON. One row per
-- (kind, value, entity); written in the entity's own transaction, so it mirrors the JSON.
CREATE TABLE entity_stable_ids (
    kind       TEXT NOT NULL,
    value      TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    PRIMARY KEY (kind, value, entity_id)
);
CREATE INDEX entity_stable_ids_entity ON entity_stable_ids (entity_id);
INSERT OR IGNORE INTO entity_stable_ids (kind, value, entity_id)
    SELECT json_extract(j.value, '$.kind'), json_extract(j.value, '$.value'), e.id
    FROM entities e, json_each(e.stable_ids) j
    WHERE json_extract(j.value, '$.kind') IS NOT NULL
      AND json_extract(j.value, '$.value') IS NOT NULL;
-- The mirror is kept by triggers, so every writer of `entities.stable_ids` (ADD, UPDATE
-- add_stable_ids, MERGE, forget and purge) keeps it in the same transaction.
CREATE TRIGGER entities_stable_ids_ins AFTER INSERT ON entities BEGIN
    INSERT OR IGNORE INTO entity_stable_ids (kind, value, entity_id)
        SELECT json_extract(j.value, '$.kind'), json_extract(j.value, '$.value'), NEW.id
        FROM json_each(NEW.stable_ids) j
        WHERE json_extract(j.value, '$.kind') IS NOT NULL
          AND json_extract(j.value, '$.value') IS NOT NULL;
END;
CREATE TRIGGER entities_stable_ids_upd AFTER UPDATE OF id, stable_ids ON entities BEGIN
    DELETE FROM entity_stable_ids WHERE entity_id = OLD.id;
    INSERT OR IGNORE INTO entity_stable_ids (kind, value, entity_id)
        SELECT json_extract(j.value, '$.kind'), json_extract(j.value, '$.value'), NEW.id
        FROM json_each(NEW.stable_ids) j
        WHERE json_extract(j.value, '$.kind') IS NOT NULL
          AND json_extract(j.value, '$.value') IS NOT NULL;
END;
CREATE TRIGGER entities_stable_ids_del AFTER DELETE ON entities BEGIN
    DELETE FROM entity_stable_ids WHERE entity_id = OLD.id;
END;
