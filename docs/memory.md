# Memory engine

`src/koob/memory/` is Koob's long-term memory: a ledger of what the assistant knows about the user, where each item came from, which version is current, and what has been forgotten. It was written spec-first: a normative specification in which every MUST names its acceptance test, and a test directory (`tests/memory/`) laid out by spec area.

**Status in one paragraph.** The engine is built behind a feature switch and is not the memory the running assistant uses today; the assistant still runs on an older note-based memory. The code and its offline tests exist for the ledger, policy, writers, recall, forgetting, and maintenance. Its quality is not yet proven: an internal benchmark in October found it behind the previous note-based memory on recall, with named correctness faults that are open work. Nothing in this document should be read as a claim that the engine works well in daily use.

## What it stores

Everything is a record in SQLite (`migrations/memory/0001_core.sql`), one table per kind, sharing a common header (`model.RecordHeader`): identity, version, status, author tier, sensitivity labels, time of validity, and lineage.

| Kind | What it holds |
|---|---|
| Entities and aliases | People, places, organizations, and the names they go by, with transliteration and merge/split (`entities.py`) |
| Facts | Typed assertions about the user and the world, with polarity, stability, and cardinality (`model.FKind`, `Polarity`, `Stability`) |
| Impressions | Soft observations, kept apart from facts |
| Decisions and justifications | What was chosen, why, the rejected alternatives, and what would revise it (`workstate.py`, `justify.py`) |
| Work states | The goal, direction, dead ends, open questions, and next step of a matter, for resuming it later (`workstate.py`) |
| Intentions, loops, and lessons | Things to do when a trigger fires, open loops, and what was learned (`intentions.py`) |
| Criteria | Scoped preferences with evidence and exceptions (`criteria.py`) |
| Episodes | Events with participants, place, time, outcome, and cause links (`episodes.py`) |
| Raw chunks | The archived conversation text records are extracted from, with a content hash (`ingest.py`) |
| Derived items | Summaries and other items computed from records, with the read set they depend on |

Three FTS5 tables index records, names, and raw text for lexical recall. A separate vectors file holds embeddings and can be rebuilt (`recall/vectors.py`).

## Provenance

Every record carries an author tier (`model.AuthorTier`): `user`, `user_confirmed`, `user_room` (speech on speakerphone), `other` (a person's message), `tool` (a page, file, mail, or tool output), and `system` (worker output). The tier bounds what a record may become: only the user's own words may become facts, preferences, constraints, and lessons; what other people or pages say becomes a claim with its source, never an instruction. Lineage (`relations`, `edges`) links a record to the raw chunk it was extracted from and to the records it was derived from, and `where_from()` in `api.py` answers "where did you get that" from the ledger. Tested in `tests/memory/security/` and `tests/memory/inspect/`.

Sensitivity labels (`model.Label`: health, money, identity documents, intimate, precise location, third-party private, legal, minor, and a fail-closed "unknown sensitive") are set at admission and inherited along lineage. `policy.route()` names which processor may read an item with given labels; a job with no allowed processor waits rather than downgrading.

## Versioning and forgetting

A record is never edited in place. A correction (`api.fix()`) writes a new version and supersedes the old one in the ledger (`ledger.py`); `retired_values` keeps what must not be shown as current, and `pending_updates` carries a correction to derived items that read the old value. Every write goes through `store.commit()` with the read set it was computed from, so a write made from stale premises is rejected (`store.py`).

Forgetting (`forget.py`, `api.forget()`) has three stages:

1. **Stop use.** One commit moves the content of the target records, what was derived from them, and the raw of their lineage into `trash`, blanks it in place, drops the FTS rows, writes suppressions so the same content is not re-derived, and journals the step.
2. **Trash.** `undo_forget()` restores everything from the trash within the window.
3. **Verified purge.** The purge walks the full scope, including backups and the vector file, and reports "deleted completely" only after a verified checkpoint and a clean token scan; otherwise it retries with backoff. `purge_status()` tells the user where it stands.

Suppression keys are keyed HMACs of the forgotten value (`forget.keyed_hmac()`), so the suppression table does not itself hold the forgotten content. Tested in `tests/memory/forget/`.

## Recall

Recall runs in code before the turn, with no model call (`recall/basket.py`). A cue is parsed from the turn (`recall/cue.py`); several finders run with per-finder deadlines (stemmed lexical FTS in `recall/lexical.py`, trigram names, vectors), with the sensitivity mask applied in SQL first; candidates are fused (`recall/fusion.py`), gated (`recall/gate.py`), and cut to a token budget. The result is a basket of lines with ids the model can cite and a read set recorded as `served`, so a later correction or forget can find the turns it affects. A core view of stable facts (`recall/core_view.py`) is rebuilt from the ledger and ordered for cache stability.

The use-now policy (`policy.py`) decides which items the model sees: the user's own items apply wherever they are relevant, with a one-line note; sensitive items from other people, chats, and files answer a direct question only; on a turn that sends something out, sensitive items stay visible to the model but held back from the message.

## How it is tested

- `tests/memory/` holds more than a thousand tests, one directory per spec area (`policy`, `forget`, `recall`, `security`, `races`, `storage`, `consistency`, `truth`, `workers`, `maintenance`, and others). They run offline against an in-memory SQLite database; there is no network and no model in the tests.
- Acceptance tests for parts not yet built are kept in the suite marked `xfail(strict=True)` with the name of the missing part, so a part that lands flips its own tests.
- `tests/memory/races/` covers correction during processing, forget during an answer, a chat-mode change, restart mid-cascade, event redelivery, and restore of an old backup.
- `tests/memory/test_boundaries.py` enforces that nothing in the package imports the rest of Koob.
- `memory/eval/` is an evaluation harness: a runner that replays timelines (messages, foreign text, corrections, forgets, mode changes, restarts, worker interleavings, backup and restore) against `api.py`, with code graders for must-include and must-not-include checks in every channel and a cross-family judge for free text. The scenarios and their results are not published here.

## Built and not yet proven

**Built, with offline tests:** the SQLite store with versions and read-set validation; admission policy with never-store spans, labels, tiers, chat modes, and use-now; the ledger with supersede and invalidation; entities and aliases; three-stage forgetting with journal and purge; the inspection commands; the hot writer (remember, fix, safety net); raw ingest with junk gate and quarantine; typed extraction through a scheduler; episodes and daily summaries; a one-way importer from the previous note store; lexical, vector, and fused recall with the core view; the justification cascade; work state and resume; intentions and loops; nightly consolidation, an exam worker, and a doctor; criteria; the turn adapters; and the evaluation harness.

**Not yet proven:** quality in daily use. The internal benchmark run in October showed the engine behind the previous note-based memory on recall hit rate and found engine faults: a correction that did not retire every carrier of an old value, a forget that did not stick on some probes, a core view not scoped to the question. These are open work, not shipped behavior.

The engine's records are written in English so that it can serve users beyond the first one.

**A note on references.** Section numbers like `§9.8` in the code's comments point to the internal memory specification, which is not published yet.
