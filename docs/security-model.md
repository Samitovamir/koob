# Security model

Koob is an assistant with hands on a personal Mac. It reads the screen, runs shell commands, drives other apps through the Accessibility API, sends messages, and opens web pages. Most of the text it reads is not the user's. This document is the threat model and what holds it up, with the enforcement point for each rule. Function names refer to the public paths in this repository.

> The gate's code (`guard.py`) is not in this repository yet: it is still wired to the author's own accounts, sites and folders, and is being generalized before it is published. The function names below refer to it. The sanitizer and the memory engine are here.

Two words used throughout:

- **Enforced in code** means a deterministic check in Python or an OS-level sandbox rule. The model cannot talk its way past it.
- **Policy** means the persona prompt and the runtime's conventions. The model is told to behave this way, and the code around it is built to make the policy hold, but the final decision is the model's.

## 1. Threat model

| Asset | Threat | Where it comes from |
|---|---|---|
| The user's secrets: passwords, API keys, tokens, keychains, browser logins, one-time codes | A secret reaches the model and leaves the Mac with the conversation, or is written into logs or memory | The model reads a secret file, the screen shows one, a page or a message contains one, a shell command prints one |
| The user's authority over the assistant | Text the user did not write steers the assistant into an action: sending, deleting, paying, changing settings, writing memory | Prompt injection in pages, mail, messages, files, screen content, speech in the room; forged "instruction" markup inside data |
| The user's money | A payment, transfer, or paid subscription without the user deciding | A page's "subscribe" button, a checkout flow reached during a task |
| Other people's data and the user's private folders | The assistant reads mail, messages, bank, medical, or private folders the user did not ask it to open | A task wanders; a background wake reads what it was not told to |
| The integrity of the assistant itself | The model edits its own rules, the gate, or persistence (launch agents, shell profiles) | A model error or an injected instruction |

The attacker model is prompt injection by anyone who can get text in front of the assistant, plus ordinary model error. The user is trusted: the user's own words are the permission. The protections do not try to protect the user from themselves, with one exception (money) and a small floor that holds even against the user's order.

## 2. Foreign text is data, never instructions

### 2.1 What counts as foreign

Everything that is not the user's own text or voice in the user's own channel: web pages, search results, mail, chat messages, files, tool output, what the eyes see on the screen, and other people's speech picked up on speakerphone. The user's own words are recognized by the runtime from the request and the conversation row, never from the content of a message.

**Enforced in code.** The guard keeps a per-conversation `SessionState` (`guard.py`) with the taint flags `tainted`, `screen_seen`, `foreign_in_chat`, and `speaker`. `taints(tool_name, tool_input, state)` says whether a tool call brings foreign text into the conversation (the `TAINTING_TOOLS` set: web research and Telegram readers; shell commands judged by `bash_taints()`; AppleScript judged by `script_reads_foreign()`); `looks()` says whether it is a look at the screen. `after_tool()` sets the flags once such a call ran. The runtime calls `tool_verdict()` from the Agent SDK's pre-tool-use hook, so no tool call reaches the Mac without passing through it.

### 2.2 Fences and neutering

Foreign text is placed in the prompt inside named data fences (the `untrusted_*` family and `screen_data`). A forged closing tag inside the data would let injected text escape the fence and read as the user's instructions.

**Enforced in code** (`sanitize.py`): `inert()` runs on every untrusted text before it is fenced. It rewrites every ASCII `<` and `>` (and their full-width and small forms, after HTML entities are decoded) to look-alike glyphs that cannot form a tag, and rewrites every fence name wherever it appears, in any spelling: the text is read through a skeleton with NFKC normalization, invisible and combining characters dropped, Cyrillic and Greek look-alikes folded to Latin, and any connector or blank between the words of a name. `unfence()` is a second layer that removes any remaining fence-like tag, iterating until none is left (a tag nested inside another). `neuter_names()` handles the one-word fence names that are also ordinary words, rewriting them only where they stand like a tag.

The guard reads the user's words through the same fences: `owner_text()` strips every fenced block (greedily, to the last closing tag of the same name) before the user's words are compared with an action, so a block of foreign text quoted inside the user's message cannot pose as the user's order.

### 2.3 The tripwire

Once a conversation is tainted, the assistant does only what the user ordered, only to whom the user pointed. Anything else that goes outward or cannot be undone is a question back to the user.

**Enforced in code** (`guard.py`): `on_order()` calls `_tripwire()` on every call in a tainted turn. The tripwire takes the action's classes from `action_classes()` (send, delete, control of another app, a write that leaves the Mac; reads, looks, and harmless calls have no class) and compares them with the user's words this turn and the task's standing order: `ordered_verbs()` for what was ordered, `concrete_targets()` and `target_named()` for what it was aimed at, `points_out()` and `points_at_addressee()` for whether the user pointed outward and at whom, `hosts_in()` and `spoken_hosts()` for hosts the user named. A step whose verb the user did not say, or whose target the user did not name, returns one of the `TRIP_REASONS`, and those are in `ALWAYS_ASK`: the gate never answers them itself, no task pass lifts them (`NEVER_PASSABLE`), and they stay a question to the user. Spans the sanitizer marked as suspicious are `withheld_spans` on the state; a target that appears only there trips too.

What stays free in a tainted turn: pure reads (`shell_reads_only()`), looking, opening a plain page for reading (`taint_gate()`: opening an address is a send only when the address can carry text to a host the user did not name), windows and clicks that stay local, and file moves that can be undone (`local_and_undoable()`). `screen_gate()` adds the same rule after a look at the screen: every send becomes a question, since what the eyes saw could be in it.

The taint lives in one conversation and is lifted by the user's next order in a clean turn; a memory note written after foreign text stays marked foreign (`memory_tainted()`), so a page cannot plant a "fact" about the user.

### 2.4 Injection detection

**Enforced in code** (`sanitize.py`): `sanitize()` turns arbitrary untrusted input into bounded, normalized plain text: `strip_html()` drops script, style, hidden elements, comments, and iframes; `normalize_text()` applies NFKC and removes zero-width and control characters; the result is capped at `MAX_CHARS`. `detect_injection()` flags text that matches `INJECTION_PATTERNS` ("ignore previous instructions", "you are an AI", "new instructions", exfiltration phrasing around keys and passwords, override markers, hidden directives, JSON breakout, and injected JSON keys), and the markup itself adds flags for hidden elements, directive-like HTML comments, and invisible characters.

The detector flags and never blocks. The reasoning is in the module docstring: a detector that blocks teaches an attacker to rephrase. Blocking is the isolation's job: the fences and the tripwire above, and the fact that the reader that digests pages has no tools. The flags surface as a badge and as `withheld_spans` for the tripwire.

### 2.5 What memory refuses

**Enforced in code** (`src/koob/memory/policy.py`): every write goes through `admit()` before anything is stored. A rejection writes one content-free audit line and nothing else. Each record carries an author tier (`model.AuthorTier`): the user's own words may become facts, preferences, constraints, and lessons; a message from another person becomes a claim with its source; a page, file, mail, or tool output becomes a claim and never an instruction; worker output never becomes a `user`-tier record. Stable identities and alias-to-person links come only from the user's tier or from platform metadata, so a message in a group chat cannot rebind a name to another account. Tests: `tests/memory/security/` (`test_tiers.py`, `test_tainted.py`, `test_alias.py`) and `tests/memory/policy/`.

## 3. Secrets never reach the model

This is the floor that holds even against the user's order.

**Enforced in code:**

- **Paths.** `is_secret()` in `guard.py` recognizes secret directories and files, credential stores (`_cred_store()`), browser credential databases (`_browser_secret()`), key files by name and content (`_key_file()`), and secret-looking globs. `path_verdict()` returns a `deny` for them, and `DENY_SECRET` is in `FLOOR_REASONS`: `on_order()` never looks at a deny, `apply_pass()` never grants one, `apply_auto()` never answers one. A directory that holds a secret store is refused for tree reads (`holds_secret()`), and a shell line that hunts for keys (`_hunts_keys()`) is refused as a whole.
- **Password fields.** `secure_field_floor()` refuses typing, pasting, menu items, and button presses while a password field has focus, whoever asks. The user types passwords and captchas.
- **Text paths.** `sanitize.redact_secrets()` replaces API keys, tokens, password lines, and one-time codes on every text path that can reach the model, and `redact_tree()` does the same for structured tool results. `scrub_secrets()` cleans memory notes and never returns what it hid, only how much. Tests: `tests/test_sanitize_secrets.py`.
- **Memory.** `policy.never_store_spans()` finds passwords, keys, tokens, one-time codes, full card and bank numbers, and private keys by pattern and entropy; the chunk is dropped, or in the user's own words the span becomes `[secret removed]` before the record is written (`tests/memory/policy/test_never_store.py`).
- **The shell.** Every shell command runs under a macOS Seatbelt sandbox with `denyRead` on the secret files and folders and on the user's private folders, no unsandboxed fallback, and loopback to the assistant's own server shut so a script cannot drive the app behind the user's back. The CLI that hosts the model is started with a clean environment, so exported secrets never reach the model's tools. (The sandbox profile is generated by the private runtime; the public guard is the layer that explains the refusal.)

**Policy:** the persona is told that keychains and password managers are never opened and that the user logs in. The code above makes the policy hold at the paths, the fields, and the text.

## 4. The user's word is the permission

Koob does not pile up confirmation dialogs. On the user's own turn, what the user said runs, with one line afterward saying what was done and an Undo where one exists.

**Enforced in code** (`guard.py`): `tool_verdict()` is the single entry point, called from the pre-tool-use hook of the Agent SDK. Its pipeline, in order:

1. `_tool_verdict()` and `_floor_holds()`: the per-tool classification (`_by_tool()`, `bash_verdict()`, `script_verdict()`, `code_verdict()`, `path_verdict()`, `site_verdict()`) into one of four decisions: `allow`, `ask`, `override` (an action a project rule forbids, named by the rule), or `deny`.
2. `screen_gate()` and `taint_gate()`: the after-a-look and after-foreign-text rules (section 2.3).
3. `on_order()`: on the user's own turn, every reason in `WORD_REASONS` (delete, install, send, network, system settings, the app's own code, personal folders the user pointed at, automation, hands, and so on) is answered by the gate itself with `auto=True`; the reason is kept for the step list and the log. What is never answered by the gate is `ALWAYS_ASK`: the money card, the tripwire reasons, the user's own "show me, do not send", and a calendar event.
4. `apply_pass()`: a task's standing order grants reasons from `PASSABLE` and nothing else; `NEVER_PASSABLE` (every floor reason, the tripwire, memory writes, facts about the user, money, an unknown tool) is never set aside by a task.
5. `apply_auto()`: the remaining asks that the one behavior mode answers.
6. `permanent_delete_floor()`: a delete past the Trash runs only when the user said so in words; a plain "delete" means the Trash.

On a turn that is not the user's (a scheduled wake, a task's own opening or report), the reasons in `BACKGROUND_ASK` stay questions; nothing runs in the background that the user did not order.

**Policy:** the one-line report after each step, and the Undo line, are the runtime's and the persona's job; the gate records which steps leave one (`DID_LINE_REASONS`).

## 5. Money is the one always-confirmed action

**Enforced in code** (`guard.py`): `R_PAY` is in `ALWAYS_ASK` and `NEVER_PASSABLE`. Money is recognized by money context, not by a verb: `_pays()` looks for a pay verb next to a price or a currency; `_pay_host()` for checkout, billing, and payment hosts; `_presses_pay_button()` and `_page_js_pays()` for a click, a Return, or a script on a checkout or subscribe page; `_pay_call()` for a payment call from the shell. A free "subscribe" is not money. Inside a task, `pay_matches()` and `step_amounts()` let a standing order cover a payment only step by step, when the task's own card names the merchant and a maximum. The confirmation card is editable before it runs.

## 6. The assistant's own integrity

**Enforced in code** (`guard.py`): `is_self()` recognizes the assistant's own code and rules, `is_instruction()` its instruction files, `is_persistence()` launch agents, shell profiles, and other places a change outlives the session, and `is_floor_file()` the tables the secret floor is built from. Writing to a floor file is a `deny` (`DENY_FLOOR_WRITE`, in `FLOOR_REASONS`); an edit that would drop a floor entry is caught by `_drops_floor_entry()`. The assistant's shell cannot reach the app's own HTTP API (`RULE_APP_API`, a floor reason), so a script cannot mark something as done behind the user's back. Model-written JavaScript in a browser tab has its own floor (`DENY_PAGE_JS` for scripts that build or send; `page_js_refusal()`).

In the user's own terminal, the assistant may type and press keys but never Enter (`sends_enter()`, `DENY_TERMINAL_ENTER`): that terminal is the one shell on the Mac outside the sandbox, and the keystroke that runs a line stays the user's. Run buttons in Script Editor, Automator, and Shortcuts are the user's for the same reason (`DENY_RUN_BY_HAND`, `DENY_SCRIPT_ENTER`).

## 7. Known limits

- The guard is heuristics over strings. It classifies commands, scripts, paths, and URLs by parsing and pattern tables, and it explains itself in plain words. It can be wrong in both directions. The OS sandbox is what enforces paths and network for the shell; the guard is the layer that routes the gray zone to the user.
- The injection detector is a flag, not a filter. The protection against injection is structural: fences, neutering, taint, and the tripwire.
- The hands act through the Accessibility API on whatever is in front. Target resolution (which window, which element, which chat) is the runtime's at decision time; in the pre-tool-use hook the guard sees only what the model claims and the last listing, which is why the hands verdict is asked again once the target is known.
- Everything above is tested offline, with fixtures, not against a live model: the gate's tests stay private with the gate for now; `tests/test_sanitize*.py` and `tests/memory/security/` are here. There is no formal proof, and no external audit yet.
