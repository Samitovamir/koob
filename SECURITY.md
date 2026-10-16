# Security policy

## Reporting a vulnerability

Please report vulnerabilities through GitHub's private vulnerability reporting for this repository (the "Security" tab, then "Report a vulnerability"). Do not open a public issue for a security problem.

Include what you found, how to reproduce it, and which component it is in. Reports are read by the maintainer; expect an acknowledgment within a week.

## Scope

In scope, and most welcome:

- A way for text the user did not write (a page, a message, a file, screen content) to steer the guard into allowing an outward or irreversible action the user did not order. See [docs/security-model.md](docs/security-model.md).
- A way for a secret to pass the redaction in `src/koob/sanitize.py` or the never-store policy in `src/koob/memory/policy.py`.
- A forged fence or instruction markup that survives `inert()` and `unfence()`.
- A forget in the memory engine that leaves content readable after it reports "deleted completely".
- A record from a non-user author tier that becomes a user-tier fact, constraint, or alias binding.

Out of scope:

- Components not in this repository (the macOS app, the local server, voice, screen control, Telegram). They are private for now; findings about them from reading the public code are still welcome in a report.
- Attacks that require the attacker to already run code as the user on the Mac.
- The behavior of the underlying language model itself, outside of what the guard, sanitizer, and memory policy are meant to contain.

The code is published for reading and review; there is no bug bounty.
