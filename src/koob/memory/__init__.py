"""Cube memory: the ledger, its policy, writers, recall and forgetting.

Normative spec: the memory spec (v0.2). The runtime, adapters, MCP tools and `cb memory` use
`koob.memory.api` only. This package MUST NOT import `koob.web`,
`koob.assistant` or island code (§2, G13; tests/memory/test_boundaries.py).

Importing the package has no side effects: no DB is opened, nothing starts.
"""
