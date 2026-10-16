"""Evaluation machinery: scenario loader, noise expander, runner, graders, canary sweep,
judge interface, report, retrieval bench; `cb memory eval`. Spec section 19.

Modules: schema/loader (strict format), noise, matchers, observe/graders (code checks), judge
(Ollama behind a flag), canary (store scan), sut (the facade driver and its hooks), runner,
report (hidden sets: per-group scores only), retrieval (embedder bench), cli.
"""
