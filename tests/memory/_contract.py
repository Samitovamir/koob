"""Shared markers for the memory acceptance stubs (the memory spec).

A stub calls the facade function it covers (it raises NotImplementedError today), then
`pending(lane)`, so it stays xfail even after that function is implemented for another stub.
Once the function is built, the body is replaced with the real test and `@not_built` removed.
"""

from __future__ import annotations

import pytest


def not_built(lane: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"{lane}: not built")


def pending(lane: str) -> None:
    raise NotImplementedError(f"{lane}: acceptance body not written")
