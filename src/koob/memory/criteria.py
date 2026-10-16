"""C2 learned criteria: hypotheses from stated reasons shared across >= 2 decisions of
different domains; scope, evidence, exceptions; three contradictions retire one.

Spec: §15.2.
"""

from __future__ import annotations

import sqlite3

from .model import Proposal


def propose_criteria(conn: sqlite3.Connection) -> Proposal:
    raise NotImplementedError("criteria")


def applicable(conn: sqlite3.Connection, domain: str, situation: dict) -> list[str]:
    """Criterion ids whose applicability matches; the answer names criterion and evidence."""
    raise NotImplementedError("criteria")
