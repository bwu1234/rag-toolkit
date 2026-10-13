"""One stable hash for identities that must match across processes and machines.

Index build records and eval provenance both identify things by hashing a
JSON-serializable description of them. They share this function so the same
description always gets the same hash: keys sorted, no whitespace, and
non-JSON values (paths, dates) rendered with ``str``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Hex characters kept: 64 bits, enough that two different inputs colliding
#: among the few thousand runs and builds this repo will ever record is not a
#: practical concern, and short enough to read in a results table.
HASH_CHARS = 16


def stable_hash(value: Any) -> str:
    """Short SHA-256 of ``value``'s canonical JSON."""

    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:HASH_CHARS]
