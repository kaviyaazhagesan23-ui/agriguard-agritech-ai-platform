"""Deterministic canonical JSON and SHA-256 hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize_record(record: dict[str, Any]) -> str:
    """Return deterministic JSON for a record."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_record(record: dict[str, Any]) -> str:
    """Return SHA-256 hex digest of the canonical record JSON."""
    canonical = canonicalize_record(record)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Return SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
