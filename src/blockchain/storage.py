"""JSON persistence for the local PaddyWise blockchain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .block import Block


def save_chain(blocks: Iterable[Block], path: Path) -> None:
    """Persist blockchain blocks as readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "blocks": [block.to_dict() for block in blocks]}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_chain(path: Path) -> list[Block]:
    """Load blockchain blocks from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Blockchain file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported blockchain storage version.")

    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Blockchain storage must contain a blocks list.")

    return [Block.from_dict(item) for item in blocks]
