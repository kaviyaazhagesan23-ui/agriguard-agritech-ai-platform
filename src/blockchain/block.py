"""Block representation for the local PaddyWise audit blockchain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .hashing import sha256_record


@dataclass(frozen=True)
class Block:
    """A single tamper-evident audit block."""

    index: int
    timestamp: str
    record_id: str
    record_hash: str
    previous_hash: str
    nonce: int
    block_hash: str

    def hash_payload(self) -> dict[str, Any]:
        """Return fields covered by the block hash."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "record_id": self.record_id,
            "record_hash": self.record_hash,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }

    @classmethod
    def create(
        cls,
        *,
        index: int,
        timestamp: str,
        record_id: str,
        record_hash: str,
        previous_hash: str,
        nonce: int = 0,
    ) -> "Block":
        """Create a block and calculate its hash."""
        payload = {
            "index": index,
            "timestamp": timestamp,
            "record_id": record_id,
            "record_hash": record_hash,
            "previous_hash": previous_hash,
            "nonce": nonce,
        }
        block_hash = sha256_record(payload)
        return cls(
            index=index,
            timestamp=timestamp,
            record_id=record_id,
            record_hash=record_hash,
            previous_hash=previous_hash,
            nonce=nonce,
            block_hash=block_hash,
        )

    def recompute_hash(self) -> str:
        """Recompute the block hash from its immutable payload fields."""
        return sha256_record(self.hash_payload())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        required = {
            "index",
            "timestamp",
            "record_id",
            "record_hash",
            "previous_hash",
            "nonce",
            "block_hash",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Block is missing fields: {sorted(missing)}")
        return cls(
            index=int(data["index"]),
            timestamp=str(data["timestamp"]),
            record_id=str(data["record_id"]),
            record_hash=str(data["record_hash"]),
            previous_hash=str(data["previous_hash"]),
            nonce=int(data["nonce"]),
            block_hash=str(data["block_hash"]),
        )
