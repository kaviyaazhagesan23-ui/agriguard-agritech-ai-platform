"""Small local blockchain used as a tamper-evident PaddyWise audit ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .block import Block
from .hashing import sha256_record, sha256_text
from .storage import load_chain, save_chain

GENESIS_PREVIOUS_HASH = "0" * 64
GENESIS_TIMESTAMP = "1970-01-01T00:00:00+00:00"
GENESIS_RECORD_ID = "GENESIS"
GENESIS_NONCE = 0
GENESIS_RECORD_HASH = sha256_text("PADDYWISE_LOCAL_AUDIT_CHAIN_V1")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_genesis_block() -> Block:
    """Create the deterministic genesis block."""
    return Block.create(
        index=0,
        timestamp=GENESIS_TIMESTAMP,
        record_id=GENESIS_RECORD_ID,
        record_hash=GENESIS_RECORD_HASH,
        previous_hash=GENESIS_PREVIOUS_HASH,
        nonce=GENESIS_NONCE,
    )


class Blockchain:
    """Local, non-mined blockchain for record-integrity auditing."""

    def __init__(self, blocks: Optional[list[Block]] = None) -> None:
        self.blocks = list(blocks) if blocks is not None else [create_genesis_block()]
        if not self.blocks:
            self.blocks = [create_genesis_block()]

    @property
    def genesis_block(self) -> Block:
        return self.blocks[0]

    def get_latest_block(self) -> Block:
        return self.blocks[-1]

    def find_record(self, record_id: str) -> Optional[Block]:
        for block in self.blocks:
            if block.record_id == record_id:
                return block
        return None

    def add_block(
        self,
        *,
        record_id: str,
        record: dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> Block:
        """Commit a record hash as a new block.

        Re-adding the exact same record ID/hash is idempotent and returns
        the existing block. Reusing a record ID for different content is
        rejected rather than creating ambiguous audit history.
        """
        if not record_id or not str(record_id).strip():
            raise ValueError("record_id must be non-empty.")
        if record_id == GENESIS_RECORD_ID:
            raise ValueError("GENESIS is reserved for the genesis block.")

        record_hash = sha256_record(record)
        existing = self.find_record(record_id)
        if existing is not None:
            if existing.record_hash == record_hash:
                return existing
            raise ValueError(
                f"record_id '{record_id}' already exists with different content."
            )

        latest = self.get_latest_block()
        block = Block.create(
            index=latest.index + 1,
            timestamp=timestamp or _utc_timestamp(),
            record_id=record_id,
            record_hash=record_hash,
            previous_hash=latest.block_hash,
            nonce=0,
        )
        self.blocks.append(block)
        return block

    def validate_chain(self) -> dict[str, Any]:
        """Validate genesis, ordering, links, and every block hash."""
        if not self.blocks:
            return {
                "valid": False,
                "blocks_checked": 0,
                "invalid_block": None,
                "reason": "Blockchain is empty.",
            }

        expected_genesis = create_genesis_block()
        actual_genesis = self.blocks[0]
        if actual_genesis != expected_genesis:
            return {
                "valid": False,
                "blocks_checked": 1,
                "invalid_block": 0,
                "reason": "Genesis block does not match the expected genesis block.",
            }

        seen_record_ids: set[str] = {GENESIS_RECORD_ID}
        for position, block in enumerate(self.blocks):
            if block.index != position:
                return {
                    "valid": False,
                    "blocks_checked": position + 1,
                    "invalid_block": block.index,
                    "reason": "Block index/order mismatch.",
                }

            if position > 0:
                previous = self.blocks[position - 1]
                if block.previous_hash != previous.block_hash:
                    return {
                        "valid": False,
                        "blocks_checked": position + 1,
                        "invalid_block": block.index,
                        "reason": "previous_hash does not match the preceding block.",
                    }

                if block.record_id in seen_record_ids:
                    return {
                        "valid": False,
                        "blocks_checked": position + 1,
                        "invalid_block": block.index,
                        "reason": "Duplicate record_id found in the chain.",
                    }
                seen_record_ids.add(block.record_id)

            if block.recompute_hash() != block.block_hash:
                return {
                    "valid": False,
                    "blocks_checked": position + 1,
                    "invalid_block": block.index,
                    "reason": "Block hash does not match block contents.",
                }

            if len(block.record_hash) != 64:
                return {
                    "valid": False,
                    "blocks_checked": position + 1,
                    "invalid_block": block.index,
                    "reason": "Record hash is not a SHA-256 hex digest.",
                }

        return {
            "valid": True,
            "blocks_checked": len(self.blocks),
            "invalid_block": None,
            "reason": None,
        }

    def verify_record(
        self,
        record_id: str,
        current_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify a current off-chain record against its committed hash."""
        chain_result = self.validate_chain()
        block = self.find_record(record_id)

        if block is None:
            return {
                "verified": False,
                "record_id": record_id,
                "block_index": None,
                "chain_valid": chain_result["valid"],
                "record_hash_matches": False,
                "status": "missing_record",
                "reason": "No committed record with this record_id was found.",
            }

        current_hash = sha256_record(current_record)
        matches = current_hash == block.record_hash

        if not chain_result["valid"]:
            status = "blockchain_corruption"
            verified = False
        elif not matches:
            status = "record_modified"
            verified = False
        else:
            status = "verified"
            verified = True

        return {
            "verified": verified,
            "record_id": record_id,
            "block_index": block.index,
            "chain_valid": chain_result["valid"],
            "record_hash_matches": matches,
            "status": status,
            "reason": chain_result["reason"] if not chain_result["valid"] else None,
        }

    def save(self, path: Path) -> None:
        save_chain(self.blocks, Path(path))

    @classmethod
    def load(cls, path: Path) -> "Blockchain":
        return cls(load_chain(Path(path)))
