"""Phase 7.5 local blockchain tamper-detection demonstration."""

from __future__ import annotations

from pathlib import Path

from src.blockchain import (
    commit_record,
    create_decision_record,
    load_or_create_blockchain,
    validate_blockchain,
    verify_record,
)
from src.decision_engine import DecisionInputs, calculate_decision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKCHAIN_PATH = PROJECT_ROOT / "data" / "blockchain" / "paddywise_blockchain.json"


def main() -> None:
    print("=" * 72)
    print("PADDYWISE BLOCKCHAIN DEMO")
    print("=" * 72)

    blockchain = load_or_create_blockchain(BLOCKCHAIN_PATH)

    # Generic demonstration data only. No real farmer identity is used.
    inputs = DecisionInputs(
        quantity_quintals=10,
        current_market="Vallam",
        current_modal_price=2000,
        forecast_price=2200,
        expected_waiting_days=5,
        transportation_cost=500,
        storage_cost_per_day=20,
        buyer_offer=2150,
    )
    decision = calculate_decision(inputs)

    record = create_decision_record(
        record_id="PW-DEMO-002",
        decision_result=decision,
        timestamp="2026-09-20T00:00:00+00:00",
    )

    block = commit_record(blockchain, record)
    blockchain.save(BLOCKCHAIN_PATH)

    print(f"Record ID: {record['record_id']}")
    print(f"Record committed: Block {block.index}")
    print(f"Record Hash: {block.record_hash}")
    print()

    original = verify_record(blockchain, record["record_id"], record)
    print("Original record verification:")
    print("PASS" if original["verified"] else "FAIL")
    print(original)
    print()

    modified_record = dict(record)
    modified_record["forecast_price"] = 2300

    modified = verify_record(
        blockchain,
        modified_record["record_id"],
        modified_record,
    )
    print("Modified record verification:")
    print("FAIL (tampering detected)" if not modified["verified"] else "PASS (unexpectedly verified)")
    print(modified)
    print()

    chain_result = validate_blockchain(blockchain)
    print("Blockchain validation:")
    print("PASS" if chain_result["valid"] else "FAIL")
    print(chain_result)
    print("=" * 72)


if __name__ == "__main__":
    main()
