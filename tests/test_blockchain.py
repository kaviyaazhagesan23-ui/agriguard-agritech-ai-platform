from pathlib import Path

import pytest

from src.blockchain import (
    Blockchain,
    commit_record,
    create_decision_record,
    sha256_record,
    validate_blockchain,
    verify_record,
)
from src.decision_engine import DecisionInputs, calculate_decision


def demo_record(record_id="PW-TEST-001"):
    result = calculate_decision(
        DecisionInputs(
            quantity_quintals=10,
            current_market="Vallam",
            current_modal_price=2000,
            forecast_price=2200,
            expected_waiting_days=5,
            transportation_cost=500,
            storage_cost_per_day=20,
            buyer_offer=2150,
        )
    )
    return create_decision_record(
        record_id=record_id,
        decision_result=result,
        timestamp="2026-09-20T00:00:00+00:00",
    )


def test_genesis_block_creation():
    chain = Blockchain()
    assert len(chain.blocks) == 1
    assert chain.genesis_block.index == 0
    assert chain.validate_chain()["valid"] is True


def test_adding_block():
    chain = Blockchain()
    record = demo_record()
    block = commit_record(chain, record)
    assert block.index == 1
    assert block.record_id == "PW-TEST-001"
    assert len(block.record_hash) == 64
    assert len(block.block_hash) == 64
    assert chain.validate_chain()["valid"] is True


def test_save_and_load(tmp_path: Path):
    path = tmp_path / "chain.json"
    chain = Blockchain()
    record = demo_record()
    commit_record(chain, record)
    chain.save(path)

    loaded = Blockchain.load(path)
    assert len(loaded.blocks) == 2
    assert loaded.validate_chain()["valid"] is True
    assert loaded.find_record(record["record_id"]).record_hash == chain.find_record(record["record_id"]).record_hash


def test_same_record_same_sha256():
    record = demo_record()
    reordered = {key: record[key] for key in reversed(list(record))}
    assert sha256_record(record) == sha256_record(reordered)


def test_changed_record_different_sha256():
    record = demo_record()
    changed = dict(record)
    changed["forecast_price"] = 2300
    assert sha256_record(record) != sha256_record(changed)


def test_valid_chain_passes_validation():
    chain = Blockchain()
    commit_record(chain, demo_record())
    commit_record(chain, demo_record("PW-TEST-002"))
    result = validate_blockchain(chain)
    assert result["valid"] is True
    assert result["blocks_checked"] == 3
    assert result["invalid_block"] is None


def test_tampered_block_fails_validation():
    chain = Blockchain()
    commit_record(chain, demo_record())
    original = chain.blocks[1]
    # Replacing record_hash without recomputing block_hash simulates storage tampering.
    from dataclasses import replace
    chain.blocks[1] = replace(original, record_hash="f" * 64)

    result = validate_blockchain(chain)
    assert result["valid"] is False
    assert result["invalid_block"] == 1


def test_tampered_record_fails_record_verification():
    chain = Blockchain()
    record = demo_record()
    commit_record(chain, record)

    assert verify_record(chain, record["record_id"], record)["verified"] is True

    changed = dict(record)
    changed["forecast_price"] = 2300
    result = verify_record(chain, changed["record_id"], changed)

    assert result["verified"] is False
    assert result["chain_valid"] is True
    assert result["record_hash_matches"] is False
    assert result["status"] == "record_modified"


def test_multiple_records_can_be_stored():
    chain = Blockchain()
    commit_record(chain, demo_record("PW-TEST-001"))
    commit_record(chain, demo_record("PW-TEST-002"))
    commit_record(chain, demo_record("PW-TEST-003"))
    assert len(chain.blocks) == 4
    assert chain.validate_chain()["valid"] is True


def test_duplicate_same_record_is_idempotent():
    chain = Blockchain()
    record = demo_record()
    first = commit_record(chain, record)
    second = commit_record(chain, record)
    assert first == second
    assert len(chain.blocks) == 2


def test_duplicate_record_id_with_changed_content_is_rejected():
    chain = Blockchain()
    record = demo_record()
    commit_record(chain, record)
    changed = dict(record)
    changed["forecast_price"] = 2300

    with pytest.raises(ValueError, match="already exists"):
        commit_record(chain, changed)


def test_missing_record_is_distinguished():
    chain = Blockchain()
    result = verify_record(chain, "PW-MISSING", demo_record("PW-MISSING"))
    assert result["verified"] is False
    assert result["status"] == "missing_record"
    assert result["block_index"] is None


def test_corrupted_chain_is_distinguished_from_modified_record():
    chain = Blockchain()
    record = demo_record()
    commit_record(chain, record)
    from dataclasses import replace
    block = chain.blocks[1]
    chain.blocks[1] = replace(block, block_hash="0" * 64)

    result = verify_record(chain, record["record_id"], record)
    assert result["verified"] is False
    assert result["chain_valid"] is False
    assert result["status"] == "blockchain_corruption"


def test_hash_rejects_non_json_nan():
    record = demo_record()
    record["bad"] = float("nan")
    with pytest.raises(ValueError):
        sha256_record(record)
