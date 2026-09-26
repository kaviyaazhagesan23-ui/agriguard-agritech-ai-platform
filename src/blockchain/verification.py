"""Public helper API for PaddyWise blockchain verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .block import Block
from .blockchain import Blockchain


def create_decision_record(
    *,
    record_id: str,
    decision_result: Any,
    forecast_horizon_days: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Create a minimal off-chain audit record from the existing Phase 7 result.

    Only fields represented by the current Phase 7 DecisionInputs/Result are
    included. No farmer PII is added. ``forecast_horizon_days`` is optional
    because Phase 7 does not currently carry a forecast horizon itself.
    """
    inputs = decision_result.inputs
    sell = decision_result.sell_today
    wait = decision_result.wait

    record: dict[str, Any] = {
        "record_id": record_id,
        "timestamp": timestamp,
        "current_market": inputs["current_market"],
        "quantity_quintals": inputs["quantity_quintals"],
        "current_modal_price": inputs["current_modal_price"],
        "forecast_price": inputs["forecast_price"],
        "expected_waiting_days": inputs["expected_waiting_days"],
        "transportation_cost": inputs["transportation_cost"],
        "storage_cost_per_day": inputs["storage_cost_per_day"],
        "buyer_offer": inputs.get("buyer_offer"),
        "sell_today": sell.to_dict() if hasattr(sell, "to_dict") else {
            "price_per_quintal": sell.price_per_quintal,
            "gross_revenue": sell.gross_revenue,
            "transportation_cost": sell.transportation_cost,
            "storage_cost": sell.storage_cost,
            "total_cost": sell.total_cost,
            "net_revenue": sell.net_revenue,
        },
        "wait": None if wait is None else {
            "price_per_quintal": wait.price_per_quintal,
            "gross_revenue": wait.gross_revenue,
            "transportation_cost": wait.transportation_cost,
            "storage_cost": wait.storage_cost,
            "total_cost": wait.total_cost,
            "net_revenue": wait.net_revenue,
        },
        "net_revenue_difference": decision_result.net_revenue_difference,
        "percentage_difference": decision_result.percentage_difference,
        "break_even_future_price": decision_result.break_even_future_price,
        "interpretation": decision_result.interpretation,
    }

    if forecast_horizon_days is not None:
        record["forecast_horizon_days"] = forecast_horizon_days

    return record


def commit_record(
    blockchain: Blockchain,
    record: dict[str, Any],
) -> Block:
    """Commit an existing off-chain audit record to the blockchain."""
    record_id = record.get("record_id")
    if not record_id:
        raise ValueError("record must contain a non-empty record_id")
    return blockchain.add_block(
        record_id=str(record_id),
        record=record,
        timestamp=record.get("timestamp"),
    )


def verify_record(
    blockchain: Blockchain,
    record_id: str,
    current_record: dict[str, Any],
) -> dict[str, Any]:
    """Verify a current off-chain record against the local chain."""
    return blockchain.verify_record(record_id, current_record)


def validate_blockchain(blockchain: Blockchain) -> dict[str, Any]:
    """Validate the complete local blockchain."""
    return blockchain.validate_chain()


def load_or_create_blockchain(path: Path) -> Blockchain:
    """Load a blockchain if present; otherwise create and persist genesis."""
    path = Path(path)
    if path.exists():
        return Blockchain.load(path)
    blockchain = Blockchain()
    blockchain.save(path)
    return blockchain
