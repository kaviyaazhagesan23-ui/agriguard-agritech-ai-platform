"""
Phase 7 — Farmer Sell vs Wait Decision Engine.

All calculations are deterministic and transparent.

Currency:
    Indian rupees (₹)

Prices and costs:
    Per quintal unless explicitly stated otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


class DecisionValidationError(ValueError):
    """Raised when decision-engine inputs are invalid."""


@dataclass(frozen=True)
class DecisionInputs:
    quantity_quintals: float
    current_market: str
    current_modal_price: float
    forecast_price: Optional[float]
    expected_waiting_days: int
    transportation_cost: float
    storage_cost_per_day: float
    buyer_offer: Optional[float] = None


@dataclass(frozen=True)
class ScenarioResult:
    price_per_quintal: float
    gross_revenue: float
    transportation_cost: float
    storage_cost: float
    total_cost: float
    net_revenue: float


@dataclass(frozen=True)
class DecisionResult:
    inputs: dict
    sell_today: ScenarioResult
    wait: Optional[ScenarioResult]
    net_revenue_difference: Optional[float]
    percentage_difference: Optional[float]
    break_even_future_price: Optional[float]
    sensitivity: list[dict]
    interpretation: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _require_finite_number(name: str, value: object) -> float:
    if value is None:
        raise DecisionValidationError(f"{name} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(
            f"{name} must be numeric."
        ) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise DecisionValidationError(f"{name} must be finite.")
    return number


def validate_inputs(inputs: DecisionInputs) -> None:
    """Validate all inputs before performing calculations."""
    quantity = _require_finite_number(
        "quantity_quintals", inputs.quantity_quintals
    )
    current_price = _require_finite_number(
        "current_modal_price", inputs.current_modal_price
    )
    transport = _require_finite_number(
        "transportation_cost", inputs.transportation_cost
    )
    storage = _require_finite_number(
        "storage_cost_per_day", inputs.storage_cost_per_day
    )

    if quantity <= 0:
        raise DecisionValidationError(
            "quantity_quintals must be greater than zero."
        )

    if current_price < 0:
        raise DecisionValidationError(
            "current_modal_price cannot be negative."
        )

    if transport < 0:
        raise DecisionValidationError(
            "transportation_cost cannot be negative."
        )

    if storage < 0:
        raise DecisionValidationError(
            "storage_cost_per_day cannot be negative."
        )

    if not isinstance(inputs.current_market, str) or not inputs.current_market.strip():
        raise DecisionValidationError(
            "current_market must be a non-empty string."
        )

    if isinstance(inputs.expected_waiting_days, bool):
        raise DecisionValidationError(
            "expected_waiting_days must be a non-negative integer."
        )

    if int(inputs.expected_waiting_days) != inputs.expected_waiting_days:
        raise DecisionValidationError(
            "expected_waiting_days must be an integer."
        )

    if inputs.expected_waiting_days < 0:
        raise DecisionValidationError(
            "expected_waiting_days cannot be negative."
        )

    if inputs.forecast_price is not None:
        forecast = _require_finite_number(
            "forecast_price", inputs.forecast_price
        )
        if forecast < 0:
            raise DecisionValidationError(
                "forecast_price cannot be negative."
            )

    if inputs.buyer_offer is not None:
        offer = _require_finite_number(
            "buyer_offer", inputs.buyer_offer
        )
        if offer < 0:
            raise DecisionValidationError(
                "buyer_offer cannot be negative."
            )


def _scenario(
    quantity: float,
    price: float,
    transportation_cost: float,
    storage_cost: float,
) -> ScenarioResult:
    gross = quantity * price
    total_cost = transportation_cost + storage_cost
    net = gross - total_cost

    return ScenarioResult(
        price_per_quintal=price,
        gross_revenue=gross,
        transportation_cost=transportation_cost,
        storage_cost=storage_cost,
        total_cost=total_cost,
        net_revenue=net,
    )


def _interpret(
    difference: Optional[float],
    tolerance: float,
) -> str:
    if difference is None:
        return "Waiting cannot be evaluated because the forecast price is missing."

    if abs(difference) <= tolerance:
        return "Difference is small/uncertain."

    if difference > 0:
        return "Estimated higher net revenue if waiting."

    return "Estimated higher net revenue if selling today."


def calculate_sensitivity(
    *,
    quantity_quintals: float,
    expected_waiting_days: int,
    transportation_cost: float,
    storage_cost_per_day: float,
    forecast_price: Optional[float],
    current_modal_price: float,
    steps: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
) -> list[dict]:
    """
    Calculate wait-vs-sell differences under forecast-price changes.

    Each step is a relative change from the supplied forecast price.
    """
    if forecast_price is None:
        return []

    storage_cost = expected_waiting_days * storage_cost_per_day
    sell_net = (
        quantity_quintals * current_modal_price
        - transportation_cost
    )

    rows = []
    for step in steps:
        scenario_price = forecast_price * (1 + step)
        wait_net = (
            quantity_quintals * scenario_price
            - transportation_cost
            - storage_cost
        )
        difference = wait_net - sell_net

        rows.append(
            {
                "forecast_adjustment": step,
                "scenario_forecast_price": scenario_price,
                "wait_net_revenue": wait_net,
                "sell_today_net_revenue": sell_net,
                "net_revenue_difference": difference,
            }
        )

    return rows



def calculate_storage_cost(
    *,
    quantity_quintals: float,
    storage_cost_per_quintal_per_day: float,
    expected_waiting_days: int,
) -> float:
    """
    Calculate total storage cost.

    Formula:
        quantity_quintals
        * storage_cost_per_quintal_per_day
        * expected_waiting_days
    """
    quantity = _require_finite_number(
        "quantity_quintals",
        quantity_quintals,
    )

    storage_per_quintal_per_day = _require_finite_number(
        "storage_cost_per_quintal_per_day",
        storage_cost_per_quintal_per_day,
    )

    if quantity <= 0:
        raise DecisionValidationError(
            "quantity_quintals must be greater than zero."
        )

    if storage_per_quintal_per_day < 0:
        raise DecisionValidationError(
            "storage_cost_per_quintal_per_day cannot be negative."
        )

    if isinstance(expected_waiting_days, bool):
        raise DecisionValidationError(
            "expected_waiting_days must be a non-negative integer."
        )

    if int(expected_waiting_days) != expected_waiting_days:
        raise DecisionValidationError(
            "expected_waiting_days must be an integer."
        )

    if expected_waiting_days < 0:
        raise DecisionValidationError(
            "expected_waiting_days cannot be negative."
        )

    return (
        quantity
        * storage_per_quintal_per_day
        * int(expected_waiting_days)
    )









def calculate_break_even_future_price(
    *,
    current_selling_price: float,
    storage_cost_per_quintal_per_day: float,
    expected_waiting_days: int,
) -> float:
    """
    Calculate the future price per quintal required for waiting
    to recover storage costs.

    Formula:
        current_selling_price
        + storage_cost_per_quintal_per_day * expected_waiting_days
    """
    current_price = _require_finite_number(
        "current_selling_price",
        current_selling_price,
    )

    storage_per_quintal_per_day = _require_finite_number(
        "storage_cost_per_quintal_per_day",
        storage_cost_per_quintal_per_day,
    )

    if current_price < 0:
        raise DecisionValidationError(
            "current_selling_price cannot be negative."
        )

    if storage_per_quintal_per_day < 0:
        raise DecisionValidationError(
            "storage_cost_per_quintal_per_day cannot be negative."
        )

    if isinstance(expected_waiting_days, bool):
        raise DecisionValidationError(
            "expected_waiting_days must be a non-negative integer."
        )

    if int(expected_waiting_days) != expected_waiting_days:
        raise DecisionValidationError(
            "expected_waiting_days must be an integer."
        )

    if expected_waiting_days < 0:
        raise DecisionValidationError(
            "expected_waiting_days cannot be negative."
        )

    return (
        current_price
        + storage_per_quintal_per_day * int(expected_waiting_days)
    )












def calculate_decision(
    inputs: DecisionInputs,
    *,
    small_difference_tolerance: float = 0.01,
) -> DecisionResult:
    """
    Calculate transparent SELL TODAY vs WAIT scenarios.

    Transportation is treated as a one-time cost in both scenarios.
    Storage is assumed to apply only to WAIT.
    Buyer offer, when supplied, is used as an informational reference
    and does not silently replace the current modal price.
    """
    validate_inputs(inputs)

    quantity = float(inputs.quantity_quintals)
    current_price = float(inputs.current_modal_price)
    waiting_days = int(inputs.expected_waiting_days)
    transport = float(inputs.transportation_cost)
    storage_per_day = float(inputs.storage_cost_per_day)

    sell_today = _scenario(
        quantity=quantity,
        price=current_price,
        transportation_cost=transport,
        storage_cost=0.0,
    )

    warnings = [
        "Results are estimates, not guaranteed profit.",
        "The forecast price is uncertain and may not occur.",
        "Transportation is assumed to be the same in both scenarios.",
        "Storage cost is assumed to apply only while waiting.",
    ]

    if inputs.buyer_offer is not None:
        warnings.append(
            "Buyer offer is shown as an informational reference; "
            "it is not automatically substituted into the calculation."
        )

    if inputs.forecast_price is None:
        return DecisionResult(
            inputs=asdict(inputs),
            sell_today=sell_today,
            wait=None,
            net_revenue_difference=None,
            percentage_difference=None,
            break_even_future_price=None,
            sensitivity=[],
            interpretation=_interpret(None, small_difference_tolerance),
            warnings=warnings + [
                "Waiting cannot be evaluated without a forecast price."
            ],
        )

    forecast_price = float(inputs.forecast_price)
    storage_total = waiting_days * storage_per_day

    wait = _scenario(
        quantity=quantity,
        price=forecast_price,
        transportation_cost=transport,
        storage_cost=storage_total,
    )

    difference = wait.net_revenue - sell_today.net_revenue

    if sell_today.net_revenue != 0:
        percentage_difference = (
            difference / abs(sell_today.net_revenue)
        ) * 100
    else:
        percentage_difference = None

    # Break-even future price:
    # quantity * future_price - transport - storage
    #     = quantity * current_price - transport
    # future_price = current_price + storage / quantity
    
    break_even = calculate_break_even_future_price(
    current_selling_price=current_price,
    storage_cost_per_quintal_per_day=storage_per_day,
    expected_waiting_days=waiting_days,
)


    sensitivity = calculate_sensitivity(
        quantity_quintals=quantity,
        expected_waiting_days=waiting_days,
        transportation_cost=transport,
        storage_cost_per_day=storage_per_day,
        forecast_price=forecast_price,
        current_modal_price=current_price,
    )

    return DecisionResult(
        inputs=asdict(inputs),
        sell_today=sell_today,
        wait=wait,
        net_revenue_difference=difference,
        percentage_difference=percentage_difference,
        break_even_future_price=break_even,
        sensitivity=sensitivity,
        interpretation=_interpret(
            difference,
            small_difference_tolerance,
        ),
        warnings=warnings,
    )


def format_currency(value: Optional[float]) -> str:
    """Format a value as Indian rupees."""
    if value is None:
        return "N/A"
    return f"₹{value:,.2f}"


def print_decision(result: DecisionResult) -> None:
    """Print a readable transparent decision summary."""
    print("=" * 72)
    print("FARMER SELL VS WAIT DECISION")
    print("=" * 72)

    print(f"Market: {result.inputs['current_market']}")
    print(f"Quantity: {result.inputs['quantity_quintals']} quintals")
    print()

    print("SELL TODAY")
    print(f"  Price/quintal: {format_currency(result.sell_today.price_per_quintal)}")
    print(f"  Gross revenue: {format_currency(result.sell_today.gross_revenue)}")
    print(f"  Transport:     {format_currency(result.sell_today.transportation_cost)}")
    print(f"  Storage:       {format_currency(result.sell_today.storage_cost)}")
    print(f"  Net revenue:   {format_currency(result.sell_today.net_revenue)}")
    print()

    if result.wait is None:
        print("WAIT")
        print("  Cannot calculate: forecast price is missing.")
    else:
        print("WAIT")
        print(f"  Price/quintal: {format_currency(result.wait.price_per_quintal)}")
        print(f"  Gross revenue: {format_currency(result.wait.gross_revenue)}")
        print(f"  Transport:     {format_currency(result.wait.transportation_cost)}")
        print(f"  Storage:       {format_currency(result.wait.storage_cost)}")
        print(f"  Net revenue:   {format_currency(result.wait.net_revenue)}")
        print()
        print(
            "Net revenue difference (WAIT - SELL TODAY): "
            f"{format_currency(result.net_revenue_difference)}"
        )
        if result.percentage_difference is not None:
            print(
                f"Percentage difference: "
                f"{result.percentage_difference:.2f}%"
            )
        else:
            print("Percentage difference: N/A")

        print(
            "Break-even future price: "
            f"{format_currency(result.break_even_future_price)}/quintal"
        )

    print()
    print(f"Interpretation: {result.interpretation}")
    print()
    print("Warnings:")
    for warning in result.warnings:
        print(f"- {warning}")
