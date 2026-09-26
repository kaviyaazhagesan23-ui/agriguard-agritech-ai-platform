
import pytest

from src.decision_engine import (
    DecisionInputs,
    calculate_break_even_future_price,
    calculate_storage_cost,
    calculate_decision,
)


def make_inputs(**overrides):
    values = {
        "quantity_quintals": 100,
        "current_market": "Vallam",
        "current_modal_price": 2200,
        "forecast_price": 2350,
        "expected_waiting_days": 7,
        "transportation_cost": 5000,
        "storage_cost_per_day": 2,
        "buyer_offer": None,
    }

    values.update(overrides)
    return DecisionInputs(**values)


def test_storage_cost():
    result = calculate_storage_cost(
        quantity_quintals=100,
        storage_cost_per_quintal_per_day=2,
        expected_waiting_days=7,
    )

    assert result == 1400


def test_break_even_future_price():
    result = calculate_break_even_future_price(
        current_selling_price=2200,
        storage_cost_per_quintal_per_day=2,
        expected_waiting_days=7,
    )

    assert result == 2214


def test_waiting_is_better_when_forecast_covers_costs():
    result = calculate_decision(
        make_inputs(
            forecast_price=2350,
        )
    )

    assert result.wait is not None
    assert result.wait.net_revenue > result.sell_today.net_revenue

    assert (
        result.interpretation
        == "Estimated higher net revenue if waiting."
    )


def test_selling_today_is_better():
    result = calculate_decision(
        make_inputs(
            forecast_price=2100,
        )
    )

    assert result.wait is not None
    assert result.sell_today.net_revenue > result.wait.net_revenue

    assert (
        result.interpretation
        == "Estimated higher net revenue if selling today."
    )


def test_buyer_offer_is_informational_only():
    result = calculate_decision(
        make_inputs(
            current_modal_price=2200,
            buyer_offer=2500,
            forecast_price=2400,
        )
    )

    assert result.sell_today.price_per_quintal == 2200

    assert any(
        "Buyer offer is shown as an informational reference"
        in warning
        for warning in result.warnings
    )


def test_zero_quantity_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                quantity_quintals=0,
            )
        )


def test_negative_transportation_cost_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                transportation_cost=-1,
            )
        )


def test_negative_storage_cost_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                storage_cost_per_day=-1,
            )
        )


def test_negative_current_price_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                current_modal_price=-100,
            )
        )


def test_negative_forecast_price_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                forecast_price=-100,
            )
        )


def test_missing_forecast_returns_sell_today_only():
    result = calculate_decision(
        make_inputs(
            forecast_price=None,
        )
    )

    assert result.wait is None
    assert result.net_revenue_difference is None
    assert result.percentage_difference is None
    assert result.break_even_future_price is None
    assert result.sensitivity == []

    assert (
        result.interpretation
        == "Waiting cannot be evaluated because the forecast price is missing."
    )


def test_missing_current_price_rejected():
    with pytest.raises((TypeError, ValueError)):
        calculate_decision(
            make_inputs(
                current_modal_price=None,
            )
        )


def test_negative_waiting_days_rejected():
    with pytest.raises(ValueError):
        calculate_decision(
            make_inputs(
                expected_waiting_days=-1,
            )
        )


def test_sensitivity_has_five_scenarios():
    result = calculate_decision(
        make_inputs()
    )

    assert len(result.sensitivity) == 5


def test_transportation_cost_cancels_in_break_even():
    result_without_transport = calculate_decision(
        make_inputs(
            transportation_cost=0,
        )
    )

    result_with_transport = calculate_decision(
        make_inputs(
            transportation_cost=10000,
        )
    )

    assert (
        result_without_transport.break_even_future_price
        == result_with_transport.break_even_future_price
    )


def test_decision_uses_same_break_even_formula():
    inputs = DecisionInputs(
        quantity_quintals=100,
        current_market="Vallam",
        current_modal_price=2200,
        forecast_price=2350,
        expected_waiting_days=7,
        transportation_cost=5000,
        storage_cost_per_day=2,
    )

    result = calculate_decision(inputs)

    expected = calculate_break_even_future_price(
        current_selling_price=2200,
        storage_cost_per_quintal_per_day=2,
        expected_waiting_days=7,
    )

    assert result.break_even_future_price == expected

