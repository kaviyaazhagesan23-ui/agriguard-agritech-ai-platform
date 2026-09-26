import math

import pandas as pd
import pytest

from src.market_comparison import (
    MARKET_COORDINATES,
    calculate_gross_revenue,
    calculate_haversine_distance,
    calculate_net_revenue,
    calculate_transportation_cost,
    compare_markets,
    get_latest_market_prices,
)


def sample_prices():
    return pd.DataFrame(
        {
            "market": ["Current", "Near", "Far", "NoGPS"],
            "modal_price_rs_per_quintal": [2200, 2250, 2300, 2400],
            "date": ["2025-03-01"] * 4,
        }
    )


def test_haversine_distance_approximately_correct():
    distance = calculate_haversine_distance(0, 0, 0, 1)
    assert distance == pytest.approx(111.195, rel=1e-3)


def test_same_coordinates_are_zero():
    assert calculate_haversine_distance(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)


def test_invalid_latitude_is_rejected():
    with pytest.raises(ValueError):
        calculate_haversine_distance(91, 0, 0, 0)


def test_invalid_longitude_is_rejected():
    with pytest.raises(ValueError):
        calculate_haversine_distance(0, 181, 0, 0)


def test_transportation_cost_calculation():
    assert calculate_transportation_cost(10, 20) == pytest.approx(200)


def test_higher_distance_increases_transport_cost():
    assert calculate_transportation_cost(20, 20) > calculate_transportation_cost(10, 20)


def test_gross_revenue_calculation():
    assert calculate_gross_revenue(100, 2200) == pytest.approx(220000)


def test_net_revenue_calculation():
    assert calculate_net_revenue(100, 2200, 5000) == pytest.approx(215000)


def test_higher_price_increases_net_revenue_when_transport_unchanged():
    assert calculate_net_revenue(100, 2300, 5000) > calculate_net_revenue(100, 2200, 5000)


def test_higher_transport_cost_decreases_net_revenue():
    assert calculate_net_revenue(100, 2200, 6000) < calculate_net_revenue(100, 2200, 5000)


def test_missing_coordinates_are_handled_safely():
    prices = pd.DataFrame(
        {
            "market": ["Thiruppananthal"],
            "modal_price_rs_per_quintal": [2200],
            "date": ["2025-03-01"],
        }
    )
    results = compare_markets(
        quantity_quintals=100,
        current_market="Thiruppananthal",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.7,
        farmer_longitude=79.1,
    )
    result = results[0]
    assert result.coordinates_available is False
    assert result.distance_km is None
    assert result.transportation_cost is None
    assert result.net_revenue is None


def test_missing_coordinates_do_not_get_fabricated_values():
    assert MARKET_COORDINATES["Thiruppananthal"] == (None, None)


def test_market_comparison_returns_all_eligible_markets():
    prices = sample_prices()
    results = compare_markets(
        quantity_quintals=100,
        current_market="Current",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
    )
    assert {result.market for result in results} == {"Current", "Near", "Far", "NoGPS"}


def test_net_revenue_difference_vs_current_is_correct():
    prices = pd.DataFrame(
        {
            "market": ["Current", "Far"],
            "modal_price_rs_per_quintal": [2200, 2300],
            "date": ["2025-03-01", "2025-03-01"],
        }
    )
    # Both markets use the same verified coordinate here by using Vallam for
    # the current market and supplying a market name with a verified point.
    prices["market"] = ["Vallam", "Thanjavur"]
    results = compare_markets(
        quantity_quintals=100,
        current_market="Vallam",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
        transport_cost_per_km=0,
    )
    by_market = {result.market: result for result in results}
    assert by_market["Vallam"].net_revenue_difference == pytest.approx(0)
    assert by_market["Thanjavur"].net_revenue_difference == pytest.approx(10000)


def test_percentage_difference_is_correct():
    prices = pd.DataFrame(
        {
            "market": ["Vallam", "Thanjavur"],
            "modal_price_rs_per_quintal": [2200, 2420],
            "date": ["2025-03-01", "2025-03-01"],
        }
    )
    results = compare_markets(
        quantity_quintals=100,
        current_market="Vallam",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
        transport_cost_per_km=0,
    )
    thanjavur = next(r for r in results if r.market == "Thanjavur")
    assert thanjavur.percentage_difference == pytest.approx(10.0)


def test_zero_quantity_is_rejected():
    with pytest.raises(ValueError):
        compare_markets(
            quantity_quintals=0,
            current_market="Vallam",
            current_market_price=2200,
            market_price_data={"Vallam": 2200},
            farmer_latitude=10.714,
            farmer_longitude=79.702,
        )


def test_negative_quantity_is_rejected():
    with pytest.raises(ValueError):
        compare_markets(
            quantity_quintals=-1,
            current_market="Vallam",
            current_market_price=2200,
            market_price_data={"Vallam": 2200},
            farmer_latitude=10.714,
            farmer_longitude=79.702,
        )


def test_negative_transport_rate_is_rejected():
    with pytest.raises(ValueError):
        compare_markets(
            quantity_quintals=100,
            current_market="Vallam",
            current_market_price=2200,
            market_price_data={"Vallam": 2200},
            farmer_latitude=10.714,
            farmer_longitude=79.702,
            transport_cost_per_km=-1,
        )


def test_source_price_dataset_is_not_modified():
    source = sample_prices()
    before = source.copy(deep=True)
    compare_markets(
        quantity_quintals=100,
        current_market="Current",
        current_market_price=2200,
        market_price_data=source,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
    )
    pd.testing.assert_frame_equal(source, before)


def test_results_are_deterministic():
    prices = sample_prices()
    kwargs = dict(
        quantity_quintals=100,
        current_market="Current",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
        transport_cost_per_km=20,
    )
    first = compare_markets(**kwargs)
    second = compare_markets(**kwargs)
    assert first == second


def test_latest_price_lookup_uses_latest_date():
    data = pd.DataFrame(
        {
            "market": ["Vallam", "Vallam", "Thanjavur"],
            "modal_price_rs_per_quintal": [2100, 2300, 2400],
            "date": ["2025-01-01", "2025-03-01", "2025-02-01"],
            "variety": ["B P T", "B P T", "B P T"],
            "grade": ["Local", "Local", "Local"],
        }
    )
    latest = get_latest_market_prices(data, variety="B P T", grade="Local")
    vallam = latest.loc[latest.market == "Vallam"].iloc[0]
    assert vallam.modal_price_rs_per_quintal == 2300


def test_like_for_like_filter_can_remove_unmatched_markets():
    data = pd.DataFrame(
        {
            "market": ["Vallam", "Thanjavur"],
            "modal_price_rs_per_quintal": [2300, 2400],
            "date": ["2025-03-01", "2025-03-01"],
            "variety": ["B P T", "Paddy"],
            "grade": ["Local", "FAQ"],
        }
    )
    latest = get_latest_market_prices(data, variety="B P T", grade="Local")
    assert latest["market"].tolist() == ["Vallam"]


def test_current_market_price_is_authoritative():
    prices = pd.DataFrame(
        {
            "market": ["Vallam"],
            "modal_price_rs_per_quintal": [9999],
            "date": ["2025-03-01"],
        }
    )
    result = compare_markets(
        quantity_quintals=1,
        current_market="Vallam",
        current_market_price=2200,
        market_price_data=prices,
        farmer_latitude=10.714,
        farmer_longitude=79.702,
        transport_cost_per_km=0,
    )[0]
    assert result.current_price == 2200
