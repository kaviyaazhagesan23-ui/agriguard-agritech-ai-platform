from pathlib import Path

import pandas as pd

from src.dashboard_helpers import latest_price_for_selection, market_coordinates


def test_latest_price_for_selection_is_latest_date():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-03"],
        "market": ["Vallam", "Vallam"],
        "variety": ["B P T", "B P T"],
        "grade": ["Local", "Local"],
        "modal_price_rs_per_quintal": [2000, 2200],
    })
    assert latest_price_for_selection(df, "Vallam", "B P T", "Local") == 2200


def test_missing_market_coordinates_are_not_fabricated():
    assert market_coordinates("Thiruppananthal") is None


def test_verified_market_coordinates_are_available():
    coords = market_coordinates("Vallam")
    assert coords is not None
    assert len(coords) == 2
