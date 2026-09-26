import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    SUPPORTED_HORIZONS,
    add_calendar_features,
    calculate_rolling_mean,
    calculate_rolling_std,
    calculate_lag_value,
    prepare_model_input,
    validate_selection,
)


def make_history():
    dates = pd.date_range(
        "2025-01-01",
        periods=40,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "market": ["TestMarket"] * 40,
            "variety": ["B P T"] * 40,
            "grade": ["FAQ"] * 40,
            "modal_price_rs_per_quintal": np.arange(
                2000,
                2040,
            ),
        }
    )


def test_supported_horizons():
    assert 7 in SUPPORTED_HORIZONS
    assert 14 in SUPPORTED_HORIZONS
    assert 30 in SUPPORTED_HORIZONS


def test_validate_selection():
    df = make_history()

    validate_selection(
        df,
        "TestMarket",
        "B P T",
        "FAQ",
    )


def test_validate_selection_rejects_unknown():
    df = make_history()

    with pytest.raises(ValueError):
        validate_selection(
            df,
            "UnknownMarket",
            "B P T",
            "FAQ",
        )


def test_calculate_lag():
    df = make_history()

    value = calculate_lag_value(
        df,
        "TestMarket",
        1,
    )

    assert value == 2039


def test_calculate_lag_7():
    df = make_history()

    value = calculate_lag_value(
        df,
        "TestMarket",
        7,
    )

    assert value == 2033


def test_rolling_mean():
    df = make_history()

    value = calculate_rolling_mean(
        df,
        "TestMarket",
        7,
    )

    expected = np.mean(
        np.arange(2033, 2040)
    )

    assert value == pytest.approx(
        expected
    )


def test_rolling_std():
    df = make_history()

    value = calculate_rolling_std(
        df,
        "TestMarket",
        7,
    )

    expected = np.std(
        np.arange(2033, 2040),
        ddof=1,
    )

    assert value == pytest.approx(
        expected
    )


def test_calendar_features():
    row = pd.Series(
        dtype=float
    )

    add_calendar_features(
        row,
        pd.Timestamp("2026-09-20"),
    )

    assert row["year"] == 2026
    assert row["month"] == 9
    assert row["quarter"] == 3
    assert row["day_of_year"] == 263
    assert -1 <= row["sin_month"] <= 1
    assert -1 <= row["cos_month"] <= 1


def test_prepare_model_input():
    row = pd.Series(
        {
            "lag_1": 2000,
            "lag_7": 1950,
            "year": 2026,
        }
    )

    columns = [
        "lag_1",
        "lag_7",
        "year",
    ]

    X = prepare_model_input(
        row,
        columns,
    )

    assert list(X.columns) == columns
    assert X.shape == (1, 3)
    assert X["lag_1"].iloc[0] == 2000


def test_prepare_model_input_missing_column():
    row = pd.Series(
        {
            "lag_1": 2000,
        }
    )

    with pytest.raises(ValueError):
        prepare_model_input(
            row,
            [
                "lag_1",
                "lag_7",
            ],
        )