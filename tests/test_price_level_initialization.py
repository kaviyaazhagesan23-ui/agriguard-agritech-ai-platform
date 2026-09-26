from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.current_price_estimator import (
    EXPECTED_FEATURES,
    get_forecast_initialization_diagnostic,
)
from src.generate_synthetic_current_data import (
    PRICE_COLUMNS,
    _reference_price,
)


ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "master_dataset_clean.csv"
)

CURRENT_PRICE_PATH = (
    ROOT
    / "data"
    / "current"
    / "synthetic_current_prices.csv"
)


def _load_historical() -> pd.DataFrame:
    df = pd.read_csv(
        HISTORICAL_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    return df


def _load_current() -> pd.DataFrame:
    df = pd.read_csv(
        CURRENT_PRICE_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    return df


def _historical_combinations(
    historical: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    return sorted(
        historical[
            [
                "market",
                "variety",
                "grade",
            ]
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )


def test_current_price_file_has_expected_columns() -> None:
    assert CURRENT_PRICE_PATH.exists()

    current = _load_current()

    assert list(current.columns) == PRICE_COLUMNS


def test_every_current_combination_is_anchored_to_exact_latest_history() -> None:
    historical = _load_historical()
    current = _load_current()

    current_combinations = sorted(
        current[
            [
                "market",
                "variety",
                "grade",
            ]
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    assert current_combinations

    for market, variety, grade in current_combinations:
        historical_exact = historical[
            (historical.market.astype(str) == market)
            & (
                historical.variety.astype(str)
                == variety
            )
            & (
                historical.grade.astype(str)
                == grade
            )
        ].sort_values("date")

        assert not historical_exact.empty

        expected_anchor = float(
            historical_exact.iloc[-1][
                "modal_price_rs_per_quintal"
            ]
        )

        current_exact = current[
            (current.market.astype(str) == market)
            & (
                current.variety.astype(str)
                == variety
            )
            & (
                current.grade.astype(str)
                == grade
            )
        ].sort_values("date")

        assert not current_exact.empty

        first_synthetic_price = float(
            current_exact.iloc[0][
                "modal_price_rs_per_quintal"
            ]
        )

        assert first_synthetic_price == expected_anchor


def test_reference_price_uses_exact_combination_only() -> None:
    historical = _load_historical()

    combinations = _historical_combinations(
        historical
    )

    assert combinations

    for market, variety, grade in combinations:
        expected = (
            historical[
                (historical.market.astype(str) == market)
                & (
                    historical.variety.astype(str)
                    == variety
                )
                & (
                    historical.grade.astype(str)
                    == grade
                )
            ]
            .sort_values("date")
            .iloc[-1][
                "modal_price_rs_per_quintal"
            ]
        )

        actual = _reference_price(
            historical,
            market,
            variety,
            grade,
            date(2026, 7, 26),
        )

        assert actual == float(expected)


def test_kumbakonam_bpt_faq_uses_exact_historical_anchor() -> None:
    historical = _load_historical()
    current = _load_current()

    market = "Kumbakonam"
    variety = "B P T"
    grade = "FAQ"

    historical_exact = historical[
        (historical.market.astype(str) == market)
        & (
            historical.variety.astype(str)
            == variety
        )
        & (
            historical.grade.astype(str)
            == grade
        )
    ].sort_values("date")

    if historical_exact.empty:
        pytest.skip(
            "Kumbakonam / B P T / FAQ is not present "
            "in the project's historical dataset."
        )

    expected_anchor = float(
        historical_exact.iloc[-1][
            "modal_price_rs_per_quintal"
        ]
    )

    current_exact = current[
        (current.market.astype(str) == market)
        & (
            current.variety.astype(str)
            == variety
        )
        & (
            current.grade.astype(str)
            == grade
        )
    ].sort_values("date")

    assert not current_exact.empty

    actual_anchor = float(
        current_exact.iloc[0][
            "modal_price_rs_per_quintal"
        ]
    )

    assert actual_anchor == expected_anchor

    if expected_anchor == 2500:
        assert actual_anchor == 2500
        assert actual_anchor != 1790


def test_initialization_diagnostic_contains_required_fields() -> None:
    historical = _load_historical()

    combinations = _historical_combinations(
        historical
    )

    current = _load_current()

    supported = []

    for market, variety, grade in combinations:
        selected = current[
            (current.market.astype(str) == market)
            & (
                current.variety.astype(str)
                == variety
            )
            & (
                current.grade.astype(str)
                == grade
            )
        ]

        if not selected.empty:
            supported.append(
                (
                    market,
                    variety,
                    grade,
                )
            )

    if not supported:
        pytest.skip(
            "No historical combinations are available "
            "in the synthetic current dataset."
        )

    market, variety, grade = supported[0]

    first_current_date = (
        current[
            (current.market.astype(str) == market)
            & (
                current.variety.astype(str)
                == variety
            )
            & (
                current.grade.astype(str)
                == grade
            )
        ]["date"]
        .min()
    )

    start_date = (
        first_current_date
        + pd.Timedelta(days=30)
    ).date()

    try:
        diagnostic = get_forecast_initialization_diagnostic(
            market,
            variety,
            grade,
            start_date,
        )
    except FileNotFoundError as exc:
        pytest.skip(
            f"Existing model artifacts are unavailable: {exc}"
        )

    required_keys = {
        "market",
        "variety",
        "grade",
        "latest_historical_price",
        "latest_historical_date",
        "initial_synthetic_price",
        "initial_synthetic_date",
        "first_xgboost_prediction",
        "historical_to_prediction_pct_difference",
        "feature_row",
        "model",
    }

    assert required_keys.issubset(
        diagnostic.keys()
    )

    assert (
        diagnostic["market"]
        == market
    )

    assert (
        diagnostic["variety"]
        == variety
    )

    assert (
        diagnostic["grade"]
        == grade
    )

    assert (
        diagnostic["latest_historical_price"]
        > 0
    )

    assert (
        diagnostic["initial_synthetic_price"]
        > 0
    )

    assert (
        diagnostic["first_xgboost_prediction"]
        > 0
    )

    assert (
        diagnostic["model"]
        == "existing XGBoost"
    )

    assert set(
        diagnostic["feature_row"].keys()
    ) == set(EXPECTED_FEATURES)


def test_initialization_diagnostic_does_not_apply_prediction_correction() -> None:
    historical = _load_historical()
    current = _load_current()

    combinations = _historical_combinations(
        historical
    )

    for market, variety, grade in combinations:
        selected = current[
            (current.market.astype(str) == market)
            & (
                current.variety.astype(str)
                == variety
            )
            & (
                current.grade.astype(str)
                == grade
            )
        ].sort_values("date")

        if selected.empty:
            continue

        start_date = (
            selected.iloc[0]["date"]
            + pd.Timedelta(days=30)
        ).date()

        try:
            diagnostic = (
                get_forecast_initialization_diagnostic(
                    market,
                    variety,
                    grade,
                    start_date,
                )
            )
        except FileNotFoundError:
            pytest.skip(
                "Existing model artifacts are unavailable."
            )

        prediction = float(
            diagnostic[
                "first_xgboost_prediction"
            ]
        )

        historical_price = float(
            diagnostic[
                "latest_historical_price"
            ]
        )

        reported_difference = float(
            diagnostic[
                "historical_to_prediction_pct_difference"
            ]
        )

        calculated_difference = (
            (
                prediction
                - historical_price
            )
            / historical_price
            * 100.0
        )

        assert reported_difference == pytest.approx(
            calculated_difference,
            rel=1e-9,
            abs=1e-9,
        )

        break