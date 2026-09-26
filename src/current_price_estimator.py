"""Current/future paddy-price forecasting using the existing 41-feature XGBoost model.

The historical dataset and trained model are read-only inputs. This module adds a
separate synthetic-current recursive forecasting layer for the application.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CURRENT_PRICE_PATH = (
    ROOT
    / "data"
    / "current"
    / "synthetic_current_prices.csv"
)

CURRENT_WEATHER_PATH = (
    ROOT
    / "data"
    / "current"
    / "synthetic_current_weather.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "phase4"
    / "xgboost.joblib"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "phase4"
    / "preprocessing_metadata.joblib"
)

HISTORICAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "master_dataset_clean.csv"
)

EXPECTED_FEATURES = [
    "sl_no",
    "min_price_rs_per_quintal",
    "max_price_rs_per_quintal",
    "latitude",
    "longitude",
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_year",
    "sin_month",
    "cos_month",
    "temperature_mean_lag_1",
    "temperature_mean_lag_3",
    "temperature_mean_lag_7",
    "temperature_max_lag_1",
    "temperature_max_lag_3",
    "temperature_max_lag_7",
    "temperature_min_lag_1",
    "temperature_min_lag_3",
    "temperature_min_lag_7",
    "rainfall_lag_1",
    "rainfall_lag_3",
    "rainfall_lag_7",
    "wind_speed_max_lag_1",
    "wind_speed_max_lag_3",
    "wind_speed_max_lag_7",
    "et0_lag_1",
    "et0_lag_3",
    "et0_lag_7",
]

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
]

PRICE_COLUMNS = [
    "date",
    "market",
    "commodity",
    "variety",
    "grade",
    "min_price_rs_per_quintal",
    "max_price_rs_per_quintal",
    "modal_price_rs_per_quintal",
]


@lru_cache(maxsize=1)
def _load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Any,
    dict,
]:
    required_paths = (
        CURRENT_PRICE_PATH,
        CURRENT_WEATHER_PATH,
        MODEL_PATH,
        METADATA_PATH,
        HISTORICAL_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required current-price forecasting file not found: {path}"
            )

    prices = pd.read_csv(
        CURRENT_PRICE_PATH
    )

    weather = pd.read_csv(
        CURRENT_WEATHER_PATH
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="raise",
    )

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="raise",
    )

    model = joblib.load(
        MODEL_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    metadata_features = list(
        metadata.get(
            "feature_columns",
            [],
        )
    )

    model_features = list(
        getattr(
            model,
            "feature_names_in_",
            [],
        )
    )

    if metadata_features != EXPECTED_FEATURES:
        raise ValueError(
            "Existing preprocessing metadata feature contract "
            "does not match the required 41-feature contract."
        )

    if (
        model_features
        and model_features != EXPECTED_FEATURES
    ):
        raise ValueError(
            "Existing XGBoost feature order does not match "
            "the required 41-feature contract."
        )

    if (
        getattr(
            model,
            "n_features_in_",
            None,
        )
        != len(EXPECTED_FEATURES)
    ):
        raise ValueError(
            "Existing XGBoost model does not accept exactly "
            "41 features."
        )

    return (
        prices,
        weather,
        model,
        metadata,
    )


def _validate_selection(
    prices: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
) -> pd.DataFrame:
    selected = prices[
        (prices.market.astype(str) == str(market))
        & (
            prices.variety.astype(str)
            == str(variety)
        )
        & (
            prices.grade.astype(str)
            == str(grade)
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "Unknown current synthetic "
            f"market/variety/grade combination: "
            f"{market}/{variety}/{grade}"
        )

    return (
        selected
        .sort_values("date")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=None)
def _historical_sl_no(
    market: str,
    variety: str,
    grade: str,
) -> float:
    hist = pd.read_csv(
        HISTORICAL_PATH,
        usecols=[
            "market",
            "variety",
            "grade",
            "date",
            "sl_no",
        ],
    )

    selected = hist[
        (hist.market.astype(str) == str(market))
        & (
            hist.variety.astype(str)
            == str(variety)
        )
        & (
            hist.grade.astype(str)
            == str(grade)
        )
    ].copy()

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="raise",
    )

    selected = selected.sort_values(
        "date"
    )

    if selected.empty:
        raise ValueError(
            f"No historical sl_no exists for "
            f"{market}/{variety}/{grade}."
        )

    value = pd.to_numeric(
        selected.iloc[-1]["sl_no"],
        errors="coerce",
    )

    if pd.isna(value):
        raise ValueError(
            f"Historical sl_no is missing for "
            f"{market}/{variety}/{grade}."
        )

    return float(value)


@lru_cache(maxsize=None)
def _market_coordinate(
    market: str,
    name: str,
) -> float:
    hist = pd.read_csv(
        HISTORICAL_PATH,
        usecols=[
            "market",
            name,
        ],
    )

    values = pd.to_numeric(
        hist.loc[
            hist.market.astype(str)
            == str(market),
            name,
        ],
        errors="coerce",
    ).dropna().unique()

    if len(values) == 0:
        raise ValueError(
            f"No verified historical {name} coordinate "
            f"exists for market '{market}'."
        )

    if len(values) > 1:
        raise ValueError(
            f"Historical {name} coordinate is inconsistent "
            f"for market '{market}'."
        )

    return float(values[0])


def _build_features(
    price_history: pd.DataFrame,
    weather_history: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build the exact 41 model features.

    Only information available before target_date is used.
    """
    prices = (
        price_history
        .sort_values("date")
        .copy()
    )

    weather = (
        weather_history
        .sort_values("date")
        .copy()
    )

    selected = prices[
        (prices.market.astype(str) == str(market))
        & (
            prices.variety.astype(str)
            == str(variety)
        )
        & (
            prices.grade.astype(str)
            == str(grade)
        )
    ].copy()

    selected = (
        selected
        .sort_values("date")
        .reset_index(drop=True)
    )

    observed = selected[
        selected.date < target_date
    ].copy()

    if len(observed) < 30:
        raise ValueError(
            "At least 30 preceding current-price observations "
            "are required to construct lag_30 without imputation."
        )

    market_weather = weather[
        weather.market.astype(str)
        == str(market)
    ].sort_values("date").copy()

    required_weather_dates = pd.date_range(
        target_date - pd.Timedelta(days=7),
        target_date - pd.Timedelta(days=1),
        freq="D",
    )

    if not set(
        required_weather_dates
    ).issubset(
        set(market_weather.date)
    ):
        raise ValueError(
            "Synthetic forecast weather coverage is incomplete "
            "for the requested forecast horizon."
        )

    values = pd.to_numeric(
        observed.modal_price_rs_per_quintal,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if (
        len(values) < 30
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Current price history is insufficient "
            "or non-finite for recursive features."
        )

    row: dict[str, float | int] = {}

    for lag in (
        1,
        3,
        7,
        14,
        30,
    ):
        row[f"lag_{lag}"] = float(
            values[-lag]
        )

    for window in (
        7,
        14,
        30,
    ):
        window_values = values[-window:]

        row[
            f"rolling_mean_{window}"
        ] = float(
            np.mean(window_values)
        )

        row[
            f"rolling_std_{window}"
        ] = (
            float(
                np.std(
                    window_values,
                    ddof=1,
                )
            )
            if len(window_values) >= 2
            else 0.0
        )

    row["year"] = int(
        target_date.year
    )

    row["month"] = int(
        target_date.month
    )

    row["quarter"] = int(
        target_date.quarter
    )

    row["week_of_year"] = int(
        target_date.isocalendar().week
    )

    row["day_of_year"] = int(
        target_date.dayofyear
    )

    row["sin_month"] = float(
        np.sin(
            2
            * np.pi
            * target_date.month
            / 12
        )
    )

    row["cos_month"] = float(
        np.cos(
            2
            * np.pi
            * target_date.month
            / 12
        )
    )

    weather_by_date = (
        market_weather
        .set_index("date")
    )

    for column in WEATHER_COLUMNS:
        for lag in (
            1,
            3,
            7,
        ):
            value = pd.to_numeric(
                weather_by_date.loc[
                    target_date
                    - pd.Timedelta(days=lag),
                    column,
                ],
                errors="coerce",
            )

            if pd.isna(value):
                raise ValueError(
                    "Synthetic forecast weather contains "
                    f"an invalid {column} lag {lag} value."
                )

            row[
                f"{column}_lag_{lag}"
            ] = float(value)

    latest = observed.iloc[-1]

    row["sl_no"] = _historical_sl_no(
        market,
        variety,
        grade,
    )

    row[
        "min_price_rs_per_quintal"
    ] = float(
        latest[
            "min_price_rs_per_quintal"
        ]
    )

    row[
        "max_price_rs_per_quintal"
    ] = float(
        latest[
            "max_price_rs_per_quintal"
        ]
    )

    row["latitude"] = _market_coordinate(
        market,
        "latitude",
    )

    row["longitude"] = _market_coordinate(
        market,
        "longitude",
    )

    features = pd.DataFrame(
        [
            [
                row[column]
                for column in EXPECTED_FEATURES
            ]
        ],
        columns=EXPECTED_FEATURES,
    )

    if (
        features.isna().any().any()
        or not np.isfinite(
            features.to_numpy(
                dtype=float
            )
        ).all()
    ):
        raise ValueError(
            "Constructed model features contain missing "
            "or non-finite values; no fallback imputation is allowed."
        )

    return features


def get_forecast_initialization_diagnostic(
    market: str,
    variety: str,
    grade: str,
    start_date: date | str,
) -> dict[str, Any]:
    """Return transparent price-level initialization diagnostics.

    The diagnostic exposes:
    - selected combination
    - latest historical price
    - latest historical date
    - synthetic starting price
    - first existing-XGBoost prediction
    - percentage difference
    """
    (
        prices,
        weather,
        model,
        _,
    ) = _load_inputs()

    start = pd.Timestamp(
        start_date
    )

    series = _validate_selection(
        prices,
        market,
        variety,
        grade,
    )

    historical = pd.read_csv(
        HISTORICAL_PATH,
        usecols=[
            "date",
            "market",
            "variety",
            "grade",
            "modal_price_rs_per_quintal",
        ],
    )

    historical["date"] = pd.to_datetime(
        historical["date"],
        errors="raise",
    )

    exact_hist = historical[
        (historical.market.astype(str) == str(market))
        & (
            historical.variety.astype(str)
            == str(variety)
        )
        & (
            historical.grade.astype(str)
            == str(grade)
        )
    ].sort_values("date")

    if exact_hist.empty:
        raise ValueError(
            f"No historical rows exist for "
            f"{market}/{variety}/{grade}."
        )

    latest_hist = float(
        exact_hist.iloc[-1][
            "modal_price_rs_per_quintal"
        ]
    )

    observed = series[
        series.date < start
    ].copy()

    if observed.empty:
        raise ValueError(
            f"No synthetic current price exists before "
            f"{start.date()} for "
            f"{market}/{variety}/{grade}."
        )

    initial_synthetic = float(
        observed.iloc[-1][
            "modal_price_rs_per_quintal"
        ]
    )

    features = _build_features(
        series,
        weather,
        market,
        variety,
        grade,
        start,
    )

    first_prediction = float(
        model.predict(features)[0]
    )

    pct_difference = (
        (
            first_prediction
            - latest_hist
        )
        / latest_hist
    ) * 100.0

    return {
        "market": market,
        "variety": variety,
        "grade": grade,
        "latest_historical_price": latest_hist,
        "latest_historical_date": (
            exact_hist.iloc[-1]["date"]
            .date()
            .isoformat()
        ),
        "initial_synthetic_price": initial_synthetic,
        "initial_synthetic_date": (
            observed.iloc[-1]["date"]
            .date()
            .isoformat()
        ),
        "first_xgboost_prediction": first_prediction,
        "historical_to_prediction_pct_difference": float(
            pct_difference
        ),
        "feature_row": features.iloc[0].to_dict(),
        "model": "existing XGBoost",
    }


def estimate_price(
    market: str,
    variety: str,
    grade: str,
    target_date: date | str | None = None,
) -> dict:
    """Backward-compatible single-date estimator.

    For future dates beyond the synthetic observed-price layer,
    forecast_future_prices must be used so predictions are recursive.
    """
    (
        prices,
        weather,
        model,
        _,
    ) = _load_inputs()

    target = pd.Timestamp(
        target_date or date.today()
    )

    series = _validate_selection(
        prices,
        market,
        variety,
        grade,
    )

    if target > series.date.max():
        raise ValueError(
            "For future dates, use forecast_future_prices "
            "so recursive predictions are used."
        )

    features = _build_features(
        series,
        weather,
        market,
        variety,
        grade,
        target,
    )

    prediction = float(
        model.predict(features)[0]
    )

    if (
        not np.isfinite(prediction)
        or prediction <= 0
    ):
        raise ValueError(
            "Existing XGBoost returned an invalid "
            f"prediction: {prediction}"
        )

    recent = (
        series[
            series.date < target
        ]
        .tail(30)
        .modal_price_rs_per_quintal
        .astype(float)
    )

    lo = float(recent.min())
    hi = float(recent.max())

    if (
        prediction < lo * 0.50
        or prediction > hi * 1.50
    ):
        raise ValueError(
            f"Existing XGBoost prediction {prediction:.2f} "
            "is outside the reasonable recent synthetic range "
            f"[{lo:.2f}, {hi:.2f}] by more than 50%."
        )

    return {
        "estimated_price_rs_per_quintal": prediction,
        "market": market,
        "variety": variety,
        "grade": grade,
        "target_date": target.date().isoformat(),
        "feature_columns": EXPECTED_FEATURES.copy(),
        "feature_row": features.iloc[0].to_dict(),
        "recent_synthetic_price_range": (
            lo,
            hi,
        ),
        "synthetic_current_data": True,
        "model": "existing XGBoost",
    }


def forecast_future_prices(
    market: str,
    variety: str,
    grade: str,
    start_date: date | str,
    horizon_days: int,
) -> pd.DataFrame:
    """Recursively forecast one price per day.

    Each prediction is appended to the selected price history before
    constructing the next day's lag and rolling features.
    """
    if int(horizon_days) <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    (
        prices,
        weather,
        model,
        _,
    ) = _load_inputs()

    start = pd.Timestamp(
        start_date
    )

    end = (
        start
        + pd.Timedelta(
            days=int(horizon_days) - 1
        )
    )

    series = _validate_selection(
        prices,
        market,
        variety,
        grade,
    )

    if start <= series.date.min():
        raise ValueError(
            "start_date must be after the first "
            "current synthetic price observation."
        )

    market_weather = weather[
        weather.market.astype(str)
        == str(market)
    ]

    if market_weather.empty:
        raise ValueError(
            f"No synthetic forecast weather exists "
            f"for market '{market}'."
        )

    weather_max_date = market_weather.date.max()

    if start > weather_max_date:
        raise ValueError(
            "Requested forecast starts after the available "
            "synthetic forecast-weather window."
        )

    if end > weather_max_date:
        raise ValueError(
            "Synthetic forecast weather is available only through "
            f"{weather_max_date.date()}."
        )

    history = (
        series[
            series.date < start
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(history) < 30:
        raise ValueError(
            "At least 30 current synthetic price observations "
            "before start_date are required."
        )

    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for step in range(
        int(horizon_days)
    ):
        target = (
            start
            + pd.Timedelta(days=step)
        )

        features = _build_features(
            history,
            weather,
            market,
            variety,
            grade,
            target,
        )

        prediction = float(
            model.predict(features)[0]
        )

        if (
            not np.isfinite(prediction)
            or prediction <= 0
        ):
            raise ValueError(
                "Existing XGBoost returned an invalid "
                f"recursive prediction for "
                f"{target.date()}: {prediction}"
            )

        feature_row = (
            features.iloc[0]
            .to_dict()
        )

        feature_rows.append(
            feature_row
        )

        rows.append(
            {
                "date": target.date(),
                "market": market,
                "variety": variety,
                "grade": grade,
                "predicted_price_rs_per_quintal": prediction,
                "forecast_day": step + 1,
                "weather_input": "synthetic forecast input",
                "model": "existing XGBoost",
            }
        )

        latest = history.iloc[-1].copy()

        latest["date"] = target

        latest[
            "modal_price_rs_per_quintal"
        ] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame([latest]),
            ],
            ignore_index=True,
        )

    result = pd.DataFrame(
        rows
    )

    result.attrs[
        "feature_rows"
    ] = feature_rows

    result.attrs[
        "model_features"
    ] = EXPECTED_FEATURES.copy()

    result.attrs[
        "recursive"
    ] = True

    historical = pd.read_csv(
        HISTORICAL_PATH,
        usecols=[
            "date",
            "market",
            "variety",
            "grade",
            "modal_price_rs_per_quintal",
        ],
    )

    historical["date"] = pd.to_datetime(
        historical["date"],
        errors="raise",
    )

    exact_hist = historical[
        (historical.market.astype(str) == str(market))
        & (
            historical.variety.astype(str)
            == str(variety)
        )
        & (
            historical.grade.astype(str)
            == str(grade)
        )
    ].sort_values("date")

    if exact_hist.empty:
        raise ValueError(
            f"No historical rows exist for "
            f"{market}/{variety}/{grade}."
        )

    latest_hist = float(
        exact_hist.iloc[-1][
            "modal_price_rs_per_quintal"
        ]
    )

    initial_synthetic = float(
        series[
            series.date < start
        ].iloc[-1][
            "modal_price_rs_per_quintal"
        ]
    )

    first_prediction = float(
        result.iloc[0][
            "predicted_price_rs_per_quintal"
        ]
    )

    result.attrs[
        "initialization_diagnostic"
    ] = {
        "market": market,
        "variety": variety,
        "grade": grade,
        "latest_historical_price": latest_hist,
        "latest_historical_date": (
            exact_hist.iloc[-1]["date"]
            .date()
            .isoformat()
        ),
        "initial_synthetic_price": initial_synthetic,
        "initial_synthetic_date": (
            start
            - pd.Timedelta(days=1)
        ).date().isoformat(),
        "first_xgboost_prediction": first_prediction,
        "historical_to_prediction_pct_difference": float(
            (
                first_prediction
                - latest_hist
            )
            / latest_hist
            * 100.0
        ),
        "feature_row": feature_rows[0].copy(),
    }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "market"
    )

    parser.add_argument(
        "variety"
    )

    parser.add_argument(
        "grade"
    )

    parser.add_argument(
        "--date",
        default=None,
    )

    args = parser.parse_args()

    result = estimate_price(
        args.market,
        args.variety,
        args.grade,
        args.date,
    )

    print(
        "Estimated price: "
        f"Rs {result['estimated_price_rs_per_quintal']:,.0f} "
        "/ quintal"
    )