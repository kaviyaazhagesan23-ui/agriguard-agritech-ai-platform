"""
Phase 6 — Future Price Forecasting

Purpose
-------
Generate recursive future paddy-price forecasts using the
Phase 5 selected model: XGBoost.

Important forecasting assumptions
---------------------------------
1. Future weather is NOT known.
2. Future weather variables are carried forward from the latest
   available observation for the selected market.
3. Weather lag features are therefore generated from this
   persistence assumption.
4. Future min/max market prices are also unavailable at forecast
   time. Because the Phase 4 model contains these contemporaneous
   features, the latest observed values are carried forward.
5. This is an inference assumption, not observed future information.
6. Prediction intervals are empirical residual bands derived from
   2025 TEST-set XGBoost errors.
7. The intervals are NOT guaranteed confidence intervals and are
   not calibrated specifically for 14-day or 30-day recursive
   forecasts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "phase4"

MODEL_PATH = MODEL_DIR / "xgboost.joblib"

METADATA_PATH = MODEL_DIR / "preprocessing_metadata.joblib"

PHASE4_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase4"
    / "predictions.csv"
)

PHASE6_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
)

FORECAST_DIR = PHASE6_DIR / "forecasts"

PLOT_DIR = PHASE6_DIR / "plots"

REPORT_DIR = PHASE6_DIR / "reports"


# ============================================================
# CONSTANTS
# ============================================================

DATE_COL = "date"

MARKET_COL = "market"

VARIETY_COL = "variety"

GRADE_COL = "grade"

TARGET_COL = "modal_price_rs_per_quintal"

MODEL_NAME = "XGBoost"

RANDOM_STATE = 42

SUPPORTED_HORIZONS = (7, 14, 30)

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
]

WEATHER_LAG_COLUMNS = [
    f"{column}_lag_{lag}"
    for column in WEATHER_COLUMNS
    for lag in (1, 3, 7)
]

PRICE_LAG_COLUMNS = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
]

ROLLING_MEAN_COLUMNS = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]

ROLLING_STD_COLUMNS = [
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_30",
]

CALENDAR_COLUMNS = [
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_year",
    "sin_month",
    "cos_month",
]

PRICE_DYNAMIC_COLUMNS = [
    "price_change_1d",
    "price_change_7d",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories() -> None:
    """Create Phase 6 output directories."""

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the actual engineered dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    if df[DATE_COL].isna().any():
        raise ValueError(
            "Feature dataset contains invalid dates."
        )

    df = df.sort_values(
        [DATE_COL, MARKET_COL]
    ).reset_index(drop=True)

    logger.info(
        "Loaded feature dataset: %d rows, %d columns",
        len(df),
        len(df.columns),
    )

    return df


def load_model():
    """Load the Phase 4 XGBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logger.info(
        "Loaded XGBoost model from %s",
        MODEL_PATH,
    )

    return model


def load_metadata():
    """
    Load preprocessing/model metadata if available.

    The Phase 4 implementation may use different metadata keys,
    so this function does not assume one exact key name.
    """

    if not METADATA_PATH.exists():
        logger.warning(
            "Preprocessing metadata not found: %s",
            METADATA_PATH,
        )
        return {}

    metadata = joblib.load(METADATA_PATH)

    if isinstance(metadata, dict):
        logger.info(
            "Loaded preprocessing metadata."
        )
        return metadata

    logger.warning(
        "Preprocessing metadata is not a dictionary."
    )

    return {}


# ============================================================
# FEATURE COLUMN DISCOVERY
# ============================================================

def discover_feature_columns(
    model,
    metadata: Dict,
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Discover the exact model feature columns.

    Priority:
    1. preprocessing metadata
    2. model feature_names_in_
    3. numeric feature inference

    The inferred list is accepted only if its length matches
    the number of features expected by the trained model.
    """

    possible_keys = [
        "feature_columns",
        "features",
        "model_features",
        "feature_names",
        "numeric_features",
        "selected_features",
    ]

    for key in possible_keys:

        value = metadata.get(key)

        if isinstance(value, (list, tuple)):

            columns = [
                str(column)
                for column in value
                if str(column) in dataframe.columns
            ]

            if columns:
                logger.info(
                    "Using feature columns from metadata key '%s'.",
                    key,
                )

                return columns

    if hasattr(model, "feature_names_in_"):

        columns = [
            str(column)
            for column in model.feature_names_in_
        ]

        if all(
            column in dataframe.columns
            for column in columns
        ):
            logger.info(
                "Using model.feature_names_in_."
            )

            return columns

    # --------------------------------------------------------
    # Fallback inference
    # --------------------------------------------------------

    excluded = {
        DATE_COL,
        MARKET_COL,
        VARIETY_COL,
        GRADE_COL,
        TARGET_COL,
        "record_id",
        "split",
        "district_y",
        "state",
    }

    numeric_candidates = []

    for column in dataframe.columns:

        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            continue

        # Current price-dynamics are not safe for direct
        # future inference unless explicitly generated.
        if column in PRICE_DYNAMIC_COLUMNS:
            continue

        numeric_candidates.append(column)

    expected = getattr(
        model,
        "n_features_in_",
        None,
    )

    if expected is not None and len(numeric_candidates) != expected:

        raise ValueError(
            "Could not safely identify the exact Phase 4 "
            f"feature columns.\n"
            f"Model expects: {expected}\n"
            f"Candidate numeric features: "
            f"{len(numeric_candidates)}\n\n"
            "The trained model metadata does not expose a "
            "recoverable feature list. Do not guess the "
            "feature ordering."
        )

    logger.warning(
        "Using numeric feature inference because explicit "
        "feature metadata was unavailable."
    )

    return numeric_candidates


# ============================================================
# USER SELECTION
# ============================================================

def available_values(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Return available market/variety/grade values."""

    return {
        MARKET_COL: sorted(
            df[MARKET_COL]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        VARIETY_COL: sorted(
            df[VARIETY_COL]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        GRADE_COL: sorted(
            df[GRADE_COL]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }


def validate_selection(
    df: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
) -> None:
    """Validate market/variety/grade selection."""

    subset = df[
        (df[MARKET_COL].astype(str) == str(market))
        & (df[VARIETY_COL].astype(str) == str(variety))
        & (df[GRADE_COL].astype(str) == str(grade))
    ]

    if subset.empty:
        raise ValueError(
            "No historical observations exist for:\n"
            f"Market: {market}\n"
            f"Variety: {variety}\n"
            f"Grade: {grade}"
        )


# ============================================================
# FEATURE ENGINEERING FOR FUTURE DATES
# ============================================================

def add_calendar_features(
    row: pd.Series,
    forecast_date: pd.Timestamp,
) -> None:
    """Update calendar features in-place."""

    row["year"] = forecast_date.year

    row["month"] = forecast_date.month

    row["quarter"] = forecast_date.quarter

    row["week_of_year"] = int(
        forecast_date.isocalendar().week
    )

    row["day_of_year"] = forecast_date.dayofyear

    row["sin_month"] = np.sin(
        2 * np.pi * forecast_date.month / 12
    )

    row["cos_month"] = np.cos(
        2 * np.pi * forecast_date.month / 12
    )


def calculate_lag_value(
    history: pd.DataFrame,
    market: str,
    lag: int,
) -> float:
    """
    Get target lag using historical/predicted market prices.

    Feature engineering in Phase 3 was performed per market.
    Therefore lag history is market-level rather than
    variety/grade-level.
    """

    market_history = history[
        history[MARKET_COL].astype(str)
        == str(market)
    ].sort_values(DATE_COL)

    if len(market_history) < lag:
        return np.nan

    return float(
        market_history[TARGET_COL]
        .iloc[-lag]
    )


def calculate_rolling_mean(
    history: pd.DataFrame,
    market: str,
    window: int,
) -> float:
    """Calculate previous market rolling mean."""

    market_history = history[
        history[MARKET_COL].astype(str)
        == str(market)
    ].sort_values(DATE_COL)

    values = market_history[TARGET_COL].dropna()

    if values.empty:
        return np.nan

    return float(
        values.tail(window).mean()
    )


def calculate_rolling_std(
    history: pd.DataFrame,
    market: str,
    window: int,
) -> float:
    """Calculate previous market rolling standard deviation."""

    market_history = history[
        history[MARKET_COL].astype(str)
        == str(market)
    ].sort_values(DATE_COL)

    values = market_history[TARGET_COL].dropna()

    if len(values) < 2:
        return np.nan

    values = values.tail(window)

    if len(values) < 2:
        return np.nan

    return float(
        values.std()
    )


def update_price_features(
    row: pd.Series,
    history: pd.DataFrame,
    market: str,
) -> None:
    """Generate future price lag/rolling features."""

    for lag in (1, 3, 7, 14, 30):

        row[f"lag_{lag}"] = calculate_lag_value(
            history,
            market,
            lag,
        )

    for window in (7, 14, 30):

        row[f"rolling_mean_{window}"] = (
            calculate_rolling_mean(
                history,
                market,
                window,
            )
        )

        row[f"rolling_std_{window}"] = (
            calculate_rolling_std(
                history,
                market,
                window,
            )
        )


def update_price_dynamics(
    row: pd.Series,
    history: pd.DataFrame,
    market: str,
) -> None:
    """
    Generate price-change features if the trained model
    requires them.

    These features are generated only from already-known
    historical/predicted values.
    """

    market_history = history[
        history[MARKET_COL].astype(str)
        == str(market)
    ].sort_values(DATE_COL)

    values = (
        market_history[TARGET_COL]
        .dropna()
        .tolist()
    )

    if len(values) >= 1:
        row["price_change_1d"] = (
            values[-1] - values[-2]
            if len(values) >= 2
            else 0.0
        )

    if len(values) >= 7:
        row["price_change_7d"] = (
            values[-1] - values[-7]
        )


def update_weather_features(
    row: pd.Series,
    latest_market_row: pd.Series,
) -> None:
    """
    Future-weather strategy:

    Carry the latest available weather observation forward.

    This is an explicit persistence assumption.
    It is NOT a weather forecast.
    """

    for column in WEATHER_COLUMNS:

        if column in latest_market_row.index:

            row[column] = latest_market_row[column]

    # Generate weather lag features using the same persistence
    # assumption. Since the latest weather observation is used
    # repeatedly, the lagged values remain equal to that
    # observation unless historical values are available.
    for column in WEATHER_COLUMNS:

        for lag in (1, 3, 7):

            lag_column = f"{column}_lag_{lag}"

            if lag_column in row.index:

                historical_value = latest_market_row.get(
                    lag_column,
                    np.nan,
                )

                if pd.notna(historical_value):
                    row[lag_column] = historical_value
                else:
                    row[lag_column] = latest_market_row.get(
                        column,
                        np.nan,
                    )


# ============================================================
# FUTURE ROW CONSTRUCTION
# ============================================================

def build_future_row(
    history: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
    forecast_date: pd.Timestamp,
) -> pd.Series:
    """
    Build one future feature row.

    The row starts from the latest historical observation for
    the selected market/variety/grade and then updates every
    feature that can be generated without future information.
    """

    selected_history = history[
        (history[MARKET_COL].astype(str) == str(market))
        & (
            history[VARIETY_COL].astype(str)
            == str(variety)
        )
        & (
            history[GRADE_COL].astype(str)
            == str(grade)
        )
    ].sort_values(DATE_COL)

    if selected_history.empty:
        raise ValueError(
            "No historical rows found for selected "
            "market/variety/grade."
        )

    latest_selected = selected_history.iloc[-1]

    # Start from latest observed row.
    row = latest_selected.copy()

    row[DATE_COL] = forecast_date

    row[MARKET_COL] = market

    row[VARIETY_COL] = variety

    row[GRADE_COL] = grade

    # Calendar features
    add_calendar_features(
        row,
        forecast_date,
    )

    # Market-level recursive price features
    update_price_features(
        row,
        history,
        market,
    )

    # Price-dynamics features
    update_price_dynamics(
        row,
        history,
        market,
    )

    # Future weather assumption
    market_history = history[
        history[MARKET_COL].astype(str)
        == str(market)
    ].sort_values(DATE_COL)

    latest_market_row = market_history.iloc[-1]

    update_weather_features(
        row,
        latest_market_row,
    )

    return row


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_model_input(
    row: pd.Series,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Prepare exactly the columns expected by the trained model.
    """

    missing = [
        column
        for column in feature_columns
        if column not in row.index
    ]

    if missing:
        raise ValueError(
            "Future row is missing required model features:\n"
            + "\n".join(missing)
        )

    X = pd.DataFrame(
        [[row[column] for column in feature_columns]],
        columns=feature_columns,
    )

    # Ensure numeric model input.
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


# ============================================================
# MISSING FEATURE HANDLING
# ============================================================

def fill_missing_features(
    X: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill remaining feature NaNs using the latest known value.

    This is only a fallback for unavailable lag history.
    It does not use future target values.
    """

    result = X.copy()

    for column in result.columns:

        if result[column].isna().any():

            historical = pd.to_numeric(
                history[column],
                errors="coerce",
            ) if column in history.columns else pd.Series(
                dtype=float
            )

            historical = historical.dropna()

            if not historical.empty:

                result[column] = result[column].fillna(
                    float(historical.iloc[-1])
                )

    remaining = result.isna().sum()

    missing = remaining[
        remaining > 0
    ]

    if not missing.empty:

        raise ValueError(
            "Unable to safely construct future model input. "
            "Missing features:\n"
            + "\n".join(
                f"{column}: {count}"
                for column, count
                in missing.items()
            )
        )

    return result


# ============================================================
# UNCERTAINTY
# ============================================================

def calculate_empirical_interval(
    predictions_path: Path = PHASE4_PREDICTIONS_PATH,
    quantile: float = 0.90,
) -> float:
    """
    Calculate empirical absolute-error uncertainty from
    2025 TEST-set XGBoost predictions.

    Returns
    -------
    float
        Absolute-error quantile in ₹/quintal.

    Important
    ---------
    This is not a formally calibrated probabilistic interval
    for recursive 7/14/30-day forecasts.
    """

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Phase 4 predictions not found:\n"
            f"{predictions_path}"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    required = {
        "split",
        "model",
        "actual",
        "prediction",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "Phase 4 predictions missing columns: "
            f"{sorted(missing)}"
        )

    test = predictions[
        (
            predictions["split"]
            .astype(str)
            .str.lower()
            == "test"
        )
        & (
            predictions["model"]
            .astype(str)
            .str.strip()
            == MODEL_NAME
        )
    ].copy()

    if test.empty:
        raise ValueError(
            "No XGBoost TEST predictions found."
        )

    actual = pd.to_numeric(
        test["actual"],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        test["prediction"],
        errors="coerce",
    )

    residuals = (
        actual - predicted
    ).dropna()

    absolute_errors = residuals.abs()

    interval = float(
        absolute_errors.quantile(
            quantile
        )
    )

    logger.info(
        "Empirical %.0f%% absolute-error band: Rs %.2f/quintal",
        quantile * 100,
        interval,
    )

    return interval


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def forecast(
    df: pd.DataFrame,
    model,
    feature_columns: List[str],
    market: str,
    variety: str,
    grade: str,
    horizon: int,
) -> Tuple[pd.DataFrame, float]:
    """
    Generate recursive forecasts.

    Each predicted price becomes available to the next
    forecast step and therefore updates lag/rolling features.
    """

    if horizon not in SUPPORTED_HORIZONS:

        raise ValueError(
            f"Horizon must be one of "
            f"{SUPPORTED_HORIZONS}."
        )

    validate_selection(
        df,
        market,
        variety,
        grade,
    )

    history = df.copy()

    history[DATE_COL] = pd.to_datetime(
        history[DATE_COL]
    )

    selected = history[
        (history[MARKET_COL].astype(str) == str(market))
        & (
            history[VARIETY_COL].astype(str)
            == str(variety)
        )
        & (
            history[GRADE_COL].astype(str)
            == str(grade)
        )
    ]

    latest_date = selected[DATE_COL].max()

    interval = calculate_empirical_interval()

    forecast_rows = []

    for step in range(1, horizon + 1):

        forecast_date = (
            latest_date
            + pd.Timedelta(days=step)
        )

        future_row = build_future_row(
            history=history,
            market=market,
            variety=variety,
            grade=grade,
            forecast_date=forecast_date,
        )

        X_future = prepare_model_input(
            future_row,
            feature_columns,
        )

        X_future = fill_missing_features(
            X_future,
            history,
        )

        prediction = float(
            np.asarray(
                model.predict(X_future)
            ).reshape(-1)[0]
        )

        lower = max(
            0.0,
            prediction - interval,
        )

        upper = (
            prediction + interval
        )

        future_row[TARGET_COL] = prediction

        # Add predicted row to recursive history.
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [future_row]
                ),
            ],
            ignore_index=True,
        )

        forecast_rows.append(
            {
                DATE_COL: forecast_date,
                MARKET_COL: market,
                VARIETY_COL: variety,
                GRADE_COL: grade,
                "forecast_day": step,
                "predicted_price_rs_per_quintal": prediction,
                "lower_bound_rs_per_quintal": lower,
                "upper_bound_rs_per_quintal": upper,
                "uncertainty_method": (
                    "2025 TEST XGBoost "
                    "empirical absolute-error "
                    "90% band"
                ),
            }
        )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    return forecast_df, interval


# ============================================================
# VISUALIZATION
# ============================================================

def create_forecast_plot(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
) -> Path:
    """Create historical + future forecast visualization."""

    selected = historical_df[
        (historical_df[MARKET_COL].astype(str) == str(market))
        & (
            historical_df[VARIETY_COL].astype(str)
            == str(variety)
        )
        & (
            historical_df[GRADE_COL].astype(str)
            == str(grade)
        )
    ].sort_values(DATE_COL)

    # Display recent history for readability.
    selected = selected.tail(120)

    path = (
        PLOT_DIR
        / (
            f"forecast_"
            f"{market}_"
            f"{variety}_"
            f"{grade}.png"
        )
    )

    # Make filename Windows-safe.
    path = Path(
        str(path).replace("/", "_").replace("\\", "_")
    )

    plt.figure(
        figsize=(13, 7)
    )

    plt.plot(
        selected[DATE_COL],
        selected[TARGET_COL],
        label="Historical modal price",
    )

    plt.plot(
        forecast_df[DATE_COL],
        forecast_df[
            "predicted_price_rs_per_quintal"
        ],
        marker="o",
        label="Forecast",
    )

    plt.fill_between(
        forecast_df[DATE_COL],
        forecast_df[
            "lower_bound_rs_per_quintal"
        ],
        forecast_df[
            "upper_bound_rs_per_quintal"
        ],
        alpha=0.20,
        label="Empirical uncertainty band",
    )

    plt.axvline(
        selected[DATE_COL].max(),
        linestyle="--",
        label="Forecast starts",
    )

    plt.title(
        "Paddy Price Forecast — "
        f"{market} / {variety} / {grade}"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Modal price (₹/quintal)"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Forecast plot written to: %s",
        path,
    )

    return path


# ============================================================
# EXPORT
# ============================================================

def export_forecast(
    forecast_df: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
    horizon: int,
) -> Path:
    """Save forecast CSV."""

    filename = (
        f"forecast_"
        f"{market}_"
        f"{variety}_"
        f"{grade}_"
        f"{horizon}d.csv"
    )

    # Windows-safe filename.
    filename = (
        filename
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    path = (
        FORECAST_DIR
        / filename
    )

    forecast_df.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Forecast CSV written to: %s",
        path,
    )

    return path


# ============================================================
# REPORT
# ============================================================

def write_forecast_report(
    forecast_df: pd.DataFrame,
    interval: float,
    market: str,
    variety: str,
    grade: str,
    horizon: int,
    csv_path: Path,
    plot_path: Path,
) -> Path:
    """Write Phase 6 forecast report."""

    first_price = float(
        forecast_df.iloc[0][
            "predicted_price_rs_per_quintal"
        ]
    )

    last_price = float(
        forecast_df.iloc[-1][
            "predicted_price_rs_per_quintal"
        ]
    )

    minimum = float(
        forecast_df[
            "predicted_price_rs_per_quintal"
        ].min()
    )

    maximum = float(
        forecast_df[
            "predicted_price_rs_per_quintal"
        ].max()
    )

    report_path = (
        REPORT_DIR
        / (
            f"forecast_report_"
            f"{market}_"
            f"{variety}_"
            f"{grade}_"
            f"{horizon}d.md"
        )
    )

    report_path = Path(
        str(report_path)
        .replace("/", "_")
        .replace("\\", "_")
    )

    content = f"""# Phase 6 — Future Price Forecast Report

## Forecast selection

- Market: `{market}`
- Variety: `{variety}`
- Grade: `{grade}`
- Horizon: `{horizon} days`
- Model: `{MODEL_NAME}`

## Forecast summary

- First forecast price: ₹{first_price:,.2f}/quintal
- Last forecast price: ₹{last_price:,.2f}/quintal
- Minimum predicted price: ₹{minimum:,.2f}/quintal
- Maximum predicted price: ₹{maximum:,.2f}/quintal

## Uncertainty

The displayed uncertainty band uses a historical empirical
absolute-error band calculated from the 2025 TEST-set XGBoost
predictions.

Approximate absolute-error band:

**±₹{interval:,.2f}/quintal**

This is NOT a guaranteed confidence interval.

It is also not specifically calibrated for recursive
14-day or 30-day forecasting. Recursive forecasts can accumulate
additional uncertainty.

## Future weather assumption

Future weather observations are unavailable.

The forecasting pipeline therefore uses a documented
last-observation-carried-forward strategy for weather-related
inputs.

This means:

- future weather is NOT treated as known;
- the latest available market weather observation is carried
  forward;
- weather lag features are generated consistently with this
  assumption.

## Contemporaneous input assumption

The Phase 4 model contains some contemporaneous market inputs
that are unavailable in advance.

For future inference, their latest available observations are
carried forward.

This is an inference assumption and should not be interpreted
as actual future market information.

## Recursive forecasting

Forecast day 1 is generated first.

The predicted price from day 1 is then added to the historical
state and used to construct lag and rolling features for day 2.

This continues recursively until the requested horizon is reached.

## Files

Forecast CSV:

`{csv_path}`

Forecast visualization:

`{plot_path}`

## Limitations

1. Future weather is not known.
2. Weather persistence is an assumption.
3. Recursive forecast errors may accumulate.
4. The uncertainty band is empirical rather than a formal
   probabilistic guarantee.
5. The forecast should not be interpreted as a guaranteed
   future market price.
6. Market conditions, policy, supply, demand, arrivals,
   transportation costs and other external factors may change
   after the latest historical observation.
"""

    report_path.write_text(
        content,
        encoding="utf-8",
    )

    logger.info(
        "Forecast report written to: %s",
        report_path,
    )

    return report_path


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_forecast(
    market: str,
    variety: str,
    grade: str,
    horizon: int,
) -> Dict:
    """Run the complete Phase 6 forecasting pipeline."""

    ensure_directories()

    logger.info(
        "========== PHASE 6 START =========="
    )

    df = load_dataset()

    model = load_model()

    metadata = load_metadata()

    feature_columns = discover_feature_columns(
        model,
        metadata,
        df,
    )

    logger.info(
        "Using %d model features.",
        len(feature_columns),
    )

    forecast_df, interval = forecast(
        df=df,
        model=model,
        feature_columns=feature_columns,
        market=market,
        variety=variety,
        grade=grade,
        horizon=horizon,
    )

    csv_path = export_forecast(
        forecast_df,
        market,
        variety,
        grade,
        horizon,
    )

    plot_path = create_forecast_plot(
        df,
        forecast_df,
        market,
        variety,
        grade,
    )

    report_path = write_forecast_report(
        forecast_df,
        interval,
        market,
        variety,
        grade,
        horizon,
        csv_path,
        plot_path,
    )

    logger.info(
        "========== PHASE 6 COMPLETE =========="
    )

    return {
        "forecast": forecast_df,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "report_path": report_path,
        "interval": interval,
        "feature_columns": feature_columns,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "PaddyWise AI — Phase 6 "
            "Future Price Forecasting"
        )
    )

    parser.add_argument(
        "--market",
        required=True,
        help="Market name.",
    )

    parser.add_argument(
        "--variety",
        required=True,
        help="Paddy variety.",
    )

    parser.add_argument(
        "--grade",
        required=True,
        help="Grade.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        choices=SUPPORTED_HORIZONS,
        default=7,
        help="Forecast horizon: 7, 14 or 30 days.",
    )

    return parser.parse_args()


def main():
    """CLI entry point."""

    args = parse_args()

    result = run_forecast(
        market=args.market,
        variety=args.variety,
        grade=args.grade,
        horizon=args.horizon,
    )

    forecast_df = result["forecast"]

    print()
    print("=" * 70)
    print("PHASE 6 FORECAST COMPLETE")
    print("=" * 70)

    print(
        f"Market : {args.market}"
    )

    print(
        f"Variety: {args.variety}"
    )

    print(
        f"Grade  : {args.grade}"
    )

    print(
        f"Horizon: {args.horizon} days"
    )

    print()

    print(
        forecast_df[
            [
                DATE_COL,
                "forecast_day",
                "predicted_price_rs_per_quintal",
                "lower_bound_rs_per_quintal",
                "upper_bound_rs_per_quintal",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Forecast CSV : {result['csv_path']}"
    )

    print(
        f"Forecast plot: {result['plot_path']}"
    )

    print(
        f"Report       : {result['report_path']}"
    )

    print(
        f"Empirical uncertainty band: "
        f"±Rs {result['interval']:,.2f}/quintal"
    )

    print()
    print(
        "IMPORTANT: This forecast does not know future weather "
        "and does not guarantee future prices."
    )


if __name__ == "__main__":
    main()