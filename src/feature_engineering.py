"""
Phase 3 - Time-Series Feature Engineering

Creates leakage-safe features from the actual processed PaddyWise dataset.

Important:
- The source dataset is never modified.
- Lag and rolling features are calculated independently by market.
- Target-derived rolling features use shift(1), so today's target is NEVER
  included in today's lag/rolling calculation.
- No random train/test splitting is performed here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "processed" / "master_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"
REPORT_PATH = ROOT / "reports" / "feature_engineering_report.md"

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

TARGET = "modal_price_rs_per_quintal"
GROUP_COL = "market"
DATE_COL = "date"

LAG_WINDOWS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
]

REQUIRED_COLUMNS = [
    DATE_COL,
    GROUP_COL,
    TARGET,
    *WEATHER_COLUMNS,
]

CALENDAR_FEATURES = [
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_year",
    "sin_month",
    "cos_month",
]

PRICE_LAG_FEATURES = [
    f"lag_{window}"
    for window in LAG_WINDOWS
]

ROLLING_FEATURES = [
    f"rolling_mean_{window}"
    for window in ROLLING_WINDOWS
] + [
    f"rolling_std_{window}"
    for window in ROLLING_WINDOWS
]

PRICE_DYNAMIC_FEATURES = [
    "price_change_1d",
    "price_change_7d",
]

# Lagged weather is included only for technically defensible historical
# modeling. Current-day weather is retained because it exists in the
# source data, but the report explicitly warns that it should not be used
# for future forecasting unless future weather forecasts are available.
WEATHER_LAG_WINDOWS = [1, 3, 7]

WEATHER_LAG_FEATURES = [
    f"{weather}_lag_{window}"
    for weather in WEATHER_COLUMNS
    for window in WEATHER_LAG_WINDOWS
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the actual processed dataset without modifying the source file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {path}"
        )

    df = pd.read_csv(path)

    logger.info(
        "Loaded %d rows and %d columns from %s",
        len(df),
        len(df.columns),
        path,
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}"
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    if df[DATE_COL].isna().any():
        raise ValueError(
            "Dataset contains invalid dates."
        )

    return df


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_source_dataset(df: pd.DataFrame) -> None:
    """Validate assumptions required for time-series feature engineering."""

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df[DATE_COL].isna().any():
        raise ValueError("Date column contains missing values.")

    if df[GROUP_COL].isna().any():
        raise ValueError("Market column contains missing values.")

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target column '{TARGET}' contains missing values."
        )

    # The original dataset should not contain duplicate
    # market/variety/grade/date records according to Phase 1.
    key_columns = [
        "market",
        "variety",
        "grade",
        "date",
    ]

    available_key_columns = [
        column
        for column in key_columns
        if column in df.columns
    ]

    duplicates = df.duplicated(
        subset=available_key_columns
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate "
            f"{available_key_columns} records."
        )


# ---------------------------------------------------------------------
# SORTING
# ---------------------------------------------------------------------

def sort_for_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort chronologically within each market.

    The original dataframe is copied so the source dataframe is never
    modified in-place.
    """

    result = df.copy()

    result = result.sort_values(
        by=[GROUP_COL, DATE_COL],
        kind="mergesort",
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# TARGET LAGS
# ---------------------------------------------------------------------

def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe target lag features.

    Example:

        date       price
        Jan 1      2000
        Jan 3      2100
        Jan 8      2200

    lag_1 on Jan 8 = 2100.

    It represents the previous available observation in the same market,
    not necessarily the previous calendar day.
    """

    result = df.copy()

    grouped_target = result.groupby(
        GROUP_COL,
        sort=False,
    )[TARGET]

    for window in LAG_WINDOWS:
        result[f"lag_{window}"] = grouped_target.shift(window)

    return result


# ---------------------------------------------------------------------
# ROLLING FEATURES
# ---------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe rolling statistics.

    shift(1) is applied BEFORE rolling so the current target is never
    included in its own rolling calculation.
    """

    result = df.copy()

    result["_previous_target"] = (
        result.groupby(GROUP_COL)[TARGET]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:

        result[f"rolling_mean_{window}"] = (
            result.groupby(GROUP_COL)["_previous_target"]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

        result[f"rolling_std_{window}"] = (
            result.groupby(GROUP_COL)["_previous_target"]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=2,
                ).std()
            )
        )

    result.drop(
        columns=["_previous_target"],
        inplace=True,
    )

    return result
# ---------------------------------------------------------------------
# CALENDAR FEATURES
# ---------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar/time features from the prediction date."""

    result = df.copy()

    result["year"] = result[DATE_COL].dt.year
    result["month"] = result[DATE_COL].dt.month
    result["quarter"] = result[DATE_COL].dt.quarter
    result["week_of_year"] = (
        result[DATE_COL].dt.isocalendar().week.astype(int)
    )
    result["day_of_year"] = result[DATE_COL].dt.dayofyear

    # Month is cyclical:
    #
    # December (12) should be close to January (1).
    #
    # A normal integer month feature does not represent that relationship.
    result["sin_month"] = np.sin(
        2 * np.pi * result["month"] / 12
    )

    result["cos_month"] = np.cos(
        2 * np.pi * result["month"] / 12
    )

    return result


# ---------------------------------------------------------------------
# PRICE DYNAMICS
# ---------------------------------------------------------------------

def add_price_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create historical price-change features.

    These use only historical observations.
    """

    result = df.copy()

    grouped_target = result.groupby(
        GROUP_COL,
        sort=False,
    )[TARGET]

    result["price_change_1d"] = (
        result[TARGET]
        - grouped_target.shift(1)
    )

    result["price_change_7d"] = (
        result[TARGET]
        - grouped_target.shift(7)
    )

    return result


# ---------------------------------------------------------------------
# WEATHER FEATURES
# ---------------------------------------------------------------------

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve the available weather observations.

    These are contemporaneous observed weather variables from the
    original dataset.

    IMPORTANT:
    They are not automatically safe for future forecasting because
    future actual weather is unknown.

    Lagged weather variables are added separately.
    """

    result = df.copy()

    for column in WEATHER_COLUMNS:
        if column not in result.columns:
            raise ValueError(
                f"Weather column missing: {column}"
            )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def add_lagged_weather_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical weather observations.

    Weather lags are grouped by market so that weather information
    comes only from previous observations of the same market.

    These features are technically justified when the model is intended
    to predict today's price using weather that was already observed
    previously.

    They also avoid requiring future weather observations.
    """

    result = df.copy()

    for weather_column in WEATHER_COLUMNS:

        grouped_weather = result.groupby(
            GROUP_COL,
            sort=False,
        )[weather_column]

        for window in WEATHER_LAG_WINDOWS:

            result[
                f"{weather_column}_lag_{window}"
            ] = grouped_weather.shift(window)

    return result


# ---------------------------------------------------------------------
# FEATURE VALIDATION
# ---------------------------------------------------------------------

def validate_feature_columns(
    df: pd.DataFrame,
) -> dict:
    """Validate that the expected feature columns exist."""

    expected = (
        PRICE_LAG_FEATURES
        + ROLLING_FEATURES
        + CALENDAR_FEATURES
        + PRICE_DYNAMIC_FEATURES
        + WEATHER_COLUMNS
        + WEATHER_LAG_FEATURES
    )

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    unexpected_duplicates = (
        df.columns[df.columns.duplicated()]
        .tolist()
    )

    if missing:
        raise ValueError(
            f"Expected feature columns missing: {missing}"
        )

    if unexpected_duplicates:
        raise ValueError(
            "Duplicate column names detected: "
            f"{unexpected_duplicates}"
        )

    return {
        "expected_feature_count": len(expected),
        "missing_features": missing,
        "duplicate_columns": unexpected_duplicates,
    }


# ---------------------------------------------------------------------
# LEAKAGE VALIDATION
# ---------------------------------------------------------------------

def validate_no_current_target_in_rolling(
    df: pd.DataFrame,
) -> None:
    """
    Independently verify that rolling features equal rolling statistics
    of historical observations only.

    This is a direct leakage check.
    """

    ordered = df.sort_values(
        [GROUP_COL, DATE_COL]
    ).reset_index(drop=True)

    for window in ROLLING_WINDOWS:

        expected = (
            ordered.groupby(GROUP_COL)[TARGET]
            .shift(1)
            .groupby(ordered[GROUP_COL])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        actual = ordered[
            f"rolling_mean_{window}"
        ]

        comparison = pd.concat(
            [
                expected.rename("expected"),
                actual.rename("actual"),
            ],
            axis=1,
        )

        valid = (
            comparison["expected"].isna()
            & comparison["actual"].isna()
        ) | np.isclose(
            comparison["expected"],
            comparison["actual"],
            equal_nan=True,
        )

        if not valid.all():
            raise AssertionError(
                f"Leakage validation failed for "
                f"rolling_mean_{window}"
            )


# ---------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# ---------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    """

    validate_source_dataset(df)

    result = sort_for_time_series(df)

    result = add_price_lags(result)

    result = add_rolling_features(result)

    result = add_calendar_features(result)

    result = add_price_dynamics(result)

    result = add_weather_features(result)

    result = add_lagged_weather_features(result)

    validate_feature_columns(result)

    validate_no_current_target_in_rolling(result)

    return result


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def _feature_status(
    df: pd.DataFrame,
    features: list[str],
) -> list[str]:

    lines = []

    for feature in features:

        if feature not in df.columns:
            continue

        missing_count = int(
            df[feature].isna().sum()
        )

        lines.append(
            f"| `{feature}` | "
            f"{missing_count} | "
            f"{missing_count / len(df) * 100:.2f}% |"
        )

    return lines


def build_report(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> str:
    """Generate the Phase 3 feature documentation report."""

    all_engineered = (
        PRICE_LAG_FEATURES
        + ROLLING_FEATURES
        + CALENDAR_FEATURES
        + PRICE_DYNAMIC_FEATURES
        + WEATHER_COLUMNS
        + WEATHER_LAG_FEATURES
    )

    lines = [
        "# Phase 3 — Feature Engineering Report",
        "",
        "## Dataset",
        "",
        f"- Source rows: **{len(source_df):,}**",
        f"- Feature rows: **{len(feature_df):,}**",
        f"- Source columns: **{len(source_df.columns)}**",
        f"- Feature dataset columns: **{len(feature_df.columns)}**",
        f"- Date range: **{source_df[DATE_COL].min().date()} "
        f"to {source_df[DATE_COL].max().date()}**",
        f"- Markets: **{source_df[GROUP_COL].nunique()}**",
        "",
        "## Target",
        "",
        f"- Target: `{TARGET}`",
        "",
        "## Grouping strategy",
        "",
        "Lag and rolling price features are calculated independently "
        "within each `market`.",
        "",
        "The source dataset contains multiple varieties and grades "
        "within markets. Because Phase 3 explicitly specifies "
        "per-market features, the implementation does not silently "
        "change the grouping to market + variety + grade.",
        "",
        "Therefore `lag_1` means the previous available observation "
        "within the same market, not necessarily the previous calendar day.",
        "",
        "## Leakage prevention",
        "",
        "- Data is sorted chronologically within each market.",
        "- Target lags use `groupby('market').shift(...)`.",
        "- Rolling statistics are calculated after `shift(1)`.",
        "- The current target is therefore excluded from its own rolling window.",
        "- No future observations are used to calculate historical features.",
        "- No random train/test split is performed.",
        "- Insufficient history remains `NaN` rather than being fabricated.",
        "",
        "## Price lag features",
        "",
    ]

    lines.extend(
        _feature_status(
            feature_df,
            PRICE_LAG_FEATURES,
        )
    )

    lines.extend(
        [
            "",
            "## Rolling features",
            "",
        ]
    )

    lines.extend(
        _feature_status(
            feature_df,
            ROLLING_FEATURES,
        )
    )

    lines.extend(
        [
            "",
            "## Calendar features",
            "",
        ]
    )

    lines.extend(
        _feature_status(
            feature_df,
            CALENDAR_FEATURES,
        )
    )

    lines.extend(
        [
            "",
            "## Price dynamics",
            "",
        ]
    )

    lines.extend(
        _feature_status(
            feature_df,
            PRICE_DYNAMIC_FEATURES,
        )
    )

    lines.extend(
        [
            "",
            "## Weather features",
            "",
            "The original dataset contains same-date weather observations.",
            "",
            "These are preserved, but they require care during future "
            "forecasting because actual future weather is not known.",
            "",
            "Lagged weather features were therefore added using only "
            "previous observations.",
            "",
        ]
    )

    lines.extend(
        _feature_status(
            feature_df,
            WEATHER_COLUMNS,
        )
    )

    lines.extend(
        [
            "",
            "## Lagged weather features",
            "",
        ]
    )

    lines.extend(
        _feature_status(
            feature_df,
            WEATHER_LAG_FEATURES,
        )
    )

    lines.extend(
        [
            "",
            "## Insufficient historical observations",
            "",
            "Early observations in each market naturally have missing "
            "lag/rolling values because sufficient historical information "
            "does not exist.",
            "",
            "These values are intentionally retained as `NaN` rather than "
            "filled using future information.",
            "",
            "Model-specific preprocessing in a later phase will decide "
            "whether rows/features with insufficient history are dropped "
            "or handled using a training-only imputation strategy.",
            "",
            "## Important modeling limitation",
            "",
            "The feature dataset is prepared for time-series modeling. "
            "It does not perform model training, hyperparameter tuning, "
            "random splitting, or final evaluation.",
            "",
            "Phase 4 is intentionally not included.",
            "",
            "## Feature inventory",
            "",
        ]
    )

    lines.extend(
        f"- `{feature}`"
        for feature in all_engineered
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

def save_feature_dataset(
    df: pd.DataFrame,
    path: Path = OUTPUT_PATH,
) -> None:
    """Save the new feature dataset."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Feature dataset written to %s",
        path,
    )


def save_report(
    report: str,
    path: Path = REPORT_PATH,
) -> None:
    """Save feature engineering documentation."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Feature engineering report written to %s",
        path,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def run_feature_engineering() -> pd.DataFrame:

    logger.info("Starting Phase 3 feature engineering.")

    source_df = load_dataset()

    feature_df = build_features(
        source_df
    )

    save_feature_dataset(
        feature_df
    )

    report = build_report(
        source_df,
        feature_df,
    )

    save_report(
        report
    )

    logger.info(
        "Phase 3 complete: %d rows, %d columns",
        len(feature_df),
        len(feature_df.columns),
    )

    return feature_df


if __name__ == "__main__":
    run_feature_engineering()