"""
Phase 4 - Baseline Models + Machine Learning

Models:
    1. Naive last-value
    2. Moving average
    3. Random Forest
    4. XGBoost

Time split:
    Train      2017-2023
    Validation 2024
    Test       2025

No random train/test split is used.

Important:
- The actual Phase 3 feature dataset is used.
- Features containing the current target are excluded.
- Lag/rolling features were created using historical observations only.
- Reproducible random seeds are used.
- Models are saved only after fitting on the training data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from src.baseline_models import (
    create_baseline_predictions,
)
from src.evaluation import (
    calculate_metrics,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "feature_dataset.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase4"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase4"
)

METRICS_JSON = REPORT_DIR / "metrics.json"
METRICS_CSV = REPORT_DIR / "metrics.csv"
COMPARISON_CSV = REPORT_DIR / "model_comparison.csv"
PREDICTIONS_CSV = REPORT_DIR / "predictions.csv"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

TARGET = "modal_price_rs_per_quintal"
DATE = "date"
MARKET = "market"

RANDOM_SEED = 42

TRAIN_START = 2017
TRAIN_END = 2023

VALIDATION_YEAR = 2024
TEST_YEAR = 2025


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# FEATURE POLICY
# ---------------------------------------------------------------------

# These features contain the current target directly or indirectly.
#
# price_change_1d:
#     current_price - previous_price
#
# price_change_7d:
#     current_price - price_7_observations_ago
#
# Both therefore leak today's target into the model.

LEAKAGE_FEATURES = {
    "price_change_1d",
    "price_change_7d",
}


# Current-day observed weather cannot automatically be known when
# making a future forecast.
CURRENT_WEATHER_FEATURES = {
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
}


NON_MODEL_COLUMNS = {
    TARGET,
    DATE,
    MARKET,
    "record_id",
    *LEAKAGE_FEATURES,
    *CURRENT_WEATHER_FEATURES,
}


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_feature_dataset() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_PATH}\n\n"
            "Run Phase 3 first."
        )

    df = pd.read_csv(DATA_PATH)

    if DATE not in df.columns:
        raise ValueError(
            f"Missing date column: {DATE}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    if MARKET not in df.columns:
        raise ValueError(
            f"Missing market column: {MARKET}"
        )

    df[DATE] = pd.to_datetime(
        df[DATE],
        errors="coerce",
    )

    if df[DATE].isna().any():
        raise ValueError(
            "Invalid dates detected."
        )

    df = df.sort_values(
        [MARKET, DATE]
    ).reset_index(drop=True)

    logger.info(
        "Loaded %d rows and %d columns",
        len(df),
        len(df.columns),
    )

    logger.info(
        "Date range: %s -> %s",
        df[DATE].min().date(),
        df[DATE].max().date(),
    )

    return df


# ---------------------------------------------------------------------
# TIME SPLIT
# ---------------------------------------------------------------------

def create_time_split(
    df: pd.DataFrame,
):
    """
    Chronological split.

    TRAIN:
        2017-01-01 through 2023-12-31

    VALIDATION:
        2024-01-01 through 2024-12-31

    TEST:
        2025-01-01 through 2025-12-31
    """

    train = df[
        df[DATE].dt.year.between(
            TRAIN_START,
            TRAIN_END,
        )
    ].copy()

    validation = df[
        df[DATE].dt.year == VALIDATION_YEAR
    ].copy()

    test = df[
        df[DATE].dt.year == TEST_YEAR
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if validation.empty:
        raise ValueError(
            "Validation split for 2024 is empty."
        )

    if test.empty:
        raise ValueError(
            "Test split for 2025 is empty."
        )

    logger.info(
        "TRAIN: %d rows | %s -> %s",
        len(train),
        train[DATE].min().date(),
        train[DATE].max().date(),
    )

    logger.info(
        "VALIDATION: %d rows | %s -> %s",
        len(validation),
        validation[DATE].min().date(),
        validation[DATE].max().date(),
    )

    logger.info(
        "TEST: %d rows | %s -> %s",
        len(test),
        test[DATE].min().date(),
        test[DATE].max().date(),
    )

    return train, validation, test


# ---------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------

def get_model_features(
    df: pd.DataFrame,
) -> list[str]:

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    features = [
        column
        for column in numeric_columns
        if column not in NON_MODEL_COLUMNS
    ]

    if not features:
        raise ValueError(
            "No usable model features were found."
        )

    # Explicitly reject known leakage features.
    leakage_present = (
        set(features)
        & LEAKAGE_FEATURES
    )

    if leakage_present:
        raise AssertionError(
            "Leakage features detected in model inputs: "
            f"{sorted(leakage_present)}"
        )

    # Current weather must not enter a future forecasting model.
    current_weather_present = (
        set(features)
        & CURRENT_WEATHER_FEATURES
    )

    if current_weather_present:
        raise AssertionError(
            "Current-day weather features detected: "
            f"{sorted(current_weather_present)}"
        )

    logger.info(
        "Using %d model features",
        len(features),
    )

    return features


# ---------------------------------------------------------------------
# PREPARE ML MATRICES
# ---------------------------------------------------------------------

def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    X = df[feature_columns].copy()

    y = df[TARGET].copy()

    # Tree models cannot reliably operate with arbitrary object values.
    # Everything selected above is numeric.

    return X, y


# ---------------------------------------------------------------------
# IMPUTATION
# ---------------------------------------------------------------------

def fill_missing_features(
    train_X: pd.DataFrame,
    validation_X: pd.DataFrame,
    test_X: pd.DataFrame,
):
    """
    Training-only median imputation.

    Median values are learned ONLY from the training set.
    """

    medians = train_X.median()

    train_filled = train_X.fillna(
        medians
    )

    validation_filled = validation_X.fillna(
        medians
    )

    test_filled = test_X.fillna(
        medians
    )

    return (
        train_filled,
        validation_filled,
        test_filled,
        medians,
    )


# ---------------------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------------------

def train_random_forest(
    X_train,
    y_train,
):

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------------------

def train_xgboost(
    X_train,
    y_train,
):

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(
    name: str,
    split_name: str,
    y_true,
    y_pred,
) -> dict:

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    return {
        "model": name,
        "split": split_name,
        **metrics,
    }


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def run_phase4():

    logger.info(
        "========== PHASE 4 START =========="
    )

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    df = load_feature_dataset()

    # -------------------------------------------------------------
    # Verify expected historical range
    # -------------------------------------------------------------

    actual_min_year = df[DATE].dt.year.min()
    actual_max_year = df[DATE].dt.year.max()

    logger.info(
        "Actual dataset year range: %d -> %d",
        actual_min_year,
        actual_max_year,
    )

    # -------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------

    baseline_df = create_baseline_predictions(
        df
    )

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    train, validation, test = create_time_split(
        baseline_df
    )

    # -------------------------------------------------------------
    # Baseline evaluation
    # -------------------------------------------------------------

    results = []

    prediction_records = []

    baseline_models = {
        "Naive Last Value": "naive_last_value",
        "Moving Average 7": "moving_average_7",
    }

    for model_name, prediction_column in (
        baseline_models.items()
    ):

        for split_name, split_df in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:

            metrics = evaluate_model(
                model_name,
                split_name,
                split_df[TARGET],
                split_df[prediction_column],
            )

            results.append(metrics)

    # -------------------------------------------------------------
    # Machine-learning features
    # -------------------------------------------------------------

    feature_columns = get_model_features(
        df
    )

    X_train, y_train = prepare_xy(
        train,
        feature_columns,
    )

    X_validation, y_validation = prepare_xy(
        validation,
        feature_columns,
    )

    X_test, y_test = prepare_xy(
        test,
        feature_columns,
    )

    # -------------------------------------------------------------
    # Training-only imputation
    # -------------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        training_medians,
    ) = fill_missing_features(
        X_train,
        X_validation,
        X_test,
    )

    # -------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------

    logger.info(
        "Training Random Forest..."
    )

    rf_model = train_random_forest(
        X_train,
        y_train,
    )

    rf_validation_pred = (
        rf_model.predict(
            X_validation
        )
    )

    rf_test_pred = (
        rf_model.predict(
            X_test
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            "validation",
            y_validation,
            rf_validation_pred,
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            "test",
            y_test,
            rf_test_pred,
        )
    )

    # -------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------

    logger.info(
        "Training XGBoost..."
    )

    xgb_model = train_xgboost(
        X_train,
        y_train,
    )

    xgb_validation_pred = (
        xgb_model.predict(
            X_validation
        )
    )

    xgb_test_pred = (
        xgb_model.predict(
            X_test
        )
    )

    results.append(
        evaluate_model(
            "XGBoost",
            "validation",
            y_validation,
            xgb_validation_pred,
        )
    )

    results.append(
        evaluate_model(
            "XGBoost",
            "test",
            y_test,
            xgb_test_pred,
        )
    )

    # -------------------------------------------------------------
    # Save model artifacts
    # -------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        rf_model,
        MODEL_DIR / "random_forest.joblib",
    )

    joblib.dump(
        xgb_model,
        MODEL_DIR / "xgboost.joblib",
    )

    joblib.dump(
        {
            "feature_columns": feature_columns,
            "training_medians": training_medians,
            "target": TARGET,
            "market": MARKET,
            "date": DATE,
            "random_seed": RANDOM_SEED,
        },
        MODEL_DIR / "preprocessing_metadata.joblib",
    )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    prediction_rows = []

    for split_name, split_df in [
        ("validation", validation),
        ("test", test),
    ]:

        for _, row in split_df.iterrows():

            date = row[DATE]
            market = row[MARKET]
            actual = row[TARGET]

            if pd.notna(
                row["naive_last_value"]
            ):
                prediction_rows.append(
                    {
                        "date": date,
                        "market": market,
                        "split": split_name,
                        "model": "Naive Last Value",
                        "actual": actual,
                        "prediction": row[
                            "naive_last_value"
                        ],
                    }
                )

            if pd.notna(
                row["moving_average_7"]
            ):
                prediction_rows.append(
                    {
                        "date": date,
                        "market": market,
                        "split": split_name,
                        "model": "Moving Average 7",
                        "actual": actual,
                        "prediction": row[
                            "moving_average_7"
                        ],
                    }
                )

    for split_name, split_df, predictions in [
        (
            "validation",
            validation,
            rf_validation_pred,
        ),
        (
            "test",
            test,
            rf_test_pred,
        ),
    ]:

        for index, (_, row) in enumerate(
            split_df.iterrows()
        ):

            prediction_rows.append(
                {
                    "date": row[DATE],
                    "market": row[MARKET],
                    "split": split_name,
                    "model": "Random Forest",
                    "actual": row[TARGET],
                    "prediction": predictions[index],
                }
            )

    for split_name, split_df, predictions in [
        (
            "validation",
            validation,
            xgb_validation_pred,
        ),
        (
            "test",
            test,
            xgb_test_pred,
        ),
    ]:

        for index, (_, row) in enumerate(
            split_df.iterrows()
        ):

            prediction_rows.append(
                {
                    "date": row[DATE],
                    "market": row[MARKET],
                    "split": split_name,
                    "model": "XGBoost",
                    "actual": row[TARGET],
                    "prediction": predictions[index],
                }
            )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    # -------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame(
        results
    )

    metrics_df.to_csv(
        METRICS_CSV,
        index=False,
    )

    metrics_json = {
        "phase": 4,
        "target": TARGET,
        "random_seed": RANDOM_SEED,
        "train_period": "2017-2023",
        "validation_period": "2024",
        "test_period": "2025",
        "results": results,
    }

    METRICS_JSON.write_text(
        json.dumps(
            metrics_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Test-set comparison table.
    comparison = (
        metrics_df[
            metrics_df["split"] == "test"
        ]
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    comparison.to_csv(
        COMPARISON_CSV,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    # -------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 MODEL RESULTS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print("\nTest-set comparison:")
    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(MODEL_DIR)
    print(METRICS_JSON)
    print(METRICS_CSV)
    print(COMPARISON_CSV)
    print(PREDICTIONS_CSV)

    logger.info(
        "========== PHASE 4 COMPLETE =========="
    )

    return metrics_df


if __name__ == "__main__":
    run_phase4()