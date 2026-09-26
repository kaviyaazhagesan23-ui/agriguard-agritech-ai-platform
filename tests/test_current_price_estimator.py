from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.current_price_estimator import EXPECTED_FEATURES, estimate_price

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "current" / "synthetic_current_prices.csv"
WEATHER_PATH = ROOT / "data" / "current" / "synthetic_current_weather.csv"
MODEL_PATH = ROOT / "models" / "phase4" / "xgboost.joblib"
META_PATH = ROOT / "models" / "phase4" / "preprocessing_metadata.joblib"


def test_synthetic_price_columns_and_order():
    df = pd.read_csv(PRICE_PATH)
    assert list(df.columns) == [
        "date", "market", "commodity", "variety", "grade",
        "min_price_rs_per_quintal", "max_price_rs_per_quintal",
        "modal_price_rs_per_quintal",
    ]
    assert (df["min_price_rs_per_quintal"] <= df["modal_price_rs_per_quintal"]).all()
    assert (df["modal_price_rs_per_quintal"] <= df["max_price_rs_per_quintal"]).all()
    assert pd.to_datetime(df.date).min().date() == date(2026, 7, 26)
    assert pd.to_datetime(df.date).max().date() == date(2026, 11, 30)
    assert df.market.nunique() == 7
    assert len(df) == 4224


def test_model_and_metadata_have_exact_41_features():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)
    assert list(metadata["feature_columns"]) == EXPECTED_FEATURES
    assert list(model.feature_names_in_) == EXPECTED_FEATURES
    assert model.n_features_in_ == 41


def test_valid_estimate_has_exact_feature_order_and_finite_positive_prediction():
    result = estimate_price("Vallam", "B P T", "Local", "2026-09-23")
    assert result["feature_columns"] == EXPECTED_FEATURES
    assert list(result["feature_row"]) == EXPECTED_FEATURES
    values = np.asarray(list(result["feature_row"].values()), dtype=float)
    assert np.isfinite(values).all()
    assert result["estimated_price_rs_per_quintal"] > 0


def test_prediction_is_within_reasonable_recent_synthetic_range():
    result = estimate_price("Vallam", "B P T", "Local", "2026-09-23")
    prediction = result["estimated_price_rs_per_quintal"]
    lo, hi = result["recent_synthetic_price_range"]
    assert lo * 0.50 <= prediction <= hi * 1.50


def test_unknown_market_errors():
    with pytest.raises(ValueError, match="Unknown current synthetic market/variety/grade"):
        estimate_price("UnknownMarket", "B P T", "FAQ", "2026-09-23")


def test_unknown_variety_errors():
    with pytest.raises(ValueError, match="Unknown current synthetic market/variety/grade"):
        estimate_price("Vallam", "UnknownVariety", "FAQ", "2026-09-23")


def test_unknown_grade_errors():
    with pytest.raises(ValueError, match="Unknown current synthetic market/variety/grade"):
        estimate_price("Vallam", "B P T", "UnknownGrade", "2026-09-23")


def test_weather_file_has_required_pipeline_fields():
    df = pd.read_csv(WEATHER_PATH)
    assert list(df.columns) == [
        "date", "market", "temperature_mean", "temperature_max", "temperature_min",
        "rainfall", "wind_speed_max", "et0",
    ]
    assert len(df) == 896
    assert df.market.nunique() == 7
    assert np.isfinite(df.iloc[:, 2:].to_numpy(dtype=float)).all()
