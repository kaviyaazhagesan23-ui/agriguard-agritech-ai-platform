from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.current_price_estimator import EXPECTED_FEATURES, forecast_future_prices

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "current" / "synthetic_current_prices.csv"


def _supported_combinations() -> list[tuple[str, str, str]]:
    df = pd.read_csv(PRICE_PATH)
    return list(df[["market", "variety", "grade"]].drop_duplicates().itertuples(index=False, name=None))


def test_recursive_forecast_returns_requested_number_of_days_and_consecutive_dates():
    result = forecast_future_prices("Vallam", "B P T", "Local", "2026-09-24", 7)
    assert len(result) == 7
    assert result["date"].tolist() == list(pd.date_range("2026-09-24", periods=7, freq="D").date)
    assert result.iloc[0]["date"] == date(2026, 9, 24)


def test_recursive_forecast_predictions_are_numeric_positive_and_model_features_are_exact():
    result = forecast_future_prices("Vallam", "B P T", "Local", "2026-09-24", 7)
    values = pd.to_numeric(result["predicted_price_rs_per_quintal"], errors="coerce")
    assert np.isfinite(values.to_numpy()).all()
    assert (values > 0).all()
    assert result.attrs["model_features"] == EXPECTED_FEATURES


def test_recursive_forecast_uses_previous_predictions_for_future_lags():
    result = forecast_future_prices("Vallam", "B P T", "Local", "2026-09-24", 3)
    feature_rows = result.attrs["feature_rows"]
    first_prediction = float(result.iloc[0]["predicted_price_rs_per_quintal"])
    second_prediction = float(result.iloc[1]["predicted_price_rs_per_quintal"])
    assert feature_rows[1]["lag_1"] == pytest.approx(first_prediction)
    assert feature_rows[2]["lag_1"] == pytest.approx(second_prediction)
    assert result.attrs["recursive"] is True


def test_every_supported_current_combination_can_forecast():
    for market, variety, grade in _supported_combinations():
        result = forecast_future_prices(market, variety, grade, "2026-09-24", 2)
        assert len(result) == 2, (market, variety, grade)
        assert (result["predicted_price_rs_per_quintal"] > 0).all(), (market, variety, grade)


def test_forecast_does_not_retrain_xgboost(monkeypatch):
    import xgboost

    def fail_fit(*args, **kwargs):
        raise AssertionError("forecasting must not retrain the existing XGBoost model")

    monkeypatch.setattr(xgboost.XGBRegressor, "fit", fail_fit)
    result = forecast_future_prices("Vallam", "B P T", "Local", "2026-09-24", 2)
    assert len(result) == 2
