from pathlib import Path

import pandas as pd
import numpy as np

from src.phase5_evaluation import (
    calculate_error_metrics,
    identify_prediction_columns,
    determine_final_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE4_DIR = PROJECT_ROOT / "reports" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "reports" / "phase5"


def test_phase4_predictions_exist():
    path = PHASE4_DIR / "predictions.csv"

    assert path.exists(), (
        "Phase 4 predictions.csv does not exist."
    )


def test_phase4_metrics_exist():
    assert (
        PHASE4_DIR / "metrics.json"
    ).exists()


def test_phase4_model_comparison_exists():
    assert (
        PHASE4_DIR / "model_comparison.csv"
    ).exists()


def test_prediction_file_is_not_empty():

    path = PHASE4_DIR / "predictions.csv"

    df = pd.read_csv(path)

    assert len(df) > 0


def test_prediction_columns_can_be_identified():

    path = PHASE4_DIR / "predictions.csv"

    df = pd.read_csv(path)

    columns = identify_prediction_columns(df)

    assert columns["actual"] is not None


def test_metrics_are_correct():

    actual = np.array(
        [100, 200, 300, 400],
        dtype=float,
    )

    predicted = np.array(
        [110, 190, 310, 380],
        dtype=float,
    )

    metrics = calculate_error_metrics(
        actual,
        predicted,
    )

    assert metrics["n"] == 4

    assert metrics["mae"] >= 0

    assert metrics["rmse"] >= 0

    assert metrics["mape"] >= 0


def test_metrics_handle_zero_actual_values():

    actual = np.array(
        [0, 100, 200],
        dtype=float,
    )

    predicted = np.array(
        [10, 110, 190],
        dtype=float,
    )

    metrics = calculate_error_metrics(
        actual,
        predicted,
    )

    assert metrics["n"] == 3

    assert np.isfinite(metrics["mae"])

    assert np.isfinite(metrics["rmse"])


def test_model_selection_uses_actual_metrics():

    comparison = pd.DataFrame(
        {
            "model": [
                "Naive Last Value",
                "Random Forest",
                "XGBoost",
            ],
            "n": [100, 100, 100],
            "mae": [100.0, 80.0, 90.0],
            "rmse": [130.0, 110.0, 120.0],
            "mape": [5.0, 4.0, 4.5],
            "r2": [0.50, 0.70, 0.65],
        }
    )

    selected, valid = determine_final_model(
        comparison
    )

    assert selected == "Random Forest"


def test_phase5_directory_can_be_created():

    PHASE5_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert PHASE5_DIR.exists()