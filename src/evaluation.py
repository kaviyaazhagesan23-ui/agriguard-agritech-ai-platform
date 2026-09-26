"""
Phase 4 model evaluation.

Metrics:
- MAE
- RMSE
- MAPE
- R2

MAPE ignores observations where the actual target is zero because
percentage error is undefined there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "MAE": None,
            "RMSE": None,
            "MAPE": None,
            "R2": None,
            "n": 0,
        }

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    non_zero = y_true != 0

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[non_zero]
                        - y_pred[non_zero]
                    )
                    / y_true[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    # R² is not meaningful with fewer than 2 observations.
    if len(y_true) >= 2:
        r2 = r2_score(
            y_true,
            y_pred,
        )
    else:
        r2 = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "n": int(len(y_true)),
    }


def evaluate_prediction_frame(
    df: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> dict:

    return calculate_metrics(
        df[target_column],
        df[prediction_column],
    )