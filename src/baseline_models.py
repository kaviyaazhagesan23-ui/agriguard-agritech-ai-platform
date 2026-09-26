"""
Phase 4 baseline forecasting models.

Baselines:
1. Naive last-value
2. Moving average

Both baselines operate independently by market.

Important:
The baselines use only observations available before the prediction
observation.
"""

from __future__ import annotations

import pandas as pd


TARGET = "modal_price_rs_per_quintal"
MARKET = "market"
DATE = "date"


def add_naive_prediction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict today's price using the previous available price
    from the same market.
    """

    result = df.copy()

    result["naive_last_value"] = (
        result
        .groupby(MARKET)[TARGET]
        .shift(1)
    )

    return result


def add_moving_average_prediction(
    df: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    """
    Predict today's price using the mean of the previous `window`
    observations from the same market.

    shift(1) ensures the current target is excluded.
    """

    result = df.copy()

    previous_target = (
        result
        .groupby(MARKET)[TARGET]
        .shift(1)
    )

    result[f"moving_average_{window}"] = (
        previous_target
        .groupby(result[MARKET])
        .transform(
            lambda values: values.rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )

    return result


def create_baseline_predictions(
    df: pd.DataFrame,
    moving_average_window: int = 7,
) -> pd.DataFrame:

    result = (
        df.sort_values(
            [MARKET, DATE]
        )
        .reset_index(drop=True)
    )

    result = add_naive_prediction(result)

    result = add_moving_average_prediction(
        result,
        window=moving_average_window,
    )

    return result