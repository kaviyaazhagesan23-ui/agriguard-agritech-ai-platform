from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
    CALENDAR_FEATURES,
    INPUT_PATH,
    PRICE_DYNAMIC_FEATURES,
    PRICE_LAG_FEATURES,
    ROLLING_FEATURES,
    WEATHER_COLUMNS,
    WEATHER_LAG_FEATURES,
    build_features,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "master_dataset.csv"


def test_actual_dataset_is_used():
    """Phase 3 must use the actual processed dataset."""

    assert DATA.exists()

    df = load_dataset(DATA)

    assert len(df) == 3024
    assert df["modal_price_rs_per_quintal"].notna().all()


def test_row_count_is_preserved():

    source = load_dataset(DATA)
    features = build_features(source)

    assert len(features) == len(source)


def test_original_dataset_is_not_modified():

    source = load_dataset(DATA)

    original_columns = source.columns.tolist()
    original_shape = source.shape

    _ = build_features(source)

    assert source.columns.tolist() == original_columns
    assert source.shape == original_shape


def test_required_price_lags_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in PRICE_LAG_FEATURES:
        assert feature in df.columns


def test_required_rolling_features_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in ROLLING_FEATURES:
        assert feature in df.columns


def test_calendar_features_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in CALENDAR_FEATURES:
        assert feature in df.columns


def test_price_dynamic_features_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in PRICE_DYNAMIC_FEATURES:
        assert feature in df.columns


def test_weather_features_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in WEATHER_COLUMNS:
        assert feature in df.columns


def test_lagged_weather_features_exist():

    df = build_features(
        load_dataset(DATA)
    )

    for feature in WEATHER_LAG_FEATURES:
        assert feature in df.columns


def test_first_observation_has_no_previous_price():

    df = build_features(
        load_dataset(DATA)
    )

    first_rows = (
        df.sort_values(
            ["market", "date"]
        )
        .groupby("market")
        .head(1)
    )

    assert first_rows["lag_1"].isna().all()


def test_lag_1_uses_previous_market_observation():

    df = build_features(
        load_dataset(DATA)
    )

    df = df.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    for market, group in df.groupby("market"):

        if len(group) < 2:
            continue

        second_row = group.iloc[1]

        first_row = group.iloc[0]

        assert second_row["lag_1"] == (
            first_row["modal_price_rs_per_quintal"]
        )


def test_rolling_mean_does_not_use_current_target():

    df = build_features(
        load_dataset(DATA)
    )

    df = df.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    for market, group in df.groupby("market"):

        if len(group) < 2:
            continue

        # Find a row with enough history.
        candidates = group.iloc[1:]

        for idx in candidates.index[:10]:

            row = df.loc[idx]

            previous_rows = (
                group.loc[:idx - 1]
                .tail(7)
            )

            expected = (
                previous_rows[
                    "modal_price_rs_per_quintal"
                ]
                .mean()
            )

            actual = row["rolling_mean_7"]

            if pd.notna(actual):

                assert np.isclose(
                    actual,
                    expected,
                )


def test_rolling_features_are_calculated_from_previous_data_only():

    df = build_features(
        load_dataset(DATA)
    )

    df = df.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    for market, group in df.groupby("market"):

        if len(group) < 2:
            continue

        row_index = group.index[1]

        current_price = df.loc[
            row_index,
            "modal_price_rs_per_quintal",
        ]

        rolling_value = df.loc[
            row_index,
            "rolling_mean_7",
        ]

        previous_price = df.loc[
            group.index[0],
            "modal_price_rs_per_quintal",
        ]

        assert rolling_value == previous_price

        # This also makes the intention explicit:
        # current target must not become the rolling value.
        assert rolling_value != current_price or (
            current_price == previous_price
        )


def test_month_cyclical_features_are_bounded():

    df = build_features(
        load_dataset(DATA)
    )

    assert df["sin_month"].between(
        -1,
        1,
    ).all()

    assert df["cos_month"].between(
        -1,
        1,
    ).all()


def test_weather_lags_are_previous_observations():

    df = build_features(
        load_dataset(DATA)
    )

    df = df.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    market = df["market"].iloc[0]

    group = df[
        df["market"] == market
    ]

    if len(group) >= 2:

        first = group.iloc[0]
        second = group.iloc[1]

        assert second[
            "temperature_mean_lag_1"
        ] == first["temperature_mean"]


def test_source_file_still_has_original_columns():

    source = pd.read_csv(DATA)

    assert len(source) == 3024

    assert "lag_1" not in source.columns
    assert "rolling_mean_7" not in source.columns
    assert "sin_month" not in source.columns