from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_models import (
    create_baseline_predictions,
)
from src.evaluation import (
    calculate_metrics,
)
from src.train_models import (
    create_time_split,
    get_model_features,
    load_feature_dataset,
)


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "processed"
    / "feature_dataset.csv"
)


def test_actual_feature_dataset_exists():

    assert DATA.exists()

    df = load_feature_dataset()

    assert len(df) == 3024


def test_date_range_supports_requested_split():

    df = load_feature_dataset()

    assert df["date"].dt.year.min() <= 2017
    assert df["date"].dt.year.max() >= 2025


def test_time_split_is_chronological():

    df = load_feature_dataset()

    train, validation, test = create_time_split(df)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_time_split_years():

    df = load_feature_dataset()

    train, validation, test = create_time_split(df)

    assert set(
        train["date"].dt.year.unique()
    ) <= set(range(2017, 2024))

    assert set(
        validation["date"].dt.year.unique()
    ) == {2024}

    assert set(
        test["date"].dt.year.unique()
    ) == {2025}


def test_no_random_split_is_required():

    df = load_feature_dataset()

    train, validation, test = create_time_split(df)

    assert len(
        set(train.index)
        & set(validation.index)
    ) == 0

    assert len(
        set(validation.index)
        & set(test.index)
    ) == 0


def test_baselines_use_previous_information_only():

    df = load_feature_dataset()

    result = create_baseline_predictions(df)

    result = result.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    for market, group in result.groupby(
        "market"
    ):

        if len(group) < 2:
            continue

        second = group.iloc[1]
        first = group.iloc[0]

        assert (
            second["naive_last_value"]
            == first[
                "modal_price_rs_per_quintal"
            ]
        )


def test_model_features_exclude_target():

    df = load_feature_dataset()

    features = get_model_features(df)

    assert (
        "modal_price_rs_per_quintal"
        not in features
    )


def test_model_features_exclude_current_target_changes():

    df = load_feature_dataset()

    features = get_model_features(df)

    assert "price_change_1d" not in features
    assert "price_change_7d" not in features


def test_model_features_exclude_current_weather():

    df = load_feature_dataset()

    features = get_model_features(df)

    weather = {
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "rainfall",
        "wind_speed_max",
        "et0",
    }

    assert not (
        set(features) & weather
    )


def test_lagged_weather_can_be_used():

    df = load_feature_dataset()

    features = get_model_features(df)

    assert (
        "temperature_mean_lag_1"
        in features
    )


def test_metrics_are_correct_for_simple_case():

    actual = np.array(
        [100.0, 200.0, 300.0]
    )

    predicted = np.array(
        [100.0, 220.0, 270.0]
    )

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert metrics["MAPE"] >= 0
    assert metrics["n"] == 3


def test_feature_dataset_is_not_modified():

    before = pd.read_csv(DATA)

    _ = load_feature_dataset()

    after = pd.read_csv(DATA)

    pd.testing.assert_frame_equal(
        before,
        after,
    )