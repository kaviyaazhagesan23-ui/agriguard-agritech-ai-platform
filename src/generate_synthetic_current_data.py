"""Maintain the deterministic synthetic current/forecast input layer.

The immutable historical dataset is only a distribution source. Existing
synthetic current rows are preserved; deterministic rows are appended through
2026-11-30 so the application has future weather and starting-price coverage.
"""
from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = ROOT / "data" / "processed" / "master_dataset_clean.csv"
CURRENT_DIR = ROOT / "data" / "current"
PRICE_OUTPUT = CURRENT_DIR / "synthetic_current_prices.csv"
WEATHER_OUTPUT = CURRENT_DIR / "synthetic_current_weather.csv"

RANDOM_SEED = 42
CURRENT_START_DATE = date(2026, 7, 26)
CURRENT_END_DATE = date(2026, 11, 30)

MARKETS = [
    "Budalur",
    "Kumbakonam",
    "Orathanadu",
    "Papanasam",
    "Pattukottai",
    "Thanjavur",
    "Vallam",
]

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
]

PRICE_COLUMNS = [
    "date",
    "market",
    "commodity",
    "variety",
    "grade",
    "min_price_rs_per_quintal",
    "max_price_rs_per_quintal",
    "modal_price_rs_per_quintal",
]


def _circular_day_distance(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 365 - d)


def _seasonal_reference(
    history: pd.DataFrame,
    target_day: date,
    columns: list[str],
) -> pd.Series:
    doy = target_day.timetuple().tm_yday

    distances = history["date"].dt.dayofyear.map(
        lambda x: _circular_day_distance(int(x), doy)
    )

    selected = history.loc[distances <= 45]

    if selected.empty:
        selected = history

    return selected[columns].median()


def _stable_rng(*parts: object) -> np.random.Generator:
    key = "|".join(
        [
            str(RANDOM_SEED),
            *(str(part) for part in parts),
        ]
    )

    seed = (
        int.from_bytes(
            sha256(key.encode("utf-8")).digest()[:8],
            "little",
        )
        % (2**63 - 1)
    )

    return np.random.default_rng(seed)


def _robust_bounds(s: pd.Series) -> tuple[float, float]:
    q05, q95 = s.quantile([0.05, 0.95])
    return float(q05), float(q95)


def _reference_price(
    history: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
    target_day: date,
) -> float:
    """Return the exact combination's latest historical modal price.

    No market-wide, variety-wide, grade-wide, or global fallback is used.
    """
    exact = history[
        (history.market == market)
        & (history.variety == variety)
        & (history.grade == grade)
    ].sort_values("date")

    if exact.empty:
        raise ValueError(
            f"No historical price distribution exists for "
            f"{market}/{variety}/{grade}."
        )

    value = pd.to_numeric(
        exact.iloc[-1]["modal_price_rs_per_quintal"],
        errors="coerce",
    )

    if pd.isna(value) or value <= 0:
        raise ValueError(
            f"Latest historical modal price is invalid for "
            f"{market}/{variety}/{grade}."
        )

    return float(value)


def _combination_price_scale(
    history: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
) -> float:
    """Estimate combination-specific short-term price movement scale.

    The scale is calculated only from the selected market/variety/grade
    historical series.
    """
    exact = history[
        (history.market == market)
        & (history.variety == variety)
        & (history.grade == grade)
    ].sort_values("date")

    changes = (
        pd.to_numeric(
            exact["modal_price_rs_per_quintal"],
            errors="coerce",
        )
        .diff()
        .dropna()
    )

    if changes.empty:
        return 0.0

    changes = changes.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if changes.empty:
        return 0.0

    median = float(changes.median())

    mad = float(
        np.median(
            np.abs(
                changes.to_numpy(dtype=float) - median
            )
        )
    )

    scale = 1.4826 * mad

    if scale <= 0:
        scale = float(changes.abs().quantile(0.75))

    if scale <= 0:
        scale = max(
            10.0,
            abs(
                float(
                    exact.iloc[-1][
                        "modal_price_rs_per_quintal"
                    ]
                )
            )
            * 0.005,
        )

    return float(scale)


def _initial_generation(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic current inputs.

    Every exact market/variety/grade series starts from its own latest
    historical modal price.

    Example:
        Kumbakonam / B P T / FAQ
        latest historical modal = 2500

    The first synthetic current observation for that combination is therefore
    exactly 2500. No global average or cross-combination value is used.
    """
    end = date(2026, 9, 23)
    start = CURRENT_START_DATE

    dates = [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
    ]

    combos = (
        history[history.market.isin(MARKETS)][
            ["market", "commodity", "variety", "grade"]
        ]
        .drop_duplicates()
        .sort_values(
            ["market", "variety", "grade"]
        )
    )

    rows = []

    for market, commodity, variety, grade in combos.itertuples(
        index=False
    ):
        exact = history[
            (history.market == market)
            & (history.variety == variety)
            & (history.grade == grade)
        ].sort_values("date")

        if exact.empty:
            raise ValueError(
                f"No historical rows exist for "
                f"{market}/{variety}/{grade}."
            )

        anchor = _reference_price(
            history,
            market,
            variety,
            grade,
            dates[0],
        )

        scale = _combination_price_scale(
            history,
            market,
            variety,
            grade,
        )

        latest = exact.iloc[-1]

        min_ratio = float(
            max(
                0.0,
                (
                    float(
                        latest[
                            "modal_price_rs_per_quintal"
                        ]
                    )
                    - float(
                        latest[
                            "min_price_rs_per_quintal"
                        ]
                    )
                )
                / anchor,
            )
        )

        max_ratio = float(
            max(
                0.0,
                (
                    float(
                        latest[
                            "max_price_rs_per_quintal"
                        ]
                    )
                    - anchor
                )
                / anchor,
            )
        )

        prev = anchor

        for day_index, day in enumerate(dates):
            if day_index == 0:
                value = anchor
            else:
                rng = _stable_rng(
                    "price-level-anchored",
                    market,
                    variety,
                    grade,
                    day,
                )

                innovation = (
                    rng.normal(0.0, scale)
                    if scale > 0
                    else 0.0
                )

                value = prev + innovation

            level_band = max(
                2.0 * scale,
                anchor * 0.08,
            )

            value = float(
                np.clip(
                    value,
                    anchor - level_band,
                    anchor + level_band,
                )
            )

            value = float(
                np.round(value / 10.0) * 10.0
            )

            min_price = float(
                np.round(
                    max(
                        1.0,
                        value * (1.0 - min_ratio),
                    )
                    / 10.0
                )
                * 10.0
            )

            max_price = float(
                np.round(
                    value * (1.0 + max_ratio)
                    / 10.0
                )
                * 10.0
            )

            rows.append(
                {
                    "date": day.isoformat(),
                    "market": market,
                    "commodity": commodity,
                    "variety": variety,
                    "grade": grade,
                    "min_price_rs_per_quintal": int(
                        min(min_price, value)
                    ),
                    "max_price_rs_per_quintal": int(
                        max(max_price, value)
                    ),
                    "modal_price_rs_per_quintal": int(
                        value
                    ),
                }
            )

            prev = value

    prices = pd.DataFrame(
        rows,
        columns=PRICE_COLUMNS,
    )

    weather_rows = []

    for market in MARKETS:
        market_history = history[
            history.market == market
        ].copy()

        for day in dates:
            ref = _seasonal_reference(
                market_history,
                day,
                WEATHER_COLUMNS,
            )

            vals = {}

            for col in WEATHER_COLUMNS:
                std = (
                    float(market_history[col].std())
                    if len(market_history) > 1
                    else float(history[col].std())
                )

                rng = _stable_rng(
                    "weather",
                    market,
                    day,
                    col,
                )

                value = float(ref[col]) + rng.normal(
                    0.0,
                    max(std * 0.08, 0.01),
                )

                vals[col] = round(
                    float(
                        np.clip(
                            value,
                            market_history[col].min(),
                            market_history[col].max(),
                        )
                    ),
                    2,
                )

            weather_rows.append(
                {
                    "date": day.isoformat(),
                    "market": market,
                    **vals,
                }
            )

    weather = pd.DataFrame(
        weather_rows,
        columns=[
            "date",
            "market",
            *WEATHER_COLUMNS,
        ],
    )

    return prices, weather


def _extend_prices(
    history: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    existing = existing.copy()

    existing["date"] = pd.to_datetime(
        existing["date"],
        errors="raise",
    )

    last_date = existing["date"].max().date()

    if last_date >= CURRENT_END_DATE:
        return existing[PRICE_COLUMNS].copy()

    future_dates = [
        last_date + timedelta(days=i)
        for i in range(
            1,
            (CURRENT_END_DATE - last_date).days + 1,
        )
    ]

    rows = []

    combos = (
        existing[
            ["market", "commodity", "variety", "grade"]
        ]
        .drop_duplicates()
        .sort_values(
            ["market", "variety", "grade"]
        )
    )

    for market, commodity, variety, grade in combos.itertuples(
        index=False
    ):
        exact_hist = history[
            (history.market == market)
            & (history.variety == variety)
            & (history.grade == grade)
        ].sort_values("date")

        if exact_hist.empty:
            raise ValueError(
                f"No historical rows exist for "
                f"{market}/{variety}/{grade}."
            )

        anchor = _reference_price(
            history,
            market,
            variety,
            grade,
            future_dates[0],
        )

        scale = _combination_price_scale(
            history,
            market,
            variety,
            grade,
        )

        latest = exact_hist.iloc[-1]

        min_ratio = float(
            max(
                0.0,
                (
                    float(
                        latest[
                            "modal_price_rs_per_quintal"
                        ]
                    )
                    - float(
                        latest[
                            "min_price_rs_per_quintal"
                        ]
                    )
                )
                / anchor,
            )
        )

        max_ratio = float(
            max(
                0.0,
                (
                    float(
                        latest[
                            "max_price_rs_per_quintal"
                        ]
                    )
                    - anchor
                )
                / anchor,
            )
        )

        combo_existing = existing[
            (existing.market == market)
            & (existing.variety == variety)
            & (existing.grade == grade)
        ].sort_values("date")

        prev = float(
            combo_existing.iloc[-1][
                "modal_price_rs_per_quintal"
            ]
        )

        for day in future_dates:
            rng = _stable_rng(
                "price-level-anchored",
                market,
                variety,
                grade,
                day,
            )

            innovation = (
                rng.normal(0.0, scale)
                if scale > 0
                else 0.0
            )

            value = prev + innovation

            level_band = max(
                2.0 * scale,
                anchor * 0.08,
            )

            value = float(
                np.clip(
                    value,
                    anchor - level_band,
                    anchor + level_band,
                )
            )

            value = float(
                np.round(value / 10.0) * 10.0
            )

            min_price = float(
                np.round(
                    max(
                        1.0,
                        value * (1.0 - min_ratio),
                    )
                    / 10.0
                )
                * 10.0
            )

            max_price = float(
                np.round(
                    value * (1.0 + max_ratio)
                    / 10.0
                )
                * 10.0
            )

            rows.append(
                {
                    "date": day.isoformat(),
                    "market": market,
                    "commodity": commodity,
                    "variety": variety,
                    "grade": grade,
                    "min_price_rs_per_quintal": int(
                        min(min_price, value)
                    ),
                    "max_price_rs_per_quintal": int(
                        max(max_price, value)
                    ),
                    "modal_price_rs_per_quintal": int(
                        value
                    ),
                }
            )

            prev = value

    extension = pd.DataFrame(
        rows,
        columns=PRICE_COLUMNS,
    )

    combined = pd.concat(
        [
            existing[PRICE_COLUMNS],
            extension,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            ["market", "variety", "grade", "date"]
        )
        .reset_index(drop=True)
    )

    combined["date"] = (
        pd.to_datetime(
            combined["date"],
            errors="raise",
        )
        .dt.strftime("%Y-%m-%d")
    )

    return combined


def _extend_weather(
    history: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    existing = existing.copy()

    existing["date"] = pd.to_datetime(
        existing["date"],
        errors="raise",
    )

    last_date = existing["date"].max().date()

    if last_date >= CURRENT_END_DATE:
        return existing[
            ["date", "market", *WEATHER_COLUMNS]
        ].copy()

    future_dates = [
        last_date + timedelta(days=i)
        for i in range(
            1,
            (CURRENT_END_DATE - last_date).days + 1,
        )
    ]

    rows = []

    for market in MARKETS:
        market_history = history[
            history.market == market
        ].copy()

        for day in future_dates:
            ref = _seasonal_reference(
                market_history,
                day,
                WEATHER_COLUMNS,
            )

            vals = {}

            for col in WEATHER_COLUMNS:
                std = (
                    float(market_history[col].std())
                    if len(market_history) > 1
                    else float(history[col].std())
                )

                rng = _stable_rng(
                    "weather",
                    market,
                    day,
                    col,
                )

                value = float(ref[col]) + rng.normal(
                    0.0,
                    max(std * 0.08, 0.01),
                )

                vals[col] = round(
                    float(
                        np.clip(
                            value,
                            market_history[col].min(),
                            market_history[col].max(),
                        )
                    ),
                    2,
                )

            rows.append(
                {
                    "date": day.isoformat(),
                    "market": market,
                    **vals,
                }
            )

    extension = pd.DataFrame(
        rows,
        columns=[
            "date",
            "market",
            *WEATHER_COLUMNS,
        ],
    )

    combined = pd.concat(
        [
            existing[
                ["date", "market", *WEATHER_COLUMNS]
            ],
            extension,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            ["market", "date"]
        )
        .reset_index(drop=True)
    )

    combined["date"] = (
        pd.to_datetime(
            combined["date"],
            errors="raise",
        )
        .dt.strftime("%Y-%m-%d")
    )

    return combined


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(HISTORICAL_PATH)

    history = pd.read_csv(HISTORICAL_PATH)

    history["date"] = pd.to_datetime(
        history["date"],
        errors="raise",
    )

    required = {
        "date",
        "market",
        "commodity",
        "variety",
        "grade",
        "min_price_rs_per_quintal",
        "max_price_rs_per_quintal",
        "modal_price_rs_per_quintal",
        *WEATHER_COLUMNS,
    }

    missing = sorted(
        required - set(history.columns)
    )

    if missing:
        raise ValueError(
            f"Historical dataset is missing required columns: {missing}"
        )

    CURRENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PRICE_OUTPUT.exists() and WEATHER_OUTPUT.exists():
        prices = pd.read_csv(PRICE_OUTPUT)
        weather = pd.read_csv(WEATHER_OUTPUT)

        prices = _extend_prices(
            history,
            prices,
        )

        weather = _extend_weather(
            history,
            weather,
        )
    else:
        prices, weather = _initial_generation(
            history
        )

        prices = _extend_prices(
            history,
            prices,
        )

        weather = _extend_weather(
            history,
            weather,
        )

    prices.to_csv(
        PRICE_OUTPUT,
        index=False,
    )

    weather.to_csv(
        WEATHER_OUTPUT,
        index=False,
    )

    return prices, weather


if __name__ == "__main__":
    p, w = generate()

    print(
        f"Wrote {len(p)} synthetic price rows to "
        f"{PRICE_OUTPUT}"
    )

    print(
        f"Wrote {len(w)} synthetic weather rows to "
        f"{WEATHER_OUTPUT}"
    )

    print(
        "Synthetic date range: "
        f"{pd.to_datetime(p['date']).min().date()} -> "
        f"{pd.to_datetime(p['date']).max().date()}"
    )

    print(
        "Disclosure: synthetic current-market demo inputs; "
        "not official/live government market data."
    )