"""
PaddyWise AI — Phase 8: Market Comparison & Net Revenue Analysis.

This module compares available paddy markets using actual market prices,
verified market coordinates, straight-line Haversine distance, and a
transparent estimated transportation-cost assumption.

Important:
    Haversine distance is geographic straight-line distance, not road distance.
    Transportation cost is an estimate, not a real-time logistics quote.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from math import asin, cos, isfinite, radians, sin, sqrt
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd


EARTH_RADIUS_KM = 6371.0
DEFAULT_TRANSPORT_COST_PER_KM = 20.0

# Verified market-yard coordinates supplied for Phase 8.
# Thiruppananthal intentionally has no coordinates.
MARKET_COORDINATES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "Budalur": (10.7968, 78.9769),
    "Kumbakonam": (10.9617, 79.3880),
    "Orathanadu": (10.6280, 79.2531),
    "Papanasam": (10.9252, 79.2708),
    "Pattukottai": (10.4250, 79.3140),
    "Thanjavur": (10.7870, 79.1378),
    "Vallam": (10.7140, 79.7020),
    "Thiruppananthal": (None, None),
}

TARGET = "modal_price_rs_per_quintal"
DATE_COLUMN = "date"
MARKET_COLUMN = "market"


@dataclass(frozen=True)
class TransportConfig:
    """Transparent transportation-cost assumptions."""

    cost_per_km: float = DEFAULT_TRANSPORT_COST_PER_KM

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.cost_per_km, "cost_per_km")


@dataclass(frozen=True)
class MarketResult:
    """Comparison result for one market."""

    market: str
    latitude: Optional[float]
    longitude: Optional[float]
    distance_km: Optional[float]
    current_price: Optional[float]
    forecast_price: Optional[float]
    gross_revenue: Optional[float]
    transportation_cost: Optional[float]
    net_revenue: Optional[float]
    net_revenue_difference: Optional[float]
    percentage_difference: Optional[float]
    coordinates_available: bool
    price_date: Optional[str]
    available_for_distance_comparison: bool


def _validate_finite(value: object, name: str) -> float:
    """Return a finite float or raise ValueError."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def _validate_non_negative_finite(value: object, name: str) -> float:
    numeric = _validate_finite(value, name)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _validate_latitude(value: object, name: str = "latitude") -> float:
    numeric = _validate_finite(value, name)
    if not -90.0 <= numeric <= 90.0:
        raise ValueError(f"{name} must be between -90 and 90 degrees.")
    return numeric


def _validate_longitude(value: object, name: str = "longitude") -> float:
    numeric = _validate_finite(value, name)
    if not -180.0 <= numeric <= 180.0:
        raise ValueError(f"{name} must be between -180 and 180 degrees.")
    return numeric


def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate straight-line geographic distance in kilometres."""
    lat1 = _validate_latitude(lat1, "lat1")
    lon1 = _validate_longitude(lon1, "lon1")
    lat2 = _validate_latitude(lat2, "lat2")
    lon2 = _validate_longitude(lon2, "lon2")

    lat1_rad, lat2_rad = radians(lat1), radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a))


def calculate_transportation_cost(
    distance_km: float,
    transport_cost_per_km: float,
) -> float:
    """Estimate transportation cost as distance_km × cost_per_km."""
    distance = _validate_non_negative_finite(distance_km, "distance_km")
    rate = _validate_non_negative_finite(
        transport_cost_per_km,
        "transport_cost_per_km",
    )
    return distance * rate


def calculate_gross_revenue(
    quantity_quintals: float,
    market_price: float,
) -> float:
    """Calculate gross revenue before transportation cost."""
    quantity = _validate_non_negative_finite(quantity_quintals, "quantity_quintals")
    if quantity <= 0:
        raise ValueError("quantity_quintals must be greater than zero.")
    price = _validate_non_negative_finite(market_price, "market_price")
    return quantity * price


def calculate_net_revenue(
    quantity_quintals: float,
    market_price: float,
    transportation_cost: float,
) -> float:
    """Calculate gross revenue minus estimated transportation cost."""
    gross = calculate_gross_revenue(quantity_quintals, market_price)
    transport = _validate_non_negative_finite(
        transportation_cost,
        "transportation_cost",
    )
    return gross - transport


def _normalise_price_data(price_data: object) -> pd.DataFrame:
    """Validate and copy market-price input without modifying the source."""
    if isinstance(price_data, pd.DataFrame):
        frame = price_data.copy(deep=True)
    elif isinstance(price_data, Mapping):
        frame = pd.DataFrame(
            [{"market": market, TARGET: price} for market, price in price_data.items()]
        )
    else:
        raise TypeError("price_data must be a pandas DataFrame or market-to-price mapping.")

    if MARKET_COLUMN not in frame.columns:
        raise ValueError("price_data must contain a 'market' column.")
    if TARGET not in frame.columns:
        raise ValueError(
            f"price_data must contain '{TARGET}' for current market prices."
        )

    frame[MARKET_COLUMN] = frame[MARKET_COLUMN].astype(str).str.strip()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")

    if DATE_COLUMN in frame.columns:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")

    return frame


def get_latest_market_prices(
    price_data: pd.DataFrame,
    *,
    as_of_date: Optional[str | date] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return one deterministic latest-price row per market.

    When variety/grade are supplied, prices are filtered to that same
    product definition before comparison. This is preferred when comparing
    like-for-like paddy observations.

    If variety/grade are omitted and several observations exist on a market's
    latest date, the modal-price field is aggregated by the median for that
    date. This avoids arbitrarily selecting one variety/grade observation.
    """
    frame = _normalise_price_data(price_data)

    if DATE_COLUMN not in frame.columns:
        raise ValueError("price_data must contain a 'date' column for latest-price lookup.")

    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(cutoff):
            raise ValueError("as_of_date must be a valid date.")
        frame = frame[frame[DATE_COLUMN] <= cutoff].copy()

    if variety is not None:
        if "variety" not in frame.columns:
            raise ValueError("price_data does not contain 'variety'.")
        frame = frame[frame["variety"].astype(str).str.strip() == str(variety).strip()].copy()

    if grade is not None:
        if "grade" not in frame.columns:
            raise ValueError("price_data does not contain 'grade'.")
        frame = frame[frame["grade"].astype(str).str.strip() == str(grade).strip()].copy()

    frame = frame.dropna(subset=[MARKET_COLUMN, DATE_COLUMN, TARGET])
    if frame.empty:
        return pd.DataFrame(
            columns=[MARKET_COLUMN, TARGET, DATE_COLUMN]
        )

    latest_dates = frame.groupby(MARKET_COLUMN)[DATE_COLUMN].transform("max")
    latest = frame.loc[frame[DATE_COLUMN].eq(latest_dates)].copy()

    if variety is not None or grade is not None:
        # With a like-for-like variety/grade filter, preserve the actual
        # modal-price observation. Duplicate market/date rows are invalid
        # for this project's cleaned dataset, so this remains deterministic.
        latest = (
            latest.sort_values([MARKET_COLUMN, DATE_COLUMN, TARGET])
            .drop_duplicates(subset=[MARKET_COLUMN], keep="last")
            [[MARKET_COLUMN, TARGET, DATE_COLUMN]]
        )
    else:
        # Multiple varieties/grades can coexist on the same market/date.
        latest = (
            latest.groupby([MARKET_COLUMN, DATE_COLUMN], as_index=False)[TARGET]
            .median()
            .sort_values([MARKET_COLUMN, DATE_COLUMN])
        )

    return latest.reset_index(drop=True)


def compare_markets(
    *,
    quantity_quintals: float,
    current_market: str,
    current_market_price: float,
    market_price_data: pd.DataFrame | Mapping[str, float],
    farmer_latitude: float,
    farmer_longitude: float,
    transport_cost_per_km: float = DEFAULT_TRANSPORT_COST_PER_KM,
    forecast_prices: Optional[Mapping[str, float]] = None,
) -> list[MarketResult]:
    """
    Compare markets by estimated net revenue.

    `market_price_data` should contain actual current modal prices. A mapping
    may also be supplied when prices have already been looked up.

    Transportation is estimated as:
        distance_km × transport_cost_per_km

    The distance is straight-line geographic distance, not road distance.
    """
    quantity = _validate_non_negative_finite(quantity_quintals, "quantity_quintals")
    if quantity <= 0:
        raise ValueError("quantity_quintals must be greater than zero.")

    current_market = str(current_market).strip()
    if not current_market:
        raise ValueError("current_market must not be empty.")

    current_price = _validate_non_negative_finite(
        current_market_price,
        "current_market_price",
    )
    farmer_lat = _validate_latitude(farmer_latitude, "farmer_latitude")
    farmer_lon = _validate_longitude(farmer_longitude, "farmer_longitude")
    transport = TransportConfig(transport_cost_per_km)

    prices = _normalise_price_data(market_price_data)

    # For a DataFrame, market price is selected deterministically. The main
    # lookup helper is preferred when dates are present.
    price_by_market: dict[str, tuple[float, Optional[str]]] = {}
    if DATE_COLUMN in prices.columns:
        latest = get_latest_market_prices(prices)
        for row in latest.itertuples(index=False):
            market = str(getattr(row, MARKET_COLUMN))
            price = float(getattr(row, TARGET))
            date_value = getattr(row, DATE_COLUMN)
            price_by_market[market] = (
                price,
                None if pd.isna(date_value) else pd.Timestamp(date_value).date().isoformat(),
            )
    else:
        for row in prices[[MARKET_COLUMN, TARGET]].itertuples(index=False):
            price_by_market[str(row[0])] = (float(row[1]), None)

    # Ensure the explicitly supplied current-market price is authoritative.
    if current_market not in price_by_market:
        price_by_market[current_market] = (current_price, None)
    else:
        price_by_market[current_market] = (
            current_price,
            price_by_market[current_market][1],
        )

    results: list[MarketResult] = []

    for market in sorted(price_by_market):
        price, price_date = price_by_market[market]
        latitude, longitude = MARKET_COORDINATES.get(market, (None, None))
        coordinates_available = latitude is not None and longitude is not None

        if coordinates_available:
            distance = calculate_haversine_distance(
                farmer_lat,
                farmer_lon,
                float(latitude),
                float(longitude),
            )
            transportation = calculate_transportation_cost(
                distance,
                transport.cost_per_km,
            )
            gross = calculate_gross_revenue(quantity, price)
            net = calculate_net_revenue(quantity, price, transportation)
        else:
            distance = None
            transportation = None
            gross = calculate_gross_revenue(quantity, price)
            net = None

        current_result = next(
            (
                item
                for item in results
                if item.market == current_market
            ),
            None,
        )

        # Difference can only be calculated once the current market itself
        # has a distance-based net revenue.
        if market == current_market and net is not None:
            current_net = net
        elif current_result is not None:
            current_net = current_result.net_revenue
        else:
            current_net = None

        if current_net is not None and net is not None:
            difference = net - current_net
            percentage = (difference / current_net * 100.0) if current_net != 0 else None
        else:
            difference = None
            percentage = None

        forecast = None
        if forecast_prices is not None and market in forecast_prices:
            forecast = _validate_non_negative_finite(
                forecast_prices[market],
                f"forecast_prices[{market!r}]",
            )

        results.append(
            MarketResult(
                market=market,
                latitude=latitude,
                longitude=longitude,
                distance_km=distance,
                current_price=price,
                forecast_price=forecast,
                gross_revenue=gross,
                transportation_cost=transportation,
                net_revenue=net,
                net_revenue_difference=difference,
                percentage_difference=percentage,
                coordinates_available=coordinates_available,
                price_date=price_date,
                available_for_distance_comparison=coordinates_available,
            )
        )

    # Recompute differences after the full result set exists, especially if
    # the current market appeared later alphabetically.
    current = next((r for r in results if r.market == current_market), None)
    if current is not None and current.net_revenue is not None:
        current_net = current.net_revenue
        results = [
            MarketResult(
                **{
                    **asdict(result),
                    "net_revenue_difference": (
                        result.net_revenue - current_net
                        if result.net_revenue is not None
                        else None
                    ),
                    "percentage_difference": (
                        ((result.net_revenue - current_net) / current_net * 100.0)
                        if result.net_revenue is not None and current_net != 0
                        else None
                    ),
                }
            )
            for result in results
        ]

    return results


def results_to_dataframe(results: Sequence[MarketResult]) -> pd.DataFrame:
    """Convert structured results to a dashboard-friendly DataFrame."""
    return pd.DataFrame([asdict(result) for result in results])


def load_master_price_data(path: str | Path) -> pd.DataFrame:
    """Load the project's actual processed master dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price dataset not found: {path}")
    return pd.read_csv(path)


def run_example(
    master_dataset_path: str | Path,
    *,
    current_market: str = "Vallam",
    quantity_quintals: float = 100.0,
    farmer_latitude: float = 10.7140,
    farmer_longitude: float = 79.7020,
    variety: Optional[str] = "B P T",
    grade: Optional[str] = "Local",
    transport_cost_per_km: float = DEFAULT_TRANSPORT_COST_PER_KM,
) -> pd.DataFrame:
    """
    Run a reproducible example using the project's actual master dataset.

    The example uses the same variety/grade where available. Markets without
    matching observations are simply absent from the current-price comparison;
    no prices are fabricated.
    """
    data = load_master_price_data(master_dataset_path)
    prices = get_latest_market_prices(
        data,
        variety=variety,
        grade=grade,
    )

    if prices.empty or current_market not in set(prices[MARKET_COLUMN]):
        raise ValueError(
            f"No current price found for market={current_market!r}, "
            f"variety={variety!r}, grade={grade!r}."
        )

    current_row = prices.loc[prices[MARKET_COLUMN].eq(current_market)].iloc[0]
    current_price = float(current_row[TARGET])

    results = compare_markets(
        quantity_quintals=quantity_quintals,
        current_market=current_market,
        current_market_price=current_price,
        market_price_data=prices,
        farmer_latitude=farmer_latitude,
        farmer_longitude=farmer_longitude,
        transport_cost_per_km=transport_cost_per_km,
    )
    return results_to_dataframe(results)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data" / "processed" / "master_dataset.csv"
    output = run_example(dataset)
    print(output.to_string(index=False))
