"""Generate a reproducible synthetic Phase 9 farmer/buyer marketplace dataset.

The generator uses the project's historical paddy price observations as the
empirical source for variety, grade, market, and price distributions. It does
not create real farmer, buyer, company, or live-demand records.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_SEED = 42
FARMER_COUNT = 500
BUYER_COUNT = 1000
SYNTHETIC_START = pd.Timestamp("2026-01-01")
SYNTHETIC_END = pd.Timestamp("2026-06-30")

FARMER_COLUMNS = [
    "listing_id", "crop", "variety", "grade", "quantity_quintals",
    "farmer_market", "minimum_expected_price", "latitude", "longitude",
    "created_date", "synthetic_demo_data",
]
BUYER_COLUMNS = [
    "buyer_id", "buyer_name", "crop", "variety", "grade",
    "required_quantity_quintals", "offered_price_per_quintal", "preferred_market",
    "latitude", "longitude", "pickup_available", "active", "synthetic_demo_data",
]


@dataclass(frozen=True)
class SourceData:
    prices: pd.DataFrame
    market_coordinates: dict[str, tuple[float | None, float | None]]
    varieties: tuple[str, ...]
    grades: tuple[str, ...]
    markets: tuple[str, ...]


def _find_master_dataset(root: Path) -> Path:
    candidates = [
        root / "data" / "processed" / "master_dataset.csv",
        root / "data" / "processed" / "master_dataset_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Neither master_dataset.csv nor master_dataset_clean.csv was found.")


def _load_market_reference(root: Path, master: pd.DataFrame) -> dict[str, tuple[float | None, float | None]]:
    """Load verified market coordinates, preferring markets.csv when present.

    If no market reference file exists, coordinates are derived only from the
    existing master dataset. Missing coordinates remain missing.
    """
    reference = root / "data" / "raw" / "reference" / "markets.csv"
    if reference.exists():
        frame = pd.read_csv(reference)
        normalized = {str(c).strip().casefold(): c for c in frame.columns}
        market_col = next((normalized[k] for k in ("market", "market_name", "name") if k in normalized), None)
        lat_col = next((normalized[k] for k in ("latitude", "lat") if k in normalized), None)
        lon_col = next((normalized[k] for k in ("longitude", "lon", "lng") if k in normalized), None)
        if market_col and lat_col and lon_col:
            coords: dict[str, tuple[float | None, float | None]] = {}
            for row in frame.itertuples(index=False):
                market = str(getattr(row, market_col)).strip()
                lat = getattr(row, lat_col)
                lon = getattr(row, lon_col)
                coords[market] = (None if pd.isna(lat) else float(lat), None if pd.isna(lon) else float(lon))
            return coords

    coords = {}
    for market, group in master.groupby("market", sort=True):
        lat_values = pd.to_numeric(group["latitude"], errors="coerce").dropna()
        lon_values = pd.to_numeric(group["longitude"], errors="coerce").dropna()
        coords[str(market).strip()] = (
            None if lat_values.empty else float(lat_values.iloc[0]),
            None if lon_values.empty else float(lon_values.iloc[0]),
        )
    return coords


def load_source_data(root: Path) -> SourceData:
    master_path = _find_master_dataset(root)
    master = pd.read_csv(master_path)
    required = {"market", "variety", "grade", "modal_price_rs_per_quintal", "latitude", "longitude"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master dataset missing columns: {sorted(missing)}")

    frame = master.copy()
    frame["market"] = frame["market"].astype(str).str.strip()
    frame["variety"] = frame["variety"].astype(str).str.strip()
    frame["grade"] = frame["grade"].astype(str).str.strip()
    frame["modal_price_rs_per_quintal"] = pd.to_numeric(frame["modal_price_rs_per_quintal"], errors="coerce")
    frame = frame.dropna(subset=["market", "variety", "grade", "modal_price_rs_per_quintal"])
    frame = frame[frame["modal_price_rs_per_quintal"] > 0].copy()
    if frame.empty:
        raise ValueError("No usable modal-price observations found in the master dataset.")

    coords = _load_market_reference(root, master)
    markets = tuple(sorted(frame["market"].unique()))
    varieties = tuple(sorted(frame["variety"].unique()))
    grades = tuple(sorted(frame["grade"].unique()))
    return SourceData(frame, coords, varieties, grades, markets)


def _sample_price(
    source: SourceData,
    rng: np.random.Generator,
    variety: str,
    grade: str,
    market: str,
) -> float:
    """Sample an empirical modal price with conditional fallback hierarchy."""
    keys = [
        ("variety", "grade", "market"),
        ("variety", "grade"),
        ("variety",),
        ("market",),
        tuple(),
    ]
    values = None
    for keyset in keys:
        subset = source.prices
        for key in keyset:
            target = {"variety": variety, "grade": grade, "market": market}[key]
            subset = subset[subset[key] == target]
        if len(subset) >= 5:
            values = subset["modal_price_rs_per_quintal"].to_numpy(dtype=float)
            break
    if values is None:
        values = source.prices["modal_price_rs_per_quintal"].to_numpy(dtype=float)
    sampled = float(rng.choice(values))
    return sampled


def _quantity(rng: np.random.Generator, *, buyer: bool) -> float:
    """Generate small/medium/large agricultural lots using documented bands."""
    if buyer:
        bands = ((5, 40, 0.35), (40, 100, 0.50), (100, 250, 0.15))
    else:
        bands = ((5, 40, 0.40), (40, 100, 0.50), (100, 200, 0.10))
    u = float(rng.random())
    cumulative = 0.0
    for low, high, probability in bands:
        cumulative += probability
        if u <= cumulative:
            return round(float(rng.uniform(low, high)), 1)
    return float(bands[-1][1])


def _date(rng: np.random.Generator) -> str:
    days = int((SYNTHETIC_END - SYNTHETIC_START).days)
    return (SYNTHETIC_START + pd.Timedelta(days=int(rng.integers(0, days + 1)))).date().isoformat()


def _coords(source: SourceData, market: str) -> tuple[float | None, float | None]:
    return source.market_coordinates.get(market, (None, None))


def generate_farmers(source: SourceData, rng: np.random.Generator) -> pd.DataFrame:
    market_weights = source.prices["market"].value_counts(normalize=True).reindex(source.markets, fill_value=0).to_numpy()
    market_weights = market_weights / market_weights.sum()
    rows = []
    for i in range(1, FARMER_COUNT + 1):
        market = str(rng.choice(source.markets, p=market_weights))
        variety = str(rng.choice(source.varieties))
        grade = str(rng.choice(source.grades, p=_grade_weights(source, variety)))
        base = _sample_price(source, rng, variety, grade, market)
        # Farmer minimum is anchored to an empirical market observation with a
        # narrow synthetic negotiation band, then clipped to the empirical
        # global 1st-99th percentile to avoid unsupported extremes.
        minimum = base * float(rng.uniform(0.94, 1.02))
        q01, q99 = source.prices["modal_price_rs_per_quintal"].quantile([0.01, 0.99])
        minimum = round(float(np.clip(minimum, q01, q99)), 2)
        lat, lon = _coords(source, market)
        rows.append({
            "listing_id": f"F-SYN-{i:04d}",
            "crop": "Paddy",
            "variety": variety,
            "grade": grade,
            "quantity_quintals": _quantity(rng, buyer=False),
            "farmer_market": market,
            "minimum_expected_price": minimum,
            "latitude": lat,
            "longitude": lon,
            "created_date": _date(rng),
            "synthetic_demo_data": True,
        })
    return pd.DataFrame(rows, columns=FARMER_COLUMNS)


def _grade_weights(source: SourceData, variety: str) -> np.ndarray:
    counts = source.prices.loc[source.prices["variety"] == variety, "grade"].value_counts().reindex(source.grades, fill_value=0).to_numpy(dtype=float)
    if counts.sum() == 0:
        return np.repeat(1 / len(source.grades), len(source.grades))
    return counts / counts.sum()


def generate_buyers(source: SourceData, farmers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    market_weights = source.prices["market"].value_counts(normalize=True).reindex(source.markets, fill_value=0).to_numpy()
    market_weights = market_weights / market_weights.sum()
    farmer_reference = farmers.sample(BUYER_COUNT, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)
    price_modes = ["below", "around", "above"]
    price_mode_probs = [0.30, 0.40, 0.30]
    rows = []
    for i in range(1, BUYER_COUNT + 1):
        reference = farmer_reference.iloc[i - 1]
        # Most buyers use the farmer's product definition to create enough
        # meaningful matches; a controlled minority tests mismatches.
        variety = str(reference["variety"]) if rng.random() < 0.72 else str(rng.choice(source.varieties))
        grade = str(reference["grade"]) if rng.random() < 0.70 else str(rng.choice(source.grades))
        market = str(rng.choice(source.markets, p=market_weights))
        base = _sample_price(source, rng, variety, grade, market)
        mode = str(rng.choice(price_modes, p=price_mode_probs))
        if mode == "below":
            factor = float(rng.uniform(0.88, 0.98))
        elif mode == "around":
            factor = float(rng.uniform(0.98, 1.03))
        else:
            factor = float(rng.uniform(1.03, 1.12))
        offer = base * factor
        q01, q99 = source.prices["modal_price_rs_per_quintal"].quantile([0.01, 0.99])
        offer = round(float(np.clip(offer, q01, q99)), 2)
        lat, lon = _coords(source, market)
        rows.append({
            "buyer_id": f"B-SYN-{i:04d}",
            "buyer_name": f"Synthetic Buyer {i:04d}",
            "crop": "Paddy",
            "variety": variety,
            "grade": grade,
            "required_quantity_quintals": _quantity(rng, buyer=True),
            "offered_price_per_quintal": offer,
            "preferred_market": market,
            "latitude": lat,
            "longitude": lon,
            "pickup_available": bool(rng.random() < 0.70),
            "active": bool(rng.random() < 0.90),
            "synthetic_demo_data": True,
        })
    return pd.DataFrame(rows, columns=BUYER_COLUMNS)


def validate_marketplace_data(farmers: pd.DataFrame, buyers: pd.DataFrame, source: SourceData) -> None:
    if len(farmers) != FARMER_COUNT:
        raise ValueError(f"Expected {FARMER_COUNT} farmer rows, got {len(farmers)}")
    if len(buyers) != BUYER_COUNT:
        raise ValueError(f"Expected {BUYER_COUNT} buyer rows, got {len(buyers)}")
    if list(farmers.columns) != FARMER_COLUMNS:
        raise ValueError("Farmer columns do not match the required schema.")
    if list(buyers.columns) != BUYER_COLUMNS:
        raise ValueError("Buyer columns do not match the required schema.")
    if farmers["listing_id"].duplicated().any() or buyers["buyer_id"].duplicated().any():
        raise ValueError("Duplicate synthetic IDs detected.")
    if not farmers["listing_id"].str.fullmatch(r"F-SYN-\d{4}").all():
        raise ValueError("Malformed farmer IDs detected.")
    if not buyers["buyer_id"].str.fullmatch(r"B-SYN-\d{4}").all():
        raise ValueError("Malformed buyer IDs detected.")
    if set(farmers["crop"]) != {"Paddy"} or set(buyers["crop"]) != {"Paddy"}:
        raise ValueError("Synthetic crop values must be Paddy.")
    valid_varieties = set(source.varieties)
    valid_grades = set(source.grades)
    valid_markets = set(source.markets)
    if not set(farmers["variety"]).issubset(valid_varieties) or not set(buyers["variety"]).issubset(valid_varieties):
        raise ValueError("Synthetic variety outside the master dataset detected.")
    if not set(farmers["grade"]).issubset(valid_grades) or not set(buyers["grade"]).issubset(valid_grades):
        raise ValueError("Synthetic grade outside the master dataset detected.")
    if not set(farmers["farmer_market"]).issubset(valid_markets) or not set(buyers["preferred_market"]).issubset(valid_markets):
        raise ValueError("Synthetic market outside the master dataset detected.")
    if (farmers["quantity_quintals"] <= 0).any() or (buyers["required_quantity_quintals"] <= 0).any():
        raise ValueError("All synthetic quantities must be positive.")
    if (farmers["minimum_expected_price"] <= 0).any() or (buyers["offered_price_per_quintal"] <= 0).any():
        raise ValueError("All synthetic prices must be positive.")
    if farmers["synthetic_demo_data"].ne(True).any() or buyers["synthetic_demo_data"].ne(True).any():
        raise ValueError("Every synthetic record must have synthetic_demo_data=True.")
    for column in ("pickup_available", "active"):
        if not buyers[column].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"Buyer column {column} must contain booleans.")
    if farmers[["latitude", "longitude"]].notna().any(axis=1).ne(
        farmers[["latitude", "longitude"]].notna().all(axis=1)
    ).any():
        raise ValueError("Farmer coordinates must be both present or both missing.")
    if buyers[["latitude", "longitude"]].notna().any(axis=1).ne(
        buyers[["latitude", "longitude"]].notna().all(axis=1)
    ).any():
        raise ValueError("Buyer coordinates must be both present or both missing.")
    if "*" in "".join(map(str, farmers.columns)) or "*" in "".join(map(str, buyers.columns)):
        raise ValueError("Asterisk artifacts found in CSV headers.")
    if farmers.isna().sum().sum() < 0 or buyers.isna().sum().sum() < 0:
        raise ValueError("Unexpected validation state.")


def write_report(root: Path, source: SourceData, farmers: pd.DataFrame, buyers: pd.DataFrame, master_path: Path) -> Path:
    report_path = root / "reports" / "phase9" / "synthetic_data_generation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    farmer_missing = float(farmers[["latitude", "longitude"]].isna().all(axis=1).mean() * 100)
    buyer_missing = float(buyers[["latitude", "longitude"]].isna().all(axis=1).mean() * 100)
    active = int(buyers["active"].sum())
    inactive = len(buyers) - active
    pickup = int(buyers["pickup_available"].sum())
    no_pickup = len(buyers) - pickup
    lines = [
        "# Phase 9 Synthetic Marketplace Data Generation Report",
        "",
        "> All farmer and buyer records are synthetic demonstration data generated from distributions derived from the project's historical market data. They do not represent real farmers, real buyers, real companies, or live market demand.",
        "",
        "## Generation summary",
        f"- Farmer records: **{len(farmers)}**",
        f"- Buyer records: **{len(buyers)}**",
        f"- Random seed: **{RANDOM_SEED}**",
        f"- Master dataset: `{master_path.relative_to(root)}`",
        f"- Market reference: `data/raw/reference/markets.csv` if present; otherwise market coordinates were derived from `{master_path.relative_to(root)}`.",
        "",
        "## Source distributions",
        f"- Varieties used ({len(source.varieties)}): {', '.join(source.varieties)}",
        f"- Grades used ({len(source.grades)}): {', '.join(source.grades)}",
        f"- Markets used ({len(source.markets)}): {', '.join(source.markets)}",
        "- Farmer minimum prices: empirical modal-price observations conditioned by variety/grade/market where at least five observations exist, with a narrow 0.94–1.02 synthetic negotiation factor and clipping to the empirical 1st–99th percentile.",
        "- Buyer offers: empirical modal-price observations conditioned by variety/grade/market where available, with below/around/above factors of 0.88–0.98, 0.98–1.03, and 1.03–1.12 respectively; clipped to the empirical 1st–99th percentile.",
        "- Quantities: documented synthetic small/medium/large lot bands. Farmer bands are 5–40 (40%), 40–100 (50%), 100–200 (10%); buyer bands are 5–40 (35%), 40–100 (50%), 100–250 (15%).",
        "- Listing dates: synthetic dates uniformly sampled from 2026-01-01 through 2026-06-30.",
        "",
        "## Coordinates and status",
        f"- Missing farmer coordinates: **{farmer_missing:.2f}%**",
        f"- Missing buyer coordinates: **{buyer_missing:.2f}%**",
        f"- Buyer active: **{active} ({active / len(buyers) * 100:.2f}%)**",
        f"- Buyer inactive: **{inactive} ({inactive / len(buyers) * 100:.2f}%)**",
        f"- Pickup available: **{pickup} ({pickup / len(buyers) * 100:.2f}%)**",
        f"- Pickup unavailable: **{no_pickup} ({no_pickup / len(buyers) * 100:.2f}%)**",
        "- Coordinates are copied from verified existing market data. Missing coordinates remain blank; no synthetic coordinates are invented.",
        "",
        "## Matching scenarios",
        "The generator deliberately creates variation in product definitions, prices, quantities, activity, pickup availability, and coordinates so the unchanged Phase 9 matching engine can exercise exact matches, grade/variety mismatches, price-compatible and price-incompatible cases, quantity incompatibility, inactive buyers, pickup differences, same/different markets, and missing-coordinate cases.",
        "",
        "## Limitations",
        "- Quantities and listing dates are synthetic assumptions because the historical master dataset does not contain farmer listing quantities or marketplace listing dates.",
        "- Buyer/farmer records are not real supply or demand observations and must not be presented as live marketplace data.",
        "- Price realism is distributional rather than a claim that any synthetic buyer would actually offer a specific price.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_marketplace(root: Path | None = None) -> tuple[Path, Path, Path]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    rng = np.random.default_rng(RANDOM_SEED)
    source = load_source_data(root)
    master_path = _find_master_dataset(root)
    farmers = generate_farmers(source, rng)
    buyers = generate_buyers(source, farmers, rng)
    validate_marketplace_data(farmers, buyers, source)
    farmer_path = root / "data" / "raw" / "synthetic" / "farmer_listings.csv"
    buyer_path = root / "data" / "raw" / "synthetic" / "buyer_listings.csv"
    farmer_path.parent.mkdir(parents=True, exist_ok=True)
    farmers.to_csv(farmer_path, index=False)
    buyers.to_csv(buyer_path, index=False)
    report_path = write_report(root, source, farmers, buyers, master_path)
    return farmer_path, buyer_path, report_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    farmer_path, buyer_path, report_path = generate_marketplace(root)
    farmers = pd.read_csv(farmer_path)
    buyers = pd.read_csv(buyer_path)
    print("=" * 72)
    print("PADDYWISE AI — SYNTHETIC MARKETPLACE DATA GENERATOR")
    print("=" * 72)
    print(f"Farmers generated: {len(farmers)}")
    print(f"Buyers generated: {len(buyers)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Farmer CSV: {farmer_path}")
    print(f"Buyer CSV: {buyer_path}")
    print(f"Report: {report_path}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
