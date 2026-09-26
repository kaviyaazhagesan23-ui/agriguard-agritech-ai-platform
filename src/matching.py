"""PaddyWise AI — Phase 9: deterministic farmer/buyer matching.

This module implements transparent, rule-based matching for synthetic
farmer and buyer demonstration listings. It is not a live marketplace.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.market_comparison import calculate_haversine_distance


@dataclass(frozen=True)
class MatchingWeights:
    """Configurable points for the non-eligibility matching components."""

    variety_match: float = 25.0
    grade_match: float = 20.0
    price_attractiveness: float = 30.0
    distance: float = 15.0
    pickup: float = 10.0

    def __post_init__(self) -> None:
        values = (self.variety_match, self.grade_match, self.price_attractiveness, self.distance, self.pickup)
        if any(not isinstance(v, (int, float)) or not isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError("Matching weights must be finite non-negative numbers.")
        if sum(values) <= 0:
            raise ValueError("At least one matching weight must be positive.")


DEFAULT_WEIGHTS = MatchingWeights()


@dataclass(frozen=True)
class FarmerListing:
    listing_id: str
    crop: str
    variety: str
    grade: str
    quantity_quintals: float
    farmer_market: str
    minimum_expected_price: float
    latitude: float | None
    longitude: float | None
    created_date: str
    synthetic_demo_data: bool

    def __post_init__(self) -> None:
        if not self.listing_id.strip():
            raise ValueError("listing_id must not be empty.")
        _require_positive(self.quantity_quintals, "quantity_quintals")
        _require_non_negative(self.minimum_expected_price, "minimum_expected_price")
        _require_text(self.crop, "crop")
        _require_text(self.variety, "variety")
        _require_text(self.grade, "grade")
        _validate_coordinates_pair(self.latitude, self.longitude, "farmer")
        if not isinstance(self.synthetic_demo_data, bool):
            raise ValueError("synthetic_demo_data must be boolean.")


@dataclass(frozen=True)
class BuyerListing:
    buyer_id: str
    buyer_name: str
    crop: str
    variety: str
    grade: str
    required_quantity_quintals: float
    offered_price_per_quintal: float
    preferred_market: str
    latitude: float | None
    longitude: float | None
    pickup_available: bool
    active: bool
    synthetic_demo_data: bool

    def __post_init__(self) -> None:
        if not self.buyer_id.strip():
            raise ValueError("buyer_id must not be empty.")
        if not self.buyer_name.strip():
            raise ValueError("buyer_name must not be empty.")
        _require_text(self.crop, "crop")
        _require_text(self.variety, "variety")
        _require_text(self.grade, "grade")
        _require_positive(self.required_quantity_quintals, "required_quantity_quintals")
        _require_non_negative(self.offered_price_per_quintal, "offered_price_per_quintal")
        _validate_coordinates_pair(self.latitude, self.longitude, "buyer")
        if not isinstance(self.pickup_available, bool) or not isinstance(self.active, bool):
            raise ValueError("pickup_available and active must be boolean.")
        if not isinstance(self.synthetic_demo_data, bool):
            raise ValueError("synthetic_demo_data must be boolean.")


@dataclass(frozen=True)
class MatchResult:
    buyer_id: str
    buyer_name: str
    eligible: bool
    match_score: float
    crop_match: bool
    variety_match: bool
    grade_match: bool
    quantity_compatible: bool
    price_compatible: bool
    offered_price: float
    farmer_minimum_price: float
    distance_km: float | None
    pickup_available: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    synthetic_demo_data: bool


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _require_positive(value: object, name: str) -> float:
    number = _require_finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return number


def _require_non_negative(value: object, name: str) -> float:
    number = _require_finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _validate_coordinates_pair(latitude: float | None, longitude: float | None, label: str) -> None:
    if latitude is None and longitude is None:
        return
    if latitude is None or longitude is None:
        raise ValueError(f"{label} latitude and longitude must both be provided or both be missing.")
    lat = _require_finite(latitude, f"{label} latitude")
    lon = _require_finite(longitude, f"{label} longitude")
    if not -90 <= lat <= 90:
        raise ValueError(f"{label} latitude must be between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError(f"{label} longitude must be between -180 and 180.")


def _price_component(offer: float, minimum: float, weight: float) -> float:
    """Scale price attractiveness from 0 to weight.

    Offers at the farmer minimum receive half the available points; an offer
    at or above twice the minimum receives the full points. This bounded,
    deterministic formula avoids unbounded scores while rewarding higher offers.
    """
    if minimum == 0:
        return weight if offer > 0 else 0.0
    ratio = offer / minimum
    return weight * min(1.0, max(0.0, ratio / 2.0))


def _distance_component(distance_km: float | None, weight: float) -> float:
    """Give full distance points at zero km and decay to zero at 100 km."""
    if distance_km is None:
        return 0.0
    return weight * max(0.0, 1.0 - min(distance_km, 100.0) / 100.0)


def match_buyer(farmer: FarmerListing, buyer: BuyerListing, weights: MatchingWeights = DEFAULT_WEIGHTS) -> MatchResult:
    """Evaluate one buyer against one farmer listing."""
    crop_match = farmer.crop.strip().casefold() == buyer.crop.strip().casefold()
    quantity_compatible = buyer.required_quantity_quintals <= farmer.quantity_quintals
    variety_match = farmer.variety.strip().casefold() == buyer.variety.strip().casefold()
    grade_match = farmer.grade.strip().casefold() == buyer.grade.strip().casefold()
    price_compatible = buyer.offered_price_per_quintal >= farmer.minimum_expected_price

    reasons: list[str] = []
    warnings: list[str] = []
    if crop_match:
        reasons.append("Crop matches")
    else:
        reasons.append("Crop does not match")
    if variety_match:
        reasons.append("Variety matches")
    else:
        reasons.append("Variety does not match")
    if grade_match:
        reasons.append("Grade matches")
    else:
        reasons.append("Grade does not match")
    if quantity_compatible:
        reasons.append("Buyer required quantity is within farmer quantity")
    else:
        reasons.append("Buyer required quantity exceeds farmer quantity")
    if price_compatible:
        reasons.append("Buyer offer meets or exceeds farmer minimum price")
    else:
        reasons.append("Buyer offer is below farmer minimum price")
    if buyer.pickup_available:
        reasons.append("Pickup available")
    else:
        reasons.append("Pickup not available")

    distance_km: float | None = None
    if farmer.latitude is not None and buyer.latitude is not None:
        distance_km = calculate_haversine_distance(
            farmer.latitude, farmer.longitude, buyer.latitude, buyer.longitude
        )
        reasons.append("Geographic distance calculated from available coordinates")
    else:
        warnings.append("Distance unavailable because farmer or buyer coordinates are missing")
        reasons.append("Distance unavailable because coordinates are missing")

    eligible = buyer.active and crop_match and quantity_compatible
    if not buyer.active:
        warnings.append("Buyer is inactive")
    if not crop_match:
        warnings.append("Buyer is not eligible because crop does not match")
    if not quantity_compatible:
        warnings.append("Buyer is not eligible because required quantity exceeds farmer quantity")

    score = 0.0
    if eligible:
        if variety_match:
            score += weights.variety_match
        if grade_match:
            score += weights.grade_match
        score += _price_component(buyer.offered_price_per_quintal, farmer.minimum_expected_price, weights.price_attractiveness)
        score += _distance_component(distance_km, weights.distance)
        if buyer.pickup_available:
            score += weights.pickup

    return MatchResult(
        buyer_id=buyer.buyer_id,
        buyer_name=buyer.buyer_name,
        eligible=eligible,
        match_score=round(score, 6),
        crop_match=crop_match,
        variety_match=variety_match,
        grade_match=grade_match,
        quantity_compatible=quantity_compatible,
        price_compatible=price_compatible,
        offered_price=buyer.offered_price_per_quintal,
        farmer_minimum_price=farmer.minimum_expected_price,
        distance_km=None if distance_km is None else round(distance_km, 6),
        pickup_available=buyer.pickup_available,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        synthetic_demo_data=buyer.synthetic_demo_data,
    )


def rank_buyers(farmer: FarmerListing, buyers: Sequence[BuyerListing], weights: MatchingWeights = DEFAULT_WEIGHTS) -> list[MatchResult]:
    """Return all buyer evaluations, with eligible buyers ranked first.

    Tie-breaking: eligible first, match score descending, offered price
    descending, buyer ID ascending. Ineligible buyers are retained rather than
    silently discarded so the caller can explain near-matches.
    """
    results = [match_buyer(farmer, buyer, weights) for buyer in buyers]
    return sorted(
        results,
        key=lambda r: (-int(r.eligible), -r.match_score, -r.offered_price, r.buyer_id),
    )


def load_farmer_listings(path: str | Path) -> list[FarmerListing]:
    """Load and validate farmer listings from CSV."""
    frame = pd.read_csv(path)
    required = {"listing_id", "crop", "variety", "grade", "quantity_quintals", "farmer_market", "minimum_expected_price", "latitude", "longitude", "created_date", "synthetic_demo_data"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Farmer CSV missing columns: {sorted(missing)}")
    return [FarmerListing(
        listing_id=str(row.listing_id), crop=str(row.crop), variety=str(row.variety), grade=str(row.grade),
        quantity_quintals=float(row.quantity_quintals), farmer_market=str(row.farmer_market),
        minimum_expected_price=float(row.minimum_expected_price), latitude=_optional_float(row.latitude),
        longitude=_optional_float(row.longitude), created_date=str(row.created_date),
        synthetic_demo_data=_parse_bool(row.synthetic_demo_data),
    ) for row in frame.itertuples(index=False)]


def load_buyer_listings(path: str | Path) -> list[BuyerListing]:
    """Load and validate buyer listings from CSV."""
    frame = pd.read_csv(path)
    required = {"buyer_id", "buyer_name", "crop", "variety", "grade", "required_quantity_quintals", "offered_price_per_quintal", "preferred_market", "latitude", "longitude", "pickup_available", "active", "synthetic_demo_data"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Buyer CSV missing columns: {sorted(missing)}")
    return [BuyerListing(
        buyer_id=str(row.buyer_id), buyer_name=str(row.buyer_name), crop=str(row.crop), variety=str(row.variety),
        grade=str(row.grade), required_quantity_quintals=float(row.required_quantity_quintals),
        offered_price_per_quintal=float(row.offered_price_per_quintal), preferred_market=str(row.preferred_market),
        latitude=_optional_float(row.latitude), longitude=_optional_float(row.longitude),
        pickup_available=_parse_bool(row.pickup_available), active=_parse_bool(row.active),
        synthetic_demo_data=_parse_bool(row.synthetic_demo_data),
    ) for row in frame.itertuples(index=False)]


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def match_to_dict(result: MatchResult) -> dict[str, object]:
    """Convert a match result into a dashboard-friendly dictionary."""
    return {
        "buyer_id": result.buyer_id,
        "buyer_name": result.buyer_name,
        "eligible": result.eligible,
        "match_score": result.match_score,
        "crop_match": result.crop_match,
        "variety_match": result.variety_match,
        "grade_match": result.grade_match,
        "quantity_compatible": result.quantity_compatible,
        "price_compatible": result.price_compatible,
        "offered_price": result.offered_price,
        "farmer_minimum_price": result.farmer_minimum_price,
        "distance_km": result.distance_km,
        "pickup_available": result.pickup_available,
        "reasons": list(result.reasons),
        "warnings": list(result.warnings),
        "synthetic_demo_data": result.synthetic_demo_data,
    }
