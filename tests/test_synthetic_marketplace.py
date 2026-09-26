from pathlib import Path

import pandas as pd

from src.generate_synthetic_marketplace import (
    BUYER_COLUMNS,
    FARMER_COLUMNS,
    RANDOM_SEED,
    validate_marketplace_data,
    load_source_data,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generated_farmer_schema_and_count():
    frame = pd.read_csv(ROOT / "data/raw/synthetic/farmer_listings.csv")
    assert len(frame) == 500
    assert list(frame.columns) == FARMER_COLUMNS
    assert frame["listing_id"].is_unique
    assert frame["synthetic_demo_data"].eq(True).all()


def test_generated_buyer_schema_and_count():
    frame = pd.read_csv(ROOT / "data/raw/synthetic/buyer_listings.csv")
    assert len(frame) == 1000
    assert list(frame.columns) == BUYER_COLUMNS
    assert frame["buyer_id"].is_unique
    assert frame["synthetic_demo_data"].eq(True).all()


def test_generated_values_are_from_project_sources():
    source = load_source_data(ROOT)
    farmers = pd.read_csv(ROOT / "data/raw/synthetic/farmer_listings.csv")
    buyers = pd.read_csv(ROOT / "data/raw/synthetic/buyer_listings.csv")
    validate_marketplace_data(farmers, buyers, source)
    assert set(farmers["variety"]).issubset(set(source.varieties))
    assert set(buyers["variety"]).issubset(set(source.varieties))
    assert set(farmers["grade"]).issubset(set(source.grades))
    assert set(buyers["grade"]).issubset(set(source.grades))
    assert set(farmers["farmer_market"]).issubset(set(source.markets))
    assert set(buyers["preferred_market"]).issubset(set(source.markets))


def test_seed_is_fixed():
    assert RANDOM_SEED == 42
