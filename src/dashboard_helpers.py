from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.decision_engine import DecisionInputs, calculate_decision
from src.market_comparison import MARKET_COORDINATES, compare_markets, results_to_dataframe
from src.matching import BuyerListing, FarmerListing, load_buyer_listings, load_farmer_listings, rank_buyers
from src.blockchain.verification import load_or_create_blockchain

APP_TITLE = "PaddyWise AI"
ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected project data paths exist: " + ", ".join(map(str, paths)))


def load_price_data_safe() -> pd.DataFrame:
    path = _first_existing(
        ROOT / "data" / "processed" / "master_dataset_clean.csv",
        ROOT / "data" / "processed" / "master_dataset.csv",
        ROOT / "data" / "processed" / "feature_dataset.csv",
    )
    df = pd.read_csv(path)
    required = {"date", "market", "variety", "grade", "modal_price_rs_per_quintal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price dataset is missing required columns: {sorted(missing)}")
    return df


def load_farmer_listings_safe() -> list[FarmerListing]:
    return load_farmer_listings(ROOT / "data" / "raw" / "synthetic" / "farmer_listings.csv")


def load_buyers_safe() -> list[BuyerListing]:
    return load_buyer_listings(ROOT / "data" / "raw" / "synthetic" / "buyer_listings.csv")


def latest_price_for_selection(df: pd.DataFrame, market: str, variety: str, grade: str) -> Optional[float]:
    subset = df[(df["market"].astype(str) == str(market)) & (df["variety"].astype(str) == str(variety)) & (df["grade"].astype(str) == str(grade))].copy()
    if subset.empty:
        return None
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset = subset.dropna(subset=["date", "modal_price_rs_per_quintal"]).sort_values("date")
    if subset.empty:
        return None
    return float(subset.iloc[-1]["modal_price_rs_per_quintal"])


def run_decision(quantity: float, market: str, current_price: float, forecast_price: Optional[float], waiting_days: int, transport: float, storage: float, buyer_offer: Optional[float]):
    inputs = DecisionInputs(
        quantity_quintals=quantity,
        current_market=market,
        current_modal_price=current_price,
        forecast_price=forecast_price,
        expected_waiting_days=waiting_days,
        transportation_cost=transport,
        storage_cost_per_day=storage,
        buyer_offer=buyer_offer,
    )
    return calculate_decision(inputs)


def market_coordinates(market: str) -> tuple[float, float] | None:
    coords = MARKET_COORDINATES.get(market)
    if not coords or coords[0] is None or coords[1] is None:
        return None
    return float(coords[0]), float(coords[1])


def run_market_comparison(quantity: float, current_market: str, current_price: float, price_data: pd.DataFrame, farmer_latitude: float, farmer_longitude: float, transport_rate: float) -> pd.DataFrame:
    results = compare_markets(
        quantity_quintals=quantity,
        current_market=current_market,
        current_market_price=current_price,
        market_price_data=price_data,
        farmer_latitude=farmer_latitude,
        farmer_longitude=farmer_longitude,
        transport_cost_per_km=transport_rate,
    )
    return results_to_dataframe(results)


def run_matching(farmer: FarmerListing, buyers: list[BuyerListing]):
    return rank_buyers(farmer, buyers)


def run_forecast_command(market: str, variety: str, grade: str, horizon: int) -> Path:
    output_dir = ROOT / "reports" / "phase6" / "forecasts"
    before = {p.resolve() for p in output_dir.glob("*.csv")} if output_dir.exists() else set()
    command = [sys.executable, "-m", "src.forecasting", "--market", market, "--variety", variety, "--grade", grade, "--horizon", str(horizon)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "forecasting command returned a non-zero status"
        raise RuntimeError(detail)
    if not output_dir.exists():
        raise FileNotFoundError("Phase 6 forecast output directory was not created.")
    candidates = [p for p in output_dir.glob("*.csv") if p.resolve() not in before]
    if not candidates:
        safe = lambda s: str(s).replace(" ", "_")
        exact = output_dir / f"forecast_{safe(market)}_{safe(variety)}_{safe(grade)}_{horizon}d.csv"
        candidates = [exact] if exact.exists() else list(output_dir.glob(f"*_{horizon}d.csv"))
    if not candidates:
        raise FileNotFoundError("Phase 6 completed but no forecast CSV could be located.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_forecast_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "predicted_price_rs_per_quintal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Forecast output is missing columns: {sorted(missing)}")
    return df


def load_blockchain_safe():
    path = ROOT / "data" / "blockchain" / "paddywise_blockchain.json"
    return load_or_create_blockchain(path)
