from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from src.database.db import init_database, make_engine, session_scope
from src.database.models import FarmerListing, Market, User, BuyerRequest
from src.auth.security import hash_password

MARKETS = {
    "Budalur": (10.7968, 78.9769, True, None),
    "Kumbakonam": (10.9617, 79.3880, True, None),
    "Orathanadu": (10.6280, 79.2531, True, None),
    "Papanasam": (10.9252, 79.2708, True, None),
    "Pattukottai": (10.4250, 79.3140, True, None),
    "Thanjavur": (10.7870, 79.1378, True, None),
    "Vallam": (10.7140, 79.7020, True, None),
    "Thiruppananthal": (None, None, False, "29, Meela Veedi, Thiruppananthal 612504"),
}


def seed_markets(session):
    for name, (lat, lon, verified, address) in MARKETS.items():
        row = session.scalar(select(Market).where(Market.name == name))
        if row is None:
            session.add(Market(name=name, district="Thanjavur", latitude=lat, longitude=lon, coordinate_verified=verified, address=address))


def seed_csv_farmers(session, path: Path):
    if not path.exists():
        return 0
    user = session.scalar(select(User).where(User.email == "demo.farmer@paddywise.local"))
    if user is None:
        user = User(name="Demo Farmer", email="demo.farmer@paddywise.local", phone=None, password_hash=hash_password("DemoFarmer123!"), role="farmer", is_active=True)
        session.add(user); session.flush()
    existing = session.scalar(select(FarmerListing.id).limit(1))
    if existing is not None:
        return 0
    count = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            session.add(FarmerListing(farmer_id=user.id, crop=row.get("crop", "Paddy"), variety=row["variety"], grade=row["grade"], quantity_quintals=Decimal(row["quantity_quintals"]), minimum_price=Decimal(row.get("minimum_expected_price", row.get("minimum_price", "0"))), market=row["farmer_market"], latitude=Decimal(row["latitude"]) if row.get("latitude") else None, longitude=Decimal(row["longitude"]) if row.get("longitude") else None, available_from=date.today()))
            count += 1
    return count



def seed_csv_buyers(session, path: Path):
    if not path.exists():
        return 0
    existing = session.scalar(select(BuyerRequest.id).limit(1))
    if existing is not None:
        return 0
    count = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buyer_key = row.get("buyer_id", f"demo-buyer-{count}")
            email = f"{buyer_key}@paddywise.demo".lower().replace(" ", "-")
            user = session.scalar(select(User).where(User.email == email))
            if user is None:
                user = User(name=row.get("buyer_name", buyer_key), email=email, password_hash=hash_password("DemoBuyer123!"), role="buyer", is_active=str(row.get("active", "True")).lower() == "true")
                session.add(user); session.flush()
            lat = Decimal(row["latitude"]) if row.get("latitude") else None
            lon = Decimal(row["longitude"]) if row.get("longitude") else None
            session.add(BuyerRequest(buyer_id=user.id, crop=row.get("crop", "Paddy"), variety=row["variety"], grade=row["grade"], quantity_quintals=Decimal(row["required_quantity_quintals"]), offered_price=Decimal(row["offered_price_per_quintal"]), preferred_market=row.get("preferred_market"), pickup_location=row.get("preferred_market"), latitude=lat, longitude=lon, required_by_date=date.today(), status="PENDING" if user.is_active else "CANCELLED"))
            count += 1
    return count


def main(csv_path: str | None = None, buyer_csv_path: str | None = None):
    engine = make_engine(); init_database(engine)
    with session_scope(engine) as session:
        seed_markets(session)
        seeded = seed_csv_farmers(session, Path(csv_path)) if csv_path else 0
        seeded_buyers = seed_csv_buyers(session, Path(buyer_csv_path)) if buyer_csv_path else 0
    print(f"Database ready. Seeded farmer listings: {seeded}; buyer requests: {seeded_buyers}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--farmer-csv")
    parser.add_argument("--buyer-csv")
    args = parser.parse_args()
    main(args.farmer_csv, args.buyer_csv)
