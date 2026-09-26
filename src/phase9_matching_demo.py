"""Readable Phase 9 demonstration using the expanded synthetic marketplace."""
from pathlib import Path

from src.matching import load_buyer_listings, load_farmer_listings, rank_buyers


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    farmer_path = root / "data" / "raw" / "synthetic" / "farmer_listings.csv"
    buyer_path = root / "data" / "raw" / "synthetic" / "buyer_listings.csv"

    farmers = load_farmer_listings(farmer_path)
    buyers = load_buyer_listings(buyer_path)
    farmer = farmers[0]
    results = rank_buyers(farmer, buyers)
    eligible = [result for result in results if result.eligible]
    ineligible = [result for result in results if not result.eligible]

    print("=" * 72)
    print("PADDYWISE AI — PHASE 9 FARMER & BUYER MATCHING")
    print("=" * 72)
    print("SYNTHETIC DEMONSTRATION DATA — NOT REAL BUYERS OR LIVE DEMAND")
    print(
        f"Farmer listing: {farmer.listing_id} | {farmer.variety} | "
        f"{farmer.grade} | {farmer.quantity_quintals:g} quintals"
    )
    print()
    print(f"Total buyers evaluated: {len(results)}")
    print(f"Eligible buyers: {len(eligible)}")
    print(f"Ineligible buyers: {len(ineligible)}")
    print("Top 10 eligible matching buyers:")

    for rank, result in enumerate(eligible[:10], start=1):
        print(
            f"{rank:>2}. {result.buyer_id} | {result.buyer_name} | "
            f"score={result.match_score:.2f} | offer=₹{result.offered_price:,.0f}/quintal | "
            f"distance={result.distance_km if result.distance_km is not None else 'N/A'} km"
        )

    if eligible:
        top = eligible[0]
        print()
        print("Highest-scoring synthetic demonstration match:")
        print(f"{top.buyer_id} — {top.buyer_name} (score {top.match_score:.2f})")

    print()
    print("Ineligible/near-match summary:")
    warning_counts: dict[str, int] = {}
    for result in ineligible:
        for warning in result.warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    for warning, count in sorted(warning_counts.items()):
        print(f"- {warning}: {count}")


if __name__ == "__main__":
    main()
