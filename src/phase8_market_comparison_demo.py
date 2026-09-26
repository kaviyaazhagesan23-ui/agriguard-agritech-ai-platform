"""Runnable Phase 8 market-comparison demonstration using project data."""

from pathlib import Path

from src.market_comparison import run_example


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data" / "processed" / "master_dataset.csv"
    report_dir = project_root / "reports" / "phase8"
    report_dir.mkdir(parents=True, exist_ok=True)

    result = run_example(
        dataset,
        current_market="Vallam",
        quantity_quintals=100,
        farmer_latitude=10.7140,
        farmer_longitude=79.7020,
        variety="B P T",
        grade="Local",
        transport_cost_per_km=20.0,
    )

    output_file = report_dir / "market_comparison_example.csv"
    result.to_csv(output_file, index=False)

    print("=" * 72)
    print("PADDYWISE AI — PHASE 8 MARKET COMPARISON")
    print("=" * 72)
    print("Source: data/processed/master_dataset.csv")
    print("Price definition: latest available modal price for B P T / Local")
    print("Quantity: 100 quintals")
    print("Farmer location: 10.7140, 79.7020")
    print("Transport estimate: ₹20 per geographic km")
    print()
    print(result.to_string(index=False))
    print()
    print(f"Example CSV written to: {output_file}")


if __name__ == "__main__":
    main()
