# Phase 9 Synthetic Marketplace Data Generation Report

> All farmer and buyer records are synthetic demonstration data generated from distributions derived from the project's historical market data. They do not represent real farmers, real buyers, real companies, or live market demand.

## Generation summary
- Farmer records: **500**
- Buyer records: **1000**
- Random seed: **42**
- Master dataset: `data\processed\master_dataset.csv`
- Market reference: `data/raw/reference/markets.csv` if present; otherwise market coordinates were derived from `data\processed\master_dataset.csv`.

## Source distributions
- Varieties used (9): 1001, ADT 37, ADT 43, B P T, Culture, I.R. 49, Other, Paddy, Ponni
- Grades used (2): FAQ, Local
- Markets used (8): Budalur, Kumbakonam, Orathanadu, Papanasam, Pattukottai, Thanjavur, Thiruppananthal, Vallam
- Farmer minimum prices: empirical modal-price observations conditioned by variety/grade/market where at least five observations exist, with a narrow 0.94–1.02 synthetic negotiation factor and clipping to the empirical 1st–99th percentile.
- Buyer offers: empirical modal-price observations conditioned by variety/grade/market where available, with below/around/above factors of 0.88–0.98, 0.98–1.03, and 1.03–1.12 respectively; clipped to the empirical 1st–99th percentile.
- Quantities: documented synthetic small/medium/large lot bands. Farmer bands are 5–40 (40%), 40–100 (50%), 100–200 (10%); buyer bands are 5–40 (35%), 40–100 (50%), 100–250 (15%).
- Listing dates: synthetic dates uniformly sampled from 2026-01-01 through 2026-06-30.

## Coordinates and status
- Missing farmer coordinates: **5.60%**
- Missing buyer coordinates: **6.50%**
- Buyer active: **907 (90.70%)**
- Buyer inactive: **93 (9.30%)**
- Pickup available: **697 (69.70%)**
- Pickup unavailable: **303 (30.30%)**
- Coordinates are copied from verified existing market data. Missing coordinates remain blank; no synthetic coordinates are invented.

## Matching scenarios
The generator deliberately creates variation in product definitions, prices, quantities, activity, pickup availability, and coordinates so the unchanged Phase 9 matching engine can exercise exact matches, grade/variety mismatches, price-compatible and price-incompatible cases, quantity incompatibility, inactive buyers, pickup differences, same/different markets, and missing-coordinate cases.

## Limitations
- Quantities and listing dates are synthetic assumptions because the historical master dataset does not contain farmer listing quantities or marketplace listing dates.
- Buyer/farmer records are not real supply or demand observations and must not be presented as live marketplace data.
- Price realism is distributional rather than a claim that any synthetic buyer would actually offer a specific price.
