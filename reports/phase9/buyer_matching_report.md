# Phase 9 — Farmer & Buyer Matching

## Purpose

PaddyWise AI can compare a farmer's paddy listing against buyer listings and identify the highest-scoring compatible synthetic demonstration buyers using transparent rule-based criteria.

This is a matching/ranking component, **not a real marketplace**.

## Data

The buyer and farmer CSV files in `data/raw/synthetic/` are synthetic demonstration data. Every record contains `synthetic_demo_data = True`.

No real people, real companies, or real buyer demand are represented.

## Matching methodology

### Eligibility

A buyer is eligible only when:

1. the buyer is active;
2. the crop matches; and
3. the buyer's required quantity is less than or equal to the farmer's available quantity.

Near-matches are retained in the output rather than silently discarded, so the application can explain why they were not eligible.

### Match components

| Component | Weight | Rule |
|---|---:|---|
| Variety match | 25 | Full points when variety matches |
| Grade match | 20 | Full points when grade matches |
| Price attractiveness | 30 | Bounded deterministic score based on offer/minimum price |
| Geographic distance | 15 | Score decays linearly to zero at 100 km |
| Pickup availability | 10 | Full points when pickup is available |

Crop match and quantity compatibility are eligibility requirements rather than scored components.

The maximum score is 100.

### Price attractiveness

For a positive farmer minimum price, the price component is:

`30 × min(1, max(0, (buyer_offer / farmer_minimum_price) / 2))`

Therefore an offer equal to the farmer minimum receives 15 points, while an offer at least twice the minimum receives the full 30 points.

### Distance

Phase 8's existing Haversine implementation is reused. The result is straight-line geographic distance, **not road distance**.

If either listing lacks coordinates, distance is unavailable and no distance value is fabricated. Matching continues using the remaining criteria.

### Ranking

Eligible buyers are ranked using:

1. eligibility first;
2. match score descending;
3. offered price descending;
4. buyer ID ascending.

The highest-ranked result means only **highest-scoring compatible buyer based on the configured criteria**. It is not a guarantee of real-world suitability or transaction success.

## Limitations

- Synthetic buyers are not real demand.
- Buyer prices are not live offers.
- Geographic distance is straight-line distance where coordinates exist, not road distance.
- The score is deterministic and rule-based, not machine-learned.
- Actual transaction suitability requires real buyer verification, current offers, logistics information, quality inspection, payment terms, and other business checks.
