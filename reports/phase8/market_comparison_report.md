# Phase 8 — Market Comparison & Net Revenue Analysis

## Purpose

Phase 8 adds a transparent market-comparison layer to PaddyWise AI. It compares available markets using the actual historical modal-price data and estimated transportation cost instead of comparing price alone.

The core calculation is:

```text
gross_revenue = quantity_quintals × market_price
net_revenue = gross_revenue − estimated_transportation_cost
```

The module is decision-support only. It does not guarantee a profit or a particular market outcome.

## Methodology

### Market prices

The module uses the project's `modal_price_rs_per_quintal` field from the processed master dataset.

For like-for-like comparison, the example filters to:

- Variety: `B P T`
- Grade: `Local`
- Current market: `Vallam`
- Quantity: `100` quintals

For each eligible market, the latest available observation for that variety/grade is used. The example is historical because the project dataset ends on 2025-03-17; these are not live market quotations.

Markets without a matching B P T / Local observation are not assigned a fabricated price.

### GPS coordinates

The implementation uses the project's verified market-yard coordinates for:

- Budalur
- Kumbakonam
- Orathanadu
- Papanasam
- Pattukottai
- Thanjavur
- Vallam

Thiruppananthal intentionally has no coordinate because no reliable verified market-yard coordinate is available.

### Haversine distance

Distance is calculated with the Haversine formula using an Earth radius of approximately 6,371 km.

**Haversine distance represents straight-line geographic distance and is not equivalent to actual road distance.**

No external routing API is used and no road distance is invented.

### Transportation-cost assumption

The initial configurable assumption is:

```text
estimated_transportation_cost = distance_km × transport_cost_per_km
```

The example uses:

```text
transport_cost_per_km = ₹20
```

This is an estimated transportation cost, **not a real-time logistics quote**.

The rate is exposed as a configuration parameter so it can be replaced later with a validated logistics-cost model.

### Revenue

For 100 quintals:

```text
gross_revenue = 100 × market_price
net_revenue = gross_revenue − estimated_transportation_cost
```

The comparison also reports the net-revenue difference and percentage difference versus the current market.

## Actual-data example

The following example uses the project's actual processed price data for B P T / Local and a farmer location at the verified Vallam coordinates (10.7140, 79.7020).

| Market | Latest modal price | Price date | Geographic distance (km) | Estimated transport | Estimated net revenue |
|---|---:|---|---:|---:|---:|
| Budalur | ₹2,400 | 2025-03-06 | 79.744 | ₹1,594.89 | ₹238,405.11 |
| Kumbakonam | ₹2,300 | 2025-03-17 | 43.984 | ₹879.68 | ₹229,120.32 |
| Orathanadu | ₹2,550 | 2025-02-28 | 49.976 | ₹999.51 | ₹254,000.49 |
| Pattukottai | ₹2,500 | 2025-03-15 | 53.211 | ₹1,064.22 | ₹248,935.78 |
| Thanjavur | ₹2,500 | 2025-03-17 | 62.167 | ₹1,243.35 | ₹248,756.65 |
| Vallam | ₹2,300 | 2025-01-09 | 0.000 | ₹0.00 | ₹230,000.00 |

These values are calculations from the actual dataset and the stated transport assumption; they are not manually entered prices.

The B P T / Local example does not contain matching observations for Papanasam or Thiruppananthal, so those markets are not given fabricated prices.

## Why price alone is insufficient

The same actual-data scenario also illustrates why transportation must remain configurable.

At the example rate of ₹20/km, Orathanadu's historical modal price is ₹2,550/quintal and its estimated net revenue for 100 quintals is ₹254,000.49.

If the configurable transport rate is increased to ₹600/km, without changing the actual market prices or coordinates, Orathanadu's estimated net revenue becomes ₹225,014.62 while Vallam remains ₹230,000 because the farmer is located at Vallam's coordinates.

This is a sensitivity illustration, not a claim that ₹600/km is a real logistics quote. It demonstrates that the market with the higher price does not necessarily have the higher net revenue once transport cost is included.

## Missing coordinates

Thiruppananthal is represented explicitly with missing coordinates:

```text
latitude = None
longitude = None
```

The module does not invent a coordinate.

When a market lacks coordinates:

- `coordinates_available` is `False`
- `distance_km` is `None`
- `transportation_cost` is `None`
- distance-based net revenue is not calculated
- the market is not silently assigned a fake distance

This prevents false precision in the dashboard.

## Forecast comparison

Phase 8 supports an optional `forecast_prices` mapping in the result object.

No forecast values are fabricated for markets that do not have real Phase 6 forecast output. The current Phase 6 workflow demonstrated in the project is market/variety/grade-specific, so Phase 8's verified implementation focuses on current-price comparison and accepts forecasts only when explicitly supplied by an upstream real forecast source.

## Files

- `src/market_comparison.py`
- `src/phase8_market_comparison_demo.py`
- `tests/test_market_comparison.py`
- `reports/phase8/market_comparison_report.md`
- `reports/phase8/market_comparison_example.csv`

## Limitations

1. Haversine distance is straight-line geographic distance, not driving distance.
2. Transportation cost is a configurable estimate, not a real-time logistics quote.
3. Historical modal prices are not live prices.
4. A like-for-like variety/grade filter may leave some markets without comparable observations.
5. The module does not infer or fabricate missing coordinates or future market forecasts.
