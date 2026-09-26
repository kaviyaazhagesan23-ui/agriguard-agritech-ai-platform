# Phase 3 — Feature Engineering Report

## Dataset

- Source rows: **3,024**
- Feature rows: **3,024**
- Source columns: **21**
- Feature dataset columns: **59**
- Date range: **2017-06-07 to 2025-03-17**
- Markets: **8**

## Target

- Target: `modal_price_rs_per_quintal`

## Grouping strategy

Lag and rolling price features are calculated independently within each `market`.

The source dataset contains multiple varieties and grades within markets. Because Phase 3 explicitly specifies per-market features, the implementation does not silently change the grouping to market + variety + grade.

Therefore `lag_1` means the previous available observation within the same market, not necessarily the previous calendar day.

## Leakage prevention

- Data is sorted chronologically within each market.
- Target lags use `groupby('market').shift(...)`.
- Rolling statistics are calculated after `shift(1)`.
- The current target is therefore excluded from its own rolling window.
- No future observations are used to calculate historical features.
- No random train/test split is performed.
- Insufficient history remains `NaN` rather than being fabricated.

## Price lag features

| `lag_1` | 8 | 0.26% |
| `lag_3` | 24 | 0.79% |
| `lag_7` | 56 | 1.85% |
| `lag_14` | 112 | 3.70% |
| `lag_30` | 240 | 7.94% |

## Rolling features

| `rolling_mean_7` | 8 | 0.26% |
| `rolling_mean_14` | 8 | 0.26% |
| `rolling_mean_30` | 8 | 0.26% |
| `rolling_std_7` | 16 | 0.53% |
| `rolling_std_14` | 16 | 0.53% |
| `rolling_std_30` | 16 | 0.53% |

## Calendar features

| `year` | 0 | 0.00% |
| `month` | 0 | 0.00% |
| `quarter` | 0 | 0.00% |
| `week_of_year` | 0 | 0.00% |
| `day_of_year` | 0 | 0.00% |
| `sin_month` | 0 | 0.00% |
| `cos_month` | 0 | 0.00% |

## Price dynamics

| `price_change_1d` | 8 | 0.26% |
| `price_change_7d` | 56 | 1.85% |

## Weather features

The original dataset contains same-date weather observations.

These are preserved, but they require care during future forecasting because actual future weather is not known.

Lagged weather features were therefore added using only previous observations.

| `temperature_mean` | 0 | 0.00% |
| `temperature_max` | 0 | 0.00% |
| `temperature_min` | 0 | 0.00% |
| `rainfall` | 0 | 0.00% |
| `wind_speed_max` | 0 | 0.00% |
| `et0` | 0 | 0.00% |

## Lagged weather features

| `temperature_mean_lag_1` | 8 | 0.26% |
| `temperature_mean_lag_3` | 24 | 0.79% |
| `temperature_mean_lag_7` | 56 | 1.85% |
| `temperature_max_lag_1` | 8 | 0.26% |
| `temperature_max_lag_3` | 24 | 0.79% |
| `temperature_max_lag_7` | 56 | 1.85% |
| `temperature_min_lag_1` | 8 | 0.26% |
| `temperature_min_lag_3` | 24 | 0.79% |
| `temperature_min_lag_7` | 56 | 1.85% |
| `rainfall_lag_1` | 8 | 0.26% |
| `rainfall_lag_3` | 24 | 0.79% |
| `rainfall_lag_7` | 56 | 1.85% |
| `wind_speed_max_lag_1` | 8 | 0.26% |
| `wind_speed_max_lag_3` | 24 | 0.79% |
| `wind_speed_max_lag_7` | 56 | 1.85% |
| `et0_lag_1` | 8 | 0.26% |
| `et0_lag_3` | 24 | 0.79% |
| `et0_lag_7` | 56 | 1.85% |

## Insufficient historical observations

Early observations in each market naturally have missing lag/rolling values because sufficient historical information does not exist.

These values are intentionally retained as `NaN` rather than filled using future information.

Model-specific preprocessing in a later phase will decide whether rows/features with insufficient history are dropped or handled using a training-only imputation strategy.

## Important modeling limitation

The feature dataset is prepared for time-series modeling. It does not perform model training, hyperparameter tuning, random splitting, or final evaluation.

Phase 4 is intentionally not included.

## Feature inventory

- `lag_1`
- `lag_3`
- `lag_7`
- `lag_14`
- `lag_30`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_30`
- `rolling_std_7`
- `rolling_std_14`
- `rolling_std_30`
- `year`
- `month`
- `quarter`
- `week_of_year`
- `day_of_year`
- `sin_month`
- `cos_month`
- `price_change_1d`
- `price_change_7d`
- `temperature_mean`
- `temperature_max`
- `temperature_min`
- `rainfall`
- `wind_speed_max`
- `et0`
- `temperature_mean_lag_1`
- `temperature_mean_lag_3`
- `temperature_mean_lag_7`
- `temperature_max_lag_1`
- `temperature_max_lag_3`
- `temperature_max_lag_7`
- `temperature_min_lag_1`
- `temperature_min_lag_3`
- `temperature_min_lag_7`
- `rainfall_lag_1`
- `rainfall_lag_3`
- `rainfall_lag_7`
- `wind_speed_max_lag_1`
- `wind_speed_max_lag_3`
- `wind_speed_max_lag_7`
- `et0_lag_1`
- `et0_lag_3`
- `et0_lag_7`
