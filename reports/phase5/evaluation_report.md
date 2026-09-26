# Phase 5 — Model Evaluation & Error Analysis

This report was generated directly from the actual Phase 4 predictions and model artifacts.

No metrics were manually entered or fabricated.

## Model Comparison

| Model | N | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| Naive Last Value | 67 | 248.3582 | 412.2056 | 9.5599% | 0.1280 |
| Random Forest | 67 | 150.1150 | 322.5109 | 5.1070% | 0.4662 |
| XGBoost | 67 | 138.0253 | 322.3174 | 4.6248% | 0.4668 |

## Model Selection

Selected model based on the lowest valid test MAE with RMSE used as the secondary criterion: **XGBoost**.

This selection is evidence-based and does not assume that XGBoost must outperform the baselines.

## Error Analysis

The following analyses were generated:

- Actual vs predicted values
- Residual analysis
- Error distribution
- Market-wise error
- Year-wise error
- Variety-wise error where sufficient observations exist
- Baseline comparison
- Feature importance
- XGBoost SHAP explanation where technically available

## Model Weaknesses

- Average prediction error is negative (-45.55), indicating overall under-prediction.
- The 95th percentile absolute error is 501.88 ₹/quintal.
- Error increases for high-price observations relative to the overall test set.

## Important Limitations

- Error analysis does not prove causality.
- High error in a market or variety may be related to limited observations, distribution changes, or unobserved factors.
- Model performance on the historical test period does not guarantee future forecasting performance.
- Forecasting has not been implemented in Phase 5.