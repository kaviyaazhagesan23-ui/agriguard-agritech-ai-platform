# Phase 5 — Model Selection Report

## Selection Principle

The final model was selected using the actual Phase 4 test predictions.

Primary criterion: lowest test MAE.
Secondary criterion: lowest test RMSE when MAE values are tied or effectively indistinguishable.

### Selected Model: **XGBoost**

- Test observations: 67
- Test MAE: 138.0253
- Test RMSE: 322.3174
- Test MAPE: 4.6248%
- Test R²: 0.4668

## Baseline Comparison

- XGBoost has lower test MAE than Naive Last Value by 110.3329 ₹/quintal.

## Important Decision Rule

XGBoost is not automatically selected. If a baseline has lower test error, the baseline remains the evidence-supported selection.

Forecasting implementation is intentionally excluded from Phase 5.