# AgriGuard AgriTech AI Platform

**Smarter farming. Better decisions.**

An AI-powered paddy price forecasting and farmer-buyer marketplace platform designed to support agricultural decision-making.

## Overview

AgriGuard AgriTech AI Platform combines machine learning-based paddy price forecasting with a farmer-buyer marketplace. It provides price estimates, sell-versus-wait decision support, market comparisons, and transaction auditing.

## Key Features

- **AI Price Forecasting:** Uses a trained XGBoost regression model to estimate paddy prices.
- **Recursive Forecasting:** Generates daily future price estimates using previous predictions as inputs.
- **Sell vs Wait:** Compares estimated selling outcomes while accounting for storage and transportation costs.
- **Market Comparison:** Compares estimated prices across supported markets.
- **Farmer-Buyer Marketplace:** Supports farmer listings, buyer requests, orders, and messaging.
- **Buyer Matching:** Helps connect buyers with relevant farmer listings.
- **Blockchain Transaction Audit:** Uses SHA-256 hashing to provide tamper-evident verification for transaction records.
- **Interactive Dashboard:** Provides a Streamlit interface for farmers and buyers.

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- SQLAlchemy
- MySQL
- SHA-256

## Machine Learning Model

The project uses an XGBoost regression model trained on historical paddy market data. The model was evaluated against a Naive Last Value baseline and a Random Forest model.

### Phase 5 Model Evaluation

Evaluation was performed on 67 historical test observations.

| Model | MAE (₹/quintal) | RMSE (₹/quintal) | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive Last Value | 248.36 | 412.21 | 9.56% | 0.1280 |
| Random Forest | 150.12 | 322.51 | 5.11% | 0.4662 |
| **XGBoost** | **138.03** | **322.32** | **4.62%** | **0.4668** |

XGBoost was selected based on the lowest test MAE, with RMSE used as the secondary criterion.

### Error Analysis

- Mean prediction error: -₹45.55, indicating overall under-prediction.
- 95th percentile absolute error: ₹501.88 per quintal.
- Errors increased for high-price observations relative to the overall test set.

### Model Limitations

- Evaluation is based on 67 historical test observations.
- Historical test performance does not guarantee future forecasting performance.
- Error analysis does not establish causal relationships.
- Performance may vary across markets, varieties, and changing market conditions.

## Data Disclosure

The model was developed using historical paddy market data. Synthetic current-market and marketplace inputs are used for demonstration where live data is unavailable. Synthetic future weather inputs are also used in the demonstration forecast. These are not official live market prices or live weather observations.

## Blockchain Audit

The transaction audit module uses SHA-256 hashing and a local blockchain structure to help detect changes to recorded transactions. It is an audit demonstration, not a public blockchain network.

## Testing

The project passed **151 automated tests** in the latest local test run.

Run the tests with:

```bash
python -m pytest
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kaviyaazhagesan23-ui/agriguard-agritech-ai-platform.git
cd agriguard-agritech-ai-platform
```

Install the project dependencies:

```bash
pip install -r requirements_marketplace.txt
```

Configure the database connection in a local `.env` file using `.env.example` as a template. Do not commit your actual `.env` file or credentials.

Initialize the database:

```bash
python -m src.database.init_db
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
agriguard-agritech-ai-platform/
├── app.py
├── src/
│   ├── auth/
│   ├── blockchain/
│   ├── database/
│   ├── marketplace/
│   ├── forecasting.py
│   ├── current_price_estimator.py
│   ├── decision_engine.py
│   └── matching.py
├── data/
│   ├── current/
│   ├── processed/
│   └── raw/
├── models/
│   └── phase4/
├── reports/
├── tests/
├── .env.example
├── .gitignore
└── requirements_marketplace.txt
```

## Project Status

Developed as an end-to-end AI and agricultural technology portfolio project.

## Author

**Kaviya Azhagesan**

GitHub: [kaviyaazhagesan23-ui](https://github.com/kaviyaazhagesan23-ui)
