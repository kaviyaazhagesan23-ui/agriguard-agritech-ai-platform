# 🌾 AgriGuard – AI-Powered Smart Agriculture Platform

**AgriGuard** is an AI-powered agriculture platform designed to support farmers with data-driven paddy price forecasting, market comparison, intelligent selling decisions, and a digital marketplace connecting farmers and buyers.

The platform combines machine learning, weather data analysis, market insights, and blockchain-inspired transaction verification to create a smarter and more transparent agricultural ecosystem.

## 🚀 Key Features

* 📈 **Paddy Price Forecasting:** Uses machine learning and XGBoost to forecast paddy prices based on historical market data.
* 🤖 **AI-Based Selling Decisions:** Provides data-driven Sell/Wait recommendations to help farmers plan their sales.
* 🏪 **Market Price Comparison:** Helps farmers compare prices across different agricultural markets.
* 💰 **Revenue Estimation:** Estimates potential revenue to support informed selling decisions.
* 🌦️ **Weather Data Integration:** Incorporates weather information into the agricultural data pipeline.
* 👨‍🌾 **Farmer Marketplace:** Provides a digital platform for farmers to connect with potential buyers.
* 🤝 **Buyer Matching:** Supports connections between farmers and buyers based on marketplace information.
* 🔐 **Blockchain-Inspired Verification:** Uses SHA-256 hashing for local transaction integrity verification.
* 📊 **Interactive Dashboard:** Presents forecasts, market insights, and agricultural information through a Streamlit interface.

## 🧠 Machine Learning

AgriGuard includes a machine learning pipeline for paddy price forecasting. Multiple models are evaluated to support model selection.

### 🤖 Models Included

* Naive Last Value
* Moving Average
* Random Forest
* XGBoost

### 📊 Model Evaluation

The Phase 5 evaluation selected XGBoost based on the lowest test-set MAE in the reported comparison.

| Metric | XGBoost |
| ------ | ------: |
| MAE    |  138.03 |
| RMSE   |  322.32 |
| MAPE   |   4.62% |
| R²     |  0.4668 |

These metrics summarize the model's performance on the evaluated test observations.

## 🛠️ Technology Stack

| Category                | Technologies          |
| ----------------------- | --------------------- |
| 💻 Programming Language | Python                |
| 🤖 Machine Learning     | XGBoost, Scikit-learn |
| 📊 Data Processing      | Pandas, NumPy         |
| 📉 Data Visualization   | Matplotlib            |
| 🌐 Web Application      | Streamlit             |
| 🗄️ Database            | MySQL                 |
| 🔐 Data Security        | SHA-256               |
| ⚙️ Development Tools    | VS Code, Git, GitHub  |

## 🏗️ Project Architecture

```text
AgriGuard
│
├── app.py
│
├── data/
│   ├── current/
│   └── processed/
│
├── models/
│   └── phase4/
│
├── reports/
│   ├── phase4/
│   └── phase5/
│
├── src/
│   ├── auth/
│   ├── blockchain/
│   ├── database/
│   ├── marketplace/
│   ├── current_price_estimator.py
│   ├── forecasting.py
│   ├── decision_engine.py
│   ├── market_comparison.py
│   └── matching.py
│
└── tests/
```

## 📊 Data Pipeline

1. 📥 Collect historical paddy market price data.
2. 🌦️ Integrate weather information.
3. 🧹 Clean and prepare the dataset.
4. ⚙️ Perform feature engineering.
5. 🤖 Train and evaluate machine learning models.
6. 🔮 Generate paddy price forecasts.
7. 💡 Produce selling recommendations and market comparisons.
8. 📊 Present insights through the farmer dashboard.

## 💻 Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kaviyaazhagesan23-ui/agriguard-agritech-ai-platform.git
cd agriguard-agritech-ai-platform
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file using the project's environment variable template, if available. Add your local database configuration.

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

## 🧪 Testing

Run the automated test suite using:

```bash
pytest
```

The project includes tests for its application components and core functionality.

## 🌱 Project Goals

* 🌾 Make agricultural market information easier to access.
* 💡 Help farmers make informed selling decisions.
* 🤝 Connect farmers and buyers through a digital marketplace.
* 🧠 Apply AI and data analytics to agricultural challenges.
* 🔐 Encourage transparency through transaction verification.

## 🔮 Future Enhancements

* 🌐 Integration with additional live agricultural market data.
* 📈 Improved forecasting with expanded datasets.
* 📱 Mobile-friendly farmer experience.
* 🛒 Enhanced buyer and farmer marketplace features.
* ⛓️ Further development of blockchain-based transaction records.

## 👩‍💻 Author

**Kaviya Azhagesan**
🎓 B.Tech Computer Science and Engineering (Artificial Intelligence and Machine Learning)
🏫 SRM Institute of Science and Technology, Tiruchirappalli

---

⭐ **AgriGuard – Empowering Agriculture Through AI and Technology** 🌾🚀
