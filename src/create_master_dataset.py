import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

PRICE_FILE = Path("data/raw/paddy_prices.csv")
WEATHER_FILE = Path("data/raw/weather.csv")
MARKET_FILE = Path("data/raw/reference/markets.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "master_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading price data...")
prices = pd.read_csv(PRICE_FILE)

print("Loading weather data...")
weather = pd.read_csv(WEATHER_FILE)

print("Loading market reference data...")
markets = pd.read_csv(MARKET_FILE)


# ============================================================
# SHOW COLUMNS
# ============================================================

print("\nPrice columns:")
print(prices.columns.tolist())

print("\nWeather columns:")
print(weather.columns.tolist())

print("\nMarket columns:")
print(markets.columns.tolist())


# ============================================================
# STANDARDIZE DATE
# ============================================================

prices["date"] = pd.to_datetime(
    prices["date"],
    errors="coerce"
)

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
)


prices = prices.dropna(subset=["date"])
weather = weather.dropna(subset=["date"])


# ============================================================
# CLEAN MARKET NAMES
# ============================================================

prices["market"] = (
    prices["market"]
    .astype(str)
    .str.strip()
)

markets["market"] = (
    markets["market"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE DUPLICATE WEATHER DATES
# ============================================================

weather = (
    weather
    .sort_values("date")
    .drop_duplicates("date")
)


# ============================================================
# MERGE PRICE + WEATHER
# ============================================================

print("\nMerging price + weather...")

master = prices.merge(
    weather,
    on="date",
    how="left"
)


# ============================================================
# MERGE MARKET INFORMATION
# ============================================================

print("Merging market coordinates...")

master = master.merge(
    markets,
    on="market",
    how="left"
)


# ============================================================
# SORT
# ============================================================

master = master.sort_values(
    ["market", "date"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n========================================")
print("MASTER DATASET CREATED")
print("========================================")

print("\nShape:")
print(master.shape)

print("\nDate range:")
print(master["date"].min())
print("to")
print(master["date"].max())

print("\nMarkets in price data:")
print(sorted(master["market"].unique()))

print("\nMissing market coordinates:")
print(
    master[
        master["latitude"].isna() |
        master["longitude"].isna()
    ]["market"].unique()
)

print("\nMissing values:")
print(master.isna().sum())

print("\nFirst 10 rows:")
print(master.head(10))

print("\nSaved to:")
print(OUTPUT_FILE)