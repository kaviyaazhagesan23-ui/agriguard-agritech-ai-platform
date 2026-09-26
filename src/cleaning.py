import pandas as pd

INPUT_FILE = "data/processed/master_dataset.csv"
OUTPUT_FILE = "data/processed/master_dataset_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Remove * from column names
df.columns = df.columns.str.replace("*", "", regex=False).str.strip()

# Remove * from text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.replace("*", "", regex=False).str.strip()

# Convert numeric columns
numeric_cols = [
    "min_price_rs_per_quintal",
    "max_price_rs_per_quintal",
    "modal_price_rs_per_quintal",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
    "latitude",
    "longitude"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sort
df = df.sort_values(["market", "date"]).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nMarkets:")
print(df["market"].value_counts())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nSaved:", OUTPUT_FILE)




print(df.shape)
print(df["market"].nunique())
print(df["market"].unique())
print(df["date"].min(), df["date"].max())
print(df.isna().sum())


print(df[df["latitude"].isna() | df["longitude"].isna()]["market"].unique())