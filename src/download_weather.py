import time
from pathlib import Path
from datetime import date

import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 10.7870
LONGITUDE = 79.1378

START_DATE = "2017-06-07"
END_DATE = "2025-03-17"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "weather.csv"

# Download one year per request
CHUNK_YEARS = 1

# Retry settings
MAX_RETRIES = 4
TIMEOUT_SECONDS = 180


DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_year_chunks(start_date, end_date):
    """
    Generate date ranges one calendar year at a time.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    current_start = start

    while current_start <= end:
        current_end = min(
            pd.Timestamp(
                year=current_start.year,
                month=12,
                day=31
            ),
            end
        )

        yield (
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        )

        current_start = current_end + pd.Timedelta(days=1)


def download_chunk(chunk_start, chunk_end):
    """
    Download one date chunk with retries.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": chunk_start,
        "end_date": chunk_end,
        "daily": ",".join(DAILY_VARIABLES),

        # Automatically resolve the local timezone
        "timezone": "auto",

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Downloading {chunk_start} to {chunk_end} "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=TIMEOUT_SECONDS
            )

            response.raise_for_status()

            result = response.json()

            if result.get("error"):
                raise RuntimeError(result.get("reason", "Unknown API error"))

            if "daily" not in result:
                raise RuntimeError("The API response does not contain daily data.")

            return result["daily"]

        except Exception as error:
            print(f"Download failed: {error}")

            if attempt < MAX_RETRIES:
                wait_seconds = attempt * 10
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                raise RuntimeError(
                    f"Could not download {chunk_start} to {chunk_end}"
                ) from error


# ============================================================
# MAIN DOWNLOAD PROCESS
# ============================================================

def main():
    all_chunks = []

    print("Starting historical weather download...")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print()

    date_chunks = list(generate_year_chunks(START_DATE, END_DATE))

    for index, (chunk_start, chunk_end) in enumerate(date_chunks, start=1):
        print(f"\nChunk {index}/{len(date_chunks)}")

        daily = download_chunk(chunk_start, chunk_end)

        chunk_df = pd.DataFrame({
            "date": daily["time"],
            "temperature_mean": daily["temperature_2m_mean"],
            "temperature_max": daily["temperature_2m_max"],
            "temperature_min": daily["temperature_2m_min"],
            "rainfall": daily["precipitation_sum"],
            "wind_speed_max": daily["wind_speed_10m_max"],
            "et0": daily["et0_fao_evapotranspiration"],
        })

        all_chunks.append(chunk_df)

        # Small pause between API requests
        time.sleep(2)

    # Combine all yearly chunks
    weather_df = pd.concat(all_chunks, ignore_index=True)

    # Clean and sort
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df.drop_duplicates(subset="date")
    weather_df = weather_df.sort_values("date")
    weather_df = weather_df.reset_index(drop=True)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUTPUT_FILE, index=False)

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nWeather dataset created successfully!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Shape: {weather_df.shape}")

    print("\nDate range:")
    print(weather_df["date"].min().date())
    print("to")
    print(weather_df["date"].max().date())

    print("\nColumns:")
    print(weather_df.columns.tolist())

    print("\nMissing values:")
    print(weather_df.isna().sum())

    print("\nFirst five rows:")
    print(weather_df.head())

    print("\nLast five rows:")
    print(weather_df.tail())


if __name__ == "__main__":
    main()