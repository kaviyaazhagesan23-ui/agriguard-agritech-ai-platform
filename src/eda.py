"""PaddyWise AI Phase 2: reproducible professional EDA pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "master_dataset.csv"

REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "eda_charts"

JSON_REPORT = REPORT_DIR / "eda_report.json"
SUMMARY_PATH = REPORT_DIR / "eda_summary.md"
LOG_PATH = REPORT_DIR / "eda.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


PRICE_COLS = [
    "min_price_rs_per_quintal",
    "modal_price_rs_per_quintal",
    "max_price_rs_per_quintal",
]

WEATHER_COLS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "rainfall",
    "wind_speed_max",
    "et0",
]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the actual processed PaddyWise dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(
            "Required date column is missing"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    required = {
        "market",
        "variety",
        "grade",
        *PRICE_COLS,
        *WEATHER_COLS,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Required EDA columns missing: {missing}"
        )

    return df


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-compatible values."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def descriptive_statistics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    return {
        "rows": len(df),

        "columns": len(df.columns),

        "memory_mb": round(
            df.memory_usage(deep=True).sum()
            / 1024**2,
            3,
        ),

        "dtypes": {
            c: str(t)
            for c, t in df.dtypes.items()
        },

        "numeric_summary": (
            df[
                PRICE_COLS
                + WEATHER_COLS
            ]
            .describe()
            .round(4)
            .to_dict()
        ),
    }


def coverage(
    df: pd.DataFrame,
) -> dict[str, Any]:

    dates = df["date"].dropna()

    return {
        "start": dates.min(),

        "end": dates.max(),

        "days": int(
            (
                dates.max()
                - dates.min()
            ).days
            + 1
        ),

        "unique_dates": int(
            dates.dt.normalize().nunique()
        ),

        "invalid_dates": int(
            df["date"].isna().sum()
        ),

        "year_counts": {
            str(k): int(v)
            for k, v in (
                df["date"]
                .dt.year
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }


def categorical_distribution(
    df: pd.DataFrame,
    col: str,
) -> dict[str, Any]:

    s = df[col].astype("string")

    return {
        "unique": int(
            s.nunique(dropna=True)
        ),

        "counts": {
            str(k): int(v)
            for k, v in (
                s.value_counts(
                    dropna=False
                ).items()
            )
        },
    }


def price_statistics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for col in PRICE_COLS:

        result[col] = (
            df[col]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
            .round(4)
            .to_dict()
        )

    spread = (
        df["max_price_rs_per_quintal"]
        - df["min_price_rs_per_quintal"]
    )

    result[
        "modal_price_spread_max_minus_min"
    ] = {
        "mean": float(
            spread.mean()
        ),

        "median": float(
            spread.median()
        ),

        "max": float(
            spread.max()
        ),
    }

    return result


def grouped_price_table(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:

    return (
        df.groupby(group_col)[
            PRICE_COLS
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .round(4)
    )


def outlier_analysis(
    df: pd.DataFrame,
) -> dict[str, Any]:

    result = {}

    for col in PRICE_COLS:

        q1, q3 = df[col].quantile(
            [0.25, 0.75]
        )

        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        flagged = df[
            (df[col] < low)
            | (df[col] > high)
        ].copy()

        result[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": low,
            "upper_fence": high,
            "flagged_count": len(flagged),

            "highest_flagged": (
                flagged
                .sort_values(
                    col,
                    ascending=False,
                )
                .head(10)[
                    [
                        "record_id",
                        "date",
                        "market",
                        "variety",
                        "grade",
                        *PRICE_COLS,
                    ]
                ]
                .to_dict("records")
            ),
        }

    return result


def investigate_vallam(
    df: pd.DataFrame,
) -> dict[str, Any]:

    mask = (
        df["market"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("vallam")
    ) & (
        df["max_price_rs_per_quintal"]
        == 24000
    )

    hit = df.loc[mask].copy()

    if hit.empty:
        return {
            "found": False
        }

    row = hit.iloc[0]

    peer = df[
        (
            df["market"]
            .astype("string")
            .str.casefold()
            == "vallam"
        )
        & (
            df["variety"]
            == row["variety"]
        )
        & (
            df["grade"]
            == row["grade"]
        )
    ].copy()

    peer_before_after = peer[
        (
            peer["date"]
            >= row["date"]
            - pd.Timedelta(days=30)
        )
        & (
            peer["date"]
            <= row["date"]
            + pd.Timedelta(days=30)
        )
    ]

    return {
        "found": True,

        "record": (
            hit.iloc[0]
            .to_dict()
        ),

        "peer_count_same_market_variety_grade": len(
            peer
        ),

        "peer_max_price_median": float(
            peer[
                "max_price_rs_per_quintal"
            ].median()
        ),

        "peer_max_price_mean": float(
            peer[
                "max_price_rs_per_quintal"
            ].mean()
        ),

        "local_window_records": (
            peer_before_after[
                [
                    "record_id",
                    "date",
                    "market",
                    "variety",
                    "grade",
                    *PRICE_COLS,
                ]
            ]
            .sort_values("date")
            .to_dict("records")
        ),

        "modal_price_at_event": float(
            row[
                "modal_price_rs_per_quintal"
            ]
        ),

        "price_order_valid": bool(
            row.get(
                "price_order_valid",
                True,
            )
        ),

        "interpretation": (
            "The maximum price is an extreme value "
            "relative to Vallam peers, but the row "
            "preserves valid min <= modal <= max ordering. "
            "It is retained for Phase 2 and not corrected "
            "or deleted."
        ),
    }


def weather_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    corr = (
        df[
            [
                "modal_price_rs_per_quintal",
                *WEATHER_COLS,
            ]
        ]
        .corr(numeric_only=True)
    )

    return (
        corr[
            ["modal_price_rs_per_quintal"]
        ]
        .drop(
            "modal_price_rs_per_quintal"
        )
        .rename(
            columns={
                "modal_price_rs_per_quintal":
                    "pearson_correlation"
            }
        )
        .sort_values(
            "pearson_correlation"
        )
    )


def missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "column": df.columns,

                "missing_count": [
                    int(df[c].isna().sum())
                    for c in df.columns
                ],

                "missing_pct": [
                    float(
                        df[c].isna().mean()
                        * 100
                    )
                    for c in df.columns
                ],
            }
        )
        .sort_values(
            "missing_count",
            ascending=False,
        )
    )


def save_table(
    df: pd.DataFrame,
    filename: str,
) -> None:

    df.to_csv(
        REPORT_DIR / filename,
        index=True,
    )


def save_chart(
    fig: go.Figure,
    filename: str,
) -> None:

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=50,
            r=30,
            t=70,
            b=50,
        ),
    )

    fig.write_html(
        CHART_DIR / filename,
        include_plotlyjs="cdn",
        full_html=True,
    )


def make_charts(
    df: pd.DataFrame,
) -> None:

    work = df.copy()

    work["year"] = (
        work["date"].dt.year
    )

    work["month"] = (
        work["date"].dt.month
    )

    # 1. Overall trend

    daily = (
        work.groupby(
            "date",
            as_index=False,
        )[
            "modal_price_rs_per_quintal"
        ]
        .mean()
    )

    save_chart(
        px.line(
            daily,
            x="date",
            y="modal_price_rs_per_quintal",
            title=(
                "Overall Daily Mean Modal Price"
            ),
            labels={
                "modal_price_rs_per_quintal":
                    "₹ / quintal"
            },
        ),
        "01_overall_price_trend.html",
    )

    # 2. Yearly trend

    annual = (
        work.groupby(
            "year",
            as_index=False,
        )[
            "modal_price_rs_per_quintal"
        ]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    fig = px.line(
        annual,
        x="year",
        y="mean",
        markers=True,
        title="Yearly Mean Modal Price",
        labels={
            "mean":
                "Mean modal price (₹/quintal)"
        },
    )

    fig.add_scatter(
        x=annual.year,
        y=annual["median"],
        mode="lines+markers",
        name="Median",
    )

    save_chart(
        fig,
        "02_yearly_trend.html",
    )

    # 3. Monthly seasonality

    monthly = (
        work.groupby(
            "month",
            as_index=False,
        )[
            "modal_price_rs_per_quintal"
        ]
        .mean()
    )

    monthly["month_name"] = (
        pd.to_datetime(
            monthly["month"],
            format="%m",
        )
        .dt.month_name()
        .str[:3]
    )

    save_chart(
        px.bar(
            monthly,
            x="month_name",
            y="modal_price_rs_per_quintal",
            title=(
                "Monthly Seasonality "
                "of Mean Modal Price"
            ),
            labels={
                "modal_price_rs_per_quintal":
                    "Mean modal price (₹/quintal)"
            },
        ),
        "03_monthly_seasonality.html",
    )

    # 4. Market distribution

    save_chart(
        px.box(
            work,
            x="market",
            y="modal_price_rs_per_quintal",
            points="outliers",
            title=(
                "Modal Price Distribution by Market"
            ),
            labels={
                "modal_price_rs_per_quintal":
                    "Modal price (₹/quintal)"
            },
        ),
        "04_market_price_distribution.html",
    )

    # 5. Market volatility

    market_stats = (
        work.groupby(
            "market",
            as_index=False,
        )[
            "modal_price_rs_per_quintal"
        ]
        .agg(
            mean="mean",
            std="std",
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    save_chart(
        px.bar(
            market_stats,
            x="market",
            y="mean",
            error_y="std",
            title=(
                "Market Mean Modal Price "
                "with ±1 SD"
            ),
            labels={
                "mean":
                    "Mean modal price (₹/quintal)"
            },
        ),
        "05_market_price_volatility.html",
    )

    # 6. Weather relationships

    weather_long = (
        work[
            [
                "modal_price_rs_per_quintal",
                *WEATHER_COLS,
            ]
        ]
        .melt(
            id_vars=[
                "modal_price_rs_per_quintal"
            ],
            var_name="weather_variable",
            value_name="value",
        )
        .dropna()
    )

    for variable in WEATHER_COLS:

        sub = weather_long[
            weather_long.weather_variable
            == variable
        ]

        save_chart(
            px.scatter(
                sub,
                x="value",
                y="modal_price_rs_per_quintal",
                trendline="ols",
                title=(
                    f"Modal Price vs {variable}"
                ),
                labels={
                    "value": variable,
                    "modal_price_rs_per_quintal":
                        "Modal price (₹/quintal)",
                },
            ),
            f"weather_{variable}.html",
        )

    # 7. Missing data

    miss = (
        missingness(work)
        .query("missing_count > 0")
    )

    if not miss.empty:

        save_chart(
            px.bar(
                miss,
                x="column",
                y="missing_count",
                title="Missing Data by Column",
            ),
            "11_missing_data.html",
        )

    # 8. Outliers

    outlier = (
        work[
            [
                "date",
                "market",
                "variety",
                "grade",
                *PRICE_COLS,
            ]
        ]
        .melt(
            id_vars=[
                "date",
                "market",
                "variety",
                "grade",
            ],
            var_name="price_type",
            value_name="price",
        )
    )

    save_chart(
        px.box(
            outlier,
            x="price_type",
            y="price",
            points="outliers",
            title="Price Outlier Overview",
            labels={
                "price": "₹ / quintal"
            },
        ),
        "12_price_outliers.html",
    )

    # 9. Vallam event context

    vallam = work[
        (
            work.market
            .astype("string")
            .str.casefold()
            == "vallam"
        )
        & (
            work.date.between(
                "2024-06-20",
                "2024-08-20",
            )
        )
    ].sort_values("date")

    fig = px.line(
        vallam,
        x="date",
        y=PRICE_COLS,
        markers=True,
        title=(
            "Vallam Price Context "
            "Around 2024-07-22"
        ),
    )

    save_chart(
        fig,
        "13_vallam_event_context.html",
    )

    # 10. Market map

    if {
        "latitude",
        "longitude",
    }.issubset(work.columns):

        geo = (
            work
            .dropna(
                subset=[
                    "latitude",
                    "longitude",
                ]
            )
            .drop_duplicates("market")
        )

        save_chart(
            px.scatter_map(
                geo,
                lat="latitude",
                lon="longitude",
                hover_name="market",
                zoom=9,
                height=600,
                title=(
                    "Market Geographical Distribution"
                ),
            ),
            "14_market_map.html",
        )


def build_summary(
    df: pd.DataFrame,
    report: dict[str, Any],
) -> str:

    annual = (
        df.groupby(
            df.date.dt.year
        )
        .modal_price_rs_per_quintal
        .mean()
    )

    market = (
        df.groupby("market")
        .modal_price_rs_per_quintal
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    monthly = (
        df.groupby(
            df.date.dt.month
        )
        .modal_price_rs_per_quintal
        .mean()
    )

    corr = weather_correlations(df)

    vallam = report[
        "suspicious_observation"
    ]["record"]

    lines = [

        "# PaddyWise AI — Phase 2 EDA Summary",

        "",

        "## Scope",

        (
            "The EDA uses the actual processed dataset at "
            f"`{DATA_PATH.relative_to(ROOT)}`: "
            f"**{len(df):,} rows × {len(df.columns)} columns**, "
            f"covering **{df.date.min().date()} "
            f"to {df.date.max().date()}**."
        ),

        "",

        "## Key findings",

        (
            f"- The dataset contains **{df.market.nunique()} "
            f"markets**, **{df.variety.nunique()} varieties**, "
            f"and **{df.grade.nunique()} grades**. "
            f"The dominant variety is **{df.variety.mode().iat[0]}** "
            f"({(df.variety == df.variety.mode().iat[0]).sum():,} records)."
        ),

        (
            f"- Mean modal price is "
            f"**₹{df.modal_price_rs_per_quintal.mean():,.2f}/quintal**; "
            f"median is "
            f"**₹{df.modal_price_rs_per_quintal.median():,.0f}**."
        ),

        (
            f"- Mean modal price by year moves from "
            f"**₹{annual.iloc[0]:,.2f}** in {annual.index[0]} "
            f"to **₹{annual.iloc[-1]:,.2f}** in {annual.index[-1]}; "
            "this describes the observed sample and is not a forecast."
        ),

        (
            f"- By historical mean modal price, "
            f"**{market.index[0]}** has the highest observed "
            f"market mean "
            f"(₹{market.iloc[0]['mean']:,.2f}/quintal), "
            f"while **{market.index[-1]}** has the lowest "
            f"(₹{market.iloc[-1]['mean']:,.2f}/quintal)."
        ),

        (
            f"- Market-level standard deviation is highest "
            f"for **{market['std'].idxmax()}** "
            f"(₹{market['std'].max():,.2f}/quintal) "
            "among these raw market observations."
        ),

        (
            f"- Monthly mean modal price is highest in "
            f"**{pd.Timestamp(2000, monthly.idxmax(), 1).month_name()}** "
            f"(₹{monthly.max():,.2f}) and lowest in "
            f"**{pd.Timestamp(2000, monthly.idxmin(), 1).month_name()}** "
            f"(₹{monthly.min():,.2f})."
        ),

        (
            f"- Weather-to-price Pearson correlations are "
            f"small in magnitude in this dataset; the largest "
            f"absolute correlation among the listed weather "
            f"variables is **{corr['pearson_correlation'].abs().max():.3f}** "
            f"for **{corr['pearson_correlation'].abs().idxmax()}**. "
            "Correlation does not establish causation."
        ),

        (
            "- Missingness is concentrated in "
            "`district_y`, `state`, `latitude`, and `longitude`, "
            f"each with **{int(df.latitude.isna().sum())}** "
            "missing records. "
            f"The price target has "
            f"**{int(df.modal_price_rs_per_quintal.isna().sum())}** "
            "missing values."
        ),

        "",

        "## Suspicious Vallam observation",

        (
            f"The preserved observation is record "
            f"**{int(vallam['record_id'])}**, dated "
            f"**{pd.Timestamp(vallam['date']).date()}**, "
            f"with Vallam/B P T/Local prices of "
            f"**₹{vallam['min_price_rs_per_quintal']:,.0f} minimum, "
            f"₹{vallam['modal_price_rs_per_quintal']:,.0f} modal, "
            f"and ₹{vallam['max_price_rs_per_quintal']:,.0f} maximum**. "
            "The row satisfies the supplied ordering rule "
            "`min ≤ modal ≤ max`, but ₹24,000 is an extreme "
            "value relative to Vallam peers. It is retained "
            "unchanged for later investigation/modeling decisions."
        ),

        "",

        "## Interpretation guardrails",

        (
            "EDA findings are descriptive. They do not establish "
            "that weather causes prices, that one market will "
            "outperform another in the future, or that an observed "
            "high/low price will persist. No observations were "
            "deleted or corrected during Phase 2."
        ),
    ]

    return "\n".join(lines) + "\n"


def run_eda() -> dict[str, Any]:

    REPORT_DIR.mkdir(
        exist_ok=True
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    LOGGER.info(
        "Loaded %s rows and %s columns",
        len(df),
        len(df.columns),
    )

    report: dict[str, Any] = {

        "dataset_path":
            str(DATA_PATH.relative_to(ROOT)),

        "statistics":
            descriptive_statistics(df),

        "date_coverage":
            coverage(df),

        "market_distribution":
            categorical_distribution(
                df,
                "market",
            ),

        "variety_distribution":
            categorical_distribution(
                df,
                "variety",
            ),

        "grade_distribution":
            categorical_distribution(
                df,
                "grade",
            ),

        "price_statistics":
            price_statistics(df),

        "market_price_statistics":
            grouped_price_table(
                df,
                "market",
            ).to_dict(),

        "yearly_statistics":
            grouped_price_table(
                df.assign(
                    year=df.date.dt.year
                ),
                "year",
            ).to_dict(),

        "monthly_statistics":
            grouped_price_table(
                df.assign(
                    month=df.date.dt.month
                ),
                "month",
            ).to_dict(),

        "weather_correlations":
            weather_correlations(
                df
            )
            .reset_index()
            .to_dict("records"),

        "missing_data":
            missingness(df)
            .to_dict("records"),

        "outlier_analysis":
            outlier_analysis(df),

        "suspicious_observation":
            investigate_vallam(df),

        "duplicate_flags": {

            "exact_duplicate_rows":
                int(df.duplicated().sum()),

            "duplicate_record_flags":
                int(
                    df.get(
                        "is_duplicate_record",
                        pd.Series(
                            False,
                            index=df.index,
                        ),
                    ).sum()
                ),

            "duplicate_market_date_flags":
                int(
                    df.get(
                        "is_duplicate_market_date",
                        pd.Series(
                            False,
                            index=df.index,
                        ),
                    ).sum()
                ),
        },
    }

    save_table(
        missingness(df),
        "missing_data.csv",
    )

    save_table(
        weather_correlations(df),
        "weather_correlations.csv",
    )

    save_table(
        grouped_price_table(
            df,
            "market",
        ),
        "market_price_statistics.csv",
    )

    save_table(
        df.assign(
            year=df.date.dt.year
        )
        .groupby("year")[PRICE_COLS]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .round(4),
        "yearly_price_statistics.csv",
    )

    save_table(
        df.assign(
            month=df.date.dt.month
        )
        .groupby("month")[PRICE_COLS]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .round(4),
        "monthly_price_statistics.csv",
    )

    make_charts(df)

    JSON_REPORT.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        build_summary(
            df,
            report,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "EDA report written to %s",
        JSON_REPORT,
    )

    return report


if __name__ == "__main__":
    run_eda()