from pathlib import Path
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATASET = PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"

PHASE4_DIR = PROJECT_ROOT / "reports" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "reports" / "phase5"
PLOTS_DIR = PHASE5_DIR / "plots"

MODELS_DIR = PROJECT_ROOT / "models" / "phase4"

PREDICTIONS_FILE = PHASE4_DIR / "predictions.csv"
METRICS_FILE = PHASE4_DIR / "metrics.json"
METRICS_CSV = PHASE4_DIR / "metrics.csv"
MODEL_COMPARISON_FILE = PHASE4_DIR / "model_comparison.csv"

RF_MODEL = MODELS_DIR / "random_forest.joblib"
XGB_MODEL = MODELS_DIR / "xgboost.joblib"
PREPROCESSING_METADATA = MODELS_DIR / "preprocessing_metadata.joblib"

EVALUATION_REPORT = PHASE5_DIR / "evaluation_report.md"
MODEL_SELECTION_REPORT = PHASE5_DIR / "model_selection_report.md"


TARGET = "modal_price_rs_per_quintal"

DATE_COL = "date"
MARKET_COL = "market"
VARIETY_COL = "variety"


RANDOM_FOREST = "Random Forest"
XGBOOST = "XGBoost"
NAIVE = "Naive Last Value"
MOVING_AVERAGE = "Moving Average"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PHASE 4 RESULTS
# ============================================================

def load_phase4_results():
    required_files = [
        PREDICTIONS_FILE,
        METRICS_FILE,
        MODEL_COMPARISON_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required Phase 4 file not found: {file}"
            )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    model_comparison = pd.read_csv(MODEL_COMPARISON_FILE)

    logger.info(
        "Loaded Phase 4 predictions: %d rows",
        len(predictions),
    )

    return predictions, metrics_json, model_comparison


# ============================================================
# HELPERS
# ============================================================

def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def identify_prediction_columns(df):
    """
    Automatically identify actual and prediction columns.

    This makes Phase 5 robust to the exact column naming
    produced by Phase 4.
    """

    actual_col = find_column(
        df,
        [
            "actual",
            "y_true",
            "true",
            TARGET,
            "actual_price",
        ],
    )

    naive_col = find_column(
        df,
        [
            "naive_prediction",
            "naive_pred",
            "naive",
        ],
    )

    moving_average_col = find_column(
        df,
        [
            "moving_average_prediction",
            "moving_average_pred",
            "moving_average",
        ],
    )

    rf_col = find_column(
        df,
        [
            "random_forest_prediction",
            "random_forest_pred",
            "random_forest",
            "rf_prediction",
            "rf_pred",
        ],
    )

    xgb_col = find_column(
        df,
        [
            "xgboost_prediction",
            "xgboost_pred",
            "xgboost",
            "xgb_prediction",
            "xgb_pred",
        ],
    )

    return {
        "actual": actual_col,
        NAIVE: naive_col,
        MOVING_AVERAGE: moving_average_col,
        RANDOM_FOREST: rf_col,
        XGBOOST: xgb_col,
    }


def calculate_error_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, MAPE and R2.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
        }

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    non_zero = y_true != 0

    if non_zero.any():
        mape = np.mean(
            np.abs(
                (y_true[non_zero] - y_pred[non_zero])
                / y_true[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    return {
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


# ============================================================
# PREPARE PREDICTION DATA
# ============================================================
def prepare_predictions(df):
    """
    Convert Phase 4 long-format predictions into wide format.

    Phase 4 format:
        date, market, split, model, actual, prediction

    Example:
        2024-01-03, Budalur, validation,
        Naive Last Value, 1850, 1850

    Phase 5 converts this into one row per
    date + market + split + actual.
    """

    required_columns = {
        DATE_COL,
        MARKET_COL,
        "split",
        "model",
        "actual",
        "prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Phase 4 predictions are missing required columns: "
            f"{sorted(missing)}"
        )

    result = df.copy()

    # Clean model names
    result["model"] = result["model"].astype(str).str.strip()

    # Convert numeric columns
    result["actual"] = pd.to_numeric(
        result["actual"],
        errors="coerce"
    )

    result["prediction"] = pd.to_numeric(
        result["prediction"],
        errors="coerce"
    )

    # Convert date
    result[DATE_COL] = pd.to_datetime(
        result[DATE_COL],
        errors="coerce"
    )

    # Keep only the four models used in Phase 4
    valid_models = [
        NAIVE,
        MOVING_AVERAGE,
        RANDOM_FOREST,
        XGBOOST,
    ]

    result = result[
        result["model"].isin(valid_models)
    ].copy()

    if result.empty:
        raise ValueError(
            "No recognized models found in Phase 4 predictions."
        )

    # --------------------------------------------------------
    # Convert LONG format -> WIDE format
    # --------------------------------------------------------

    wide = result.pivot_table(
        index=[
            DATE_COL,
            MARKET_COL,
            "split",
            "actual",
        ],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    # Remove pandas column-axis name
    wide.columns.name = None

    # Make sure every expected model column exists
    for model_name in valid_models:
        if model_name not in wide.columns:
            wide[model_name] = np.nan

    # --------------------------------------------------------
    # Create internal Phase 5 prediction columns
    # --------------------------------------------------------

    wide["_actual"] = pd.to_numeric(
        wide["actual"],
        errors="coerce",
    )

    for model_name in valid_models:

        wide[f"_pred_{model_name}"] = pd.to_numeric(
            wide[model_name],
            errors="coerce",
        )

    # Sort chronologically
    wide = wide.sort_values(
        [DATE_COL, MARKET_COL]
    ).reset_index(drop=True)

    logger.info(
        "Converted Phase 4 predictions from LONG format "
        "to WIDE format."
    )

    logger.info(
        "Prepared prediction rows: %d",
        len(wide),
    )

    logger.info(
        "Models available: %s",
        ", ".join(valid_models),
    )

    return wide, {
        "actual": "actual",
        NAIVE: NAIVE,
        MOVING_AVERAGE: MOVING_AVERAGE,
        RANDOM_FOREST: RANDOM_FOREST,
        XGBOOST: XGBOOST,
    }


# ============================================================
# ERROR DATA
# ============================================================

def add_error_columns(df, model_name):
    pred_col = f"_pred_{model_name}"

    if pred_col not in df.columns:
        return df

    result = df.copy()

    result[f"{model_name}_error"] = (
        result[pred_col] - result["_actual"]
    )

    result[f"{model_name}_absolute_error"] = (
        result[f"{model_name}_error"].abs()
    )

    result[f"{model_name}_percentage_error"] = np.where(
        result["_actual"] != 0,
        (
            result[f"{model_name}_absolute_error"]
            / result["_actual"].abs()
        ) * 100,
        np.nan,
    )

    return result


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(df, model_name):
    pred_col = f"_pred_{model_name}"

    if pred_col not in df.columns:
        return

    data = df[
        ["_actual", pred_col]
    ].dropna()

    if data.empty:
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        data["_actual"],
        data[pred_col],
        alpha=0.6,
    )

    minimum = min(
        data["_actual"].min(),
        data[pred_col].min(),
    )

    maximum = max(
        data["_actual"].max(),
        data[pred_col].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel("Actual Modal Price")
    plt.ylabel("Predicted Modal Price")
    plt.title(f"Actual vs Predicted — {model_name}")

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

def plot_residuals(df, model_name):
    error_col = f"{model_name}_error"

    if error_col not in df.columns:
        return

    data = df[
        ["_actual", error_col]
    ].dropna()

    if data.empty:
        return

    plt.figure(figsize=(9, 5))

    plt.scatter(
        data["_actual"],
        data[error_col],
        alpha=0.6,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Actual Modal Price")
    plt.ylabel("Prediction Error")
    plt.title(f"Residual Analysis — {model_name}")

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / f"residuals_{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()


def plot_error_distribution(df, model_name):
    error_col = f"{model_name}_error"

    if error_col not in df.columns:
        return

    data = df[error_col].dropna()

    if data.empty:
        return

    plt.figure(figsize=(9, 5))

    plt.hist(
        data,
        bins=30,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution — {model_name}")

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / f"error_distribution_{model_name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()


# ============================================================
# ERROR METRICS BY GROUP
# ============================================================

def grouped_error_analysis(
    df,
    model_name,
    group_column,
    minimum_samples=5,
):
    error_col = f"{model_name}_absolute_error"
    pred_col = f"_pred_{model_name}"

    if (
        group_column not in df.columns
        or error_col not in df.columns
        or pred_col not in df.columns
    ):
        return pd.DataFrame()

    data = df[
        [
            group_column,
            "_actual",
            pred_col,
            error_col,
        ]
    ].dropna()

    if data.empty:
        return pd.DataFrame()

    grouped = []

    for group_value, group in data.groupby(group_column):

        if len(group) < minimum_samples:
            continue

        metrics = calculate_error_metrics(
            group["_actual"],
            group[pred_col],
        )

        grouped.append(
            {
                group_column: group_value,
                **metrics,
            }
        )

    return pd.DataFrame(grouped)


def save_grouped_error_analysis(df, model_name):
    """
    Generate grouped error analysis for the TEST set.

    Groups:
    - Market
    - Year
    - Variety (if available)

    The returned group-column names are normalized so that
    plotting functions receive the expected names.
    """

    # --------------------------------------------------------
    # Use TEST data only
    # --------------------------------------------------------

    if "split" in df.columns:

        test_df = df[
            df["split"].astype(str).str.lower() == "test"
        ].copy()

    else:
        raise ValueError(
            "split column is required for grouped error analysis."
        )

    if test_df.empty:
        logger.warning(
            "No TEST observations available for grouped error analysis."
        )
        return []

    analyses = []

    # --------------------------------------------------------
    # Market-wise
    # --------------------------------------------------------

    market = grouped_error_analysis(
        test_df,
        model_name,
        MARKET_COL,
        minimum_samples=5,
    )

    if not market.empty:

        market["model"] = model_name

        market.to_csv(
            PHASE5_DIR
            / f"{model_name.lower().replace(' ', '_')}_market_errors.csv",
            index=False,
        )

        analyses.append(
            ("market", market)
        )

    # --------------------------------------------------------
    # Year-wise
    # --------------------------------------------------------

    if DATE_COL in test_df.columns:

        data = test_df.copy()

        data["year"] = data[DATE_COL].dt.year

        year = grouped_error_analysis(
            data,
            model_name,
            "year",
            minimum_samples=5,
        )

        if not year.empty:

            year["model"] = model_name

            year.to_csv(
                PHASE5_DIR
                / f"{model_name.lower().replace(' ', '_')}_year_errors.csv",
                index=False,
            )

            analyses.append(
                ("year", year)
            )

    # --------------------------------------------------------
    # Variety-wise
    # --------------------------------------------------------

    if VARIETY_COL in test_df.columns:

        variety = grouped_error_analysis(
            test_df,
            model_name,
            VARIETY_COL,
            minimum_samples=10,
        )

        if not variety.empty:

            variety["model"] = model_name

            variety.to_csv(
                PHASE5_DIR
                / f"{model_name.lower().replace(' ', '_')}_variety_errors.csv",
                index=False,
            )

            analyses.append(
                ("variety", variety)
            )

    return analyses


# ============================================================
# GROUPED ERROR PLOTS
# ============================================================

def plot_grouped_mae(group_df, group_column, model_name):
    if group_df.empty:
        return

    plt.figure(figsize=(10, 6))

    ordered = group_df.sort_values("mae")

    plt.bar(
        ordered[group_column].astype(str),
        ordered["mae"],
    )

    plt.xlabel(group_column.replace("_", " ").title())
    plt.ylabel("MAE")
    plt.title(
        f"{model_name} — MAE by {group_column.replace('_', ' ').title()}"
    )

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / (
            f"{model_name.lower().replace(' ', '_')}"
            f"_mae_by_{group_column}.png"
        )
    )

    plt.savefig(output, dpi=150)
    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================
def calculate_model_comparison(df):
    """
    Calculate overall TEST-set performance for all models.
    """

    comparison_rows = []

    # Use TEST data only
    if "split" in df.columns:
        test_df = df[
            df["split"].astype(str).str.lower() == "test"
        ].copy()
    else:
        raise ValueError(
            "split column is required for model comparison."
        )

    if test_df.empty:
        logger.warning(
            "No TEST observations available for model comparison."
        )
        return pd.DataFrame(
            columns=[
                "model",
                "n",
                "mae",
                "rmse",
                "mape",
                "r2",
            ]
        )

    models = [
        NAIVE,
        MOVING_AVERAGE,
        RANDOM_FOREST,
        XGBOOST,
    ]

    for model_name in models:

        pred_col = f"_pred_{model_name}"

        if pred_col not in test_df.columns:
            continue

        data = test_df[
            ["_actual", pred_col]
        ].dropna()

        if data.empty:
            continue

        metrics = calculate_error_metrics(
            data["_actual"],
            data[pred_col],
        )

        comparison_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    output_path = (
        PHASE5_DIR
        / "phase5_model_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Model comparison written to: %s",
        output_path,
    )

    return comparison


def plot_model_comparison(comparison):
    """
    Plot test-set MAE comparison across all models.
    """

    if comparison.empty:
        logger.warning(
            "No model comparison data available for plotting."
        )
        return

    plot_data = comparison.dropna(
        subset=["mae"]
    ).copy()

    if plot_data.empty:
        logger.warning(
            "No valid MAE values available for model comparison plot."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_data["model"].astype(str),
        plot_data["mae"],
    )

    plt.xlabel("Model")
    plt.ylabel("Test MAE")
    plt.title("Model Comparison — Test MAE")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / "model_comparison_mae.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    logger.info(
        "Model comparison plot written to: %s",
        output,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================










# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def load_model(path):
    if not path.exists():
        logger.warning(
            "Model not found: %s",
            path,
        )
        return None

    return joblib.load(path)


def get_feature_names():
    """
    Retrieve feature names from Phase 4 preprocessing metadata.
    """

    if not PREPROCESSING_METADATA.exists():
        logger.warning(
            "Preprocessing metadata not found."
        )
        return None

    metadata = joblib.load(
        PREPROCESSING_METADATA
    )

    if isinstance(metadata, dict):

        for key in [
            "feature_names",
            "features",
            "model_features",
            "numeric_features",
        ]:
            if key in metadata:
                return list(metadata[key])

    return None


def save_feature_importance(model, model_name):
    if model is None:
        return pd.DataFrame()

    if not hasattr(model, "feature_importances_"):
        logger.warning(
            "%s does not expose feature_importances_.",
            model_name,
        )
        return pd.DataFrame()

    importance = np.asarray(
        model.feature_importances_
    )

    feature_names = get_feature_names()

    if feature_names is None:
        logger.warning(
            "Feature names unavailable for %s.",
            model_name,
        )

        feature_names = [
            f"feature_{i}"
            for i in range(len(importance))
        ]

    if len(feature_names) != len(importance):
        logger.warning(
            "Feature-name count does not match importance count for %s.",
            model_name,
        )

        feature_names = [
            f"feature_{i}"
            for i in range(len(importance))
        ]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    )

    output = (
        PHASE5_DIR
        / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    top = result.head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top["feature"][::-1],
        top["importance"][::-1],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Top Feature Importance — {model_name}"
    )

    plt.tight_layout()

    plot_output = (
        PLOTS_DIR
        / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    )

    plt.savefig(
        plot_output,
        dpi=150,
    )

    plt.close()

    return result


# ============================================================
# SHAP ANALYSIS
# ============================================================

def run_shap_analysis():
    """
    SHAP is optional.

    If SHAP is installed and the required preprocessing
    metadata is available, generate a global explanation.

    If SHAP is unavailable, Phase 5 continues normally.
    """

    try:
        import shap
    except ImportError:
        logger.warning(
            "SHAP is not installed. Skipping SHAP analysis."
        )
        return False

    model = load_model(XGB_MODEL)

    if model is None:
        return False

    if not FEATURE_DATASET.exists():
        return False

    if not PREPROCESSING_METADATA.exists():
        logger.warning(
            "Preprocessing metadata unavailable. "
            "Skipping SHAP."
        )
        return False

    metadata = joblib.load(
        PREPROCESSING_METADATA
    )

    if not isinstance(metadata, dict):
        return False

    feature_names = metadata.get(
        "feature_names"
    )

    if feature_names is None:
        feature_names = metadata.get(
            "features"
        )

    if feature_names is None:
        logger.warning(
            "Feature names unavailable. Skipping SHAP."
        )
        return False

    df = pd.read_csv(
        FEATURE_DATASET
    )

    # Reproduce the Phase 4 feature selection.
    excluded = {
        TARGET,
        DATE_COL,
        MARKET_COL,
        "record_id",
        "price_change_1d",
        "price_change_7d",
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "rainfall",
        "wind_speed_max",
        "et0",
    }

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    features = [
        col
        for col in numeric_columns
        if col not in excluded
    ]

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        logger.warning(
            "SHAP feature mismatch. Skipping."
        )
        return False

    X = df[features].copy()

    # Use training medians from preprocessing metadata
    medians = metadata.get(
        "training_medians"
    )

    if medians is not None:
        for column in X.columns:
            if column in medians:
                X[column] = X[column].fillna(
                    medians[column]
                )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Limit SHAP sample size for practical runtime.
    sample_size = min(
        500,
        len(X),
    )

    X_sample = X.sample(
        n=sample_size,
        random_state=42,
    )

    try:
        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X_sample
        )

        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
        )

        plt.tight_layout()

        output = (
            PLOTS_DIR
            / "xgboost_shap_summary.png"
        )

        plt.savefig(
            output,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "SHAP summary generated: %s",
            output,
        )

        return True

    except Exception as exc:
        logger.warning(
            "SHAP analysis failed: %s",
            exc,
        )
        return False


# ============================================================
# MODEL SELECTION
# ============================================================

def determine_final_model(comparison):
    """
    Determine the model based on actual test MAE/RMSE evidence.

    Important:
    This function does NOT force XGBoost.

    The model with the lowest test MAE is selected, provided
    it has valid test metrics.

    The result is evidence-based.
    """

    if comparison.empty:
        raise ValueError(
            "No model comparison results available."
        )

    valid = comparison.dropna(
        subset=["mae", "rmse"]
    ).copy()

    if valid.empty:
        raise ValueError(
            "No valid model metrics available."
        )

    # Primary metric: MAE
    selected = valid.sort_values(
        ["mae", "rmse"],
        ascending=[True, True],
    ).iloc[0]

    return selected["model"], valid


# ============================================================
# WEAKNESS DETECTION
# ============================================================

def identify_model_weaknesses(
    df,
    model_name,
):
    weaknesses = []

    pred_col = f"_pred_{model_name}"
    error_col = f"{model_name}_absolute_error"

    if pred_col not in df.columns:
        return weaknesses

    data = df[
        ["_actual", pred_col]
    ].dropna()

    if data.empty:
        return weaknesses

    metrics = calculate_error_metrics(
        data["_actual"],
        data[pred_col],
    )

    # Bias
    signed_error = (
        data[pred_col] - data["_actual"]
    )

    mean_error = signed_error.mean()

    if mean_error > 0:
        weaknesses.append(
            f"Average prediction error is positive "
            f"({mean_error:.2f}), indicating overall over-prediction."
        )

    elif mean_error < 0:
        weaknesses.append(
            f"Average prediction error is negative "
            f"({mean_error:.2f}), indicating overall under-prediction."
        )

    # Large errors
    absolute_error = signed_error.abs()

    q95 = absolute_error.quantile(
        0.95
    )

    weaknesses.append(
        f"The 95th percentile absolute error is "
        f"{q95:.2f} ₹/quintal."
    )

    # High-price behavior
    high_price_threshold = data["_actual"].quantile(
        0.90
    )

    high_price = data[
        data["_actual"] >= high_price_threshold
    ]

    if len(high_price) >= 5:

        high_price_metrics = calculate_error_metrics(
            high_price["_actual"],
            high_price[pred_col],
        )

        if high_price_metrics["mae"] > metrics["mae"]:

            weaknesses.append(
                "Error increases for high-price observations "
                "relative to the overall test set."
            )

    # Low-price behavior
    low_price_threshold = data["_actual"].quantile(
        0.10
    )

    low_price = data[
        data["_actual"] <= low_price_threshold
    ]

    if len(low_price) >= 5:

        low_price_metrics = calculate_error_metrics(
            low_price["_actual"],
            low_price[pred_col],
        )

        if low_price_metrics["mae"] > metrics["mae"]:

            weaknesses.append(
                "Error increases for low-price observations "
                "relative to the overall test set."
            )

    return weaknesses


# ============================================================
# REPORT GENERATION
# ============================================================

def format_metric(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.4f}"


def generate_evaluation_report(
    comparison,
    selected_model,
    weaknesses,
):
    lines = []

    lines.append(
        "# Phase 5 — Model Evaluation & Error Analysis"
    )

    lines.append("")

    lines.append(
        "This report was generated directly from the actual "
        "Phase 4 predictions and model artifacts."
    )

    lines.append("")

    lines.append(
        "No metrics were manually entered or fabricated."
    )

    lines.append("")

    lines.append("## Model Comparison")

    lines.append("")

    if comparison.empty:
        lines.append(
            "No valid comparison results were available."
        )

    else:

        lines.append(
            "| Model | N | MAE | RMSE | MAPE | R² |"
        )

        lines.append(
            "|---|---:|---:|---:|---:|---:|"
        )

        for _, row in comparison.iterrows():

            lines.append(
                f"| {row['model']} "
                f"| {int(row['n'])} "
                f"| {format_metric(row['mae'])} "
                f"| {format_metric(row['rmse'])} "
                f"| {format_metric(row['mape'])}% "
                f"| {format_metric(row['r2'])} |"
            )

    lines.append("")

    lines.append("## Model Selection")

    lines.append("")

    lines.append(
        f"Selected model based on the lowest valid test MAE "
        f"with RMSE used as the secondary criterion: "
        f"**{selected_model}**."
    )

    lines.append("")

    lines.append(
        "This selection is evidence-based and does not assume "
        "that XGBoost must outperform the baselines."
    )

    lines.append("")

    lines.append("## Error Analysis")

    lines.append("")

    lines.append(
        "The following analyses were generated:"
    )

    lines.append("")

    lines.append(
        "- Actual vs predicted values"
    )
    lines.append(
        "- Residual analysis"
    )
    lines.append(
        "- Error distribution"
    )
    lines.append(
        "- Market-wise error"
    )
    lines.append(
        "- Year-wise error"
    )
    lines.append(
        "- Variety-wise error where sufficient observations exist"
    )
    lines.append(
        "- Baseline comparison"
    )
    lines.append(
        "- Feature importance"
    )
    lines.append(
        "- XGBoost SHAP explanation where technically available"
    )

    lines.append("")

    lines.append("## Model Weaknesses")

    lines.append("")

    if weaknesses:

        for weakness in weaknesses:
            lines.append(
                f"- {weakness}"
            )

    else:
        lines.append(
            "No automatic weakness indicators were triggered."
        )

    lines.append("")

    lines.append("## Important Limitations")

    lines.append("")

    lines.append(
        "- Error analysis does not prove causality."
    )

    lines.append(
        "- High error in a market or variety may be related "
        "to limited observations, distribution changes, "
        "or unobserved factors."
    )

    lines.append(
        "- Model performance on the historical test period "
        "does not guarantee future forecasting performance."
    )

    lines.append(
        "- Forecasting has not been implemented in Phase 5."
    )

    with open(
        EVALUATION_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )


def generate_model_selection_report(
    comparison,
    selected_model,
):
    lines = []

    lines.append(
        "# Phase 5 — Model Selection Report"
    )

    lines.append("")

    lines.append(
        "## Selection Principle"
    )

    lines.append("")

    lines.append(
        "The final model was selected using the actual "
        "Phase 4 test predictions."
    )

    lines.append("")

    lines.append(
        "Primary criterion: lowest test MAE."
    )

    lines.append(
        "Secondary criterion: lowest test RMSE when MAE values "
        "are tied or effectively indistinguishable."
    )

    lines.append("")

    lines.append(
        f"### Selected Model: **{selected_model}**"
    )

    lines.append("")

    if not comparison.empty:

        selected = comparison[
            comparison["model"] == selected_model
        ]

        if not selected.empty:

            row = selected.iloc[0]

            lines.append(
                f"- Test observations: {int(row['n'])}"
            )

            lines.append(
                f"- Test MAE: {format_metric(row['mae'])}"
            )

            lines.append(
                f"- Test RMSE: {format_metric(row['rmse'])}"
            )

            lines.append(
                f"- Test MAPE: {format_metric(row['mape'])}%"
            )

            lines.append(
                f"- Test R²: {format_metric(row['r2'])}"
            )

    lines.append("")

    lines.append(
        "## Baseline Comparison"
    )

    lines.append("")

    if not comparison.empty:

        selected_row = comparison[
            comparison["model"] == selected_model
        ]

        baseline_rows = comparison[
            comparison["model"].isin(
                [
                    NAIVE,
                    MOVING_AVERAGE,
                ]
            )
        ]

        if not selected_row.empty and not baseline_rows.empty:

            selected_mae = selected_row.iloc[0]["mae"]

            for _, baseline in baseline_rows.iterrows():

                difference = (
                    baseline["mae"]
                    - selected_mae
                )

                if difference > 0:

                    lines.append(
                        f"- {selected_model} has lower test MAE "
                        f"than {baseline['model']} by "
                        f"{difference:.4f} ₹/quintal."
                    )

                elif difference < 0:

                    lines.append(
                        f"- {baseline['model']} has lower test MAE "
                        f"than {selected_model} by "
                        f"{abs(difference):.4f} ₹/quintal."
                    )

                else:

                    lines.append(
                        f"- {selected_model} and "
                        f"{baseline['model']} have identical "
                        f"test MAE."
                    )

    lines.append("")

    lines.append(
        "## Important Decision Rule"
    )

    lines.append("")

    lines.append(
        "XGBoost is not automatically selected. "
        "If a baseline has lower test error, the baseline remains "
        "the evidence-supported selection."
    )

    lines.append("")

    lines.append(
        "Forecasting implementation is intentionally excluded "
        "from Phase 5."
    )

    with open(
        MODEL_SELECTION_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting Phase 5 — Model Evaluation & Error Analysis"
    )

    create_directories()

    predictions, metrics_json, model_comparison = (
        load_phase4_results()
    )

    predictions, prediction_columns = (
        prepare_predictions(predictions)
    )

    models = [
        NAIVE,
        MOVING_AVERAGE,
        RANDOM_FOREST,
        XGBOOST,
    ]

    # --------------------------------------------------------
    # Add error columns
    # --------------------------------------------------------

    for model_name in models:

        if f"_pred_{model_name}" in predictions.columns:

            predictions = add_error_columns(
                predictions,
                model_name,
            )

    # --------------------------------------------------------
    # Plots + grouped analysis
    # --------------------------------------------------------

    for model_name in models:

        pred_col = f"_pred_{model_name}"

        if pred_col not in predictions.columns:
            continue

        logger.info(
            "Analysing model: %s",
            model_name,
        )

        plot_actual_vs_predicted(
            predictions,
            model_name,
        )

        plot_residuals(
            predictions,
            model_name,
        )

        plot_error_distribution(
            predictions,
            model_name,
        )

        grouped_results = save_grouped_error_analysis(
            predictions,
            model_name,
        )

        for group_name, group_df in grouped_results:

            plot_grouped_mae(
                group_df,
                group_name,
                model_name,
            )

    # --------------------------------------------------------
    # Overall comparison
    # --------------------------------------------------------

    comparison = calculate_model_comparison(
        predictions
    )

    plot_model_comparison(
        comparison
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    logger.info(
        "Generating Random Forest feature importance."
    )

    rf_model = load_model(
        RF_MODEL
    )

    save_feature_importance(
        rf_model,
        RANDOM_FOREST,
    )

    logger.info(
        "Generating XGBoost feature importance."
    )

    xgb_model = load_model(
        XGB_MODEL
    )

    save_feature_importance(
        xgb_model,
        XGBOOST,
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_available = run_shap_analysis()

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    selected_model, valid_comparison = (
        determine_final_model(
            comparison
        )
    )

    logger.info(
        "Evidence-based selected model: %s",
        selected_model,
    )

    # --------------------------------------------------------
    # Weakness analysis
    # --------------------------------------------------------

    weaknesses = identify_model_weaknesses(
        predictions,
        selected_model,
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    generate_evaluation_report(
        comparison,
        selected_model,
        weaknesses,
    )

    generate_model_selection_report(
        comparison,
        selected_model,
    )

    # --------------------------------------------------------
    # Save complete error dataset
    # --------------------------------------------------------

    predictions.to_csv(
        PHASE5_DIR / "prediction_error_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    phase5_metadata = {
        "selected_model": selected_model,
        "shap_generated": bool(shap_available),
        "source_predictions": str(PREDICTIONS_FILE),
        "source_metrics": str(METRICS_FILE),
        "source_model_comparison": str(
            MODEL_COMPARISON_FILE
        ),
    }

    with open(
        PHASE5_DIR / "phase5_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            phase5_metadata,
            f,
            indent=2,
        )

    logger.info(
        "Phase 5 completed successfully."
    )

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)

    print(
        f"\nEvidence-based selected model: {selected_model}"
    )

    print(
        f"\nEvaluation report:"
        f"\n{EVALUATION_REPORT}"
    )

    print(
        f"\nModel selection report:"
        f"\n{MODEL_SELECTION_REPORT}"
    )

    print(
        f"\nPlots:"
        f"\n{PLOTS_DIR}"
    )

    print(
        f"\nSHAP generated: {shap_available}"
    )

    print(
        "\nForecasting has NOT been implemented."
    )


if __name__ == "__main__":
    main()