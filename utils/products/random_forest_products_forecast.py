from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "data" / "products" / "split"
OUTPUT_DIR = BASE_DIR / "outputs" / "products" / "random_forest"

TOP_PRODUCTS = ["P0007", "P0004", "P0009"]

FEATURES = [
    "Demand",
    "Competitor Pricing",
    "Epidemic",
    "UnitsSold_Lag_7",
    "UnitsSold_Lag_28",
    "UnitsSold_RollingMean_7",
    "UnitsSold_RollingMean_14",
]

TARGET = "Units Sold"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_metrics(actual, forecast):

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mae = mean_absolute_error(
        actual,
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            forecast
        )
    )

    mape = np.mean(
        np.abs(
            (actual - forecast) / actual
        )
    ) * 100

    bias = np.mean(
        forecast - actual
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias,
    }

all_metrics = []

for product_id in TOP_PRODUCTS:

    print("\n" + "=" * 70)
    print(f"RANDOM FOREST — {product_id}")
    print("=" * 70)

    product_input_dir = INPUT_DIR / product_id

    train = pd.read_csv(
        product_input_dir / "train.csv"
    )

    validation = pd.read_csv(
        product_input_dir / "validation.csv"
    )

    train["Date"] = pd.to_datetime(
        train["Date"]
    )

    validation["Date"] = pd.to_datetime(
        validation["Date"]
    )

    train = (
        train
        .sort_values("Date")
        .reset_index(drop=True)
    )

    validation = (
        validation
        .sort_values("Date")
        .reset_index(drop=True)
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[FEATURES],
        train[TARGET]
    )

    history = (
        train[TARGET]
        .astype(float)
        .tolist()
    )

    predictions = []

    for _, row in validation.iterrows():

        feature_values = {
            "Demand": row["Demand"],
            "Competitor Pricing": row["Competitor Pricing"],
            "Epidemic": row["Epidemic"],

            "UnitsSold_Lag_7":
                history[-7],

            "UnitsSold_Lag_28":
                history[-28],

            "UnitsSold_RollingMean_7":
                np.mean(history[-7:]),

            "UnitsSold_RollingMean_14":
                np.mean(history[-14:]),
        }

        X_future = pd.DataFrame(
            [feature_values],
            columns=FEATURES
        )

        prediction = model.predict(
            X_future
        )[0]

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )
    predictions = np.rint(
        predictions
    ).astype(int)

    actual = (
        validation[TARGET]
        .to_numpy()
    )

    metrics = calculate_metrics(
        actual,
        predictions
    )

    metrics_row = {
        "Product ID": product_id,
        **metrics
    }

    all_metrics.append(
        metrics_row
    )

    forecast_df = pd.DataFrame({
        "Date": validation["Date"],
        "Actual": actual.astype(int),
        "Forecast": predictions,
    })

    forecast_df["Error"] = (
        forecast_df["Forecast"]
        - forecast_df["Actual"]
    )

    forecast_df["Absolute_Error"] = (
        forecast_df["Error"]
        .abs()
    )

    product_output_dir = (
        OUTPUT_DIR /
        product_id
    )

    product_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast_file = (
        product_output_dir /
        "forecast.csv"
    )

    metrics_file = (
        product_output_dir /
        "metrics.csv"
    )

    forecast_df.to_csv(
        forecast_file,
        index=False
    )

    pd.DataFrame(
        [metrics_row]
    ).to_csv(
        metrics_file,
        index=False
    )

    print(f"\nMAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"Bias: {metrics['Bias']:+.2f}")

    print(
        f"\nPrognozę zapisano do: {forecast_file}"
    )

    print(
        f"Metryki zapisano do: {metrics_file}"
    )

all_metrics_df = pd.DataFrame(
    all_metrics
)

all_metrics_file = (
    OUTPUT_DIR /
    "all_products_metrics.csv"
)

all_metrics_df.to_csv(
    all_metrics_file,
    index=False
)

print("WYNIKI TOP 3 PRODUKTÓW")

print(
    all_metrics_df
    .round(2)
    .to_string(index=False)
)

print(
    f"\nZbiorcze metryki zapisano do: "
    f"{all_metrics_file}"
)