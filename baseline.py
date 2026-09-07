import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

OUTPUT_DIR = "baseline_results"

TARGET = "Units Sold"
LAG = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

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

print("FINAL BASELINE EVALUATION")

print(
    f"\nTrain: "
    f"{train['Date'].min().date()} - "
    f"{train['Date'].max().date()}"
)

print(
    f"Validation: "
    f"{validation['Date'].min().date()} - "
    f"{validation['Date'].max().date()}"
)

print(f"\nForecast horizon: {len(validation)} days")

history = train[TARGET].tolist()

predictions = []


for _ in range(len(validation)):


    prediction = history[-LAG]

    predictions.append(prediction)

    history.append(prediction)


validation_results = validation[
    ["Date", TARGET]
].copy()

validation_results["Baseline Forecast"] = predictions


y_true = validation_results[TARGET].to_numpy()

y_pred = validation_results[
    "Baseline Forecast"
].to_numpy()


mae = mean_absolute_error(
    y_true,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)


# MAPE
non_zero = y_true != 0

mape = np.mean(
    np.abs(
        (
            y_true[non_zero]
            - y_pred[non_zero]
        )
        / y_true[non_zero]
    )
) * 100


bias = np.mean(
    y_pred - y_true
)

metrics = pd.DataFrame({
    "Model": [
        "Seasonal Naive (lag 7)"
    ],
    "MAE": [
        mae
    ],
    "RMSE": [
        rmse
    ],
    "MAPE": [
        mape
    ],
    "Bias": [
        bias
    ]
})

print("BASELINE METRICS")

print(
    metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print(
    validation_results.to_string(
        index=False
    )
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "baseline_validation_metrics.csv"
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "baseline_validation_predictions.csv"
)


metrics.to_csv(
    metrics_file,
    index=False
)

validation_results.to_csv(
    predictions_file,
    index=False
)


print(metrics_file)
print(predictions_file)