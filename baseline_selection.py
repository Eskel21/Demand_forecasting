import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_FILE = "data/train.csv"
OUTPUT_DIR = "baseline_results"

TARGET = "Units Sold"

FORECAST_HORIZON = 28
N_BACKTEST_WINDOWS = 6

TREND_WINDOWS = [28, 56, 84]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("=" * 70)
print("BASELINE SELECTION")
print("=" * 70)

print(
    f"\nTrain range: "
    f"{df['Date'].min().date()} - "
    f"{df['Date'].max().date()}"
)

print(f"Train observations: {len(df)}")

def calculate_metrics(y_true, y_pred):

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

    return mae, rmse, mape, bias


def seasonal_naive_forecast(
    history,
    horizon,
    lag=7
):

    values = history[TARGET].tolist()

    predictions = []

    for _ in range(horizon):

        prediction = values[-lag]

        predictions.append(prediction)

        values.append(prediction)

    return np.array(predictions)


def linear_trend_forecast(
    history,
    horizon,
    trend_window
):

    trend_data = (
        history
        .tail(trend_window)
        .copy()
    )
    X_train = np.arange(
        len(trend_data)
    ).reshape(-1, 1)

    y_train = (
        trend_data[TARGET]
        .to_numpy()
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    X_future = np.arange(
        len(trend_data),
        len(trend_data) + horizon
    ).reshape(-1, 1)

    predictions = model.predict(
        X_future
    )

    predictions = np.maximum(
        predictions,
        0
    )

    return predictions

backtest_starts = []

for i in range(
    N_BACKTEST_WINDOWS,
    0,
    -1
):

    start_index = (
        len(df)
        - i * FORECAST_HORIZON
    )

    backtest_starts.append(
        start_index
    )


print("\nBacktest periods:")

for start_index in backtest_starts:

    end_index = (
        start_index
        + FORECAST_HORIZON
        - 1
    )

    print(
        f" - "
        f"{df.iloc[start_index]['Date'].date()} "
        f"to "
        f"{df.iloc[end_index]['Date'].date()}"
    )

baseline_models = [
    {
        "name": "Seasonal Naive (lag 7)",
        "type": "seasonal_naive"
    }
]

for window in TREND_WINDOWS:

    baseline_models.append({
        "name": f"Linear Trend ({window} days)",
        "type": "linear_trend",
        "window": window
    })

all_results = []
all_predictions = []


for baseline_config in baseline_models:

    model_name = baseline_config["name"]

    print("\n" + "-" * 70)
    print(f"Testing: {model_name}")
    print("-" * 70)

    for fold_number, start_index in enumerate(
        backtest_starts,
        start=1
    ):

        end_index = (
            start_index
            + FORECAST_HORIZON
        )

        history = df.iloc[
            :start_index
        ].copy()

        test = df.iloc[
            start_index:end_index
        ].copy()

        if baseline_config["type"] == "seasonal_naive":

            predictions = seasonal_naive_forecast(
                history=history,
                horizon=len(test),
                lag=7
            )

        elif baseline_config["type"] == "linear_trend":

            predictions = linear_trend_forecast(
                history=history,
                horizon=len(test),
                trend_window=baseline_config["window"]
            )

        else:

            raise ValueError(
                "Unknown baseline type."
            )

        y_true = (
            test[TARGET]
            .to_numpy()
        )

        mae, rmse, mape, bias = calculate_metrics(
            y_true,
            predictions
        )

        all_results.append({
            "Model": model_name,
            "Fold": fold_number,
            "Start Date": test["Date"].min(),
            "End Date": test["Date"].max(),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Bias": bias
        })

        fold_predictions = pd.DataFrame({
            "Date": test["Date"].to_numpy(),
            "Actual": y_true,
            "Prediction": predictions,
            "Model": model_name,
            "Fold": fold_number
        })

        all_predictions.append(
            fold_predictions
        )


        print(
            f"Fold {fold_number}: "
            f"MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}, "
            f"MAPE={mape:.2f}%, "
            f"Bias={bias:.2f}"
        )


results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

summary = (
    results_df
    .groupby("Model")
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        MAPE=("MAPE", "mean"),
        Bias=("Bias", "mean")
    )
    .reset_index()
)


summary = (
    summary
    .sort_values("MAE")
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

best_model = summary.iloc[0]

print("\n" + "=" * 70)
print("BEST BASELINE")
print("=" * 70)

print(
    f"\nModel: {best_model['Model']}"
)

print(
    f"Average MAE: "
    f"{best_model['MAE']:.2f}"
)

print(
    f"Average RMSE: "
    f"{best_model['RMSE']:.2f}"
)

print(
    f"Average MAPE: "
    f"{best_model['MAPE']:.2f}%"
)

print(
    f"Average Bias: "
    f"{best_model['Bias']:.2f}"
)

results_file = os.path.join(
    OUTPUT_DIR,
    "baseline_backtest_results.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "baseline_summary.csv"
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "baseline_backtest_predictions.csv"
)

best_baseline_file = os.path.join(
    OUTPUT_DIR,
    "best_baseline.txt"
)


results_df.to_csv(
    results_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


with open(
    best_baseline_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"model={best_model['Model']}\n"
    )

    file.write(
        f"MAE={best_model['MAE']}\n"
    )

    file.write(
        f"RMSE={best_model['RMSE']}\n"
    )

    file.write(
        f"MAPE={best_model['MAPE']}\n"
    )

    file.write(
        f"Bias={best_model['Bias']}\n"
    )

print(results_file)
print(summary_file)
print(predictions_file)
print(best_baseline_file)