import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# USTAWIENIA
# ============================================================

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

OUTPUT_DIR = "outputs/random_forest"

FORECAST_FILE = os.path.join(
    OUTPUT_DIR,
    "random_forest_forecast.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "random_forest_metrics.csv"
)

TARGET = "Units Sold"

SELECTED_FEATURES = [
    "Demand",
    "Competitor Pricing",
    "Epidemic",
    "UnitsSold_Lag_7",
    "UnitsSold_Lag_28",
    "UnitsSold_RollingMean_7",
    "UnitsSold_RollingMean_14",
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# METRYKI
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

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

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    bias = np.mean(
        y_pred - y_true
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias
    }

def create_recursive_features(
    row,
    sales_history
):

    if len(sales_history) < 28:

        raise ValueError(
            "Za mało danych historycznych "
            "do utworzenia Lag_28."
        )

    features = {

        "Demand":
            row["Demand"],

        "Competitor Pricing":
            row["Competitor Pricing"],

        "Epidemic":
            row["Epidemic"],

        "UnitsSold_Lag_7":
            sales_history[-7],

        "UnitsSold_Lag_28":
            sales_history[-28],

        "UnitsSold_RollingMean_7":
            np.mean(
                sales_history[-7:]
            ),

        "UnitsSold_RollingMean_14":
            np.mean(
                sales_history[-14:]
            ),
    }

    return pd.DataFrame(
        [features],
        columns=SELECTED_FEATURES
    )


train = pd.read_csv(
    TRAIN_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
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


print("RANDOM FOREST - Prognoza")

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

print(
    f"Liczba dni validation: {len(validation)}"
)

required_train_columns = (
    [TARGET]
    + SELECTED_FEATURES
)

required_validation_columns = [
    "Date",
    TARGET,
    "Demand",
    "Competitor Pricing",
    "Epidemic"
]


missing_train = [
    column
    for column in required_train_columns
    if column not in train.columns
]

missing_validation = [
    column
    for column in required_validation_columns
    if column not in validation.columns
]


if missing_train:

    raise ValueError(
        "Brak kolumn w train:\n"
        + "\n".join(missing_train)
    )


if missing_validation:

    raise ValueError(
        "Brak kolumn w validation:\n"
        + "\n".join(missing_validation)
    )


if len(validation) != 28:

    print(
        "\nUWAGA: validation nie zawiera "
        "dokładnie 28 obserwacji."
    )

X_train = train[
    SELECTED_FEATURES
].copy()

y_train = train[
    TARGET
].copy()


model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print("\nTrenowanie Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Model wytrenowany.")

sales_history = (
    train[TARGET]
    .astype(float)
    .tolist()
)

predictions = []


print("\nGenerowanie prognozy...\n")


for _, row in validation.iterrows():

    X_future = create_recursive_features(
        row=row,
        sales_history=sales_history
    )

    prediction = round(
        model.predict(
            X_future
        )[0]
    )

    predictions.append(
        prediction
    )

    sales_history.append(
        prediction
    )

    print(
        f"{row['Date'].date()} "
        f"-> {prediction:.2f}"
    )

actual = (
    validation[TARGET]
    .to_numpy()
)

metrics = calculate_metrics(
    actual,
    predictions
)

forecast_df = pd.DataFrame({

    "Date":
        validation["Date"],

    "Actual":
        actual,

    "Forecast":
        predictions,

})

forecast_df["Error"] = (
    forecast_df["Forecast"]
    - forecast_df["Actual"]
)

forecast_df["Absolute_Error"] = (
    np.abs(
        forecast_df["Error"]
    )
)

metrics_df = pd.DataFrame([
    {
        "Model": "Random Forest",
        **metrics
    }
])

print("\n" + "=" * 70)
print("WYNIKI NA ZBIORZE WALIDACYJNYM")
print("=" * 70)

print(
    f"\nMAE:  {metrics['MAE']:.2f}"
)

print(
    f"RMSE: {metrics['RMSE']:.2f}"
)

print(
    f"MAPE: {metrics['MAPE']:.2f}%"
)

print(
    f"Bias: {metrics['Bias']:.2f}"
)


print("\n" + "=" * 70)
print("PROGNOZA")
print("=" * 70)

print(
    forecast_df.to_string(
        index=False,
        formatters={
            "Forecast":
                lambda x: f"{x:.2f}",

            "Error":
                lambda x: f"{x:.2f}",

            "Absolute_Error":
                lambda x: f"{x:.2f}",
        }
    )
)

forecast_df.to_csv(
    FORECAST_FILE,
    index=False
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


print("\n" + "=" * 70)

print(
    f"Prognoza zapisana do:\n"
    f"{FORECAST_FILE}"
)

print(
    f"\nMetryki zapisane do:\n"
    f"{METRICS_FILE}"
)