import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# USTAWIENIA
# ============================================================

TRAIN_FILE = "../data/train.csv"
VALIDATION_FILE = "../data/validation.csv"

OUTPUT_DIR = "../outputs/baseline"

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_predictions.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_metrics.csv"
)

TARGET = "Units Sold"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WCZYTANIE DANYCH
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

train = train.sort_values("Date").reset_index(drop=True)
validation = validation.sort_values("Date").reset_index(drop=True)


print("\n" + "=" * 60)
print("BASELINE - SEASONAL NAIVE (PREVIOUS YEAR)")
print("=" * 60)

print(
    "\nTrain:",
    train["Date"].min().date(),
    "-",
    train["Date"].max().date()
)

print(
    "Validation:",
    validation["Date"].min().date(),
    "-",
    validation["Date"].max().date()
)


# ============================================================
# ANALOGICZNY OKRES ROK WCZEŚNIEJ
# ============================================================

# Dla każdego dnia validation wyznaczamy analogiczną
# datę rok wcześniej.
validation_results = validation[
    ["Date", TARGET]
].copy()

validation_results["ReferenceDate"] = (
    validation_results["Date"]
    - pd.DateOffset(years=1)
)


# ============================================================
# POBRANIE SPRZEDAŻY Z POPRZEDNIEGO ROKU
# ============================================================

historical_sales = train[
    ["Date", TARGET]
].copy()

historical_sales = historical_sales.rename(
    columns={
        "Date": "ReferenceDate",
        TARGET: "Prediction"
    }
)

validation_results = validation_results.merge(
    historical_sales,
    on="ReferenceDate",
    how="left"
)


# ============================================================
# KONTROLA
# ============================================================

if validation_results["Prediction"].isna().any():

    missing_dates = validation_results.loc[
        validation_results["Prediction"].isna(),
        "ReferenceDate"
    ]

    raise ValueError(
        "Brak danych historycznych dla dat:\n"
        + missing_dates.to_string(index=False)
    )


# ============================================================
# BŁĄD
# ============================================================

validation_results["Error"] = (
    validation_results["Prediction"]
    - validation_results[TARGET]
)


# ============================================================
# METRYKI
# ============================================================

y_true = validation_results[TARGET]
y_pred = validation_results["Prediction"]

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

non_zero_mask = y_true != 0

mape = (
    np.mean(
        np.abs(
            (
                y_true[non_zero_mask]
                - y_pred[non_zero_mask]
            )
            / y_true[non_zero_mask]
        )
    )
    * 100
)

# Prediction - Actual
bias = np.mean(
    y_pred - y_true
)


# ============================================================
# WYNIKI
# ============================================================

metrics = pd.DataFrame(
    {
        "Model": [
            "Seasonal Naive (previous year)"
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
    }
)

print("\nMetryki baseline:")
print(metrics.to_string(index=False))

print("\nPrognozy:")
print(
    validation_results[
        [
            "Date",
            "ReferenceDate",
            TARGET,
            "Prediction",
            "Error"
        ]
    ].to_string(index=False)
)


# ============================================================
# ZAPIS
# ============================================================

validation_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

metrics.to_csv(
    METRICS_FILE,
    index=False
)

print("\nZapisano:")
print("-", PREDICTIONS_FILE)
print("-", METRICS_FILE)