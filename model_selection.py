import os

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

INPUT_FILE = "data/train.csv"

OUTPUT_DIR = "outputs/model_selection"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "model_selection_results.csv"
)

TARGET = "Units Sold"

FORECAST_HORIZON = 28
N_SPLITS = 5

SELECTED_FEATURES = [
    "Demand",
    "Competitor Pricing",
    "Epidemic",
    "UnitsSold_Lag_7",
    "UnitsSold_Lag_28",
    "UnitsSold_RollingMean_7",
    "UnitsSold_RollingMean_14",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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
            "Za mało danych historycznych do utworzenia Lag_28."
        )

    features = {

        "Demand": row["Demand"],

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

def recursive_forecast(
    model,
    history_df,
    future_df
):

    sales_history = (
        history_df[TARGET]
        .astype(float)
        .tolist()
    )

    predictions = []

    for _, row in future_df.iterrows():

        X_future = create_recursive_features(
            row=row,
            sales_history=sales_history
        )

        prediction = float(
            model.predict(X_future)[0]
        )

        predictions.append(
            prediction
        )

        sales_history.append(
            prediction
        )

    return np.asarray(predictions)

models = {

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "HistGradientBoosting":
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
}

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("=" * 70)
print("MODEL SELECTION - TRAIN ONLY")
print("=" * 70)

print(
    f"\nLiczba obserwacji: {len(df)}"
)

print(
    f"Zakres: "
    f"{df['Date'].min().date()} - "
    f"{df['Date'].max().date()}"
)


required_columns = (
    ["Date", TARGET]
    + SELECTED_FEATURES
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Brakuje wymaganych kolumn:\n"
        + "\n".join(missing_columns)
    )


required_rows = (
    N_SPLITS
    * FORECAST_HORIZON
)

if len(df) <= required_rows:

    raise ValueError(
        "Za mało danych do wykonania "
        f"{N_SPLITS} foldów po "
        f"{FORECAST_HORIZON} dni."
    )

first_test_start = (
    len(df)
    - N_SPLITS * FORECAST_HORIZON
)

all_results = []

for fold in range(N_SPLITS):

    test_start = (
        first_test_start
        + fold * FORECAST_HORIZON
    )

    test_end = (
        test_start
        + FORECAST_HORIZON
    )

    fold_train = (
        df
        .iloc[:test_start]
        .copy()
    )

    fold_test = (
        df
        .iloc[test_start:test_end]
        .copy()
    )

    print(
        f"FOLD {fold + 1}/{N_SPLITS}"
    )

    print(
        "Train:",
        fold_train["Date"].min().date(),
        "-",
        fold_train["Date"].max().date()
    )

    print(
        "Test:",
        fold_test["Date"].min().date(),
        "-",
        fold_test["Date"].max().date()
    )

    X_train = (
        fold_train[
            SELECTED_FEATURES
        ]
        .copy()
    )

    y_train = (
        fold_train[TARGET]
        .copy()
    )

    for model_name, base_model in models.items():

        print(
            f"\n  {model_name}"
        )

        model = clone(
            base_model
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = recursive_forecast(
            model=model,
            history_df=fold_train,
            future_df=fold_test
        )

        actual = (
            fold_test[TARGET]
            .to_numpy()
        )

        metrics = calculate_metrics(
            actual,
            predictions
        )

        print(
            f"    MAE:  {metrics['MAE']:.2f}"
        )

        print(
            f"    RMSE: {metrics['RMSE']:.2f}"
        )

        print(
            f"    MAPE: {metrics['MAPE']:.2f}%"
        )

        print(
            f"    Bias: {metrics['Bias']:.2f}"
        )

        all_results.append({
            "Model": model_name,
            "Fold": fold + 1,

            "Train_Start":
                fold_train["Date"]
                .min()
                .date(),

            "Train_End":
                fold_train["Date"]
                .max()
                .date(),

            "Test_Start":
                fold_test["Date"]
                .min()
                .date(),

            "Test_End":
                fold_test["Date"]
                .max()
                .date(),

            **metrics
        })

results_df = pd.DataFrame(
    all_results
)

summary = (
    results_df
    .groupby("Model")
    .agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        MAPE=("MAPE", "mean"),
        Bias=("Bias", "mean"),
        MAE_Std=("MAE", "std")
    )
    .reset_index()
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n")
print("ŚREDNIE WYNIKI CROSS-VALIDATION")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

best_model = (
    summary
    .iloc[0]["Model"]
)

print(
    f"NAJLEPSZY MODEL WG MAE: {best_model}"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "model_selection_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"\nSzczegółowe wyniki: {OUTPUT_FILE}"
)

print(
    f"Podsumowanie:       {summary_file}"
)