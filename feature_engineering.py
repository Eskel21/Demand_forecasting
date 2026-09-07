import os
import pandas as pd

INPUT_FILE = "aggregated_data/demand_forecasting_daily.csv"
OUTPUT_FILE = "data/demand_forecasting_features.csv"

os.makedirs("data", exist_ok=True)


df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\nLiczba obserwacji przed feature engineering:", len(df))


df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayOfMonth"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

df["WeekOfYear"] = (
    df["Date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

df["IsWeekend"] = (
    df["DayOfWeek"]
    .isin([5, 6])
    .astype(int)
)

lags = [
    1,
    7,
    14,
    21,
    28
]

for lag in lags:

    df[f"UnitsSold_Lag_{lag}"] = (
        df["Units Sold"]
        .shift(lag)
    )

rolling_windows = [
    7,
    14,
    28
]

for window in rolling_windows:

    df[f"UnitsSold_RollingMean_{window}"] = (
        df["Units Sold"]
        .shift(1)
        .rolling(window=window)
        .mean()
    )


for window in rolling_windows:

    df[f"UnitsSold_RollingStd_{window}"] = (
        df["Units Sold"]
        .shift(1)
        .rolling(window=window)
        .std()
    )

feature_columns = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "WeekOfYear",
    "IsWeekend",

    *[
        f"UnitsSold_Lag_{lag}"
        for lag in lags
    ],

    *[
        f"UnitsSold_RollingMean_{window}"
        for window in rolling_windows
    ],

    *[
        f"UnitsSold_RollingStd_{window}"
        for window in rolling_windows
    ]
]

print("\nBrakujące wartości po utworzeniu cech:")

print(
    df[feature_columns]
    .isna()
    .sum()
)

rows_before = len(df)

df = (
    df
    .dropna(subset=feature_columns)
    .reset_index(drop=True)
)

rows_after = len(df)

print(
    "\nUsunięte początkowe obserwacje:",
    rows_before - rows_after
)

print(
    "Liczba obserwacji po feature engineering:",
    rows_after
)


print("\nZakres dat po feature engineering:")

print(
    df["Date"].min().date(),
    "-",
    df["Date"].max().date()
)

print("\nUtworzone cechy:")

for column in feature_columns:
    print("-", column)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nZapisano dane z cechami do: {OUTPUT_FILE}"
)