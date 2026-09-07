import os

import pandas as pd

INPUT_FILE = "aggregated_data/demand_forecasting_daily_reduced.csv"
OUTPUT_FILE = "data/demand_forecasting_features.csv"

TARGET = "Units Sold"


df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print(f"\nLiczba obserwacji przed feature engineering: {len(df)}")
print(f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}")


df["day_of_week"] = df["Date"].dt.dayofweek

df["month"] = df["Date"].dt.month

df["week_of_year"] = (
    df["Date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

df["is_weekend"] = (
    df["Date"]
    .dt
    .dayofweek
    .isin([5, 6])
    .astype(int)
)

df["lag_1"] = (
    df[TARGET]
    .shift(1)
)

df["lag_7"] = (
    df[TARGET]
    .shift(7)
)

df["lag_14"] = (
    df[TARGET]
    .shift(14)
)

df["lag_28"] = (
    df[TARGET]
    .shift(28)
)


df["rolling_mean_7"] = (
    df[TARGET]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["rolling_mean_14"] = (
    df[TARGET]
    .shift(1)
    .rolling(window=14)
    .mean()
)

df["rolling_mean_28"] = (
    df[TARGET]
    .shift(1)
    .rolling(window=28)
    .mean()
)

if "Seasonality" in df.columns:

    seasonality_columns = pd.get_dummies(
        df["Seasonality"],
        prefix="Season",
        dtype=int
    )

    df = pd.concat(
        [df, seasonality_columns],
        axis=1
    )
    df = df.drop(
        columns=["Seasonality"]
    )


print("\nBrakujące wartości po utworzeniu lagów:")

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28"
]

print(df[feature_columns].isna().sum())


df = (
    df
    .dropna(subset=feature_columns)
    .reset_index(drop=True)
)

print("WYNIK")

print(f"\nLiczba obserwacji po feature engineering: {len(df)}")

print(
    f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}"
)

print("\nUtworzone cechy:")

for column in feature_columns:
    print(f" - {column}")

print(" - day_of_week")
print(" - month")
print(" - week_of_year")
print(" - is_weekend")

season_columns = [
    column
    for column in df.columns
    if column.startswith("Season_")
]

for column in season_columns:
    print(f" - {column}")

missing = df.isna().sum()

missing = missing[
    missing > 0
]

print("\nBrakujące wartości:")

if len(missing) == 0:
    print("Brak")
else:
    print(missing)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nZapisano dane do: {OUTPUT_FILE}"
)