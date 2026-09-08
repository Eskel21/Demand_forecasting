import os

import pandas as pd

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


INPUT_FILE = "../data/train.csv"

OUTPUT_DIR = "../outputs/feature_selection"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "boruta_results.csv"
)

TARGET = "Units Sold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print("BORUTA FEATURE SELECTION")

print(f"\nLiczba obserwacji treningowych: {len(df)}")
print(
    f"Zakres treningowy: "
    f"{df['Date'].min().date()} - "
    f"{df['Date'].max().date()}"
)

feature_columns = [
    "Demand",
    "Competitor Pricing",
    "Epidemic",

    "Weather_Sunny",
    "Weather_Cloudy",
    "Weather_Rainy",
    "Weather_Snowy",

    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "WeekOfYear",
    "IsWeekend",

    "UnitsSold_Lag_1",
    "UnitsSold_Lag_7",
    "UnitsSold_Lag_14",
    "UnitsSold_Lag_21",
    "UnitsSold_Lag_28",

    "UnitsSold_RollingMean_7",
    "UnitsSold_RollingMean_14",
    "UnitsSold_RollingMean_28",

    "UnitsSold_RollingStd_7",
    "UnitsSold_RollingStd_14",
    "UnitsSold_RollingStd_28",
]

seasonality_dummies = pd.get_dummies(
    df["Seasonality"],
    prefix="Seasonality",
    dtype=int
)

df = pd.concat(
    [
        df,
        seasonality_dummies
    ],
    axis=1
)

feature_columns.extend(
    seasonality_dummies.columns.tolist()
)

missing_columns = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Brak wymaganych kolumn:\n"
        + "\n".join(missing_columns)
    )

X = df[feature_columns].copy()

y = df[TARGET].copy()

if X.isna().any().any():

    missing = (
        X.isna()
        .sum()
        .loc[lambda x: x > 0]
    )

    raise ValueError(
        "W cechach znajdują się braki danych:\n"
        f"{missing}"
    )

if y.isna().any():
    raise ValueError(
        f"Target {TARGET} zawiera braki danych."
    )

random_forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=7,
    n_jobs=-1,
    random_state=42
)

boruta = BorutaPy(
    estimator=random_forest,
    n_estimators="auto",
    max_iter=100,
    random_state=42,
    verbose=1
)

boruta.fit(
    X.values,
    y.values
)

results = pd.DataFrame({
    "Feature": feature_columns,
    "Selected": boruta.support_,
    "Tentative": boruta.support_weak_,
    "Ranking": boruta.ranking_
})

results = results.sort_values(
    by=[
        "Selected",
        "Tentative",
        "Ranking"
    ],
    ascending=[
        False,
        False,
        True
    ]
).reset_index(drop=True)


print("WYNIKI")
print(
    results.to_string(index=False)
)

selected_features = results.loc[
    results["Selected"],
    "Feature"
].tolist()

tentative_features = results.loc[
    results["Tentative"],
    "Feature"
].tolist()

rejected_features = results.loc[
    ~results["Selected"]
    & ~results["Tentative"],
    "Feature"
].tolist()

print("WYBRANE CECHY")

for feature in selected_features:
    print(f"[SELECTED] {feature}")

print("CECHY NIEPEWNE")

if tentative_features:
    for feature in tentative_features:
        print(f"[TENTATIVE] {feature}")
else:
    print("Brak.")

print("ODRZUCONE CECHY")

for feature in rejected_features:
    print(f"[REJECTED] {feature}")


results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nWyniki zapisano do: {OUTPUT_FILE}"
)