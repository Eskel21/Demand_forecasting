import os

import pandas as pd

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

TRAIN_FILE = "data/train.csv"
OUTPUT_DIR = "feature_selection_results"

TARGET = "Units Sold"

os.makedirs(OUTPUT_DIR, exist_ok=True)


train_df = pd.read_csv(TRAIN_FILE)

train_df["Date"] = pd.to_datetime(train_df["Date"])

train_df = (
    train_df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("=" * 60)
print("BORUTA FEATURE SELECTION")
print("=" * 60)

print(
    f"\nTrain range: "
    f"{train_df['Date'].min().date()} - "
    f"{train_df['Date'].max().date()}"
)

print(f"Train observations: {len(train_df)}")

excluded_columns = [
    "Date",
    TARGET
]

feature_columns = [
    column
    for column in train_df.columns
    if column not in excluded_columns
]

X_train = train_df[feature_columns].copy()
y_train = train_df[TARGET].copy()

non_numeric_columns = (
    X_train
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)

if non_numeric_columns:
    raise ValueError(
        "Boruta received non-numeric columns: "
        + ", ".join(non_numeric_columns)
    )

missing_values = X_train.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:
    raise ValueError(
        "Missing values detected:\n"
        + missing_values.to_string()
    )


print(f"\nNumber of candidate features: {len(feature_columns)}")

print("\nCandidate features:")

for feature in feature_columns:
    print(f" - {feature}")

random_forest = RandomForestRegressor(
    n_jobs=-1,
    max_depth=5,
    random_state=42
)

boruta = BorutaPy(
    estimator=random_forest,
    n_estimators="auto",
    max_iter=100,
    verbose=2,
    random_state=42
)

boruta.fit(
    X_train.to_numpy(),
    y_train.to_numpy()
)

results = pd.DataFrame({
    "Feature": feature_columns,
    "Selected": boruta.support_,
    "Tentative": boruta.support_weak_,
    "Ranking": boruta.ranking_
})


def get_status(row):

    if row["Selected"]:
        return "Selected"

    if row["Tentative"]:
        return "Tentative"

    return "Rejected"


results["Status"] = results.apply(
    get_status,
    axis=1
)

results = (
    results
    .sort_values(
        ["Ranking", "Feature"]
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("BORUTA RESULTS")
print("=" * 60)

print(
    results[
        ["Feature", "Status", "Ranking"]
    ].to_string(index=False)
)

selected_features = results.loc[
    results["Status"] == "Selected",
    "Feature"
].tolist()

tentative_features = results.loc[
    results["Status"] == "Tentative",
    "Feature"
].tolist()

rejected_features = results.loc[
    results["Status"] == "Rejected",
    "Feature"
].tolist()

print("WYBRANE")

if selected_features:
    for feature in selected_features:
        print(f" + {feature}")
else:
    print("None")


print("NIEPEWNE")

if tentative_features:
    for feature in tentative_features:
        print(f" ? {feature}")
else:
    print("None")

print("ODRZUCONE")

if rejected_features:
    for feature in rejected_features:
        print(f" - {feature}")
else:
    print("None")

results_file = os.path.join(
    OUTPUT_DIR,
    "boruta_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

selected_features_file = os.path.join(
    OUTPUT_DIR,
    "selected_features.txt"
)

with open(
    selected_features_file,
    "w",
    encoding="utf-8"
) as file:

    for feature in selected_features:
        file.write(feature + "\n")

print(results_file)
print(selected_features_file)