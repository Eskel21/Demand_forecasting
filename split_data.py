import os
import pandas as pd


INPUT_FILE = "data/demand_forecasting_features.csv"

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

VALIDATION_DAYS = 28

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

train_df = df.iloc[:-VALIDATION_DAYS].copy()
validation_df = df.iloc[-VALIDATION_DAYS:].copy()


print(
    f"\nTRAIN: "
    f"{train_df['Date'].min().date()} - "
    f"{train_df['Date'].max().date()}"
)

print(
    f"VALIDATION: "
    f"{validation_df['Date'].min().date()} - "
    f"{validation_df['Date'].max().date()}"
)

print(f"\nTrain observations: {len(train_df)}")
print(f"Validation observations: {len(validation_df)}")


assert len(validation_df) == VALIDATION_DAYS

assert (
    train_df["Date"].max()
    <
    validation_df["Date"].min()
)


os.makedirs("data", exist_ok=True)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

print(f"\nSaved: {TRAIN_FILE}")
print(f"Saved: {VALIDATION_FILE}")