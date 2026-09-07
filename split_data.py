import os
import pandas as pd

INPUT_FILE = "data/demand_forecasting_features.csv"

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

VALIDATION_DAYS = 28

os.makedirs("data", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

print("\nLiczba wszystkich obserwacji:", len(df))
print(
    "Zakres danych:",
    df["Date"].min().date(),
    "-",
    df["Date"].max().date()
)

train = df.iloc[:-VALIDATION_DAYS].copy()

validation = df.iloc[-VALIDATION_DAYS:].copy()

print("\nTRAIN")
print("-" * 40)

print("Liczba obserwacji:", len(train))

print(
    "Zakres:",
    train["Date"].min().date(),
    "-",
    train["Date"].max().date()
)


print("\nVALIDATION")
print("-" * 40)

print("Liczba obserwacji:", len(validation))

print(
    "Zakres:",
    validation["Date"].min().date(),
    "-",
    validation["Date"].max().date()
)

assert len(validation) == VALIDATION_DAYS, (
    "Validation powinien zawierać dokładnie 28 dni."
)

assert train["Date"].max() < validation["Date"].min(), (
    "Train i validation nachodzą na siebie."
)

assert len(train) + len(validation) == len(df), (
    "Liczba obserwacji po podziale nie zgadza się z wejściem."
)

print("\nPodział danych jest poprawny.")

train.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)

print("\nZapisano:")
print("-", TRAIN_FILE)
print("-", VALIDATION_FILE)