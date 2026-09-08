import os
import pandas as pd

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

OUTPUT_DIR = "outputs/baseline"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_forecast.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

train = train.sort_values("Date")
validation = validation.sort_values("Date")

historical_sales = (
    train
    .set_index("Date")["Units Sold"]
)

predictions = []

for date in validation["Date"]:

    previous_year_date = date - pd.DateOffset(years=1)

    if previous_year_date not in historical_sales.index:
        raise ValueError(
            f"Brak danych historycznych dla "
            f"{previous_year_date.date()}"
        )

    prediction = historical_sales.loc[
        previous_year_date
    ]

    predictions.append(prediction)

forecast = pd.DataFrame({
    "Date": validation["Date"],
    "Actual": validation["Units Sold"],
    "Forecast": predictions
})

forecast["Error"] = (
    forecast["Forecast"]
    - forecast["Actual"]
)

forecast["Absolute_Error"] = (
    forecast["Error"].abs()
)

forecast.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Zapisano: {OUTPUT_FILE}")

print(
    forecast.to_string(
        index=False
    )
)