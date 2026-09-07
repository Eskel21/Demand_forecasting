import pandas as pd

# WCZYTANIE DANYCH

INPUT_FILE = "data/demand_forecasting.csv"
OUTPUT_FILE = "aggregated_data/demand_forecasting_daily.csv"

df = pd.read_csv(INPUT_FILE)

#Konwersja daty
df["Date"] = pd.to_datetime(df["Date"])

#Zamiana kolumny weather conditions na kolumny o wartościach 0/1 w celu obliczenia średniej, czyli udziału procentowego danych warunków atmosferycznych w danym dniu
weather_columns = pd.get_dummies(
    df["Weather Condition"],
    prefix="Weather",
    dtype=int
)

df = pd.concat(
    [df, weather_columns],
    axis=1
)

aggregation = {
    "Units Sold": "sum",
    "Inventory Level": "sum",
    "Units Ordered": "sum",
    "Demand": "sum",
    "Price": "mean",
    "Competitor Pricing": "mean",
    "Discount": "mean",
    "Promotion": "mean",
    "Epidemic": "first",
    "Seasonality": "first"
}

for column in weather_columns.columns:
    aggregation[column] = "mean"

daily_df = (
    df
    .groupby("Date", as_index=False)
    .agg(aggregation)
    .sort_values("Date")
    .reset_index(drop=True)
)


daily_df["Promotion"] = (
    daily_df["Promotion"] * 100
)

for column in weather_columns.columns:
    daily_df[column] = (
        daily_df[column] * 100
    )

columns_to_round = [
    "Price",
    "Competitor Pricing",
    "Discount",
    "Promotion",
    *weather_columns.columns
]

daily_df[columns_to_round] = (
    daily_df[columns_to_round]
    .round(2)
)


print("\n--- DANE PO AGREGACJI ---")

print(daily_df.head())

print("\nLiczba wierszy:", len(daily_df))
print("Liczba kolumn:", len(daily_df.columns))

print(
    "Zakres dat:",
    daily_df["Date"].min(),
    "-",
    daily_df["Date"].max()
)

print(
    "\nŁączna sprzedaż przed agregacją:",
    df["Units Sold"].sum()
)

print(
    "Łączna sprzedaż po agregacji:",
    daily_df["Units Sold"].sum()
)

daily_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nZapisano zagregowane dane do: {OUTPUT_FILE}"
)