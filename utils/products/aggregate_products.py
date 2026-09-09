from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = BASE_DIR / "data" / "demand_forecasting.csv"
OUTPUT_DIR = BASE_DIR / "aggregated_data" / "products"

TOP_PRODUCTS = ["P0007", "P0004", "P0009"]


df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for product_id in TOP_PRODUCTS:

    print(f"\n{'=' * 60}")
    print(f"AGREGACJA PRODUKTU: {product_id}")
    print(f"{'=' * 60}")

    product_df = df[
        df["Product ID"] == product_id
    ].copy()

    weather_columns = pd.get_dummies(
        product_df["Weather Condition"],
        prefix="Weather",
        dtype=int
    )

    product_df = pd.concat(
        [product_df, weather_columns],
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
        product_df
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
        product_df["Units Sold"].sum()
    )

    print(
        "Łączna sprzedaż po agregacji:",
        daily_df["Units Sold"].sum()
    )


    output_file = (
        OUTPUT_DIR /
        f"demand_forecasting_{product_id}_daily.csv"
    )

    daily_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nZapisano zagregowane dane do: {output_file}"
    )