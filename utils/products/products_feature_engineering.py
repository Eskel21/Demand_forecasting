from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "aggregated_data" / "products"
OUTPUT_DIR = BASE_DIR / "data" / "products" / "features"

TOP_PRODUCTS = ["P0007", "P0004", "P0009"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for product_id in TOP_PRODUCTS:

    print(f"FEATURE ENGINEERING — {product_id}")
    input_file = (
        INPUT_DIR /
        f"demand_forecasting_{product_id}_daily.csv"
    )

    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    for lag in [1, 7, 14, 21, 28]:
        df[f"UnitsSold_Lag_{lag}"] = (
            df["Units Sold"].shift(lag)
        )

    shifted_sales = df["Units Sold"].shift(1)

    for window in [7, 14, 28]:
        df[f"UnitsSold_RollingMean_{window}"] = (
            shifted_sales
            .rolling(window=window)
            .mean()
        )

    for window in [7, 14, 28]:
        df[f"UnitsSold_RollingStd_{window}"] = (
            shifted_sales
            .rolling(window=window)
            .std()
        )

    rows_before = len(df)

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    rows_after = len(df)

    output_file = (
        OUTPUT_DIR /
        f"{product_id}_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Liczba wierszy przed: {rows_before}")
    print(f"Liczba wierszy po:    {rows_after}")
    print(f"Usunięto:             {rows_before - rows_after}")

    print(
        "Zakres dat:",
        df["Date"].min(),
        "-",
        df["Date"].max()
    )

    print(f"Zapisano do: {output_file}")
