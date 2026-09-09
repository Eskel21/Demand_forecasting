from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data" / "demand_forecasting.csv"
OUTPUT_PATH = BASE_DIR / "data" / "top3_products.csv"

df = pd.read_csv(INPUT_PATH)

df["Date"] = pd.to_datetime(df["Date"])

daily_product_sales = (
    df.groupby(["Date", "Product ID"], as_index=False)
    .agg(
        Units_Sold=("Units Sold", "sum")
    )
)

product_ranking = (
    daily_product_sales
    .groupby("Product ID", as_index=False)
    .agg(
        Average_Daily_Sales=("Units_Sold", "mean"),
        Total_Units_Sold=("Units_Sold", "sum"),
        Number_of_Days=("Date", "nunique"),
    )
    .sort_values(
        by="Average_Daily_Sales",
        ascending=False,
    )
    .reset_index(drop=True)
)

product_ranking["Rank"] = product_ranking.index + 1

top3_products = product_ranking.head(3).copy()

top3_products = top3_products[
    [
        "Rank",
        "Product ID",
        "Average_Daily_Sales",
        "Total_Units_Sold",
        "Number_of_Days",
    ]
]

top3_products["Average_Daily_Sales"] = (
    top3_products["Average_Daily_Sales"].round(2)
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

top3_products.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nTOP 3 najlepiej sprzedające się produkty:")
print(top3_products.to_string(index=False))

print(f"\nZapisano wyniki do: {OUTPUT_PATH}")