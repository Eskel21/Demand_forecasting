import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


INPUT_FILE = "aggregated_data/demand_forecasting_daily_reduced.csv"
OUTPUT_DIR = "eda_results"

TARGET = "Units Sold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nLiczba obserwacji: {len(df)}")
print(f"Liczba kolumn: {len(df.columns)}")
print(f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}")


print("STATYSTYKI OPISOWE")

numeric_columns = df.select_dtypes(include="number").columns

statistics = df[numeric_columns].describe().T

statistics["median"] = df[numeric_columns].median()
statistics["skewness"] = df[numeric_columns].skew()

statistics = statistics[
    [
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "skewness"
    ]
]

print(statistics)

statistics.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
)

plt.figure(figsize=(10, 6))

plt.hist(
    df[TARGET],
    bins=30,
    edgecolor="black"
)

plt.title("Rozkład dziennej sprzedaży")
plt.xlabel("Ilość sprzedanych sztuk")
plt.ylabel("Częstotliwość")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "units_sold_distribution.png"),
    dpi=150
)

plt.close()

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df[TARGET]
)

plt.title("Dzienna liczba sprzedanych sztuk w czasie")
plt.xlabel("Data")
plt.ylabel("Ilość sprzedanych sztuk")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "units_sold_over_time.png"),
    dpi=150
)

plt.close()

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df[TARGET],
    alpha=0.4,
    label="Dzienna sprzedaż"
)

plt.plot(
    df["Date"],
    df[TARGET].rolling(7).mean(),
    label="Średnia krocząca z tygodnia"
)

plt.plot(
    df["Date"],
    df[TARGET].rolling(28).mean(),
    label="Średnia krocząca z 4 tygodni"
)

plt.title("Dzienna sprzedaż ze średnimi kroczącymi")
plt.xlabel("Data")
plt.ylabel("Ilość sprzedanych sztuk")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "units_sold_moving_average.png"),
    dpi=150
)

plt.close()

pearson_corr = (
    df[numeric_columns]
    .corr(method="pearson")
)

pearson_target = (
    pearson_corr[TARGET]
    .drop(TARGET)
    .sort_values(ascending=False)
)

print("KORELACJA PEARSONA Z ILOŚCIĄ SPRZEDANYCH SZTUK")

print(pearson_target)

pearson_target.to_csv(
    os.path.join(OUTPUT_DIR, "pearson_target.csv"),
    header=["Pearson correlation"]
)

spearman_corr = (
    df[numeric_columns]
    .corr(method="spearman")
)

spearman_target = (
    spearman_corr[TARGET]
    .drop(TARGET)
    .sort_values(ascending=False)
)

print("KORELACJA SPEARMANA Z ILOŚCIĄ SPRZEDANYCH SZTUK")

print(spearman_target)

spearman_target.to_csv(
    os.path.join(OUTPUT_DIR, "spearman_target.csv"),
    header=["Spearman correlation"]
)

correlation_results = []

for column in numeric_columns:

    if column == TARGET:
        continue

    pair = df[[column, TARGET]].dropna()

    if pair[column].nunique() <= 1:
        continue

    # Pearson
    pearson_r, pearson_p = pearsonr(
        pair[column],
        pair[TARGET]
    )

    # Spearman
    spearman_r, spearman_p = spearmanr(
        pair[column],
        pair[TARGET]
    )

    correlation_results.append({
        "Variable": column,

        "Pearson r": pearson_r,
        "Pearson p-value": pearson_p,
        "Pearson significant (p < 0.05)": pearson_p < 0.05,

        "Spearman r": spearman_r,
        "Spearman p-value": spearman_p,
        "Spearman significant (p < 0.05)": spearman_p < 0.05
    })


correlation_results = pd.DataFrame(correlation_results)

correlation_results["Absolute Pearson r"] = (
    correlation_results["Pearson r"].abs()
)

correlation_results = (
    correlation_results
    .sort_values(
        "Absolute Pearson r",
        ascending=False
    )
    .drop(columns="Absolute Pearson r")
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("KORELACJE I ISTOTNOŚĆ STATYSTYCZNA")
print("=" * 60)

print(
    correlation_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)
correlation_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_significance.csv"
    ),
    index=False
)

fig, ax = plt.subplots(figsize=(12, 10))

correlation_matrix = pearson_corr

image = ax.imshow(
    correlation_matrix,
    aspect="auto",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))

ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=90
)

ax.set_yticklabels(
    correlation_matrix.columns
)
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):

        value = correlation_matrix.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

fig.colorbar(image)

plt.title("MACIERZ KORELACJI PEARSONA")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_matrix.png"),
    dpi=150
)

plt.close()

if "Seasonality" in df.columns:

    seasonality_sales = (
        df.groupby("Seasonality")[TARGET]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )


    print("SPRZEDAŻ WEDŁUG PORY ROKU")


    print(seasonality_sales)

    seasonality_sales.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sales_by_seasonality.csv"
        )
    )

if "Epidemic" in df.columns:

    epidemic_sales = (
        df.groupby("Epidemic")[TARGET]
        .agg(["mean", "median", "std", "count"])
    )

    print("SPRZEDAŻ: EPIDEMIA KONTRA BEZ EPIDEMII")

    print(epidemic_sales)

    epidemic_sales.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sales_by_epidemic.csv"
        )
    )

df["Day of Week"] = df["Date"].dt.day_name()

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_sales = (
    df.groupby("Day of Week")[TARGET]
    .agg(["mean", "median", "std", "count"])
    .reindex(day_order)
)

print("SPRZEDAŻ WEDŁUG DNIA TYGODNIA")

print(day_sales)

day_sales.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sales_by_day_of_week.csv"
    )
)


df["YearMonth"] = df["Date"].dt.to_period("M")

monthly_sales = (
    df.groupby("YearMonth")[TARGET]
    .mean()
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)

plt.title("Średnia sprzedaż miesięcznie")
plt.xlabel("Miesiąc")
plt.ylabel("Średnia sprzedaż")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_sales_by_month.png"
    ),
    dpi=150
)

plt.close()

print(
    f"Wyniki zapisano w katalogu: {OUTPUT_DIR}"
)