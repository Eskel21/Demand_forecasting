import os

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = "aggregated_data/demand_forecasting_daily.csv"
OUTPUT_DIR = "outputs/external_factors_analysis"

TARGET = "Units Sold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print("ANALIZA CZYNNIKÓW ZEWNĘTRZNYCH")


print("\nZakres danych:")
print(df["Date"].min().date(), "-", df["Date"].max().date())

print("\nLiczba dni:", len(df))

external_numeric = [
    "Demand",
    "Competitor Pricing"
]

external_binary = [
    "Epidemic"
]

external_categorical = [
    "Seasonality"
]

weather_columns = [
    column
    for column in df.columns
    if column.startswith("Weather_")
]


print("\nAnalizowane czynniki:")

print("\nLiczbowe:")
for column in external_numeric:
    print("-", column)

print("\nBinarne:")
for column in external_binary:
    print("-", column)

print("\nKategoryczne:")
for column in external_categorical:
    print("-", column)

print("\nPogodowe:")
for column in weather_columns:
    print("-", column)

print("KORELACJE ZE SPRZEDAŻĄ")

correlation_columns = (
    [TARGET]
    + external_numeric
    + external_binary
    + weather_columns
)

correlations = (
    df[correlation_columns]
    .corr(method="pearson")[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)

print("\nKorelacja Pearsona z Units Sold:")
print(correlations)

correlations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlations_with_units_sold.csv"
    ),
    header=["Pearson Correlation"]
)

for column in external_numeric:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[column],
        df[TARGET],
        alpha=0.6
    )

    correlation = df[[column, TARGET]].corr().iloc[0, 1]

    plt.title(
        f"{TARGET} vs {column}\n"
        f"Pearson correlation = {correlation:.3f}"
    )

    plt.xlabel(column)
    plt.ylabel(TARGET)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{column.lower().replace(' ', '_')}_scatter.png"
        ),
        dpi=150
    )

    plt.close()

print("EPIDEMIC")

epidemic_summary = (
    df
    .groupby("Epidemic")[TARGET]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std"
    )
    .round(2)
)

print("\nSprzedaż według Epidemic:")
print(epidemic_summary)

epidemic_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "epidemic_summary.csv"
    )
)


plt.figure(figsize=(7, 5))

epidemic_means = (
    df
    .groupby("Epidemic")[TARGET]
    .mean()
)

epidemic_means.plot(
    kind="bar"
)

plt.title("Średnia dzienna sprzedaż - Epidemic")
plt.xlabel("Epidemic")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "epidemic_average_sales.png"
    ),
    dpi=150
)

plt.close()

print("SEASONALITY")

seasonality_summary = (
    df
    .groupby("Seasonality")[TARGET]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std"
    )
    .sort_values("Mean", ascending=False)
    .round(2)
)

print("\nSprzedaż według pory roku:")
print(seasonality_summary)

seasonality_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seasonality_summary.csv"
    )
)


plt.figure(figsize=(8, 5))

seasonality_summary["Mean"].plot(
    kind="bar"
)

plt.title("Średnia dzienna sprzedaż według pory roku")
plt.xlabel("Seasonality")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "seasonality_average_sales.png"
    ),
    dpi=150
)

plt.close()

print("WEATHER")

if weather_columns:

    weather_correlations = (
        df[
            [TARGET] + weather_columns
        ]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )

    print("\nKorelacja udziału warunków pogodowych ze sprzedażą:")
    print(weather_correlations)

    weather_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "weather_correlations.csv"
        ),
        header=["Pearson Correlation"]
    )

    for column in weather_columns:

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[column],
            df[TARGET],
            alpha=0.6
        )

        correlation = (
            df[[column, TARGET]]
            .corr()
            .iloc[0, 1]
        )

        weather_name = column.replace(
            "Weather_",
            ""
        )

        plt.title(
            f"Units Sold vs {weather_name}\n"
            f"Pearson correlation = {correlation:.3f}"
        )

        plt.xlabel(
            f"{weather_name} share [%]"
        )

        plt.ylabel(TARGET)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{column.lower()}_scatter.png"
            ),
            dpi=150
        )

        plt.close()

else:

    print("\nNie znaleziono kolumn Weather_*")

print("RANKING CZYNNIKÓW")

ranking = (
    correlations
    .rename("Correlation")
    .to_frame()
)

ranking["Absolute Correlation"] = (
    ranking["Correlation"].abs()
)

ranking = (
    ranking
    .sort_values(
        "Absolute Correlation",
        ascending=False
    )
)

print(ranking.round(4))

ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "external_factors_ranking.csv"
    )
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "analysis_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ANALIZA CZYNNIKÓW ZEWNĘTRZNYCH\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        "Korelacje Pearsona z Units Sold:\n\n"
    )

    file.write(
        correlations
        .round(4)
        .to_string()
    )

    file.write("\n\n")

    file.write(
        "Sprzedaż według Epidemic:\n\n"
    )

    file.write(
        epidemic_summary.to_string()
    )

    file.write("\n\n")

    file.write(
        "Sprzedaż według Seasonality:\n\n"
    )

    file.write(
        seasonality_summary.to_string()
    )

print("ANALIZA ZAKOŃCZONA")

print(
    f"\nWyniki zapisano w: {OUTPUT_DIR}"
)