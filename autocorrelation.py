import os

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = "data/demand_forecasting_features.csv"
OUTPUT_DIR = "eda_results"
TARGET = "Units Sold"
MAX_LAG = 28

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

results = []

for lag in range(1, MAX_LAG + 1):
    correlation = df[TARGET].autocorr(lag=lag)

    results.append({
        "Lag": lag,
        "Autocorrelation": correlation
    })

autocorrelation_df = pd.DataFrame(results)

print("\nAutokorelacja sprzedaży:")
print(
    autocorrelation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

autocorrelation_df.to_csv(
    os.path.join(OUTPUT_DIR, "sales_autocorrelation.csv"),
    index=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    autocorrelation_df["Lag"],
    autocorrelation_df["Autocorrelation"]
)

plt.axhline(0, linewidth=1)

plt.xlabel("Opóźnienie [dni]")
plt.ylabel("Autokorelacja")
plt.title("Autokorelacja ilości sprzedanych sztuk")

plt.xticks(range(1, MAX_LAG + 1))

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "sales_autocorrelation.png"),
    dpi=200
)

plt.close()