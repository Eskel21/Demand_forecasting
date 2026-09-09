from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "data" / "products" / "features"
OUTPUT_DIR = BASE_DIR / "data" / "products" / "split"

TOP_PRODUCTS = ["P0007", "P0004", "P0009"]

VALIDATION_DAYS = 28

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for product_id in TOP_PRODUCTS:

    print("\n" + "=" * 70)
    print(f"PODZIAŁ DANYCH — {product_id}")
    print("=" * 70)

    input_file = (
        INPUT_DIR /
        f"{product_id}_features.csv"
    )

    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    train_df = df.iloc[:-VALIDATION_DAYS].copy()
    validation_df = df.iloc[-VALIDATION_DAYS:].copy()

    assert len(validation_df) == VALIDATION_DAYS

    assert (
        train_df["Date"].max()
        <
        validation_df["Date"].min()
    )

    assert len(train_df) + len(validation_df) == len(df)

    product_output_dir = OUTPUT_DIR / product_id

    product_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = product_output_dir / "train.csv"
    validation_file = product_output_dir / "validation.csv"

    train_df.to_csv(
        train_file,
        index=False
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    print(f"Wszystkie obserwacje: {len(df)}")
    print(f"Train:                {len(train_df)}")
    print(f"Validation:           {len(validation_df)}")

    print(
        "\nTrain:",
        train_df["Date"].min(),
        "-",
        train_df["Date"].max()
    )

    print(
        "Validation:",
        validation_df["Date"].min(),
        "-",
        validation_df["Date"].max()
    )

    print(f"\nTrain zapisano do:      {train_file}")
    print(f"Validation zapisano do: {validation_file}")
