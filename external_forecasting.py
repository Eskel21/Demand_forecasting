import os
import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_INPUT_FILE = "data/train.csv"

OUTPUT_DIR = "outputs/external_forecasting"

TARGETS = [
    "Demand",
    "Competitor Pricing"
]

LAGS = [
    1,
    7,
    14,
    28
]

ROLLING_WINDOWS = [
    7,
    14,
    28
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

def add_calendar_features(df):

    df = df.copy()

    df["DayOfWeek"] = (
        df["Date"].dt.dayofweek
    )

    df["DayOfMonth"] = (
        df["Date"].dt.day
    )

    df["Month"] = (
        df["Date"].dt.month
    )

    df["WeekOfYear"] = (
        df["Date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["IsWeekend"] = (
        df["DayOfWeek"]
        .isin([5, 6])
        .astype(int)
    )

    return df

def create_training_features(
    data,
    target
):

    df = data[
        ["Date", target]
    ].copy()

    df = add_calendar_features(df)

    for lag in LAGS:

        df[f"Lag_{lag}"] = (
            df[target]
            .shift(lag)
        )

    for window in ROLLING_WINDOWS:

        shifted = (
            df[target]
            .shift(1)
        )

        df[
            f"RollingMean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        df[
            f"RollingStd_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "Date",
            target
        ]
    ]

    X = df[
        feature_columns
    ]

    y = df[
        target
    ]

    return X, y, feature_columns

def create_future_features(
    history,
    date
):

    features = {
        "DayOfWeek": date.dayofweek,
        "DayOfMonth": date.day,
        "Month": date.month,
        "WeekOfYear": int(
            date.isocalendar().week
        ),
        "IsWeekend": int(
            date.dayofweek >= 5
        )
    }

    for lag in LAGS:

        features[
            f"Lag_{lag}"
        ] = history[-lag]

    for window in ROLLING_WINDOWS:

        values = history[
            -window:
        ]

        features[
            f"RollingMean_{window}"
        ] = np.mean(values)

        features[
            f"RollingStd_{window}"
        ] = np.std(
            values,
            ddof=1
        )

    return features

def train_external_model(
    data,
    target
):

    X, y, feature_columns = (
        create_training_features(
            data,
            target
        )
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(
                    alpha=1.0
                )
            )
        ]
    )

    model.fit(
        X,
        y
    )

    return model, feature_columns


def forecast_variable(
    data,
    target,
    future_dates
):

    model, feature_columns = (
        train_external_model(
            data,
            target
        )
    )

    history = (
        data[target]
        .astype(float)
        .tolist()
    )

    predictions = []

    for date in future_dates:

        features = create_future_features(
            history,
            date
        )

        X_future = pd.DataFrame(
            [features]
        )

        X_future = X_future[
            feature_columns
        ]

        prediction = (
            model
            .predict(X_future)[0]
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.array(
        predictions
    )

def forecast_external_factors(
    historical_data,
    start_date,
    end_date
):

    data = historical_data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    start_date = pd.Timestamp(
        start_date
    )

    end_date = pd.Timestamp(
        end_date
    )

    last_known_date = (
        data["Date"].max()
    )

    expected_start = (
        last_known_date
        + pd.Timedelta(days=1)
    )

    if start_date != expected_start:

        raise ValueError(
            f"Prognoza musi rozpoczynać się dzień po "
            f"ostatniej znanej obserwacji "
            f"({expected_start.date()})."
        )

    if end_date < start_date:

        raise ValueError(
            "Data końcowa nie może być wcześniejsza "
            "od daty początkowej."
        )

    future_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    forecast = pd.DataFrame(
        {
            "Date": future_dates
        }
    )

    for target in TARGETS:

        print(
            f"Prognozowanie: {target}"
        )

        forecast[
            f"{target}_Forecast"
        ] = forecast_variable(
            data=data,
            target=target,
            future_dates=future_dates
        )

    return forecast


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prognoza zewnętrznych czynników "
            "wpływających na sprzedaż."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Plik z danymi historycznymi."
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help=(
            "Ostatni dzień prognozy "
            "w formacie YYYY-MM-DD."
        )
    )

    args = parser.parse_args()

    data = pd.read_csv(
        args.input
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    last_known_date = (
        data["Date"].max()
    )

    start_date = (
        last_known_date
        + pd.Timedelta(days=1)
    )

    forecast = forecast_external_factors(
        historical_data=data,
        start_date=start_date,
        end_date=args.end_date
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "external_factor_forecasts.csv"
    )

    forecast.to_csv(
        output_file,
        index=False
    )

    print("PROGNOZA ZAKOŃCZONA")

    print(
        "\nZakres prognozy:",
        forecast["Date"].min().date(),
        "-",
        forecast["Date"].max().date()
    )

    print(
        "Liczba prognozowanych dni:",
        len(forecast)
    )

    print("\nPierwsze prognozy:")

    print(
        forecast
        .head()
        .to_string(index=False)
    )

    print(
        f"\nZapisano: {output_file}"
    )