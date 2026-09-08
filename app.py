from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Prognoza sprzedaży w szstukach",
    page_icon="assets/logo.webp",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "data" / "train.csv"
VALIDATION_FILE = BASE_DIR / "data" / "validation.csv"

RF_FORECAST_FILE = (
    BASE_DIR
    / "outputs"
    / "random_forest"
    / "random_forest_forecast.csv"
)

RF_METRICS_FILE = (
    BASE_DIR
    / "outputs"
    / "random_forest"
    / "random_forest_metrics.csv"
)

BASELINE_FORECAST_FILE = (
    BASE_DIR
    / "outputs"
    / "baseline"
    / "baseline_forecast.csv"
)

BASELINE_METRICS_FILE = (
    BASE_DIR
    / "outputs"
    / "baseline"
    / "baseline_metrics.csv"
)



@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def load_data():

    train = load_csv(TRAIN_FILE)
    validation = load_csv(VALIDATION_FILE)

    rf_forecast = load_csv(RF_FORECAST_FILE)
    rf_metrics = load_csv(RF_METRICS_FILE)

    baseline_forecast = load_csv(BASELINE_FORECAST_FILE)
    baseline_metrics = load_csv(BASELINE_METRICS_FILE)

    date_dataframes = [
        train,
        validation,
        rf_forecast,
        baseline_forecast
    ]

    for df in date_dataframes:
        df["Date"] = pd.to_datetime(df["Date"])

    return (
        train,
        validation,
        rf_forecast,
        rf_metrics,
        baseline_forecast,
        baseline_metrics
    )


try:
    (
        train,
        validation,
        rf_forecast,
        rf_metrics,
        baseline_forecast,
        baseline_metrics
    ) = load_data()

except FileNotFoundError as error:

    st.error(
        "Nie znaleziono jednego z wymaganych plików."
    )

    st.code(str(error))

    st.stop()

baseline_prediction_candidates = [
    "Forecast",
    "Prediction",
    "Baseline",
    "Baseline_Forecast"
]

baseline_prediction_column = None

for column in baseline_prediction_candidates:

    if column in baseline_forecast.columns:
        baseline_prediction_column = column
        break


if baseline_prediction_column is None:

    st.error(
        "Nie znaleziono kolumny z prognozą "
        "w pliku baseline_forecast.csv."
    )

    st.write(
        "Dostępne kolumny:",
        baseline_forecast.columns.tolist()
    )

    st.stop()

st.title("Prognoza sprzedaży")

st.markdown(
    """
    Wyniki **28-dniowej prognozy sprzedaży**
    wygenerowanej modelem **Random Forest** oraz porównanie
    z modelem bazowym **Seasonal Naive (previous year)**.
    """
)

st.divider()

st.subheader("Prognoza na zbiorze walidacyjnym")

min_date = rf_forecast["Date"].min().date()
max_date = rf_forecast["Date"].max().date()

selected_dates = st.date_input(
    "Zakres dat",
    value=(
        min_date,
        max_date
    ),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, (tuple, list)):

    if len(selected_dates) == 2:

        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = pd.Timestamp(
            selected_dates[1]
        )

    else:

        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = start_date

else:

    start_date = pd.Timestamp(
        selected_dates
    )

    end_date = start_date


filtered_forecast = rf_forecast[
    (
        rf_forecast["Date"] >= start_date
    )
    &
    (
        rf_forecast["Date"] <= end_date
    )
].copy()


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_forecast["Date"],
        y=filtered_forecast["Actual"],
        mode="lines+markers",
        name="Rzeczywista sprzedaż"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered_forecast["Date"],
        y=filtered_forecast["Forecast"],
        mode="lines+markers",
        name="Random Forest"
    )
)

fig.update_layout(
    title="Rzeczywista sprzedaż vs prognoza",
    xaxis_title="Data",
    yaxis_title="Sprzedaż [szt.]",
    hovermode="x unified",
    height=550
)


st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

rf = rf_metrics.iloc[0]

st.subheader("Wyniki modelu Random Forest")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MAE",
    f"{rf['MAE']:.2f}"
)

col2.metric(
    "RMSE",
    f"{rf['RMSE']:.2f}"
)

col3.metric(
    "MAPE",
    f"{rf['MAPE']:.2f}%"
)

col4.metric(
    "Bias",
    f"{rf['Bias']:+.2f}"
)

st.caption(
    "MAE i RMSE podano w sztukach. "
    "MAPE przedstawia średni błąd procentowy. "
)

st.markdown(
    f"""
    Wyniki wskazują, że model **dobrze odwzorowuje rzeczywisty poziom i zmienność sprzedaży**.
    MAPE na poziomie **{rf['MAPE']:.2f}%** oznacza, że prognozy przeciętnie odbiegają
    od rzeczywistych wartości o mniej niż **{round(rf['MAPE'])}%**, co przy dziennej
    sprzedaży liczonej w tysiącach sztuk jest niewielkim błędem.

    Wyższa wartość RMSE (**{rf['RMSE']:.2f}**) względem MAE (**{rf['MAE']:.2f}**)
    sugeruje występowanie pojedynczych dni, w których błąd prognozy był większy,
    jednak nie dominują one w całym okresie.

    Bias na poziomie **{rf['Bias']:+.2f}** wskazuje na
    {"lekką tendencję do zawyżania prognoz" if rf["Bias"] > 0 else "lekką tendencję do zaniżania prognoz"}.
    """
)

with st.expander(
    "Pokaż szczegółowe wartości prognozy"
):

    table = filtered_forecast.copy()

    table["Date"] = (
        table["Date"]
        .dt
        .strftime("%Y-%m-%d")
    )

    numeric_columns = [
        "Actual",
        "Forecast",
        "Error",
        "Absolute_Error"
    ]

    for column in numeric_columns:

        if column in table.columns:

            table[column] = (
                table[column]
                .round(2)
            )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


st.divider()

st.subheader("Porównanie z modelem bazowym")

baseline = baseline_metrics.iloc[0]


comparison = pd.DataFrame({
    "Metryka": [
        "MAE",
        "RMSE",
        "MAPE",
        "Bias"
    ],

    "Random Forest": [
        rf["MAE"],
        rf["RMSE"],
        rf["MAPE"],
        rf["Bias"]
    ],

    "Seasonal Naive": [
        baseline["MAE"],
        baseline["RMSE"],
        baseline["MAPE"],
        baseline["Bias"]
    ]
})


st.dataframe(
    comparison.style.format({
        "Random Forest": "{:.2f}",
        "Seasonal Naive": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

comparison_forecast = (
    rf_forecast[
        [
            "Date",
            "Actual",
            "Forecast"
        ]
    ]
    .rename(
        columns={
            "Forecast":
                "Random Forest"
        }
    )
)


baseline_for_merge = (
    baseline_forecast[
        [
            "Date",
            baseline_prediction_column
        ]
    ]
    .rename(
        columns={
            baseline_prediction_column:
                "Seasonal Naive"
        }
    )
)


comparison_forecast = (
    comparison_forecast
    .merge(
        baseline_for_merge,
        on="Date",
        how="left"
    )
)


comparison_filtered = comparison_forecast[
    (
        comparison_forecast["Date"]
        >= start_date
    )
    &
    (
        comparison_forecast["Date"]
        <= end_date
    )
]


fig_comparison = go.Figure()


fig_comparison.add_trace(
    go.Scatter(
        x=comparison_filtered["Date"],
        y=comparison_filtered["Actual"],
        mode="lines",
        name="Rzeczywista sprzedaż",
        line=dict(width=3)
    )
)


fig_comparison.add_trace(
    go.Scatter(
        x=comparison_filtered["Date"],
        y=comparison_filtered["Random Forest"],
        mode="lines",
        name="Random Forest"
    )
)


fig_comparison.add_trace(
    go.Scatter(
        x=comparison_filtered["Date"],
        y=comparison_filtered["Seasonal Naive"],
        mode="lines",
        name="Seasonal Naive"
    )
)


fig_comparison.update_layout(
    title="Random Forest vs model bazowy",
    xaxis_title="Data",
    yaxis_title="Sprzedaż [szt.]",
    hovermode="x unified",
    height=500
)



st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


st.divider()

st.subheader("Prognoza w kontekście danych historycznych")


history = (
    train
    .tail(120)
    [
        [
            "Date",
            "Units Sold"
        ]
    ]
    .copy()
)


fig_history = go.Figure()


fig_history.add_trace(
    go.Scatter(
        x=history["Date"],
        y=history["Units Sold"],
        mode="lines",
        name="Dane treningowe"
    )
)


fig_history.add_trace(
    go.Scatter(
        x=rf_forecast["Date"],
        y=rf_forecast["Actual"],
        mode="lines",
        name="Validation - rzeczywiste"
    )
)


fig_history.add_trace(
    go.Scatter(
        x=rf_forecast["Date"],
        y=rf_forecast["Forecast"],
        mode="lines",
        name="Validation - prognoza",
        line=dict(dash="dash")
    )
)


fig_history.add_vline(
    x=validation["Date"].min().timestamp() * 1000,
    line_dash="dash",
    annotation_text="Początek validation",
    annotation_position="top"
)


fig_history.update_layout(
    title="Dane treningowe i 28-dniowa prognoza",
    xaxis_title="Data",
    yaxis_title="Sprzedaż [szt.]",
    hovermode="x unified",
    height=500
)

st.plotly_chart(
    fig_history,
    use_container_width=True
)

