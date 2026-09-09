from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# KONFIGURACJA STRONY
# =========================================================

st.set_page_config(
    page_title="Prognoza sprzedaży w sztukach",
    page_icon="assets/logo.webp",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

TOP_PRODUCTS = ["P0007", "P0004", "P0009"]


# =========================================================
# WCZYTYWANIE CSV
# =========================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def prepare_dates(*dataframes):
    for df in dataframes:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])


def get_date_range(dataframe, key):
    min_date = dataframe["Date"].min().date()
    max_date = dataframe["Date"].max().date()

    selected_dates = st.date_input(
        "Zakres dat",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=key
    )

    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date = pd.Timestamp(selected_dates[0])
            end_date = pd.Timestamp(selected_dates[1])
        else:
            start_date = pd.Timestamp(selected_dates[0])
            end_date = start_date
    else:
        start_date = pd.Timestamp(selected_dates)
        end_date = start_date

    return start_date, end_date


def filter_dates(dataframe, start_date, end_date):
    return dataframe[
        (dataframe["Date"] >= start_date)
        & (dataframe["Date"] <= end_date)
    ].copy()



st.sidebar.title("Menu")

page = st.sidebar.radio(
    "Wybierz widok",
    [
        "**Całkowita dzienna sprzedaż**",
        "**3 najlepsze produkty**"
    ]
)


# =========================================================
# STRONA GŁÓWNA
# =========================================================

def show_total_sales():

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

    try:
        train = load_csv(TRAIN_FILE)
        validation = load_csv(VALIDATION_FILE)
        rf_forecast = load_csv(RF_FORECAST_FILE)
        rf_metrics = load_csv(RF_METRICS_FILE)
        baseline_forecast = load_csv(BASELINE_FORECAST_FILE)
        baseline_metrics = load_csv(BASELINE_METRICS_FILE)

    except FileNotFoundError as error:
        st.error("Nie znaleziono jednego z wymaganych plików.")
        st.code(str(error))
        st.stop()

    prepare_dates(
        train,
        validation,
        rf_forecast,
        baseline_forecast
    )

    baseline_prediction_candidates = [
        "Forecast",
        "Prediction",
        "Baseline",
        "Baseline_Forecast"
    ]

    baseline_prediction_column = next(
        (
            column
            for column in baseline_prediction_candidates
            if column in baseline_forecast.columns
        ),
        None
    )

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

    st.title("Całkowita dzienna sprzedaż")

    st.markdown(
        """
        Wyniki **28-dniowej prognozy całkowitej dziennej sprzedaży**
        wygenerowanej modelem **Random Forest** oraz porównanie
        z modelem bazowym **Seasonal Naive (previous year)**.
        """
    )

    st.divider()
    st.subheader("Prognoza na zbiorze walidacyjnym")

    start_date, end_date = get_date_range(
        rf_forecast,
        "total_date_range"
    )

    filtered_forecast = filter_dates(
        rf_forecast,
        start_date,
        end_date
    )

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

    col1.metric("MAE", f"{rf['MAE']:.2f}")
    col2.metric("RMSE", f"{rf['RMSE']:.2f}")
    col3.metric("MAPE", f"{rf['MAPE']:.2f}%")
    col4.metric("Bias", f"{rf['Bias']:+.2f}")

    st.caption(
        "MAE i RMSE podano w sztukach. "
        "MAPE przedstawia średni błąd procentowy."
    )

    st.markdown(
        f"""
        Wyniki wskazują, że model **dobrze odwzorowuje rzeczywisty poziom i zmienność sprzedaży**.
        MAPE na poziomie **{rf['MAPE']:.2f}%** oznacza, że prognozy przeciętnie odbiegają
        od rzeczywistych wartości o mniej niż **{round(rf['MAPE'])}%**.

        Wyższa wartość RMSE (**{rf['RMSE']:.2f}**) względem MAE (**{rf['MAE']:.2f}**)
        sugeruje występowanie pojedynczych dni, w których błąd prognozy był większy.

        Bias na poziomie **{rf['Bias']:+.2f}** wskazuje na
        {"lekką tendencję do zawyżania prognoz" if rf["Bias"] > 0 else "lekką tendencję do zaniżania prognoz"}.
        """
    )

    with st.expander("Pokaż szczegółowe wartości prognozy"):

        table = filtered_forecast.copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")

        numeric_columns = [
            "Actual",
            "Forecast",
            "Error",
            "Absolute_Error"
        ]

        for column in numeric_columns:
            if column in table.columns:
                table[column] = table[column].round(2)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("Porównanie z modelem bazowym")

    baseline = baseline_metrics.iloc[0]

    comparison = pd.DataFrame({
        "Metryka": ["MAE", "RMSE", "MAPE", "Bias"],
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
            ["Date", "Actual", "Forecast"]
        ]
        .rename(
            columns={"Forecast": "Random Forest"}
        )
    )

    baseline_for_merge = (
        baseline_forecast[
            ["Date", baseline_prediction_column]
        ]
        .rename(
            columns={
                baseline_prediction_column: "Seasonal Naive"
            }
        )
    )

    comparison_forecast = comparison_forecast.merge(
        baseline_for_merge,
        on="Date",
        how="left"
    )

    comparison_filtered = filter_dates(
        comparison_forecast,
        start_date,
        end_date
    )

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
        .tail(120)[["Date", "Units Sold"]]
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


# =========================================================
# BONUS — TOP 3 PRODUKTÓW
# =========================================================

def show_top_products():

    st.title(
        "Dzienna sprzedaż 3 najlepiej sprzedających się produktów"
    )

    st.markdown(
        """
        Wyniki **28-dniowych prognoz dziennej sprzedaży**
        dla produktów **P0007, P0004 i P0009**.
        Dla każdego produktu wytrenowano osobny model
        **Random Forest**.
        """
    )

    forecasts = {}
    metrics = {}
    trains = {}
    validations = {}

    try:
        for product_id in TOP_PRODUCTS:

            product_split_dir = (
                BASE_DIR
                / "data"
                / "products"
                / "split"
                / product_id
            )

            product_output_dir = (
                BASE_DIR
                / "outputs"
                / "products"
                / "random_forest"
                / product_id
            )

            train = load_csv(
                product_split_dir / "train.csv"
            )

            validation = load_csv(
                product_split_dir / "validation.csv"
            )

            forecast = load_csv(
                product_output_dir / "forecast.csv"
            )

            metric = load_csv(
                product_output_dir / "metrics.csv"
            )

            prepare_dates(
                train,
                validation,
                forecast
            )

            trains[product_id] = train
            validations[product_id] = validation
            forecasts[product_id] = forecast
            metrics[product_id] = metric.iloc[0]

    except FileNotFoundError as error:
        st.error(
            "Nie znaleziono jednego z plików bonusu."
        )
        st.code(str(error))
        st.stop()

    st.divider()
    st.subheader("Prognoza na zbiorze walidacyjnym")

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        selected_products = st.multiselect(
            "Produkty na wykresach",
            options=TOP_PRODUCTS,
            default=TOP_PRODUCTS,
            key="product_selector"
        )

    reference_forecast = forecasts[TOP_PRODUCTS[0]]

    with col_filter2:
        start_date, end_date = get_date_range(
            reference_forecast,
            "products_date_range"
        )

    if not selected_products:
        st.warning(
            "Wybierz co najmniej jeden produkt, aby wyświetlić wykresy."
        )
        return

    # =====================================================
    # PROGNOZA VS RZECZYWISTA SPRZEDAŻ
    # =====================================================

    fig = go.Figure()

    for product_id in selected_products:

        filtered = filter_dates(
            forecasts[product_id],
            start_date,
            end_date
        )

        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["Actual"],
                mode="lines+markers",
                name=f"{product_id} - rzeczywista"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["Forecast"],
                mode="lines+markers",
                name=f"{product_id} - Random Forest",
                line=dict(dash="dash")
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

    # =====================================================
    # METRYKI
    # =====================================================

    st.divider()
    st.subheader("Wyniki modelu Random Forest")

    metrics_rows = []

    for product_id in selected_products:

        metric = metrics[product_id]

        metrics_rows.append({
            "Produkt": product_id,
            "MAE": metric["MAE"],
            "RMSE": metric["RMSE"],
            "MAPE": metric["MAPE"],
            "Bias": metric["Bias"]
        })

    metrics_table = pd.DataFrame(metrics_rows)

    st.dataframe(
        metrics_table.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "MAPE": "{:.2f}%",
            "Bias": "{:+.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE i RMSE podano w sztukach. "
        "MAPE przedstawia średni błąd procentowy. "
        "Każdy produkt posiada osobno wytrenowany model Random Forest."
    )
    st.divider()

    # =====================================================
    # INTERPRETACJA WYNIKÓW
    # =====================================================

    st.markdown("#### Interpretacja wyników")

    for product_id in selected_products:

        metric = metrics[product_id]

        mae = metric["MAE"]
        rmse = metric["RMSE"]
        mape = metric["MAPE"]
        bias = metric["Bias"]

        if mape < 10:
            accuracy_text = (
                "Model osiąga dobrą dokładność prognoz – "
                "średni błąd procentowy nie przekracza 10%."
            )
        elif mape < 15:
            accuracy_text = (
                "Model osiąga zadowalającą dokładność prognoz – "
                "średni błąd procentowy mieści się w przedziale 10–15%."
            )
        else:
            accuracy_text = (
                "Model charakteryzuje się większym błędem procentowym, "
                "dlatego prognozy należy interpretować z większą ostrożnością."
            )

        if bias > 0:
            bias_text = (
                f"Dodatni Bias ({bias:+.2f}) wskazuje na niewielką "
                "tendencję modelu do zawyżania prognoz."
            )
        elif bias < 0:
            bias_text = (
                f"Ujemny Bias ({bias:+.2f}) wskazuje na niewielką "
                "tendencję modelu do zaniżania prognoz."
            )
        else:
            bias_text = (
                "Bias równy 0 wskazuje na brak systematycznej "
                "tendencji do zawyżania lub zaniżania prognoz."
            )

        st.markdown(
            f"""
            **{product_id}** — średni bezwzględny błąd prognozy wynosi
            **{mae:.2f} szt.**, natomiast RMSE wynosi **{rmse:.2f} szt.**.
            MAPE na poziomie **{mape:.2f}%** oznacza, że prognoza odbiega
            przeciętnie od rzeczywistej sprzedaży o około **{mape:.1f}%**.
            {accuracy_text} {bias_text}
            """
        )
    # =====================================================
    # SZCZEGÓŁOWE WARTOŚCI
    # =====================================================

    with st.expander(
        "Pokaż szczegółowe wartości prognozy"
    ):

        detailed_tables = []

        for product_id in selected_products:

            table = filter_dates(
                forecasts[product_id],
                start_date,
                end_date
            )

            table.insert(
                0,
                "Product ID",
                product_id
            )

            detailed_tables.append(table)

        detailed_table = pd.concat(
            detailed_tables,
            ignore_index=True
        )

        detailed_table["Date"] = (
            detailed_table["Date"]
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
            if column in detailed_table.columns:
                detailed_table[column] = (
                    detailed_table[column]
                    .round(2)
                )

        st.dataframe(
            detailed_table,
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader(
        "Prognoza w kontekście danych historycznych"
    )

    fig_history = go.Figure()

    for product_id in selected_products:

        history = (
            trains[product_id]
            .tail(120)[["Date", "Units Sold"]]
            .copy()
        )

        forecast = forecasts[product_id]

        fig_history.add_trace(
            go.Scatter(
                x=history["Date"],
                y=history["Units Sold"],
                mode="lines",
                name=f"{product_id} - dane treningowe"
            )
        )

        fig_history.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Actual"],
                mode="lines",
                name=f"{product_id} - validation"
            )
        )

        fig_history.add_trace(
            go.Scatter(
                x=forecast["Date"],
                y=forecast["Forecast"],
                mode="lines",
                name=f"{product_id} - prognoza",
                line=dict(dash="dash")
            )
        )

    validation_start = min(
        validations[product_id]["Date"].min()
        for product_id in selected_products
    )

    fig_history.add_vline(
        x=validation_start.timestamp() * 1000,
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

if page == "**Całkowita dzienna sprzedaż**":
    show_total_sales()

else:
    show_top_products()
