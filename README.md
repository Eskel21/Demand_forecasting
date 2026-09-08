# Demand Forecasting

Projekt przedstawia narzędzie do prognozowania dziennej sprzedaży w sztukach na okres 4 tygodni. Dane zostały zagregowane do poziomu dziennego poprzez zsumowanie `Units Sold` dla wszystkich sklepów i produktów. Do prognozowania wykorzystano historyczne wartości sprzedaży oraz wybrane czynniki zewnętrzne.

Końcowa prognoza generowana jest za pomocą modelu **Random Forest**, a jej wyniki prezentowane są w interaktywnej aplikacji **Streamlit** z wykresami **Plotly**.

## Wybór cech

Na pierwszym etapie odrzuciłem czynniki wewnętrzne zależne od decyzji firmy:
`Inventory Level`, `Units Ordered`, `Discount`, `Promotion` oraz `Price`.

Następnie przeanalizowałem potencjalne czynniki zewnętrzne. Najsilniejszą zależność
ze sprzedażą wykazały `Demand` (0,9813), `Competitor Pricing` (0,9052) oraz
`Epidemic` (-0,8978). Uwzględniłem również `Seasonality`, natomiast zmienne
pogodowe wykazały znacznie słabszą zależność ze sprzedażą.

W ramach feature engineering utworzono cechy kalendarzowe oraz historyczne wartości
sprzedaży: lagi 1, 7, 14, 21 i 28 dni, a także średnie kroczące i odchylenia
standardowe dla 7, 14 i 28 dni.

Ostateczną selekcję przeprowadzono metodą **Boruta** na zbiorze treningowym.
Do modelu wybrano 7 cech:

- `Demand`
- `Competitor Pricing`
- `Epidemic`
- `UnitsSold_Lag_7`
- `UnitsSold_Lag_28`
- `UnitsSold_RollingMean_7`
- `UnitsSold_RollingMean_14`

Pozostałe cechy, w tym `Seasonality`, cechy kalendarzowe i pogodowe, zostały
odrzucone przez algorytm Boruta.

## Podział danych

Ostatnie **28 dni** danych zostało wydzielone jako zbiór walidacyjny. Model był trenowany wyłącznie na wcześniejszych obserwacjach. Zbiór walidacyjny pozostał niewykorzystany podczas wyboru modelu i został użyty do jego końcowej oceny.

## Baseline

Jako model bazowy zastosowałem **Seasonal Naive**, w którym prognoza dla danego dnia odpowiada sprzedaży z analogicznego dnia rok wcześniej.

Baseline osiągnął następujące wyniki:

| Metryka | Wynik |
|---|---:|
| MAE | 1513.61 |
| RMSE | 1951.90 |
| MAPE | 20.99% |
| Bias | 887.61 |

## Wybór modelu

Porównałem trzy modele: **Random Forest**, **HistGradientBoosting** oraz **XGBoost**. Do ich oceny wykorzystałem walidację szeregu czasowego typu expanding window z pięcioma kolejnymi okresami testowymi po 28 dni.

Najlepsze i najbardziej stabilne wyniki uzyskał **Random Forest**, który osiągnął najniższy MAE w 4 z 5 okresów. Z tego względu został wybrany do wygenerowania końcowej prognozy.

## Wyniki

Random Forest wytrenowany na całym zbiorze treningowym osiągnął na 28-dniowym zbiorze walidacyjnym:

| Metryka | Random Forest | Baseline |
|---|---:|---:|
| MAE | 218.83 | 1513.61 |
| RMSE | 265.20 | 1951.90 |
| MAPE | 2.67% | 20.99% |
| Bias | 33.76 | 887.61 |

Model znacząco poprawił wyniki względem baseline'u. MAPE na poziomie **2,67%** oznacza, że prognozy odbiegały od rzeczywistej sprzedaży średnio o mniej niż 3%. Niewielki dodatni Bias wskazuje na lekką tendencję modelu do zawyżania prognoz.

## Aplikacja

Do prezentacji wyników przygotowałem aplikację webową w **Streamlit** z wykorzystaniem wykresów **Plotly**. Aplikacja umożliwia porównanie prognozy z rzeczywistą sprzedażą i modelem bazowym, prezentuje metryki jakości oraz pozwala filtrować wyniki według wybranego zakresu dat.

## Uruchomienie

### 1. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 2. Uruchomienie aplikacji

```bash
streamlit run app.py
```

Po uruchomieniu aplikacja będzie dostępna lokalnie w przeglądarce, domyślnie pod adresem `localhost:8501`.
