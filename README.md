## Wybór cech

W pierwszym etapie odrzuciłem czynniki wewnętrzne zależne od decyzji firmy:
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
