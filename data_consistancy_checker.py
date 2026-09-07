import pandas as pd

# Ścieżka do pliku
FILE_PATH = "data/demand_forecasting.csv"

# WCZYTANIE DANYCH
df = pd.read_csv(FILE_PATH)

print("PODSTAWOWA ANALIZA DANYCH")

print("\nLiczba wierszy:", len(df))
print("Liczba kolumn:", len(df.columns))

# TYPY DANYCH

print("\nTYPY DANYCH")

print(df.dtypes)

# BRAKUJĄCE WARTOŚCI

print("\nBRAKUJĄCE WARTOŚCI")

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "Liczba braków": missing,
    "Procent braków": (missing / len(df) * 100).round(2)
})

print(missing_table)

print(
    "\nŁączna liczba brakujących wartości:",
    missing.sum()
)

# DUPLIKATY

print("\nDUPLIKATY")

duplicates = df.duplicated().sum()

print("Liczba zduplikowanych wierszy:", duplicates)

if duplicates > 0:
    print("\nPrzykładowe duplikaty:")
    print(df[df.duplicated(keep=False)].head(10))

# AUTOMATYCZNE WYKRYWANIE DAT

print("\nKOLUMNY Z DATAMI")

date_columns = []

date_keywords = [
    "date",
    "datetime",
    "timestamp",
    "time"
]

text_columns = df.select_dtypes(
    include=["object", "str"]
).columns

for column in text_columns:

    column_name = column.lower()
    if any(keyword in column_name for keyword in date_keywords):

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()
        if valid_ratio >= 0.9:

            df[column] = converted
            date_columns.append(column)


if date_columns:

    for column in date_columns:

        print(f"\n{column}")
        print("Najstarsza data:", df[column].min())
        print("Najnowsza data:", df[column].max())

else:

    print("Nie znaleziono kolumn z datami.")

# KOLUMNY LICZBOWE

print("\nSTATYSTYKI KOLUMN LICZBOWYCH")

numeric_columns = df.select_dtypes(
    include="number"
).columns

if len(numeric_columns) > 0:

    numeric_stats = (
        df[numeric_columns]
        .describe()
        .T[
            [
                "count",
                "min",
                "max",
                "mean",
                "50%",
                "std"
            ]
        ]
        .rename(columns={"50%": "median"})
    )

    print(numeric_stats)

else:
    print("Nie znaleziono kolumn liczbowych.")

# POTENCJALNIE PODEJRZANE WARTOŚCI LICZBOWE
print("\nWARTOŚCI UJEMNE I ZEROWE")

for column in numeric_columns:

    negative = (df[column] < 0).sum()
    zero = (df[column] == 0).sum()

    print(
        f"{column}: "
        f"ujemne = {negative}, "
        f"zera = {zero}"
    )

# KOLUMNY KATEGORYCZNE / TEKSTOWE
print("\nKOLUMNY KATEGORYCZNE")

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

if len(categorical_columns) > 0:

    for column in categorical_columns:

        unique_count = df[column].nunique()

        print(f"\n{column}")
        print("Liczba unikalnych wartości:", unique_count)
        if unique_count <= 30:
            print(df[column].value_counts(dropna=False))

        else:
            print("Najczęstsze wartości:")
            print(
                df[column]
                .value_counts(dropna=False)
                .head(10)
            )

else:
    print("Nie znaleziono kolumn kategorycznych.")

# POTENCJALNE KOLUMNY BINARNE

print("\nPOTENCJALNE ZMIENNE BINARNE")

binary_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=True) == 2
]

if binary_columns:

    for column in binary_columns:

        print(f"\n{column}")
        print(df[column].value_counts(dropna=False))

else:
    print("Nie znaleziono zmiennych binarnych.")

# PODSUMOWANIE

print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)

print("Wiersze:", len(df))
print("Kolumny:", len(df.columns))
print("Braki:", df.isna().sum().sum())
print("Duplikaty:", duplicates)

print("Kolumny liczbowe:", len(numeric_columns))
print("Kolumny kategoryczne:", len(categorical_columns))
print("Kolumny z datami:", len(date_columns))
print("Kolumny binarne:", len(binary_columns))
