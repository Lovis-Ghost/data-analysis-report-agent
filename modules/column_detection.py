import pandas as pd


def detect_column_types(df):
    id_cols = []

    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / len(df)

        if "id" in col.lower() or unique_ratio > 0.95:
            id_cols.append(col)

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in id_cols:
            continue

        unique_count = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]) and unique_count > 10:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, id_cols
