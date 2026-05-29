import pandas as pd


def analyze_correlations(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None

    corr_matrix = df[numeric_cols].corr()
    top_correlations = []

    for i, col_a in enumerate(corr_matrix.columns):
        for col_b in corr_matrix.columns[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value):
                top_correlations.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": value
                })

    top_correlations = sorted(
        top_correlations,
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )[:3]

    return {
        "corr_matrix": corr_matrix,
        "top_correlations": top_correlations
    }
