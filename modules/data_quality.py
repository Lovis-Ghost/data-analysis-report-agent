from modules.column_detection import detect_column_types


def calculate_data_quality_score(df):
    rows, cols = df.shape
    total_cells = rows * cols
    missing_ratio = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / rows if rows > 0 else 0
    _, _, id_cols = detect_column_types(df)
    id_ratio = len(id_cols) / cols if cols > 0 else 0

    score = 100
    suggestions = []

    if missing_ratio > 0:
        missing_deduction = min(40, missing_ratio * 100)
        score -= missing_deduction
        suggestions.append(
            f"Missing values affect {missing_ratio:.1%} of all cells. "
            "Consider filling or removing missing values before modeling."
        )
    else:
        suggestions.append("No missing values were detected.")

    if duplicate_ratio > 0:
        duplicate_deduction = min(30, duplicate_ratio * 100)
        score -= duplicate_deduction
        suggestions.append(
            f"Duplicate rows make up {duplicate_ratio:.1%} of the dataset. "
            "Review whether these records should be removed."
        )
    else:
        suggestions.append("No duplicate rows were detected.")

    if len(id_cols) > 1 or id_ratio > 0.2:
        id_deduction = min(10, len(id_cols) * 2)
        score -= id_deduction
        suggestions.append(
            f"{len(id_cols)} ID-like columns were detected. "
            "Exclude ID columns from charts and machine learning features."
        )
    elif len(id_cols) == 1:
        suggestions.append(
            f"1 ID-like column was detected: {id_cols[0]}. "
            "It should usually be excluded from modeling."
        )

    score = max(0, round(score))
    return score, suggestions
