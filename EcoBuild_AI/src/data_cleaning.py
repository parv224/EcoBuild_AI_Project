"""
data_cleaning.py
----------------
Data quality checks and pre-processing steps:
  - Missing value report
  - Duplicate detection
  - Descriptive statistics
  - Clean data export to CSV
"""

import pandas as pd
from pathlib import Path
from src.data_loader import load_clean, FEATURE_COLS, TARGET_COLS

PROCESSED_PATH = Path("data/processed_dataset.csv")


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with data-quality metrics for the dashboard.
    """
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "feature_count": len(FEATURE_COLS),
        "target_count": len(TARGET_COLS),
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a nicely formatted descriptive statistics table.
    """
    stats = df.describe().T.round(4)
    stats.index.name = "Feature"
    return stats


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    """
    Export the cleaned DataFrame to CSV for reproducibility.
    Does not modify the original Excel file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = load_clean()

    report = get_data_quality_report(df)
    print("=== Data Quality Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n=== Descriptive Statistics ===")
    print(get_descriptive_stats(df))

    out = save_processed(df)
    print(f"\nProcessed dataset saved to: {out}")
