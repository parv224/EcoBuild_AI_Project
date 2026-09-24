"""
data_loader.py
--------------
Loads the raw building energy efficiency Excel dataset and returns a
clean, properly typed DataFrame with meaningful column names.

Original file is never modified.
"""

import pandas as pd
from pathlib import Path

# ── Column rename map (X1–X8 → physical names, Y1–Y2 → load names) ────────
COLUMN_RENAME = {
    "X1": "Relative_Compactness",
    "X2": "Surface_Area",
    "X3": "Wall_Area",
    "X4": "Roof_Area",
    "X5": "Overall_Height",
    "X6": "Orientation",
    "X7": "Glazing_Area",
    "X8": "Glazing_Area_Distribution",
    "Y1": "Heating_Load",
    "Y2": "Cooling_Load",
}

FEATURE_COLS = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
]

TARGET_COLS = ["Heating_Load", "Cooling_Load"]

# Default path relative to project root
DEFAULT_DATA_PATH = Path("Data/building_energy_efficiency.xlsx")


def load_raw(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Read the Excel file and return a raw DataFrame.
    Only the first sheet (Φύλλο1) is used; columns K and L are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    # Keep only the 10 data columns (A–J); drop anything beyond J
    df = df.iloc[:, :10]

    return df


def load_clean(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load, clean, and rename the dataset.

    Steps:
      1. Read raw Excel (first sheet, first 10 columns).
      2. Drop completely empty rows (the 528 trailing blank rows).
      3. Rename columns from X1–X8 / Y1–Y2 to meaningful names.
      4. Reset index.
      5. Cast all columns to float64.

    Returns a 768-row, 10-column DataFrame ready for analysis.
    """
    df = load_raw(path)

    # Drop rows where every column is NaN (the 528 trailing blank rows)
    df = df.dropna(how="all").reset_index(drop=True)

    # Rename columns
    df = df.rename(columns=COLUMN_RENAME)

    # Ensure correct column selection (only the 10 expected columns)
    expected = list(COLUMN_RENAME.values())
    df = df[[c for c in expected if c in df.columns]]

    # Cast to float64 for numerical consistency
    df = df.astype(float)

    return df


if __name__ == "__main__":
    df = load_clean()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
