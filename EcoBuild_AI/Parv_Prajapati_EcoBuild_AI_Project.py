"""
EcoBuild_AI_Project.py
======================
EcoBuild AI — Building Energy Efficiency Analysis & Prediction
Complete consolidated project file for internship submission.

This single file contains the full implementation of:
  Section 1 — Constants & Configuration
  Section 2 — Data Loader
  Section 3 — Data Cleaning
  Section 4 — Exploratory Data Analysis (EDA)
  Section 5 — Machine Learning Models (Linear Regression + Random Forest)
  Section 6 — Prediction
  Section 7 — Business Insights & Recommendations
  Section 8 — Streamlit Application (all 5 pages)

Dataset : Data/building_energy_efficiency.xlsx
          UCI Energy Efficiency dataset (Tsanas & Xifara, 2012)
          768 building simulations, 8 features, 2 targets

Run with:
    streamlit run EcoBuild_AI_Project.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import streamlit as st


# SECTION 1 — CONSTANTS & CONFIGURATION

# Column rename map: raw Excel names → meaningful physical names
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

# Paths — all relative to the directory containing this file
_ROOT        = Path(__file__).parent
DATA_PATH    = _ROOT / "Data" / "building_energy_efficiency.xlsx"
MODELS_DIR   = _ROOT / "models"
LR_PATH      = MODELS_DIR / "linear_regression.joblib"
RF_PATH      = MODELS_DIR / "random_forest.joblib"
SCALER_PATH  = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
PROCESSED_CSV = _ROOT / "data" / "processed_dataset.csv"

# ML hyper-parameters
RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 80 / 20 train-test split

# Plot style
FIGURE_DPI = 110
SHORT_LABELS = {
    "Relative_Compactness":       "Rel. Compactness",
    "Surface_Area":               "Surface Area",
    "Wall_Area":                  "Wall Area",
    "Roof_Area":                  "Roof Area",
    "Overall_Height":             "Overall Height",
    "Orientation":                "Orientation",
    "Glazing_Area":               "Glazing Area",
    "Glazing_Area_Distribution":  "Glazing Distribution",
    "Heating_Load":               "Heating Load",
    "Cooling_Load":               "Cooling Load",
}


# SECTION 2 — DATA LOADER

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Read the Excel file and return a raw DataFrame.
    Only the first sheet is used; any columns beyond column J are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}\n"
            "Place 'building_energy_efficiency.xlsx' inside the 'Data/' folder."
        )
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.iloc[:, :10]   # keep only the 10 data columns (A-J)
    return df


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load, clean, and rename the dataset.

    Steps:
      1. Read raw Excel (first sheet, first 10 columns).
      2. Drop completely empty rows (528 trailing blank rows).
      3. Rename columns from X1-X8 / Y1-Y2 to meaningful names.
      4. Reset index.
      5. Cast all columns to float64.

    Returns a 768-row, 10-column DataFrame ready for analysis.
    """
    df = load_raw(path)
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.rename(columns=COLUMN_RENAME)
    expected = list(COLUMN_RENAME.values())
    df = df[[c for c in expected if c in df.columns]]
    df = df.astype(float)
    return df


# SECTION 3 — DATA CLEANING

def get_data_quality_report(df: pd.DataFrame) -> dict:
    """Return a dictionary of data-quality metrics."""
    return {
        "total_rows":     len(df),
        "total_cols":     len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "feature_count":  len(FEATURE_COLS),
        "target_count":   len(TARGET_COLS),
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted descriptive statistics table (transposed)."""
    stats = df.describe().T.round(4)
    stats.index.name = "Feature"
    return stats


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_CSV) -> Path:
    """Export the cleaned DataFrame to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA)

def _eda_style():
    """Apply a consistent, clean style to all EDA plots."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi":        FIGURE_DPI,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })


def plot_dataset_overview(df: pd.DataFrame) -> plt.Figure:
    """Box-plots of all 8 input features in a 2x4 grid."""
    _eda_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].boxplot(
            df[col].dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor="#cfe2f3", color="#3b82d4"),
            medianprops=dict(color="#e05c5c", linewidth=2),
            whiskerprops=dict(color="#3b82d4"),
            capprops=dict(color="#3b82d4"),
            flierprops=dict(marker="o", color="#3b82d4", alpha=0.4),
        )
        axes[i].set_title(SHORT_LABELS.get(col, col))
        axes[i].set_ylabel("Value")
    fig.suptitle("Building Feature Distributions (Box Plots)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_heating_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE for Heating Load (Y1)."""
    _eda_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Heating_Load"], bins=30, kde=True,
                 color="#e05c5c", ax=ax, edgecolor="white", alpha=0.8)
    ax.axvline(df["Heating_Load"].mean(), color="#333", linestyle="--",
               linewidth=1.5, label=f"Mean = {df['Heating_Load'].mean():.1f}")
    ax.axvline(df["Heating_Load"].median(), color="#555", linestyle=":",
               linewidth=1.5, label=f"Median = {df['Heating_Load'].median():.1f}")
    ax.set_xlabel("Heating Load (kWh/m\u00b2)")
    ax.set_ylabel("Count")
    ax.set_title("Heating Load Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cooling_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE for Cooling Load (Y2)."""
    _eda_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Cooling_Load"], bins=30, kde=True,
                 color="#3b82d4", ax=ax, edgecolor="white", alpha=0.8)
    ax.axvline(df["Cooling_Load"].mean(), color="#333", linestyle="--",
               linewidth=1.5, label=f"Mean = {df['Cooling_Load'].mean():.1f}")
    ax.axvline(df["Cooling_Load"].median(), color="#555", linestyle=":",
               linewidth=1.5, label=f"Median = {df['Cooling_Load'].median():.1f}")
    ax.set_xlabel("Cooling Load (kWh/m\u00b2)")
    ax.set_ylabel("Count")
    ax.set_title("Cooling Load Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Full Pearson correlation heatmap of all 10 columns (lower triangle)."""
    _eda_style()
    corr = df.corr(numeric_only=True)
    corr.columns = [SHORT_LABELS.get(c, c) for c in corr.columns]
    corr.index   = [SHORT_LABELS.get(c, c) for c in corr.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False   # show diagonal too

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_features_vs_heating(df: pd.DataFrame) -> plt.Figure:
    """Scatter plots of each of the 8 features against Heating Load with trend lines."""
    _eda_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df["Heating_Load"],
                        alpha=0.35, s=15, color="#e05c5c", edgecolors="none")
        z = np.polyfit(df[col], df["Heating_Load"], 1)
        xr = np.linspace(df[col].min(), df[col].max(), 100)
        axes[i].plot(xr, np.poly1d(z)(xr), color="#333", linewidth=1.2, linestyle="--")
        axes[i].set_xlabel(SHORT_LABELS.get(col, col), fontsize=9)
        axes[i].set_ylabel("Heating Load", fontsize=9)
    fig.suptitle("Building Features vs Heating Load (kWh/m\u00b2)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_features_vs_cooling(df: pd.DataFrame) -> plt.Figure:
    """Scatter plots of each of the 8 features against Cooling Load with trend lines."""
    _eda_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df["Cooling_Load"],
                        alpha=0.35, s=15, color="#3b82d4", edgecolors="none")
        z = np.polyfit(df[col], df["Cooling_Load"], 1)
        xr = np.linspace(df[col].min(), df[col].max(), 100)
        axes[i].plot(xr, np.poly1d(z)(xr), color="#333", linewidth=1.2, linestyle="--")
        axes[i].set_xlabel(SHORT_LABELS.get(col, col), fontsize=9)
        axes[i].set_ylabel("Cooling Load", fontsize=9)
    fig.suptitle("Building Features vs Cooling Load (kWh/m\u00b2)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_glazing_relationships(df: pd.DataFrame) -> plt.Figure:
    """Box plots of Heating and Cooling Load grouped by Glazing Area category."""
    _eda_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    glazing_labels = {0.0: "0%", 0.1: "10%", 0.25: "25%", 0.4: "40%"}
    df_plot = df.copy()
    df_plot["Glazing_Label"] = df_plot["Glazing_Area"].map(glazing_labels)
    order = ["0%", "10%", "25%", "40%"]

    sns.boxplot(data=df_plot, x="Glazing_Label", y="Heating_Load",
                order=order, hue="Glazing_Label", palette="Reds",
                legend=False, ax=axes[0])
    axes[0].set_title("Glazing Area vs Heating Load")
    axes[0].set_xlabel("Glazing Area (% of floor area)")
    axes[0].set_ylabel("Heating Load (kWh/m\u00b2)")

    sns.boxplot(data=df_plot, x="Glazing_Label", y="Cooling_Load",
                order=order, hue="Glazing_Label", palette="Blues",
                legend=False, ax=axes[1])
    axes[1].set_title("Glazing Area vs Cooling Load")
    axes[1].set_xlabel("Glazing Area (% of floor area)")
    axes[1].set_ylabel("Cooling Load (kWh/m\u00b2)")

    fig.suptitle("Impact of Glazing Area on Energy Loads",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_orientation_comparison(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of mean Heating and Cooling Load by orientation."""
    _eda_style()
    orient_map = {2: "North", 3: "East", 4: "South", 5: "West"}
    df_plot = df.copy()
    df_plot["Orientation_Label"] = df_plot["Orientation"].map(orient_map)
    grouped = df_plot.groupby("Orientation_Label")[["Heating_Load", "Cooling_Load"]].mean()
    grouped = grouped.reindex(["North", "East", "South", "West"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(grouped))
    width = 0.38
    bars1 = ax.bar(x - width / 2, grouped["Heating_Load"], width,
                   label="Heating Load", color="#e05c5c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, grouped["Cooling_Load"], width,
                   label="Cooling Load", color="#3b82d4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel("Mean Energy Load (kWh/m\u00b2)")
    ax.set_title("Mean Energy Loads by Building Orientation")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_heating_vs_cooling(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of Heating Load vs Cooling Load, coloured by Overall Height."""
    _eda_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {3.5: "#f4a261", 7.0: "#264653"}
    for height, grp in df.groupby("Overall_Height"):
        ax.scatter(grp["Heating_Load"], grp["Cooling_Load"],
                   label=f"Height = {height} m",
                   color=colors.get(height, "gray"),
                   alpha=0.5, s=20, edgecolors="none")
    lo = min(df["Heating_Load"].min(), df["Cooling_Load"].min())
    hi = max(df["Heating_Load"].max(), df["Cooling_Load"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="1:1 line")
    ax.set_xlabel("Heating Load (kWh/m\u00b2)")
    ax.set_ylabel("Cooling Load (kWh/m\u00b2)")
    ax.set_title("Heating Load vs Cooling Load\n(coloured by Building Height)")
    ax.legend()
    fig.tight_layout()
    return fig


def generate_all_figures(df: pd.DataFrame) -> dict:
    """
    Generate all 9 EDA figures at once.
    Returns a dict of {name: matplotlib.Figure}.
    """
    return {
        "overview":         plot_dataset_overview(df),
        "heating_dist":     plot_heating_distribution(df),
        "cooling_dist":     plot_cooling_distribution(df),
        "correlation":      plot_correlation_heatmap(df),
        "features_heating": plot_features_vs_heating(df),
        "features_cooling": plot_features_vs_cooling(df),
        "glazing":          plot_glazing_relationships(df),
        "orientation":      plot_orientation_comparison(df),
        "h_vs_c":           plot_heating_vs_cooling(df),
    }

