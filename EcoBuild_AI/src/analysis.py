"""
analysis.py
-----------
Exploratory Data Analysis functions.
Each function returns a Matplotlib Figure (or dict of Figures) that can be
rendered either inline in a notebook or displayed in Streamlit via
st.pyplot(fig).

No figures are displayed or saved here — callers decide what to do.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from src.data_loader import FEATURE_COLS, TARGET_COLS

# ── Shared style settings ──────────────────────────────────────────────────
PALETTE = "Blues_d"
TARGET_COLORS = {"Heating_Load": "#e05c5c", "Cooling_Load": "#3b82d4"}
FIGURE_DPI = 110

# Friendly short labels for axes
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


def _style():
    """Apply a consistent, clean style to all plots."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


# ── 1. Dataset Overview ────────────────────────────────────────────────────
def plot_dataset_overview(df: pd.DataFrame) -> plt.Figure:
    """
    Box-plots of all 8 features in a 2×4 grid.
    Shows the range and spread of every input variable.
    """
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#cfe2f3", color="#3b82d4"),
                        medianprops=dict(color="#e05c5c", linewidth=2),
                        whiskerprops=dict(color="#3b82d4"),
                        capprops=dict(color="#3b82d4"),
                        flierprops=dict(marker="o", color="#3b82d4", alpha=0.4))
        axes[i].set_title(SHORT_LABELS.get(col, col))
        axes[i].set_ylabel("Value")

    fig.suptitle("Building Feature Distributions (Box Plots)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 2. Heating Load Distribution ──────────────────────────────────────────
def plot_heating_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE for Heating Load (Y1)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Heating_Load"], bins=30, kde=True, color="#e05c5c",
                 ax=ax, edgecolor="white", alpha=0.8)
    ax.axvline(df["Heating_Load"].mean(), color="#333", linestyle="--",
               linewidth=1.5, label=f"Mean = {df['Heating_Load'].mean():.1f}")
    ax.axvline(df["Heating_Load"].median(), color="#555", linestyle=":",
               linewidth=1.5, label=f"Median = {df['Heating_Load'].median():.1f}")
    ax.set_xlabel("Heating Load (kWh/m²)")
    ax.set_ylabel("Count")
    ax.set_title("Heating Load Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 3. Cooling Load Distribution ──────────────────────────────────────────
def plot_cooling_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE for Cooling Load (Y2)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Cooling_Load"], bins=30, kde=True, color="#3b82d4",
                 ax=ax, edgecolor="white", alpha=0.8)
    ax.axvline(df["Cooling_Load"].mean(), color="#333", linestyle="--",
               linewidth=1.5, label=f"Mean = {df['Cooling_Load'].mean():.1f}")
    ax.axvline(df["Cooling_Load"].median(), color="#555", linestyle=":",
               linewidth=1.5, label=f"Median = {df['Cooling_Load'].median():.1f}")
    ax.set_xlabel("Cooling Load (kWh/m²)")
    ax.set_ylabel("Count")
    ax.set_title("Cooling Load Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 4. Correlation Heatmap ─────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Full Pearson correlation heatmap of all 10 columns."""
    _style()
    corr = df.corr(numeric_only=True)

    # Rename for shorter axis labels
    corr.columns = [SHORT_LABELS.get(c, c) for c in corr.columns]
    corr.index   = [SHORT_LABELS.get(c, c) for c in corr.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True   # show lower triangle + diagonal
    mask[np.diag_indices_from(mask)] = False  # keep diagonal

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 5. Features vs Heating Load ────────────────────────────────────────────
def plot_features_vs_heating(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plots of each of the 8 features against Heating Load.
    """
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df["Heating_Load"],
                        alpha=0.35, s=15, color="#e05c5c", edgecolors="none")
        # trend line
        z = np.polyfit(df[col], df["Heating_Load"], 1)
        p = np.poly1d(z)
        xr = np.linspace(df[col].min(), df[col].max(), 100)
        axes[i].plot(xr, p(xr), color="#333", linewidth=1.2, linestyle="--")
        axes[i].set_xlabel(SHORT_LABELS.get(col, col), fontsize=9)
        axes[i].set_ylabel("Heating Load", fontsize=9)

    fig.suptitle("Building Features vs Heating Load (kWh/m²)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 6. Features vs Cooling Load ────────────────────────────────────────────
def plot_features_vs_cooling(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plots of each of the 8 features against Cooling Load.
    """
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df["Cooling_Load"],
                        alpha=0.35, s=15, color="#3b82d4", edgecolors="none")
        z = np.polyfit(df[col], df["Cooling_Load"], 1)
        p = np.poly1d(z)
        xr = np.linspace(df[col].min(), df[col].max(), 100)
        axes[i].plot(xr, p(xr), color="#333", linewidth=1.2, linestyle="--")
        axes[i].set_xlabel(SHORT_LABELS.get(col, col), fontsize=9)
        axes[i].set_ylabel("Cooling Load", fontsize=9)

    fig.suptitle("Building Features vs Cooling Load (kWh/m²)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 7. Glazing Area Relationships ──────────────────────────────────────────
def plot_glazing_relationships(df: pd.DataFrame) -> plt.Figure:
    """
    Box plots of Heating and Cooling Load grouped by Glazing Area.
    Side-by-side subplots.
    """
    _style()
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
    axes[0].set_ylabel("Heating Load (kWh/m²)")

    sns.boxplot(data=df_plot, x="Glazing_Label", y="Cooling_Load",
                order=order, hue="Glazing_Label", palette="Blues",
                legend=False, ax=axes[1])
    axes[1].set_title("Glazing Area vs Cooling Load")
    axes[1].set_xlabel("Glazing Area (% of floor area)")
    axes[1].set_ylabel("Cooling Load (kWh/m²)")

    fig.suptitle("Impact of Glazing Area on Energy Loads",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 8. Orientation Comparison ──────────────────────────────────────────────
def plot_orientation_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Bar charts of mean Heating and Cooling Load by Orientation (2=N,3=E,4=S,5=W).
    """
    _style()
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
    ax.set_ylabel("Mean Energy Load (kWh/m²)")
    ax.set_title("Mean Energy Loads by Building Orientation")
    ax.legend()

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ── 9. Heating Load vs Cooling Load ───────────────────────────────────────
def plot_heating_vs_cooling(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of Y1 vs Y2 coloured by Overall Height.
    Shows the relationship between the two targets.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {3.5: "#f4a261", 7.0: "#264653"}
    for height, grp in df.groupby("Overall_Height"):
        ax.scatter(grp["Heating_Load"], grp["Cooling_Load"],
                   label=f"Height = {height} m",
                   color=colors.get(height, "gray"),
                   alpha=0.5, s=20, edgecolors="none")

    # Add 1:1 reference line
    lo = min(df["Heating_Load"].min(), df["Cooling_Load"].min())
    hi = max(df["Heating_Load"].max(), df["Cooling_Load"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="1:1 line")

    ax.set_xlabel("Heating Load (kWh/m²)")
    ax.set_ylabel("Cooling Load (kWh/m²)")
    ax.set_title("Heating Load vs Cooling Load\n(coloured by Building Height)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Convenience: return all figures as a dict ─────────────────────────────
def generate_all_figures(df: pd.DataFrame) -> dict:
    """
    Return a dict of {name: Figure} for every EDA plot.
    Used by the Streamlit app to cache all charts at once.
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
