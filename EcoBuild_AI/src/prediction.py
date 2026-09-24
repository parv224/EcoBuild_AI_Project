"""
prediction.py
-------------
Provides:
  1. Single-sample prediction using the trained Random Forest model.
  2. Feature importance visualization functions.

Used by app.py for the interactive prediction section.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_loader import FEATURE_COLS, TARGET_COLS
from src.model import load_models, get_feature_importances


# ── Prediction ────────────────────────────────────────────────────────────
def predict_loads(input_values: dict) -> dict:
    """
    Predict Heating Load and Cooling Load for a single building.

    Parameters
    ----------
    input_values : dict
        Keys must match FEATURE_COLS exactly.
        Example:
            {
              "Relative_Compactness": 0.76,
              "Surface_Area": 661.5,
              ...
            }

    Returns
    -------
    dict  {"Heating_Load": float, "Cooling_Load": float}
    """
    _, rf_model, _ = load_models()   # Random Forest (no scaling needed)

    # Build input array in the correct feature order
    x = np.array([[input_values[col] for col in FEATURE_COLS]])

    preds = rf_model.predict(x)[0]   # shape: (2,)
    return {
        "Heating_Load": round(float(preds[0]), 2),
        "Cooling_Load": round(float(preds[1]), 2),
    }


# ── Feature Importance Plots ──────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })


SHORT_LABELS = {
    "Relative_Compactness":      "Rel. Compactness",
    "Surface_Area":              "Surface Area",
    "Wall_Area":                 "Wall Area",
    "Roof_Area":                 "Roof Area",
    "Overall_Height":            "Overall Height",
    "Orientation":               "Orientation",
    "Glazing_Area":              "Glazing Area",
    "Glazing_Area_Distribution": "Glazing Distribution",
}


def plot_feature_importance_heating() -> plt.Figure:
    """
    Horizontal bar chart of feature importances for Heating Load.
    """
    _style()
    importances = get_feature_importances()
    imp = importances["Heating_Load"]
    labels = [SHORT_LABELS.get(i, i) for i in imp.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e05c5c" if v >= imp.median() else "#f4a4a4" for v in imp.values]
    ax.barh(labels, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Feature Importance — Heating Load")
    ax.axvline(0, color="gray", linewidth=0.8)

    for i, v in enumerate(imp.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def plot_feature_importance_cooling() -> plt.Figure:
    """
    Horizontal bar chart of feature importances for Cooling Load.
    """
    _style()
    importances = get_feature_importances()
    imp = importances["Cooling_Load"]
    labels = [SHORT_LABELS.get(i, i) for i in imp.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3b82d4" if v >= imp.median() else "#a4c4e8" for v in imp.values]
    ax.barh(labels, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Feature Importance — Cooling Load")
    ax.axvline(0, color="gray", linewidth=0.8)

    for i, v in enumerate(imp.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig
