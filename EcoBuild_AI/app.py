"""
app.py
------
EcoBuild AI — Building Energy Efficiency Analysis & Prediction
Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project modules ────────────────────────────────────────────────────────
from src.data_loader import load_clean, FEATURE_COLS, TARGET_COLS
from src.data_cleaning import get_data_quality_report, get_descriptive_stats
from src.analysis import (
    plot_dataset_overview,
    plot_heating_distribution,
    plot_cooling_distribution,
    plot_correlation_heatmap,
    plot_features_vs_heating,
    plot_features_vs_cooling,
    plot_glazing_relationships,
    plot_orientation_comparison,
    plot_heating_vs_cooling,
)
from src.model import train_and_evaluate, load_metrics, get_feature_importances
from src.prediction import predict_loads, plot_feature_importance_heating, plot_feature_importance_cooling
from src.insights import generate_insights

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EcoBuild AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a clean, professional look ─────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2563a8 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 6px;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 14px;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82d4;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Insight card */
    .insight-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    /* Recommendation card */
    .rec-card {
        background: #f0fdf4;
        border: 1px solid #d1fae5;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    /* Hypothesis card */
    .hyp-card {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    /* Prediction result */
    .pred-result {
        background: #ffffff;
        border: 2px solid #3b82d4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA / MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset…")
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data(show_spinner="Running data quality checks…")
def get_quality_report(_df: pd.DataFrame) -> dict:
    return get_data_quality_report(_df)


@st.cache_data(show_spinner="Computing descriptive statistics…")
def get_stats(_df: pd.DataFrame) -> pd.DataFrame:
    return get_descriptive_stats(_df)


@st.cache_resource(show_spinner="Training models (first run only)…")
def get_metrics() -> dict:
    return load_metrics()


@st.cache_data(show_spinner="Generating insights…")
def get_insights(_df: pd.DataFrame) -> dict:
    return generate_insights(_df)


# ── Figure caching (each plot cached individually) ────────────────────────
@st.cache_data(show_spinner=False)
def fig_overview(_df):         return plot_dataset_overview(_df)
@st.cache_data(show_spinner=False)
def fig_heating_dist(_df):     return plot_heating_distribution(_df)
@st.cache_data(show_spinner=False)
def fig_cooling_dist(_df):     return plot_cooling_distribution(_df)
@st.cache_data(show_spinner=False)
def fig_correlation(_df):      return plot_correlation_heatmap(_df)
@st.cache_data(show_spinner=False)
def fig_feat_heating(_df):     return plot_features_vs_heating(_df)
@st.cache_data(show_spinner=False)
def fig_feat_cooling(_df):     return plot_features_vs_cooling(_df)
@st.cache_data(show_spinner=False)
def fig_glazing(_df):          return plot_glazing_relationships(_df)
@st.cache_data(show_spinner=False)
def fig_orientation(_df):      return plot_orientation_comparison(_df)
@st.cache_data(show_spinner=False)
def fig_h_vs_c(_df):           return plot_heating_vs_cooling(_df)
@st.cache_resource(show_spinner=False)
def fig_imp_heating():         return plot_feature_importance_heating()
@st.cache_resource(show_spinner=False)
def fig_imp_cooling():         return plot_feature_importance_cooling()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/building.png", width=60)
    st.title("EcoBuild AI")
    st.caption("Building Energy Efficiency\nAnalysis & Prediction")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Overview",
            "📊 Data Analysis",
            "🤖 Model Performance",
            "🔍 Feature Importance",
            "⚡ Energy Prediction",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: UCI Energy Efficiency\n768 observations · 10 variables")
    st.caption("Models: Linear Regression · Random Forest")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS (once, cached)
# ══════════════════════════════════════════════════════════════════════════════
df       = get_data()
quality  = get_quality_report(df)
stats    = get_stats(df)
metrics  = get_metrics()
insights = get_insights(df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("## 🏢 EcoBuild AI — Building Energy Efficiency")
    st.markdown(
        "<div class='info-box'>"
        "<b>Project Objective:</b> Analyse the structural characteristics of buildings "
        "and predict their heating and cooling energy loads using machine learning. "
        "This supports early-stage design decisions to reduce energy consumption and carbon emissions."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── KPI row ───────────────────────────────────────────────────────────
    st.markdown("### 📌 Key Dataset KPIs")
    kpis = insights["kpis"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{quality['total_rows']:,}")
    c2.metric("Input Features", quality["feature_count"])
    c3.metric("Target Variables", quality["target_count"])
    c4.metric("Missing Values", quality["missing_values"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Heating Load",  f"{kpis['heating_load_mean']} kWh/m²")
    c2.metric("Mean Cooling Load",  f"{kpis['cooling_load_mean']} kWh/m²")
    c3.metric("Heating Load Range", kpis["heating_load_range"])
    c4.metric("Cooling Load Range", kpis["cooling_load_range"])

    st.divider()

    # ── Dataset description ───────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### 📋 Dataset Description")
        st.markdown("""
        This dataset is the **UCI Energy Efficiency** dataset (Tsanas & Xifara, 2012),
        generated by simulating 768 building configurations in **Ecotect**, a building
        energy simulation tool.

        | Column | Name | Role |
        |--------|------|------|
        | X1 | Relative Compactness | Feature |
        | X2 | Surface Area (m²) | Feature |
        | X3 | Wall Area (m²) | Feature |
        | X4 | Roof Area (m²) | Feature |
        | X5 | Overall Height (m) | Feature |
        | X6 | Orientation | Feature |
        | X7 | Glazing Area (fraction) | Feature |
        | X8 | Glazing Area Distribution | Feature |
        | Y1 | **Heating Load (kWh/m²)** | **Target** |
        | Y2 | **Cooling Load (kWh/m²)** | **Target** |
        """)

    with col_right:
        st.markdown("### 🔬 Methodology")
        st.markdown("""
        **Phase 1 — Data Preparation**
        - Load Excel dataset, strip 528 empty trailing rows
        - Rename columns to physical names
        - Verify zero missing values and no duplicates

        **Phase 2 — Exploratory Data Analysis**
        - Distribution analysis of targets
        - Correlation heatmap
        - Feature-target scatter plots
        - Glazing and orientation comparisons

        **Phase 3 — Machine Learning**
        - Baseline: Linear Regression (scaled features)
        - Main model: Random Forest (200 estimators)
        - 80 / 20 train-test split, random state = 42
        - Metrics: MAE, RMSE, R²

        **Phase 4 — Insights**
        - Feature importance from Random Forest
        - Data-driven observations, hypotheses, recommendations
        """)

    st.divider()

    # ── Business Insights preview ─────────────────────────────────────────
    st.markdown("### 💡 Key Insights")
    for title, detail in insights["observations"][:3]:
        st.markdown(
            f"<div class='insight-card'><b>{title}</b><br>{detail}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("📖 Show all observations"):
        for title, detail in insights["observations"][3:]:
            st.markdown(
                f"<div class='insight-card'><b>{title}</b><br>{detail}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 🧪 Testable Hypotheses")
    for hyp, approach in insights["hypotheses"]:
        st.markdown(
            f"<div class='hyp-card'><b>{hyp}</b><br><i>Test approach:</i> {approach}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### ✅ Recommendations")
    for title, detail in insights["recommendations"]:
        st.markdown(
            f"<div class='rec-card'><b>{title}</b><br>{detail}</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis & Exploratory Visualizations")

    # ── Data Quality ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔎 Data Quality Report</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows (clean)",    f"{quality['total_rows']:,}")
    q2.metric("Columns",         quality["total_cols"])
    q3.metric("Missing Values",  quality["missing_values"], delta="✓ None" if quality["missing_values"] == 0 else None)
    q4.metric("Duplicate Rows",  quality["duplicate_rows"], delta="✓ None" if quality["duplicate_rows"] == 0 else None)

    with st.expander("📑 Descriptive Statistics Table"):
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)

    with st.expander("👁 Raw Data Preview (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # ── Feature Distributions ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>📦 Building Feature Distributions</div>", unsafe_allow_html=True)
    st.pyplot(fig_overview(df), use_container_width=True)

    st.divider()

    # ── Target Distributions ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>🌡 Energy Load Distributions</div>", unsafe_allow_html=True)
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.pyplot(fig_heating_dist(df), use_container_width=True)
    with dcol2:
        st.pyplot(fig_cooling_dist(df), use_container_width=True)

    st.divider()

    # ── Correlation ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔗 Correlation Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>The Pearson correlation matrix below shows the linear "
        "relationships between all features and the two targets. "
        "Values close to ±1 indicate strong linear associations.</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(fig_correlation(df), use_container_width=True)

    st.divider()

    # ── Features vs targets ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Features vs Heating Load</div>", unsafe_allow_html=True)
    st.pyplot(fig_feat_heating(df), use_container_width=True)

    st.markdown("<div class='section-header'>📈 Features vs Cooling Load</div>", unsafe_allow_html=True)
    st.pyplot(fig_feat_cooling(df), use_container_width=True)

    st.divider()

    # ── Glazing & Orientation ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🪟 Glazing Area Analysis</div>", unsafe_allow_html=True)
    st.pyplot(fig_glazing(df), use_container_width=True)

    st.markdown("<div class='section-header'>🧭 Orientation Comparison</div>", unsafe_allow_html=True)
    st.pyplot(fig_orientation(df), use_container_width=True)

    st.markdown("<div class='section-header'>⚡ Heating Load vs Cooling Load</div>", unsafe_allow_html=True)
    st.pyplot(fig_h_vs_c(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Machine Learning Model Performance")
    st.markdown(
        "<div class='info-box'>"
        "Two regression models were trained on <b>80%</b> of the data (614 samples) "
        "and evaluated on the remaining <b>20%</b> (154 samples). "
        "Random state = 42 ensures reproducibility."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Metrics tables ────────────────────────────────────────────────────
    for target in TARGET_COLS:
        nice = target.replace("_", " ")
        st.markdown(f"<div class='section-header'>🎯 {nice}</div>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            st.markdown("**Linear Regression**")
            lr_vals = metrics["Linear_Regression"][target]
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("MAE",  f"{lr_vals['MAE']:.4f}")
            lc2.metric("RMSE", f"{lr_vals['RMSE']:.4f}")
            lc3.metric("R²",   f"{lr_vals['R2']:.4f}")

        with m2:
            st.markdown("**Random Forest**")
            rf_vals = metrics["Random_Forest"][target]
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("MAE",  f"{rf_vals['MAE']:.4f}",
                       delta=f"{rf_vals['MAE'] - lr_vals['MAE']:+.4f}",
                       delta_color="inverse")
            rc2.metric("RMSE", f"{rf_vals['RMSE']:.4f}",
                       delta=f"{rf_vals['RMSE'] - lr_vals['RMSE']:+.4f}",
                       delta_color="inverse")
            rc3.metric("R²",   f"{rf_vals['R2']:.4f}",
                       delta=f"{rf_vals['R2'] - lr_vals['R2']:+.4f}")

        st.divider()

    # ── Side-by-side comparison bar chart ─────────────────────────────────
    st.markdown("<div class='section-header'>📊 Model Comparison — R² Score</div>", unsafe_allow_html=True)

    fig_cmp, axes = plt.subplots(1, 2, figsize=(11, 4))
    model_names = ["Linear\nRegression", "Random\nForest"]
    colors_lr = ["#a0c4ff", "#ffd6a5"]
    colors_rf = ["#3b82d4", "#e05c5c"]

    for ax_idx, target in enumerate(TARGET_COLS):
        r2_vals = [
            metrics["Linear_Regression"][target]["R2"],
            metrics["Random_Forest"][target]["R2"],
        ]
        bars = axes[ax_idx].bar(model_names, r2_vals,
                                color=[colors_lr[ax_idx], colors_rf[ax_idx]],
                                edgecolor="white", width=0.45)
        for bar, val in zip(bars, r2_vals):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.005,
                              f"{val:.4f}", ha="center", va="bottom", fontsize=11)
        axes[ax_idx].set_ylim(0, 1.08)
        axes[ax_idx].set_ylabel("R² Score")
        axes[ax_idx].set_title(f"R² — {target.replace('_', ' ')}")
        axes[ax_idx].spines["top"].set_visible(False)
        axes[ax_idx].spines["right"].set_visible(False)
        axes[ax_idx].axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig_cmp.tight_layout()
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    # ── RMSE comparison ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Model Comparison — RMSE (lower is better)</div>", unsafe_allow_html=True)
    fig_rmse, axes2 = plt.subplots(1, 2, figsize=(11, 4))

    for ax_idx, target in enumerate(TARGET_COLS):
        rmse_vals = [
            metrics["Linear_Regression"][target]["RMSE"],
            metrics["Random_Forest"][target]["RMSE"],
        ]
        bars = axes2[ax_idx].bar(model_names, rmse_vals,
                                 color=[colors_lr[ax_idx], colors_rf[ax_idx]],
                                 edgecolor="white", width=0.45)
        for bar, val in zip(bars, rmse_vals):
            axes2[ax_idx].text(bar.get_x() + bar.get_width() / 2,
                               bar.get_height() + 0.02,
                               f"{val:.4f}", ha="center", va="bottom", fontsize=11)
        axes2[ax_idx].set_ylabel("RMSE (kWh/m²)")
        axes2[ax_idx].set_title(f"RMSE — {target.replace('_', ' ')}")
        axes2[ax_idx].spines["top"].set_visible(False)
        axes2[ax_idx].spines["right"].set_visible(False)

    fig_rmse.tight_layout()
    st.pyplot(fig_rmse, use_container_width=True)
    plt.close(fig_rmse)

    st.markdown(
        "<div class='info-box'>"
        "<b>Interpretation:</b> R² measures what fraction of variance in the target "
        "is explained by the model (1.0 = perfect fit). RMSE is in the same units "
        "as the target variable (kWh/m²) — lower is better."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("## 🔍 Feature Importance — Random Forest")
    st.markdown(
        "<div class='info-box'>"
        "Feature importance is measured as the <b>Mean Decrease in Impurity (MDI)</b> "
        "across all 200 decision trees in the Random Forest. "
        "Higher values indicate that a feature contributes more to predicting the target."
        "</div>",
        unsafe_allow_html=True,
    )

    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        st.markdown("<div class='section-header'>🔥 Heating Load</div>", unsafe_allow_html=True)
        st.pyplot(fig_imp_heating(), use_container_width=True)
    with fi_col2:
        st.markdown("<div class='section-header'>❄️ Cooling Load</div>", unsafe_allow_html=True)
        st.pyplot(fig_imp_cooling(), use_container_width=True)

    # ── Importance table ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Feature Importance Values")

    importances = get_feature_importances()
    imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Heating Load Importance": [
            round(float(importances["Heating_Load"].get(f, 0)), 4)
            for f in FEATURE_COLS
        ],
        "Cooling Load Importance": [
            round(float(importances["Cooling_Load"].get(f, 0)), 4)
            for f in FEATURE_COLS
        ],
    })
    imp_df = imp_df.sort_values("Heating Load Importance", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    st.dataframe(
        imp_df.style.background_gradient(
            cmap="Reds", subset=["Heating Load Importance"]
        ).background_gradient(
            cmap="Blues", subset=["Cooling Load Importance"]
        ).format("{:.4f}", subset=["Heating Load Importance", "Cooling Load Importance"]),
        use_container_width=True,
    )

    st.markdown(
        "<div class='info-box'>"
        "<b>Key takeaway:</b> Overall Height and Relative Compactness consistently "
        "dominate both targets. Glazing Area becomes more prominent for Cooling Load, "
        "while Orientation contributes the least — consistent with the correlation analysis."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ENERGY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Energy Prediction":
    st.markdown("## ⚡ Building Energy Load Predictor")
    st.markdown(
        "<div class='info-box'>"
        "Enter the structural characteristics of a building below. "
        "The trained <b>Random Forest</b> model will predict its "
        "<b>Heating Load</b> and <b>Cooling Load</b> in kWh/m²."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Input sliders ─────────────────────────────────────────────────────
    st.markdown("### 🏗️ Building Parameters")
    inp_col1, inp_col2 = st.columns(2)

    with inp_col1:
        rc = st.slider(
            "Relative Compactness",
            min_value=0.62, max_value=0.98, value=0.76, step=0.01,
            help="Ratio of building volume to surface area. More compact = lower heat loss.",
        )
        sa = st.slider(
            "Surface Area (m²)",
            min_value=514.5, max_value=808.5, value=661.5, step=0.5,
            help="Total external surface area of the building.",
        )
        wa = st.slider(
            "Wall Area (m²)",
            min_value=245.0, max_value=416.5, value=318.5, step=0.5,
            help="Total wall surface area.",
        )
        ra = st.slider(
            "Roof Area (m²)",
            min_value=110.25, max_value=220.5, value=147.0, step=0.25,
            help="Horizontal roof area. Inversely related to building height.",
        )

    with inp_col2:
        oh = st.selectbox(
            "Overall Height (m)",
            options=[3.5, 7.0],
            index=1,
            help="Building height: 3.5 m = single storey, 7.0 m = double storey.",
        )
        orient = st.selectbox(
            "Orientation",
            options=[2, 3, 4, 5],
            format_func=lambda x: {2: "2 — North", 3: "3 — East", 4: "4 — South", 5: "5 — West"}[x],
            index=1,
            help="Compass orientation of the building facade.",
        )
        ga = st.select_slider(
            "Glazing Area (fraction of floor area)",
            options=[0.0, 0.10, 0.25, 0.40],
            value=0.25,
            format_func=lambda x: f"{x:.0%}",
            help="Proportion of floor area covered by glazing (windows).",
        )
        gad = st.selectbox(
            "Glazing Area Distribution",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: {
                0: "0 — None (no glazing)",
                1: "1 — Uniform",
                2: "2 — North",
                3: "3 — East",
                4: "4 — South",
                5: "5 — West",
            }[x],
            index=3,
            help="Location of glazing: which facade carries the glass area.",
        )

    # ── Prediction button ─────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Predict Energy Loads", type="primary", use_container_width=True):
        input_values = {
            "Relative_Compactness":      rc,
            "Surface_Area":              sa,
            "Wall_Area":                 wa,
            "Roof_Area":                 ra,
            "Overall_Height":            oh,
            "Orientation":               float(orient),
            "Glazing_Area":              ga,
            "Glazing_Area_Distribution": float(gad),
        }

        with st.spinner("Running prediction…"):
            result = predict_loads(input_values)

        # ── Results display ───────────────────────────────────────────────
        st.markdown("### 📊 Prediction Results")
        res_col1, res_col2 = st.columns(2)

        hl = result["Heating_Load"]
        cl = result["Cooling_Load"]

        # Contextual labels based on dataset range
        def energy_label(val, mn, mx):
            pct = (val - mn) / (mx - mn)
            if pct < 0.33:   return "🟢 Low"
            elif pct < 0.66: return "🟡 Medium"
            else:             return "🔴 High"

        hl_label = energy_label(hl, df["Heating_Load"].min(), df["Heating_Load"].max())
        cl_label = energy_label(cl, df["Cooling_Load"].min(), df["Cooling_Load"].max())

        with res_col1:
            st.markdown(
                f"""<div class='pred-result'>
                    <h3 style='color:#e05c5c;'>🔥 Heating Load</h3>
                    <h1 style='color:#e05c5c;font-size:3rem;'>{hl} <span style='font-size:1.2rem;'>kWh/m²</span></h1>
                    <p style='font-size:1.1rem;'>{hl_label}</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""<div class='pred-result'>
                    <h3 style='color:#3b82d4;'>❄️ Cooling Load</h3>
                    <h1 style='color:#3b82d4;font-size:3rem;'>{cl} <span style='font-size:1.2rem;'>kWh/m²</span></h1>
                    <p style='font-size:1.1rem;'>{cl_label}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Comparison to dataset averages ────────────────────────────────
        st.markdown("### 📐 Comparison to Dataset Averages")
        cmp1, cmp2, cmp3, cmp4 = st.columns(4)
        cmp1.metric("Predicted Heating",  f"{hl:.2f} kWh/m²",
                    delta=f"{hl - df['Heating_Load'].mean():.2f} vs mean",
                    delta_color="inverse")
        cmp2.metric("Predicted Cooling",  f"{cl:.2f} kWh/m²",
                    delta=f"{cl - df['Cooling_Load'].mean():.2f} vs mean",
                    delta_color="inverse")
        cmp3.metric("Dataset Mean Heating", f"{df['Heating_Load'].mean():.2f} kWh/m²")
        cmp4.metric("Dataset Mean Cooling", f"{df['Cooling_Load'].mean():.2f} kWh/m²")

        # ── Input summary ─────────────────────────────────────────────────
        with st.expander("🔍 Input Summary"):
            input_df = pd.DataFrame([input_values]).T
            input_df.columns = ["Value"]
            st.dataframe(input_df, use_container_width=True)

    else:
        st.markdown(
            "<div class='info-box'>"
            "👆 Adjust the sliders above and click <b>Predict Energy Loads</b> to see results."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Reference ranges ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📏 Dataset Reference Ranges")
    ref_data = {
        "Feature": [c.replace("_", " ") for c in FEATURE_COLS + TARGET_COLS],
        "Min":  [round(df[c].min(), 2) for c in FEATURE_COLS + TARGET_COLS],
        "Mean": [round(df[c].mean(), 2) for c in FEATURE_COLS + TARGET_COLS],
        "Max":  [round(df[c].max(), 2) for c in FEATURE_COLS + TARGET_COLS],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("EcoBuild AI · v1.0\nData Analytics & AI/ML Internship Project")
