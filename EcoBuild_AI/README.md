# 🏢 EcoBuild AI — Building Energy Efficiency Analysis & Prediction

> **Data Analytics & AI/ML Internship Project**

---

## Problem Statement

Buildings account for approximately 40% of global energy consumption. The ability to
predict a building's energy demand at the design stage — before construction — enables
architects and engineers to make informed, energy-efficient design decisions. This
project develops a machine-learning system that predicts the **Heating Load** and
**Cooling Load** of residential buildings based on eight structural characteristics.

---

## Objectives

1. Perform exploratory data analysis on the building energy efficiency dataset.
2. Identify the structural features most strongly associated with energy loads.
3. Train regression models to predict Heating Load (Y1) and Cooling Load (Y2).
4. Compare model performance using MAE, RMSE, and R².
5. Generate evidence-based insights and design recommendations.
6. Deliver an interactive Streamlit application for real-time prediction.

---

## Dataset

| Property | Value |
|----------|-------|
| **File** | `Data/building_energy_efficiency.xlsx` |
| **Source** | UCI Machine Learning Repository — Energy Efficiency Dataset 'https://archive.ics.uci.edu/dataset/242/energy%2Befficiency?'|
| **Citation** | Tsanas, A. & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings, 49, 560–567. |
| **Observations** | 768 (768 unique building configurations) |
| **Features** | 8 input variables (X1–X8) |
| **Targets** | 2 output variables (Y1 = Heating Load, Y2 = Cooling Load) |
| **Generation** | Simulated using Ecotect building energy simulation software |

### Column Descriptions

| Original | Renamed | Description |
|----------|---------|-------------|
| X1 | Relative_Compactness | Ratio of building volume to surface area (0.62–0.98) |
| X2 | Surface_Area | Total external surface area in m² (514.5–808.5) |
| X3 | Wall_Area | Total wall surface area in m² (245–416.5) |
| X4 | Roof_Area | Horizontal roof area in m² (110.25–220.5) |
| X5 | Overall_Height | Building height: 3.5 m (1 storey) or 7.0 m (2 storeys) |
| X6 | Orientation | Compass direction: 2=N, 3=E, 4=S, 5=W |
| X7 | Glazing_Area | Window area as fraction of floor area (0, 10%, 25%, 40%) |
| X8 | Glazing_Area_Distribution | Glazing location: 0=none, 1=uniform, 2=N, 3=E, 4=S, 5=W |
| Y1 | Heating_Load | Annual heating energy demand in kWh/m² |
| Y2 | Cooling_Load | Annual cooling energy demand in kWh/m² |

---

## Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Core language |
| pandas | ≥ 1.5.0 | Data manipulation |
| numpy | ≥ 1.23.0 | Numerical computing |
| matplotlib | ≥ 3.6.0 | Visualizations |
| seaborn | ≥ 0.12.0 | Statistical plots |
| scikit-learn | ≥ 1.1.0 | ML models (LinearRegression, RandomForest) |
| streamlit | ≥ 1.25.0 | Interactive web application |
| openpyxl | ≥ 3.0.10 | Excel file reading |
| joblib | ≥ 1.2.0 | Model serialization |

---

## Project Structure

```
EcoBuild_AI/
│
├── Data/
│   └── building_energy_efficiency.xlsx   ← original dataset (unchanged)
│
├── data/
│   └── processed_dataset.csv             ← cleaned export (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     ← load and rename dataset
│   ├── data_cleaning.py   ← quality checks, descriptive stats, CSV export
│   ├── analysis.py        ← all EDA visualization functions
│   ├── model.py           ← train, evaluate, and persist models
│   ├── prediction.py      ← single-sample prediction + importance charts
│   └── insights.py        ← data-driven observations, hypotheses, recommendations
│
├── models/
│   ├── linear_regression.joblib  ← trained Linear Regression
│   ├── random_forest.joblib      ← trained Random Forest
│   ├── scaler.joblib             ← StandardScaler for LR
│   └── metrics.json              ← evaluation metrics
│
├── app.py                 ← Streamlit application entry point
├── requirements.txt       ← Python dependencies
├── README.md              ← this file
└── Project_Report.docx    ← full project documentation
```

---

## Methodology

### 1. Data Preparation
- Raw Excel file contains 1,297 rows; rows 770–1,297 are empty trailing rows (pre-formatted, no values).
- `load_clean()` drops all rows where every column is `NaN`, retaining exactly 768 observations.
- Columns are renamed from `X1–X8 / Y1–Y2` to descriptive physical names.
- No imputation is needed — there are zero missing values in the 768 data rows.

### 2. Exploratory Data Analysis
- Distribution analysis (histograms + KDE) for both targets.
- Box plots for all 8 input features.
- Pearson correlation heatmap.
- Scatter plots of each feature against both targets.
- Grouped box plots by Glazing Area and Orientation.
- Scatter plot of Heating Load vs Cooling Load (coloured by building height).

### 3. Machine Learning
- **Problem type:** Multi-output regression (two continuous targets).
- **Train/test split:** 80% / 20%, random state = 42.
- **Baseline model:** Multiple Linear Regression (features scaled with `StandardScaler`).
- **Main model:** Random Forest Regressor (200 estimators, default depth, random state = 42).
- Both models use `MultiOutputRegressor` to predict Y1 and Y2 simultaneously.

### 4. Evaluation Metrics
- **MAE** (Mean Absolute Error) — average absolute error in kWh/m²
- **RMSE** (Root Mean Squared Error) — penalises large errors
- **R²** (Coefficient of Determination) — proportion of variance explained (1.0 = perfect)

### 5. Feature Importance
- Extracted from the Random Forest using Mean Decrease in Impurity (MDI).
- Reported separately for Heating Load and Cooling Load.

---

## Installation

```bash
# 1. Clone or download the project
cd EcoBuild_AI

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**On first run**, the application will:
1. Load and clean the Excel dataset.
2. Train both ML models (takes ~10 seconds).
3. Save trained models to `models/` for instant reuse on subsequent runs.

---

## Expected Outputs

| Section | Output |
|---------|--------|
| Overview | Project KPIs, dataset description, business insights |
| Data Analysis | 9 visualizations covering distributions, correlations, and comparisons |
| Model Performance | MAE / RMSE / R² for Linear Regression and Random Forest |
| Feature Importance | Ranked bar charts for Heating and Cooling Load |
| Energy Prediction | Real-time prediction from user-supplied building parameters |

---

## Limitations

1. **Simulated data only.** The dataset was generated by a building simulation tool (Ecotect), not collected from real buildings. Results may differ from field measurements due to simplified boundary conditions.
2. **Small dataset.** 768 observations cover a structured grid of parameter combinations. The model may not generalise well to building configurations outside this grid.
3. **No time-series data.** Loads are annual totals — the model cannot predict seasonal or hourly demand profiles.
4. **Climate not modelled.** The simulation uses a single climate dataset; predictions may not transfer to other climate zones without retraining.
5. **Two building heights only.** X5 is binary (3.5 m or 7.0 m), limiting applicability to non-standard storey heights.
6. **No embodied carbon.** The project covers operational energy only; lifecycle carbon and embodied energy are out of scope.

---

## References

- Tsanas, A. & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings, 49, 560–567.
- UCI Machine Learning Repository — [Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)
- Streamlit Documentation — https://docs.streamlit.io
- Scikit-learn Documentation — https://scikit-learn.org
