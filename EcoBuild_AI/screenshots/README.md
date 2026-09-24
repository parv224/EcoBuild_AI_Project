# EcoBuild AI — Screenshots

Place the 8 application screenshots in this folder with these exact filenames:

| Filename                    | Page in App              | Content                                           |
|-----------------------------|--------------------------|---------------------------------------------------|
| 01_app_overview.png         | 🏠 Overview              | Full dashboard with KPI metrics row               |
| 02_data_quality.png         | 📊 Data Analysis         | Data Quality Report (768 rows, 0 missing)         |
| 03_eda_distributions.png    | 📊 Data Analysis         | Heating + Cooling Load histogram/KDE plots        |
| 04_correlation_heatmap.png  | 📊 Data Analysis         | Pearson Correlation Matrix heatmap                |
| 05_eda_relationship.png     | 📊 Data Analysis         | Features vs Heating Load scatter grid             |
| 06_model_performance.png    | 🤖 Model Performance     | MAE / RMSE / R² metrics for both models           |
| 07_feature_importance.png   | 🔍 Feature Importance    | Heating + Cooling importance bar charts           |
| 08_prediction_demo.png      | ⚡ Energy Prediction     | Sliders + prediction result (Heating + Cooling)   |

## How to capture

1. Run:  `streamlit run app.py`
2. Open `http://localhost:8501` in your browser
3. Navigate to each page and take a screenshot (Windows: Win+Shift+S, then save as PNG)
4. Rename to the filenames above and place in this folder

## Verified model values (for reference)

| Model              | Target       | MAE    | RMSE   | R²     |
|--------------------|--------------|--------|--------|--------|
| Linear Regression  | Heating Load | 2.1821 | 3.0254 | 0.9122 |
| Linear Regression  | Cooling Load | 2.1953 | 3.1454 | 0.8932 |
| Random Forest      | Heating Load | 0.3528 | 0.4903 | 0.9977 |
| Random Forest      | Cooling Load | 1.0760 | 1.7382 | 0.9674 |
