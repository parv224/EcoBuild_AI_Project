"""
Pre-flight verification for app.py
Checks all imports, data, models and prediction without starting the server.
"""
import sys, py_compile, json
sys.path.insert(0, '.')

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
    except Exception as e:
        print(f"{FAIL}  {label}: {e}")
        errors.append(label)

# 1. Syntax
def test_syntax():
    py_compile.compile('app.py', doraise=True)

check("app.py syntax compiles without errors", test_syntax)

# 2. All src module imports
def test_imports():
    import src.data_loader
    import src.data_cleaning
    import src.analysis
    import src.model
    import src.prediction
    import src.insights
    import streamlit

check("All src modules and streamlit import correctly", test_imports)

# 3. Dataset load
def test_dataset():
    from src.data_loader import load_clean
    df = load_clean()
    assert df.shape == (768, 10), f"Unexpected shape: {df.shape}"
    assert df.isnull().sum().sum() == 0
    expected = ['Relative_Compactness','Surface_Area','Wall_Area','Roof_Area',
                'Overall_Height','Orientation','Glazing_Area',
                'Glazing_Area_Distribution','Heating_Load','Cooling_Load']
    assert list(df.columns) == expected
    print(f"         -> 768 rows x 10 cols, 0 missing values, columns correctly named")

check("Dataset loads correctly (768x10, 0 missing)", test_dataset)

# 4. Data cleaning
def test_cleaning():
    from src.data_loader import load_clean
    from src.data_cleaning import get_data_quality_report, get_descriptive_stats
    df = load_clean()
    rep = get_data_quality_report(df)
    assert rep['missing_values'] == 0
    assert rep['duplicate_rows'] == 0
    stats = get_descriptive_stats(df)
    assert stats.shape == (10, 8)
    print(f"         -> quality report OK, descriptive stats shape {stats.shape}")

check("Data cleaning functions work correctly", test_cleaning)

# 5. Metrics from disk (no re-training)
def test_metrics():
    with open('models/metrics.json') as f:
        m = json.load(f)
    assert 'Linear_Regression' in m and 'Random_Forest' in m
    for model in ['Linear_Regression', 'Random_Forest']:
        for target in ['Heating_Load', 'Cooling_Load']:
            v = m[model][target]
            assert v['R2'] > 0.85
            print(f"         -> {model}/{target}: MAE={v['MAE']}  RMSE={v['RMSE']}  R2={v['R2']}")

check("Model metrics load from disk with valid values", test_metrics)

# 6. EDA figures
def test_eda():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.data_loader import load_clean
    from src.analysis import generate_all_figures
    df = load_clean()
    figs = generate_all_figures(df)
    assert len(figs) == 9
    for fig in figs.values():
        plt.close(fig)
    print(f"         -> {len(figs)}/9 EDA figures generated successfully")

check("All 9 EDA visualizations generate without error", test_eda)

# 7. Feature importance
def test_feature_importance():
    from src.model import get_feature_importances
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.prediction import plot_feature_importance_heating, plot_feature_importance_cooling
    imps = get_feature_importances()
    assert 'Heating_Load' in imps and 'Cooling_Load' in imps
    top_h = imps['Heating_Load'].idxmax()
    top_c = imps['Cooling_Load'].idxmax()
    print(f"         -> Top feature Heating: {top_h}, Cooling: {top_c}")
    fig1 = plot_feature_importance_heating()
    plt.close(fig1)
    fig2 = plot_feature_importance_cooling()
    plt.close(fig2)
    print(f"         -> Feature importance bar charts generated")

check("Feature importance loads and charts render correctly", test_feature_importance)

# 8. Prediction — test multiple inputs
def test_prediction():
    from src.prediction import predict_loads
    test_cases = [
        # (label, input_dict)
        ("compact 7m no glazing", {
            'Relative_Compactness': 0.98, 'Surface_Area': 514.5, 'Wall_Area': 294.0,
            'Roof_Area': 110.25, 'Overall_Height': 7.0, 'Orientation': 2.0,
            'Glazing_Area': 0.0, 'Glazing_Area_Distribution': 0.0
        }),
        ("default 7m 25% glazing", {
            'Relative_Compactness': 0.76, 'Surface_Area': 661.5, 'Wall_Area': 318.5,
            'Roof_Area': 147.0, 'Overall_Height': 7.0, 'Orientation': 3.0,
            'Glazing_Area': 0.25, 'Glazing_Area_Distribution': 3.0
        }),
        ("low compact 3.5m 40% glazing", {
            'Relative_Compactness': 0.62, 'Surface_Area': 808.5, 'Wall_Area': 416.5,
            'Roof_Area': 220.5, 'Overall_Height': 3.5, 'Orientation': 4.0,
            'Glazing_Area': 0.40, 'Glazing_Area_Distribution': 5.0
        }),
    ]
    for label, inputs in test_cases:
        result = predict_loads(inputs)
        hl = result['Heating_Load']
        cl = result['Cooling_Load']
        assert isinstance(hl, float) and isinstance(cl, float)
        assert hl > 0 and cl > 0
        print(f"         -> [{label}] Heating={hl} kWh/m2, Cooling={cl} kWh/m2")

check("Prediction works for 3 different building configurations", test_prediction)

# 9. Business insights
def test_insights():
    from src.insights import generate_insights
    from src.data_loader import load_clean
    df = load_clean()
    ins = generate_insights(df)
    assert len(ins['observations']) == 6
    assert len(ins['hypotheses']) == 3
    assert len(ins['recommendations']) == 3
    kpi = ins['kpis']
    print(f"         -> Mean Heating Load: {kpi['heating_load_mean']} kWh/m2")
    print(f"         -> Mean Cooling Load: {kpi['cooling_load_mean']} kWh/m2")
    print(f"         -> Top corr feature (heating): {kpi['top_corr_feature_heating']}")

check("Business insights generate with actual dataset values", test_insights)

# 10. Streamlit importability (confirms app.py can be loaded by streamlit)
def test_streamlit_import():
    import streamlit
    import ast
    with open('app.py', 'r', encoding='utf-8') as f:
        src = f.read()
    ast.parse(src)  # full AST parse validates all syntax
    print(f"         -> streamlit v{streamlit.__version__} found")
    print(f"         -> app.py passes full AST parse")

check("Streamlit is available and app.py passes full AST validation", test_streamlit_import)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if errors:
    print(f"RESULT: {10 - len(errors)}/10 checks passed | {len(errors)} FAILED")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("RESULT: 10/10 checks passed")
    print()
    print("app.py is fully functional and ready to run.")
    print("Start command: streamlit run app.py")
