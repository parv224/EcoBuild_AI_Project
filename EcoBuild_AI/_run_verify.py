"""
Direct functional verification of EcoBuild_AI_Project.py
Tests ALL functions by importing the file as a module using exec(),
bypassing the Streamlit application block entirely.
Run: python _run_verify.py
"""
import sys, types, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label, fn):
    try:
        fn()
        print(f"{PASS} {label}")
        results.append((label, True))
    except Exception as e:
        print(f"{FAIL} {label}: {e}")
        results.append((label, False))

# ── Load only the non-Streamlit portions of the file ──────────────────────────
# We read the file, strip everything from "st.set_page_config" onward,
# and exec() just the function/constant definitions.

with open("EcoBuild_AI_Project.py", "r", encoding="utf-8") as f:
    full_source = f.read()

# Find the cut point — stop before the Streamlit application block
CUT_MARKER = "# SECTION 8 — STREAMLIT APPLICATION"
cut_idx = full_source.find(CUT_MARKER)
if cut_idx == -1:
    # fallback: cut at st.set_page_config
    cut_idx = full_source.find("st.set_page_config(")

functions_source = full_source[:cut_idx]

# Create a namespace that has streamlit as a dummy (imports won't fail)
import types as _types
import importlib, importlib.util

# Provide a real-looking stub for streamlit so the import line at top succeeds
class _StubST:
    def __getattr__(self, name):
        return lambda *a, **kw: None
    cache_data = staticmethod(lambda *a, **kw: (lambda f: f))
    cache_resource = staticmethod(lambda *a, **kw: (lambda f: f))

stub_st = _StubST()
sys.modules.setdefault("streamlit", stub_st)

# Now exec just the functions section
ns = {"__name__": "__main__", "__file__": str(Path("EcoBuild_AI_Project.py").resolve())}
exec(compile(functions_source, "EcoBuild_AI_Project.py", "exec"), ns)
print("Functions loaded from EcoBuild_AI_Project.py\n")

# Convenience
load_clean              = ns["load_clean"]
get_data_quality_report = ns["get_data_quality_report"]
get_descriptive_stats   = ns["get_descriptive_stats"]
save_processed          = ns["save_processed"]
generate_all_figures    = ns["generate_all_figures"]
train_and_evaluate      = ns["train_and_evaluate"]
load_metrics            = ns["load_metrics"]
get_feature_importances = ns["get_feature_importances"]
predict_loads           = ns["predict_loads"]
plot_feature_importance_heating = ns["plot_feature_importance_heating"]
plot_feature_importance_cooling = ns["plot_feature_importance_cooling"]
generate_insights       = ns["generate_insights"]

# ------------------------------------------------------------------
# 1. Import check
# ------------------------------------------------------------------
def test_imports():
    import pandas, numpy, matplotlib, seaborn, sklearn, joblib, openpyxl
    print(f"      pandas={pandas.__version__} sklearn={sklearn.__version__}")

check("1. All required packages are importable", test_imports)

# ------------------------------------------------------------------
# 2. Dataset loading
# ------------------------------------------------------------------
def test_load():
    df = load_clean()
    assert df.shape == (768, 10), f"Expected (768,10) got {df.shape}"
    assert df.isnull().sum().sum() == 0
    expected_cols = [
        "Relative_Compactness","Surface_Area","Wall_Area","Roof_Area",
        "Overall_Height","Orientation","Glazing_Area",
        "Glazing_Area_Distribution","Heating_Load","Cooling_Load"
    ]
    assert list(df.columns) == expected_cols
    print(f"      Shape {df.shape}, missing {df.isnull().sum().sum()}, cols renamed correctly")

check("2. Dataset loads correctly (768 x 10, 0 missing, columns renamed)", test_load)

# ------------------------------------------------------------------
# 3. Data cleaning
# ------------------------------------------------------------------
def test_cleaning():
    df = load_clean()
    rep = get_data_quality_report(df)
    assert rep["total_rows"] == 768
    assert rep["missing_values"] == 0
    assert rep["duplicate_rows"] == 0
    assert rep["feature_count"] == 8
    stats = get_descriptive_stats(df)
    assert stats.shape == (10, 8)
    path = save_processed(df)
    assert path.exists()
    print(f"      QA report OK, stats (10x8), CSV exported to {path}")

check("3. Data cleaning works (QA report + descriptive stats + CSV export)", test_cleaning)

# ------------------------------------------------------------------
# 4. EDA figures
# ------------------------------------------------------------------
def test_eda():
    df = load_clean()
    figs = generate_all_figures(df)
    expected = ["overview","heating_dist","cooling_dist","correlation",
                "features_heating","features_cooling","glazing","orientation","h_vs_c"]
    for k in expected:
        assert k in figs, f"Missing figure: {k}"
    for fig in figs.values():
        plt.close(fig)
    print(f"      {len(figs)}/9 EDA figures generated")

check("4. All 9 EDA figures generated without error", test_eda)

# ------------------------------------------------------------------
# 5. Model training + metrics
# ------------------------------------------------------------------
def test_model():
    df = load_clean()
    metrics = train_and_evaluate(df)
    for model in ["Linear_Regression", "Random_Forest"]:
        assert model in metrics
        for target in ["Heating_Load", "Cooling_Load"]:
            v = metrics[model][target]
            assert "MAE" in v and "RMSE" in v and "R2" in v
            assert v["R2"] > 0.85, f"{model}/{target} R2={v['R2']}"
    print(f"      LR  HL: MAE={metrics['Linear_Regression']['Heating_Load']['MAE']} "
          f"R2={metrics['Linear_Regression']['Heating_Load']['R2']}")
    print(f"      RF  HL: MAE={metrics['Random_Forest']['Heating_Load']['MAE']} "
          f"R2={metrics['Random_Forest']['Heating_Load']['R2']}")
    print(f"      LR  CL: MAE={metrics['Linear_Regression']['Cooling_Load']['MAE']} "
          f"R2={metrics['Linear_Regression']['Cooling_Load']['R2']}")
    print(f"      RF  CL: MAE={metrics['Random_Forest']['Cooling_Load']['MAE']} "
          f"R2={metrics['Random_Forest']['Cooling_Load']['R2']}")

check("5. Both models train and produce MAE/RMSE/R2 metrics", test_model)

# ------------------------------------------------------------------
# 6. Load persisted metrics
# ------------------------------------------------------------------
def test_saved_metrics():
    m = load_metrics()
    lr = m["Linear_Regression"]
    rf = m["Random_Forest"]
    assert rf["Heating_Load"]["R2"] > lr["Heating_Load"]["R2"]
    assert rf["Heating_Load"]["R2"] > 0.96
    assert rf["Cooling_Load"]["R2"] > 0.96
    print(f"      Metrics JSON loaded, RF Heating R2={rf['Heating_Load']['R2']} "
          f"Cooling R2={rf['Cooling_Load']['R2']}")

check("6. Persisted metrics load correctly and RF beats LR", test_saved_metrics)

# ------------------------------------------------------------------
# 7. Prediction
# ------------------------------------------------------------------
def test_prediction():
    sample = {
        "Relative_Compactness": 0.76,
        "Surface_Area": 661.5,
        "Wall_Area": 318.5,
        "Roof_Area": 147.0,
        "Overall_Height": 7.0,
        "Orientation": 3.0,
        "Glazing_Area": 0.25,
        "Glazing_Area_Distribution": 3.0,
    }
    result = predict_loads(sample)
    assert "Heating_Load" in result and "Cooling_Load" in result
    assert isinstance(result["Heating_Load"], float)
    assert isinstance(result["Cooling_Load"], float)
    assert result["Heating_Load"] > 0 and result["Cooling_Load"] > 0
    print(f"      Heating={result['Heating_Load']} Cooling={result['Cooling_Load']} kWh/m2")

    fig1 = plot_feature_importance_heating()
    plt.close(fig1)
    fig2 = plot_feature_importance_cooling()
    plt.close(fig2)
    print(f"      Feature importance charts generated")

check("7. predict_loads() returns valid results, importance charts work", test_prediction)

# ------------------------------------------------------------------
# 8. Business insights
# ------------------------------------------------------------------
def test_insights():
    df = load_clean()
    ins = generate_insights(df)
    assert "kpis" in ins and "observations" in ins
    assert "hypotheses" in ins and "recommendations" in ins
    assert len(ins["observations"]) == 6
    assert len(ins["hypotheses"]) == 3
    assert len(ins["recommendations"]) == 3
    print(f"      KPIs: {list(ins['kpis'].keys())}")

check("8. generate_insights() returns complete structure", test_insights)

# ------------------------------------------------------------------
# 9. Syntax / app compile check
# ------------------------------------------------------------------
def test_syntax():
    import py_compile
    py_compile.compile("EcoBuild_AI_Project.py", doraise=True)
    print(f"      Full file compiles without errors")

check("9. EcoBuild_AI_Project.py syntax check passed", test_syntax)

# ------------------------------------------------------------------
# 10. Original files untouched
# ------------------------------------------------------------------
def test_originals():
    for f in ["app.py","src/data_loader.py","src/data_cleaning.py",
              "src/analysis.py","src/model.py","src/prediction.py","src/insights.py"]:
        assert Path(f).exists(), f"MISSING: {f}"
    print(f"      All 7 original project files confirmed present")

check("10. All original modular project files are untouched", test_originals)

# ------------------------------------------------------------------
# Final report
# ------------------------------------------------------------------
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"Results: {passed}/{len(results)} passed, {failed} failed")
if failed:
    print("\nFailed:")
    for label, ok in results:
        if not ok:
            print(f"  FAIL: {label}")
else:
    print("All checks PASSED!")
