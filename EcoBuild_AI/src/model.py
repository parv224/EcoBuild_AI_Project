"""
model.py
--------
Trains Linear Regression and Random Forest models for multi-output
regression on Heating Load (Y1) and Cooling Load (Y2).

Responsibilities:
  - Train / evaluate both models
  - Persist trained models to disk (models/)
  - Return metrics dictionaries keyed by model name and target

Usage:
  python -m src.model          # train and save models
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_clean, FEATURE_COLS, TARGET_COLS

# ── Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
LR_PATH    = MODELS_DIR / "linear_regression.joblib"
RF_PATH    = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 / 20 split


# ── Helpers ───────────────────────────────────────────────────────────────
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     target_names: list) -> dict:
    """
    Compute MAE, RMSE, and R² for each target.
    Returns a nested dict: {target: {"MAE": ..., "RMSE": ..., "R2": ...}}
    """
    results = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        results[name] = {
            "MAE":  round(float(mean_absolute_error(yt, yp)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "R2":   round(float(r2_score(yt, yp)), 4),
        }
    return results


# ── Main training function ─────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame = None) -> dict:
    """
    Train both models on 80 % of the data, evaluate on 20 %, and
    persist models + metrics to disk.

    Returns:
        {
          "Linear_Regression": {"Heating_Load": {...}, "Cooling_Load": {...}},
          "Random_Forest":     {"Heating_Load": {...}, "Cooling_Load": {...}},
        }
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_clean()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Scale features (used for Linear Regression; RF sees unscaled data) ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Linear Regression ─────────────────────────────────────────────────
    lr = MultiOutputRegressor(LinearRegression())
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    metrics_lr = _compute_metrics(y_test, y_pred_lr, TARGET_COLS)

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # MultiOutputRegressor is not needed because RandomForest natively supports
    # multi-output, but we wrap to keep API consistent
    rf_multi = MultiOutputRegressor(rf)
    rf_multi.fit(X_train, y_train)   # Random Forest does not require scaling
    y_pred_rf = rf_multi.predict(X_test)
    metrics_rf = _compute_metrics(y_test, y_pred_rf, TARGET_COLS)

    # ── Persist ───────────────────────────────────────────────────────────
    joblib.dump(lr,     LR_PATH)
    joblib.dump(rf_multi, RF_PATH)
    joblib.dump(scaler, SCALER_PATH)

    all_metrics = {
        "Linear_Regression": metrics_lr,
        "Random_Forest":     metrics_rf,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


# ── Load persisted models ──────────────────────────────────────────────────
def load_models() -> tuple:
    """
    Load persisted models and scaler from disk.
    Returns (lr_model, rf_model, scaler).
    Trains automatically if models are not found.
    """
    if not (LR_PATH.exists() and RF_PATH.exists() and SCALER_PATH.exists()):
        train_and_evaluate()

    lr     = joblib.load(LR_PATH)
    rf     = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return lr, rf, scaler


def load_metrics() -> dict:
    """
    Load persisted metrics from JSON.
    Trains automatically if metrics file is not found.
    """
    if not METRICS_PATH.exists():
        return train_and_evaluate()
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Feature importance ────────────────────────────────────────────────────
def get_feature_importances() -> dict:
    """
    Extract feature importances from the persisted Random Forest model.
    Returns:
        {
          "Heating_Load": pd.Series (importance per feature),
          "Cooling_Load": pd.Series (importance per feature),
        }
    """
    _, rf_multi, _ = load_models()

    importances = {}
    for i, target in enumerate(TARGET_COLS):
        estimator = rf_multi.estimators_[i]  # individual RF for each target
        imp = pd.Series(
            estimator.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=True)
        importances[target] = imp

    return importances


if __name__ == "__main__":
    print("Training models …")
    metrics = train_and_evaluate()
    print("\n=== Model Metrics ===")
    for model_name, targets in metrics.items():
        print(f"\n{model_name}:")
        for target, vals in targets.items():
            print(f"  {target}: MAE={vals['MAE']:.4f}  RMSE={vals['RMSE']:.4f}  R²={vals['R2']:.4f}")
