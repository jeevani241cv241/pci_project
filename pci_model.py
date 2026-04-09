"""
PCI PREDICTION MODELLING  –  OPTIMISED VERSION
===============================================
Dataset  : cleaned_PCI_dataset.csv
Target   : PCI  (Pavement Condition Index – continuous 0-100)
Task     : Regression
Models   : LinearSVR · MLP · XGBoost · LightGBM · Random Forest

HOW TO RUN IN VSCode
---------------------
1.  Install requirements (run once in terminal):
        pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib optuna

2.  Place this file in the same folder as cleaned_PCI_dataset.csv
    OR update DATA_PATH below to the full path.

3.  Run:
        python pci_modelling.py

Output:
    - Prints metrics table in terminal
    - Saves comparison chart      → pci_model_comparison.png
    - Saves predictions CSV       → pci_predictions.csv
    - Saves feature importance    → pci_feature_importance.png
    - Saves trained models        → models/ folder
"""

# ──────────────────────────────────────────────
#  CONFIGURATION  (edit these if needed)
# ──────────────────────────────────────────────
DATA_PATH    = "cleaned_PCI_dataset.csv"
TARGET_COL   = "PCI"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
CV_FOLDS     = 5          # cross-validation folds
TUNE_MODELS  = False      # set False to skip Optuna tuning (faster run)
OPTUNA_TRIALS = 30        # number of tuning trials per model

# ──────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection    import train_test_split, cross_val_score, KFold
from sklearn.preprocessing      import StandardScaler, OrdinalEncoder
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm                import LinearSVR
from sklearn.neural_network     import MLPRegressor
from sklearn.ensemble           import RandomForestRegressor
from sklearn.pipeline           import Pipeline
import xgboost  as xgb
import lightgbm as lgb

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("  PCI PREDICTION  –  5-Model Comparison  (Optimised)")
print("=" * 60)

# ──────────────────────────────────────────────
#  STEP 1: LOAD DATA
# ──────────────────────────────────────────────
print("\n[1/6]  Loading data ...")
df = pd.read_csv(DATA_PATH)
print(f"       Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"       Target '{TARGET_COL}': min={df[TARGET_COL].min():.1f}  "
      f"max={df[TARGET_COL].max():.1f}  mean={df[TARGET_COL].mean():.1f}")

# ──────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n[2/6]  Preparing features ...")

DROP_COLS = ["Segment_ID", "Latitude", "Longitude"]
df = df.drop(columns=DROP_COLS, errors="ignore")

# FIX: Use OrdinalEncoder (handles multiple columns correctly)
CAT_COLS = df.select_dtypes(include=["object", "category"]).columns.tolist()
if CAT_COLS:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CAT_COLS] = enc.fit_transform(df[CAT_COLS].astype(str))
    print(f"       Categorical columns encoded: {CAT_COLS}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

FEATURE_NAMES = X.columns.tolist()
print(f"       Features used ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# ──────────────────────────────────────────────
#  STEP 3: TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[3/6]  Splitting data ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ──────────────────────────────────────────────
#  HELPER: METRICS
# ──────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"       {name:<18}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return {"Model": name, "MAE": round(mae, 3),
            "RMSE": round(rmse, 3), "R2": round(r2, 4)}

results     = []
predictions = {"y_test": y_test.values}
importances = {}

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ──────────────────────────────────────────────
#  STEP 4: TRAIN ALL MODELS
# ──────────────────────────────────────────────
print("\n[4/6]  Training models ...\n")

# ── 4a. RANDOM FOREST ──────────────────────────
print("  ▶  Random Forest")

if TUNE_MODELS:
    def rf_objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 400),
            "max_depth":       trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features":    trial.suggest_float("max_features", 0.5, 1.0),
        }
        m = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        return cross_val_score(m, X_train, y_train, cv=3, scoring="r2").mean()

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    rf_params = study_rf.best_params
    print(f"       Best params: {rf_params}")
else:
    rf_params = {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "max_features": 1.0}

rf = RandomForestRegressor(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(get_metrics("Random Forest", y_test, y_pred_rf))
predictions["Random Forest"] = y_pred_rf
importances["Random Forest"] = rf.feature_importances_
joblib.dump(rf, "models/random_forest.pkl")

# Cross-val score
cv_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R² (5-fold): {cv_rf:.4f}")

# ── 4b. LinearSVR (replaces slow RBF SVR) ─────
print("\n  ▶  LinearSVR (fast SVM)")

# FIX: Use Pipeline so scaler is properly encapsulated
if TUNE_MODELS:
    def svr_objective(trial):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    LinearSVR(C=C, max_iter=5000, random_state=RANDOM_SEED))
        ])
        return cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2").mean()

    study_svr = optuna.create_study(direction="maximize")
    study_svr.optimize(svr_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_C = study_svr.best_params["C"]
    print(f"       Best C: {best_C:.4f}")
else:
    best_C = 1.0

svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    LinearSVR(C=best_C, max_iter=5000, random_state=RANDOM_SEED))
])
svr_pipe.fit(X_train, y_train)
y_pred_svm = svr_pipe.predict(X_test)
results.append(get_metrics("LinearSVR", y_test, y_pred_svm))
predictions["LinearSVR"] = y_pred_svm
joblib.dump(svr_pipe, "models/linear_svr.pkl")

cv_svr = cross_val_score(svr_pipe, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R² (5-fold): {cv_svr:.4f}")

# ── 4c. XGBoost ────────────────────────────────
print("\n  ▶  XGBoost")

if TUNE_MODELS:
    def xgb_objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }
        m = xgb.XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
        return cross_val_score(m, X_train, y_train, cv=3, scoring="r2").mean()

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    xgb_params = study_xgb.best_params
    print(f"       Best params: {xgb_params}")
else:
    xgb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0}

# FIX: Use a proper internal val split — NOT the test set
X_tr2, X_val, y_tr2, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_SEED)
xgb_model = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
xgb_model.fit(X_tr2, y_tr2,
              eval_set=[(X_val, y_val)],
              verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
results.append(get_metrics("XGBoost", y_test, y_pred_xgb))
predictions["XGBoost"] = y_pred_xgb
importances["XGBoost"] = xgb_model.feature_importances_
joblib.dump(xgb_model, "models/xgboost.pkl")

cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R² (5-fold): {cv_xgb:.4f}")

# ── 4d. LightGBM ───────────────────────────────
print("\n  ▶  LightGBM")

if TUNE_MODELS:
    def lgb_objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }
        m = lgb.LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        return cross_val_score(m, X_train, y_train, cv=3, scoring="r2").mean()

    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    lgb_params = study_lgb.best_params
    print(f"       Best params: {lgb_params}")
else:
    lgb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                  "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_alpha": 0.1, "reg_lambda": 1.0}

lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
lgb_model.fit(X_tr2, y_tr2,
              eval_set=[(X_val, y_val)])
y_pred_lgb = lgb_model.predict(X_test)
results.append(get_metrics("LightGBM", y_test, y_pred_lgb))
predictions["LightGBM"] = y_pred_lgb
importances["LightGBM"] = lgb_model.feature_importances_
joblib.dump(lgb_model, "models/lightgbm.pkl")

cv_lgb = cross_val_score(lgb_model, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R² (5-fold): {cv_lgb:.4f}")

# ── 4e. MLP (replaces LSTM for tabular data) ───
print("\n  ▶  MLP Neural Network (replaces LSTM for tabular data)")

# FIX: LSTM on (samples,1,features) has zero temporal benefit for tabular PCI data.
#      A proper MLP is faster, more accurate, and makes more sense here.
if TUNE_MODELS:
    def mlp_objective(trial):
        layer1 = trial.suggest_int("layer1", 64, 256)
        layer2 = trial.suggest_int("layer2", 32, 128)
        alpha  = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        lr     = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(layer1, layer2),
                activation="relu",
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=200,
                early_stopping=True,
                random_state=RANDOM_SEED
            ))
        ])
        return cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2").mean()

    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(mlp_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    bp = study_mlp.best_params
    print(f"       Best params: {bp}")
    mlp_layers = (bp["layer1"], bp["layer2"])
    mlp_alpha  = bp["alpha"]
    mlp_lr     = bp["learning_rate_init"]
else:
    mlp_layers, mlp_alpha, mlp_lr = (128, 64), 1e-4, 1e-3

mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=mlp_layers,
        activation="relu",
        alpha=mlp_alpha,
        learning_rate_init=mlp_lr,
        max_iter=300,
        early_stopping=True,
        random_state=RANDOM_SEED,
        verbose=False
    ))
])
mlp_pipe.fit(X_train, y_train)
y_pred_mlp = mlp_pipe.predict(X_test)
results.append(get_metrics("MLP", y_test, y_pred_mlp))
predictions["MLP"] = y_pred_mlp
joblib.dump(mlp_pipe, "models/mlp.pkl")

cv_mlp = cross_val_score(mlp_pipe, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R² (5-fold): {cv_mlp:.4f}")

# ──────────────────────────────────────────────
#  STEP 5: RESULTS TABLE
# ──────────────────────────────────────────────
print("\n[5/6]  Results Summary")
print("-" * 58)
results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
results_df.index += 1
print(results_df.to_string())
print("-" * 58)
best = results_df.iloc[0]["Model"]
print(f"\n  ★  Best model: {best}  (R² = {results_df.iloc[0]['R2']})")

# ──────────────────────────────────────────────
#  STEP 6: PLOTS
# ──────────────────────────────────────────────
print("\n[6/6]  Generating charts ...")

MODELS = ["Random Forest", "LinearSVR", "XGBoost", "LightGBM", "MLP"]
COLORS = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]

# ── Plot 1: Actual vs Predicted scatter ────────
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("PCI Prediction – 5-Model Comparison (Optimised)", fontsize=18, fontweight="bold", y=0.98)

for ax, model, color in zip(axes.flat[:5], MODELS, COLORS):
    preds  = predictions[model]
    actual = predictions["y_test"]
    ax.scatter(actual, preds, alpha=0.15, s=4, color=color, label="Predictions")
    lo, hi = actual.min(), actual.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect fit")
    r2  = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    ax.set_title(f"{model}\nR² = {r2:.4f}  |  MAE = {mae:.3f}", fontsize=12)
    ax.set_xlabel("Actual PCI")
    ax.set_ylabel("Predicted PCI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Bar chart comparison
ax6 = axes[2, 1]
x = np.arange(len(MODELS))
w = 0.35
r2_vals  = [r2_score(predictions["y_test"], predictions[m]) for m in MODELS]
mae_vals = [mean_absolute_error(predictions["y_test"], predictions[m]) for m in MODELS]

ax6.bar(x - w/2, r2_vals,  w, label="R²",  color="#2196F3", alpha=0.85)
ax6b = ax6.twinx()
ax6b.bar(x + w/2, mae_vals, w, label="MAE", color="#FF9800",  alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=10)
ax6.set_ylabel("R² Score",  color="#2196F3")
ax6b.set_ylabel("MAE",      color="#FF9800")
ax6.set_title("Model Comparison: R² vs MAE", fontsize=12)
ax6.legend(loc="upper left",  fontsize=9)
ax6b.legend(loc="upper right", fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pci_model_comparison.png", dpi=150, bbox_inches="tight")
print("       Chart saved → pci_model_comparison.png")

# ── Plot 2: Feature Importance ─────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("Feature Importance by Model", fontsize=15, fontweight="bold")

for ax, (mname, imp) in zip(axes2, importances.items()):
    fi = pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=True).tail(15)
    fi.plot(kind="barh", ax=ax, color="#2196F3", alpha=0.8)
    ax.set_title(mname)
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pci_feature_importance.png", dpi=150, bbox_inches="tight")
print("       Feature importance saved → pci_feature_importance.png")

# Save predictions & results
pd.DataFrame(predictions).to_csv("pci_predictions.csv", index=False)
results_df.to_csv("pci_results_summary.csv", index=False)
print("       Predictions saved → pci_predictions.csv")
print("       Results summary  → pci_results_summary.csv")
print("       Trained models   → models/ folder")

print("\n" + "=" * 60)
print("  DONE! All 5 models trained, tuned, and evaluated.")
print("=" * 60)
