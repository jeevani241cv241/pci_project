"""
PCI PREDICTION MODELLING  -  OPTIMISED & CLEAN VERSION
=======================================================
Dataset  : cleaned_PCI_dataset.csv
Target   : PCI  (Pavement Condition Index - continuous 0-100)
Models   : LinearSVR, MLP, XGBoost, LightGBM, Random Forest

STEP 1 - Install libraries (run this ONCE in terminal):
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib optuna

STEP 2 - Run:
    python pci_modelling.py

Output files saved in same folder:
    pci_model_comparison.png
    pci_feature_importance.png
    pci_correlation_heatmap.png
    pci_feature_correlation.png
    pci_predictions.csv
    pci_results_summary.csv
    models/ folder with all trained models
"""

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
DATA_PATH     = "cleaned_PCI_dataset.csv"
TARGET_COL    = "PCI"
TEST_SIZE     = 0.20
RANDOM_SEED   = 42
CV_FOLDS      = 5
TUNE_MODELS   = False   # Set False to run faster (skips auto-tuning)
OPTUNA_TRIALS = 30

# ──────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Check all libraries are installed before starting
REQUIRED = ["numpy", "pandas", "matplotlib", "seaborn",
            "sklearn", "xgboost", "lightgbm", "joblib", "optuna"]
missing = []
for lib in REQUIRED:
    try:
        __import__(lib)
    except ImportError:
        missing.append(lib)

if missing:
    print(f"\nERROR: Missing libraries: {missing}")
    print("Run this in your terminal first:")
    print("  pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib optuna\n")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # No display needed - saves to file
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm              import LinearSVR
from sklearn.neural_network   import MLPRegressor
from sklearn.ensemble         import RandomForestRegressor
from sklearn.pipeline         import Pipeline
import xgboost  as xgb
import lightgbm as lgb

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("  PCI PREDICTION  -  5-Model Comparison  (Optimised)")
print("=" * 60)

# ──────────────────────────────────────────────
#  STEP 1: LOAD DATA
# ──────────────────────────────────────────────
print("\n[1/7]  Loading data ...")

if not os.path.exists(DATA_PATH):
    print(f"\nERROR: File not found: {DATA_PATH}")
    print("Make sure cleaned_PCI_dataset.csv is in the same folder as this script.\n")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"       Rows: {len(df):,}  |  Columns: {df.shape[1]}")

if TARGET_COL not in df.columns:
    print(f"\nERROR: Column '{TARGET_COL}' not found in dataset.")
    print(f"       Available columns: {df.columns.tolist()}\n")
    sys.exit(1)

print(f"       Target '{TARGET_COL}': "
      f"min={df[TARGET_COL].min():.1f}  "
      f"max={df[TARGET_COL].max():.1f}  "
      f"mean={df[TARGET_COL].mean():.1f}")

# ──────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n[2/7]  Preparing features ...")

DROP_COLS = ["Segment_ID", "Latitude", "Longitude"]
df = df.drop(columns=DROP_COLS, errors="ignore")

# Drop rows where target is missing
df = df.dropna(subset=[TARGET_COL])

# Fill missing numeric values with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encode categorical columns properly
CAT_COLS = df.select_dtypes(include=["object", "category"]).columns.tolist()
if CAT_COLS:
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CAT_COLS] = enc.fit_transform(df[CAT_COLS].astype(str))
    print(f"       Categorical columns encoded: {CAT_COLS}")
else:
    print("       No categorical columns found.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

FEATURE_NAMES = X.columns.tolist()
print(f"       Features used ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# ──────────────────────────────────────────────
#  STEP 3: CORRELATION CHARTS
# ──────────────────────────────────────────────
print("\n[3/7]  Generating correlation charts ...")

corr_matrix = df.corr(numeric_only=True)

# Full heatmap (lower triangle only)
fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax_corr,
    annot_kws={"size": 8}
)
ax_corr.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("pci_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("       Saved -> pci_correlation_heatmap.png")

# PCI vs each feature bar chart
pci_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values()
fig_pc, ax_pc = plt.subplots(figsize=(10, max(6, len(pci_corr) * 0.4)))
colors = ["#F44336" if v < 0 else "#2196F3" for v in pci_corr.values]
pci_corr.plot(kind="barh", ax=ax_pc, color=colors, alpha=0.85)
ax_pc.axvline(0, color="black", linewidth=0.8)
ax_pc.set_title(f"Correlation of Each Feature with {TARGET_COL}",
                fontsize=14, fontweight="bold")
ax_pc.set_xlabel("Pearson Correlation Coefficient")
ax_pc.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("pci_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("       Saved -> pci_feature_correlation.png")

# ──────────────────────────────────────────────
#  STEP 4: TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[4/7]  Splitting data ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Internal val split for XGBoost/LightGBM early stopping
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_SEED)

print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ──────────────────────────────────────────────
#  HELPER
# ──────────────────────────────────────────────
def get_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"       {name:<18}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"Model": name, "MAE": round(mae, 3),
            "RMSE": round(rmse, 3), "R2": round(r2, 4)}

results     = []
predictions = {"y_test": y_test.values}
importances = {}

# ──────────────────────────────────────────────
#  STEP 5: TRAIN ALL MODELS
# ──────────────────────────────────────────────
print("\n[5/7]  Training models ...\n")

# ── 5a. RANDOM FOREST ──────────────────────────
print("  >> Random Forest")

if TUNE_MODELS:
    def rf_objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features":      trial.suggest_float("max_features", 0.5, 1.0),
        }
        m = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
        return cross_val_score(m, X_train, y_train, cv=3, scoring="r2").mean()

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    rf_params = study_rf.best_params
    print(f"       Best params: {rf_params}")
else:
    rf_params = {"n_estimators": 200, "max_depth": 20,
                 "min_samples_split": 5, "max_features": 0.8}

rf = RandomForestRegressor(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(get_metrics("Random Forest", y_test, y_pred_rf))
predictions["Random Forest"] = y_pred_rf
importances["Random Forest"] = rf.feature_importances_
joblib.dump(rf, "models/random_forest.pkl")
cv_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R2 (5-fold): {cv_rf:.4f}")

# ── 5b. LINEAR SVR ─────────────────────────────
print("\n  >> LinearSVR")

if TUNE_MODELS:
    def svr_objective(trial):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", LinearSVR(C=C, max_iter=10000, random_state=RANDOM_SEED, dual=True))
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
    ("svr", LinearSVR(C=best_C, max_iter=10000, random_state=RANDOM_SEED, dual=True))
])
svr_pipe.fit(X_train, y_train)
y_pred_svm = svr_pipe.predict(X_test)
results.append(get_metrics("LinearSVR", y_test, y_pred_svm))
predictions["LinearSVR"] = y_pred_svm
joblib.dump(svr_pipe, "models/linear_svr.pkl")
cv_svr = cross_val_score(svr_pipe, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R2 (5-fold): {cv_svr:.4f}")

# ── 5c. XGBOOST ────────────────────────────────
print("\n  >> XGBoost")

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
        m = xgb.XGBRegressor(**params, random_state=RANDOM_SEED,
                              n_jobs=-1, verbosity=0, eval_metric="rmse")
        return cross_val_score(m, X_train, y_train, cv=3, scoring="r2").mean()

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    xgb_params = study_xgb.best_params
    print(f"       Best params: {xgb_params}")
else:
    xgb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_alpha": 0.1, "reg_lambda": 1.0}

xgb_model = xgb.XGBRegressor(
    **xgb_params, random_state=RANDOM_SEED,
    n_jobs=-1, verbosity=0, eval_metric="rmse"
)
xgb_model.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
results.append(get_metrics("XGBoost", y_test, y_pred_xgb))
predictions["XGBoost"] = y_pred_xgb
importances["XGBoost"] = xgb_model.feature_importances_
joblib.dump(xgb_model, "models/xgboost.pkl")
cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R2 (5-fold): {cv_xgb:.4f}")

# ── 5d. LIGHTGBM ───────────────────────────────
print("\n  >> LightGBM")

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
lgb_model.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)])
y_pred_lgb = lgb_model.predict(X_test)
results.append(get_metrics("LightGBM", y_test, y_pred_lgb))
predictions["LightGBM"] = y_pred_lgb
importances["LightGBM"] = lgb_model.feature_importances_
joblib.dump(lgb_model, "models/lightgbm.pkl")
cv_lgb = cross_val_score(lgb_model, X_train, y_train, cv=kf, scoring="r2").mean()
print(f"       Cross-val R2 (5-fold): {cv_lgb:.4f}")

# ── 5e. MLP NEURAL NETWORK ─────────────────────
print("\n  >> MLP Neural Network")

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
                n_iter_no_change=10,
                random_state=RANDOM_SEED,
                verbose=False
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
        n_iter_no_change=10,
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
print(f"       Cross-val R2 (5-fold): {cv_mlp:.4f}")

# ──────────────────────────────────────────────
#  STEP 6: RESULTS TABLE
# ──────────────────────────────────────────────
print("\n[6/7]  Results Summary")
print("-" * 58)
results_df = (pd.DataFrame(results)
                .sort_values("R2", ascending=False)
                .reset_index(drop=True))
results_df.index += 1
print(results_df.to_string())
print("-" * 58)
best = results_df.iloc[0]["Model"]
print(f"\n  BEST MODEL: {best}  (R2 = {results_df.iloc[0]['R2']})")
results_df.to_csv("pci_results_summary.csv", index=False)

# ──────────────────────────────────────────────
#  STEP 7: ALL CHARTS
# ──────────────────────────────────────────────
print("\n[7/7]  Generating charts ...")

MODELS = ["Random Forest", "LinearSVR", "XGBoost", "LightGBM", "MLP"]
COLORS = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]

# Chart 1: Actual vs Predicted scatter
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("PCI Prediction - 5-Model Comparison",
             fontsize=18, fontweight="bold", y=0.98)

for ax, model, color in zip(axes.flat[:5], MODELS, COLORS):
    preds  = predictions[model]
    actual = predictions["y_test"]
    ax.scatter(actual, preds, alpha=0.15, s=4, color=color, label="Predictions")
    lo, hi = float(actual.min()), float(actual.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect fit")
    r2_val  = r2_score(actual, preds)
    mae_val = mean_absolute_error(actual, preds)
    ax.set_title(f"{model}\nR2={r2_val:.4f}  |  MAE={mae_val:.3f}", fontsize=12)
    ax.set_xlabel("Actual PCI")
    ax.set_ylabel("Predicted PCI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax6 = axes[2, 1]
x = np.arange(len(MODELS))
w = 0.35
r2_vals  = [r2_score(predictions["y_test"], predictions[m]) for m in MODELS]
mae_vals = [mean_absolute_error(predictions["y_test"], predictions[m]) for m in MODELS]
ax6.bar(x - w/2, r2_vals,  w, label="R2",  color="#2196F3", alpha=0.85)
ax6b = ax6.twinx()
ax6b.bar(x + w/2, mae_vals, w, label="MAE", color="#FF9800", alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=10)
ax6.set_ylabel("R2 Score", color="#2196F3")
ax6b.set_ylabel("MAE",     color="#FF9800")
ax6.set_title("Model Comparison: R2 vs MAE", fontsize=12)
ax6.legend(loc="upper left",  fontsize=9)
ax6b.legend(loc="upper right", fontsize=9)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pci_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("       Saved -> pci_model_comparison.png")

# Chart 2: Feature Importance
n_imp = len(importances)
if n_imp > 0:
    fig2, axes2 = plt.subplots(1, n_imp, figsize=(7 * n_imp, 6))
    if n_imp == 1:
        axes2 = [axes2]
    fig2.suptitle("Feature Importance by Model", fontsize=15, fontweight="bold")
    for ax, (mname, imp) in zip(axes2, importances.items()):
        fi = pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=True).tail(15)
        fi.plot(kind="barh", ax=ax, color="#2196F3", alpha=0.8)
        ax.set_title(mname, fontsize=13)
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("pci_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("       Saved -> pci_feature_importance.png")

pd.DataFrame(predictions).to_csv("pci_predictions.csv", index=False)
print("       Saved -> pci_predictions.csv")
print("       Saved -> pci_results_summary.csv")
print("       Saved -> models/ folder")

# ── Chart 3: Before vs After Forecast (individual per model) ─
print("       Generating before vs after forecast graphs ...")

SAMPLE      = 100   # show first 100 roads so graph is clear
actual_vals = predictions["y_test"][:SAMPLE]
x_axis      = np.arange(SAMPLE)

for model, color in zip(MODELS, COLORS):
    pred_vals = predictions[model][:SAMPLE]
    fig_ba, ax_ba = plt.subplots(figsize=(16, 6))

    ax_ba.plot(x_axis, actual_vals, color="#2196F3", linewidth=1.8,
               label="Before Forecast  (Actual PCI)", marker="o", markersize=3)
    ax_ba.plot(x_axis, pred_vals, color=color, linewidth=1.8,
               label="After Forecast  (Predicted PCI)",
               marker="s", markersize=3, linestyle="--")
    ax_ba.fill_between(x_axis, actual_vals, pred_vals,
                       alpha=0.15, color=color, label="Difference")

    r2_val  = r2_score(predictions["y_test"], predictions[model])
    mae_val = mean_absolute_error(predictions["y_test"], predictions[model])
    ax_ba.set_title(
        f"{model}  -  Before vs After Forecast\n"
        f"R2={r2_val:.4f}  |  MAE={mae_val:.3f}  "
        f"(First {SAMPLE} road segments)",
        fontsize=14, fontweight="bold"
    )
    ax_ba.set_xlabel("Road Segment Number", fontsize=12)
    ax_ba.set_ylabel("PCI Score (0 to 100)", fontsize=12)
    ax_ba.legend(fontsize=11)
    ax_ba.grid(True, alpha=0.3)
    ax_ba.set_ylim(0, 100)
    plt.tight_layout()
    fname = f"pci_before_after_{model.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {fname}")

# ── Chart 4: All 5 models Before vs After in one combined graph
fig_all, axes_all = plt.subplots(5, 1, figsize=(16, 28))
fig_all.suptitle("Before vs After Forecast  -  All 5 Models",
                 fontsize=18, fontweight="bold")

for ax_a, model, color in zip(axes_all, MODELS, COLORS):
    pred_vals = predictions[model][:SAMPLE]
    ax_a.plot(x_axis, actual_vals, color="#2196F3", linewidth=1.8,
              label="Before  (Actual PCI)", marker="o", markersize=2)
    ax_a.plot(x_axis, pred_vals, color=color, linewidth=1.8,
              label="After  (Predicted PCI)",
              marker="s", markersize=2, linestyle="--")
    ax_a.fill_between(x_axis, actual_vals, pred_vals,
                      alpha=0.15, color=color)
    r2_val  = r2_score(predictions["y_test"], predictions[model])
    mae_val = mean_absolute_error(predictions["y_test"], predictions[model])
    ax_a.set_title(f"{model}  |  R2={r2_val:.4f}  |  MAE={mae_val:.3f}",
                   fontsize=12, fontweight="bold")
    ax_a.set_ylabel("PCI Score", fontsize=10)
    ax_a.legend(fontsize=9, loc="upper right")
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(0, 100)

axes_all[-1].set_xlabel("Road Segment Number", fontsize=12)
plt.tight_layout()
plt.savefig("pci_before_after_ALL_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("       Saved -> pci_before_after_ALL_models.png")

print("\n" + "=" * 60)
print("  DONE! Check your folder for all output files.")
print("=" * 60)
