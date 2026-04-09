"""
PCI PREDICTION MODELLING
========================
Dataset  : cleaned_PCI_dataset.csv
Target   : PCI (Pavement Condition Index - continuous 0-100)
Task     : Regression
Models   : Tuned SVR, XGBoost, LightGBM, Random Forest

HOW TO RUN IN VSCode
--------------------
1. Install requirements (run once in terminal):
       pip install pandas numpy scikit-learn xgboost lightgbm matplotlib

2. Place this file in the same folder as cleaned_PCI_dataset.csv
   OR update DATA_PATH below to the full path.

3. Run:
       python pci_modelling.py

Output:
   - Prints CV + test metrics table in terminal
   - Saves comparison chart  -> pci_model_comparison.png
   - Saves predictions CSV   -> pci_predictions.csv
"""

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_PATH = "cleaned_PCI_dataset.csv"
TARGET_COL = "PCI"
TEST_SIZE = 0.20
RANDOM_SEED = 42

CV_FOLDS = 5
SEARCH_ITERS = 25
N_JOBS = -1

# SVR can be slow on large datasets; tune/train on a capped sample.
SVR_MAX_TRAIN_SAMPLES = 25000

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import warnings
import math

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


print("=" * 72)
print("  PCI PREDICTION - Tuned Models with Cross-Validation")
print("=" * 72)


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
print("\n[1/6] Loading data ...")
df = pd.read_csv(DATA_PATH)
print(f"       Rows: {len(df):,} | Columns: {df.shape[1]}")
print(
    f"       Target '{TARGET_COL}': min={df[TARGET_COL].min():.1f} "
    f"max={df[TARGET_COL].max():.1f} mean={df[TARGET_COL].mean():.1f}"
)


# ------------------------------------------------------------
# STEP 2: FEATURE PREP
# ------------------------------------------------------------
print("\n[2/6] Preparing features ...")

DROP_COLS = ["Segment_ID", "Latitude", "Longitude"]
df = df.drop(columns=DROP_COLS, errors="ignore")

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

print(f"       Categorical columns encoded: {cat_cols}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print(f"       Features used ({X.shape[1]}): {X.columns.tolist()}")


# ------------------------------------------------------------
# STEP 3: SPLIT
# ------------------------------------------------------------
print("\n[3/6] Splitting data ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"       Train: {len(X_train):,} | Test: {len(X_test):,}")

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def get_metrics(name, y_true, y_pred, cv_r2):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(
        f"       {name:<15} CV_R2={cv_r2:.4f} "
        f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}"
    )
    return {
        "Model": name,
        "CV_R2": round(cv_r2, 4),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
    }


def tune_and_evaluate(model_name, estimator, param_dist, X_fit, y_fit, X_eval, y_eval):
    print(f"\n  -> Tuning {model_name}")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=SEARCH_ITERS,
        scoring="r2",
        n_jobs=N_JOBS,
        cv=cv,
        random_state=RANDOM_SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X_fit, y_fit)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_eval)

    print(f"       Best CV R2: {search.best_score_:.4f}")
    print(f"       Best params: {search.best_params_}")

    row = get_metrics(model_name, y_eval, y_pred, search.best_score_)
    return row, y_pred


results = []
predictions = {"y_test": y_test.values}


# ------------------------------------------------------------
# STEP 4: CV + TUNING
# ------------------------------------------------------------
print("\n[4/6] Hyperparameter tuning with CV ...")

# 4a) Random Forest
rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS)
rf_params = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.7, 0.9],
}
row, preds = tune_and_evaluate("Random Forest", rf, rf_params, X_train, y_train, X_test, y_test)
results.append(row)
predictions["Random Forest"] = preds

# 4b) SVR (pipeline with scaler)
svr_train_x = X_train
svr_train_y = y_train
if len(X_train) > SVR_MAX_TRAIN_SAMPLES:
    idx = np.random.RandomState(RANDOM_SEED).choice(
        len(X_train), size=SVR_MAX_TRAIN_SAMPLES, replace=False
    )
    svr_train_x = X_train.iloc[idx]
    svr_train_y = y_train.iloc[idx]
    print(
        f"\n  -> SVR uses sampled train set for tuning/training: "
        f"{len(svr_train_x):,} rows"
    )

svr_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ]
)
svr_params = {
    "svr__C": [1, 10, 30, 100, 300],
    "svr__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
}
row, preds = tune_and_evaluate("SVM", svr_pipe, svr_params, svr_train_x, svr_train_y, X_test, y_test)
results.append(row)
predictions["SVM"] = preds

# 4c) XGBoost
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    n_jobs=N_JOBS,
    verbosity=0,
)
xgb_params = {
    "n_estimators": [200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}
row, preds = tune_and_evaluate("XGBoost", xgb_model, xgb_params, X_train, y_train, X_test, y_test)
results.append(row)
predictions["XGBoost"] = preds

# 4d) LightGBM
lgb_model = lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=-1)
lgb_params = {
    "n_estimators": [200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [-1, 6, 10, 14],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}
row, preds = tune_and_evaluate("LightGBM", lgb_model, lgb_params, X_train, y_train, X_test, y_test)
results.append(row)
predictions["LightGBM"] = preds


# ------------------------------------------------------------
# STEP 5: RESULTS TABLE
# ------------------------------------------------------------
print("\n[5/6] Results Summary")
print("-" * 72)
results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
results_df.index += 1
print(results_df.to_string())
print("-" * 72)
best = results_df.iloc[0]["Model"]
print(f"\n  Best model: {best} (Test R2 = {results_df.iloc[0]['R2']})")


# ------------------------------------------------------------
# STEP 6: PLOTS + FILE OUTPUT
# ------------------------------------------------------------
print("\n[6/6] Generating charts and files ...")

models = ["Random Forest", "SVM", "XGBoost", "LightGBM"]
colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

n_cols = 2
n_rows = math.ceil((len(models) + 1) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
fig.suptitle("PCI Prediction - Tuned Model Comparison", fontsize=16, fontweight="bold", y=0.99)
axes_flat = np.array(axes).reshape(-1)

for i, (model, color) in enumerate(zip(models, colors)):
    ax = axes_flat[i]
    preds = predictions[model]
    actual = predictions["y_test"]

    ax.scatter(actual, preds, alpha=0.2, s=8, color=color, label="Predictions")
    lo, hi = actual.min(), actual.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect fit")

    r2 = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    ax.set_title(f"{model} | R2={r2:.4f} MAE={mae:.3f}", fontsize=11)
    ax.set_xlabel("Actual PCI")
    ax.set_ylabel("Predicted PCI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

bar_ax = axes_flat[len(models)]
x = np.arange(len(models))
w = 0.35
r2_vals = [r2_score(predictions["y_test"], predictions[m]) for m in models]
mae_vals = [mean_absolute_error(predictions["y_test"], predictions[m]) for m in models]

bar_ax.bar(x - w / 2, r2_vals, w, label="R2", color="#1f77b4", alpha=0.85)
bar_ax2 = bar_ax.twinx()
bar_ax2.bar(x + w / 2, mae_vals, w, label="MAE", color="#ff7f0e", alpha=0.85)

bar_ax.set_xticks(x)
bar_ax.set_xticklabels(models, rotation=12, ha="right", fontsize=10)
bar_ax.set_ylabel("R2", color="#1f77b4")
bar_ax2.set_ylabel("MAE", color="#ff7f0e")
bar_ax.set_title("Model Comparison: R2 vs MAE", fontsize=11)
bar_ax.legend(loc="upper left", fontsize=9)
bar_ax2.legend(loc="upper right", fontsize=9)
bar_ax.grid(True, alpha=0.3)

for j in range(len(models) + 1, len(axes_flat)):
    axes_flat[j].axis("off")

plt.tight_layout()
plt.savefig("pci_model_comparison.png", dpi=150, bbox_inches="tight")
print("       Chart saved -> pci_model_comparison.png")

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("pci_predictions.csv", index=False)
print("       Predictions saved -> pci_predictions.csv")

print("\n" + "=" * 72)
print("  DONE! Tuned models trained and evaluated with CV.")
print("=" * 72)
