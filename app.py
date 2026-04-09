"""
PCI PREDICTION - STREAMLIT WEB APP (FAST VERSION)
===================================================
Run with:  python -m streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="PCI Road Condition Predictor", page_icon="🛣️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
.hero-box { background: linear-gradient(135deg, #1a3a2a, #0d2818, #1a2f1a); border: 1px solid #2ea043; border-radius: 16px; padding: 30px; margin-bottom: 20px; text-align: center; box-shadow: 0 0 40px rgba(46,160,67,0.15); }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.5em; font-weight: 800; color: #3fb950; margin: 0; }
.hero-sub { color: #8b949e; font-size: 1em; margin-top: 8px; }
.metric-card { background: linear-gradient(135deg, #1c2128, #21262d); border: 1px solid #30363d; border-radius: 12px; padding: 15px; text-align: center; margin: 5px; }
.metric-number { font-family: 'Syne', sans-serif; font-size: 2em; font-weight: 800; color: #3fb950; }
.metric-label { color: #8b949e; font-size: 0.8em; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
.info-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px 20px; margin: 10px 0; }
.result-best { background: linear-gradient(135deg, #1a3a2a, #0d2818); border: 2px solid #3fb950; border-radius: 12px; padding: 20px; text-align: center; margin: 15px 0; }
.stButton > button { background: linear-gradient(135deg, #238636, #2ea043) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1.05em !important; padding: 12px 30px !important; width: 100% !important; }
.divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #30363d, transparent); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🛣️ PCI Road Condition Predictor</div>
    <div class="hero-sub">Upload your road data → Choose models → Get instant predictions<br>
    <span style="color:#3fb950">PCI</span> ranges from <span style="color:#f85149">0 (Failed)</span> to <span style="color:#3fb950">100 (Perfect)</span></div>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 What is PCI? (Click to learn)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-number" style="color:#3fb950">70–100</div><div class="metric-label">🟢 Good</div><div style="color:#8b949e;font-size:0.8em;margin-top:6px">Routine maintenance only</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-number" style="color:#f0b429">40–69</div><div class="metric-label">🟡 Fair</div><div style="color:#8b949e;font-size:0.8em;margin-top:6px">Plan repairs soon</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-number" style="color:#f85149">0–39</div><div class="metric-label">🔴 Poor</div><div style="color:#8b949e;font-size:0.8em;margin-top:6px">Immediate repair needed</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("**<span style='color:#3fb950'>Step 1 – Upload CSV</span>**", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**<span style='color:#3fb950'>Step 2 – Choose Models</span>**", unsafe_allow_html=True)
    run_rf  = st.checkbox("🌲 Random Forest",  value=True)
    run_xgb = st.checkbox("⚡ XGBoost",        value=True)
    run_lgb = st.checkbox("💡 LightGBM",       value=True)
    run_svr = st.checkbox("📐 Linear SVR",     value=True)
    run_mlp = st.checkbox("🧠 Neural Network", value=True)
    st.markdown("---")
    st.markdown("**<span style='color:#3fb950'>Step 3 – Run</span>**", unsafe_allow_html=True)
    run_button = st.button("🚀 Run Prediction Models")

if uploaded_file is None:
    st.markdown('<div class="info-card" style="text-align:center;padding:40px"><div style="font-size:3.5em">📂</div><div style="font-family:Syne,sans-serif;font-size:1.3em;color:#3fb950;margin:10px 0">Upload your CSV file to get started</div><div style="color:#8b949e">Use the sidebar → Upload CSV → Choose models → Click Run</div></div>', unsafe_allow_html=True)
    st.stop()

@st.cache_data
def load_data(f): return pd.read_csv(f)

df_raw = load_data(uploaded_file)

st.markdown("## 📊 Your Dataset")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-number">{len(df_raw):,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-number">{df_raw.shape[1]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
with c3:
    if "PCI" in df_raw.columns:
        avg = df_raw["PCI"].mean()
        col = "#3fb950" if avg >= 70 else "#f0b429" if avg >= 40 else "#f85149"
        st.markdown(f'<div class="metric-card"><div class="metric-number" style="color:{col}">{avg:.1f}</div><div class="metric-label">Average PCI</div></div>', unsafe_allow_html=True)

st.dataframe(df_raw.head(), use_container_width=True)

if "PCI" not in df_raw.columns:
    st.error("❌ Column 'PCI' not found!")
    st.stop()

selected_models = []
if run_rf:  selected_models.append("Random Forest")
if run_xgb: selected_models.append("XGBoost")
if run_lgb: selected_models.append("LightGBM")
if run_svr: selected_models.append("LinearSVR")
if run_mlp: selected_models.append("MLP Neural Network")

if not selected_models:
    st.warning("⚠️ Select at least one model.")
    st.stop()

if not run_button:
    st.markdown('<div class="info-card" style="text-align:center;padding:25px"><div style="font-size:1.8em">👈</div><div style="color:#8b949e">Data loaded! Click <b style="color:#3fb950">🚀 Run Prediction Models</b> in the sidebar.</div></div>', unsafe_allow_html=True)
    st.stop()

# PREPROCESS
@st.cache_data
def preprocess(df_raw):
    df = df_raw.copy()
    df = df.drop(columns=["Segment_ID","Latitude","Longitude"], errors="ignore")
    df = df.dropna(subset=["PCI"])
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("Unknown")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    X = df.drop(columns=["PCI"])
    y = df["PCI"]
    return X, y, X.columns.tolist(), df

X, y, FEATURE_NAMES, df_proc = preprocess(df_raw)
# Use max 3000 rows for speed — still gives accurate results
if len(X) > 3000:
    X = X.sample(3000, random_state=42)
    y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_tr2, X_val, y_tr2, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# FAST MODEL CONFIGS
def get_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=-1)
    elif name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=50, learning_rate=0.2, max_depth=4, random_state=42, n_jobs=-1, verbosity=0, eval_metric="rmse")
    elif name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=50, learning_rate=0.2, max_depth=4, random_state=42, n_jobs=-1, verbose=-1)
    elif name == "LinearSVR":
        return Pipeline([("sc", StandardScaler()), ("svr", LinearSVR(C=1.0, max_iter=1000, random_state=42, dual=True))])
    elif name == "MLP Neural Network":
        return Pipeline([("sc", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=50, early_stopping=True, random_state=42, verbose=False))])

COLORS = ["#3fb950","#58a6ff","#f0b429","#f85149","#bc8cff"]

# TRAIN
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## 🤖 Training Models...")
prog = st.progress(0)
status = st.empty()

results, predictions, importances = [], {"y_test": y_test.values}, {}

for i, mname in enumerate(selected_models):
    status.markdown(f"⚙️ *Training **{mname}**... ({i+1}/{len(selected_models)})*")
    m = get_model(mname)
    if mname == "XGBoost":
        m.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
    elif mname == "LightGBM":
        m.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)])
    else:
        m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    predictions[mname] = y_pred
    if hasattr(m, "feature_importances_"):
        importances[mname] = m.feature_importances_
    results.append({"Model": mname, "MAE": round(mean_absolute_error(y_test,y_pred),3),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test,y_pred)),3),
                    "R² Score": round(r2_score(y_test,y_pred),4)})
    prog.progress((i+1)/len(selected_models))

status.markdown("✅ *All models trained!*")

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True)
results_df.index += 1
best_model = results_df.iloc[0]["Model"]
best_r2    = results_df.iloc[0]["R² Score"]

# RESULTS TABLE
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## 🏆 Results")
st.markdown(f'<div class="result-best"><div style="color:#8b949e;font-size:0.85em;text-transform:uppercase;letter-spacing:2px">Best Model</div><div style="font-family:Syne,sans-serif;font-size:2.2em;font-weight:800;color:#3fb950">{best_model}</div><div style="color:#8b949e">R² Score: <span style="color:#3fb950;font-weight:700">{best_r2}</span> | Closer to 1.0 = better!</div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-card" style="font-size:0.88em"><b style="color:#3fb950">How to read:</b> R² closer to 1.0 = accurate | MAE & RMSE lower = better</div>', unsafe_allow_html=True)

def color_r2(val):
    if val >= 0.9: return "background-color:#1a3a2a;color:#3fb950;font-weight:bold"
    elif val >= 0.7: return "background-color:#2a2a1a;color:#f0b429;font-weight:bold"
    else: return "background-color:#2a1a1a;color:#f85149"

st.dataframe(results_df.style.applymap(color_r2, subset=["R² Score"]), use_container_width=True)

# CHARTS
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## 📈 Charts")

model_list = [m for m in predictions if m != "y_test"]

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    return fig, ax

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Actual vs Predicted","📊 Feature Importance","🔥 Correlation Heatmap","📉 Before vs After"])

with tab1:
    st.caption("Dots close to the diagonal = accurate predictions.")
    for i in range((len(model_list)+1)//2):
        cols = st.columns(2)
        for j in range(2):
            idx = i*2+j
            if idx >= len(model_list): break
            mname = model_list[idx]
            color = COLORS[idx % len(COLORS)]
            with cols[j]:
                fig, ax = dark_fig(5, 4)
                actual, preds = predictions["y_test"], predictions[mname]
                ax.scatter(actual, preds, alpha=0.3, s=6, color=color)
                lo, hi = float(actual.min()), float(actual.max())
                ax.plot([lo,hi],[lo,hi],"w--",lw=1.5)
                ax.set_title(f"{mname}\nR²={r2_score(actual,preds):.4f} | MAE={mean_absolute_error(actual,preds):.3f}", color="white", fontsize=10)
                ax.set_xlabel("Actual PCI", color="#8b949e", fontsize=9)
                ax.set_ylabel("Predicted PCI", color="#8b949e", fontsize=9)
                ax.tick_params(colors="#8b949e", labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#30363d")
                ax.grid(True, alpha=0.2, color="#30363d")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

with tab2:
    if not importances:
        st.info("Feature importance not available for SVR and Neural Network.")
    for idx, (mname, imp) in enumerate(importances.items()):
        fi = pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=True).tail(12)
        fig, ax = dark_fig(8, max(3, len(fi)*0.35))
        fi.plot(kind="barh", ax=ax, color=COLORS[idx % len(COLORS)], alpha=0.85)
        ax.set_title(f"Feature Importance — {mname}", color="white", fontsize=12)
        ax.set_xlabel("Importance", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.grid(True, alpha=0.2, axis="x", color="#30363d")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

with tab3:
    corr = df_proc.corr(numeric_only=True)
    fig, ax = dark_fig(12, 8)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.4, cbar_kws={"shrink":0.8}, ax=ax, annot_kws={"size":7})
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    pci_corr = corr["PCI"].drop("PCI").sort_values()
    fig2, ax2 = dark_fig(9, max(4, len(pci_corr)*0.35))
    pci_corr.plot(kind="barh", ax=ax2, color=["#f85149" if v<0 else "#3fb950" for v in pci_corr.values], alpha=0.85)
    ax2.axvline(0, color="white", lw=0.8)
    ax2.set_title("Which features affect PCI the most?", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Pearson Correlation", color="#8b949e", fontsize=9)
    ax2.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#30363d")
    ax2.grid(True, alpha=0.2, axis="x", color="#30363d")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with tab4:
    SAMPLE = min(80, len(y_test))
    actual_arr = predictions["y_test"][:SAMPLE]
    x_axis = np.arange(SAMPLE)
    for idx, mname in enumerate(model_list):
        pred_arr = predictions[mname][:SAMPLE]
        fig, ax = dark_fig(13, 4)
        ax.plot(x_axis, actual_arr, color="#58a6ff", lw=1.8, label="Actual PCI", marker="o", markersize=2)
        ax.plot(x_axis, pred_arr, color=COLORS[idx%len(COLORS)], lw=1.8, label="Predicted PCI", marker="s", markersize=2, ls="--")
        ax.fill_between(x_axis, actual_arr, pred_arr, alpha=0.12, color=COLORS[idx%len(COLORS)])
        ax.set_title(f"{mname}  |  R²={r2_score(predictions['y_test'],predictions[mname]):.4f}  |  MAE={mean_absolute_error(predictions['y_test'],predictions[mname]):.3f}", color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Road Segment", color="#8b949e", fontsize=9)
        ax.set_ylabel("PCI Score", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
        ax.grid(True, alpha=0.2, color="#30363d")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# DOWNLOADS
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## 💾 Download Results")
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Download Results Summary", results_df.to_csv(index=False).encode(), "pci_results_summary.csv", "text/csv")
with c2:
    pred_df = pd.DataFrame({"Actual_PCI": predictions["y_test"]})
    for m in model_list: pred_df[f"Predicted_{m.replace(' ','_')}"] = predictions[m]
    st.download_button("📥 Download All Predictions", pred_df.to_csv(index=False).encode(), "pci_predictions.csv", "text/csv")

st.markdown('<div style="text-align:center;color:#8b949e;padding:20px;font-size:0.8em">PCI Road Condition Predictor | Built with Streamlit & scikit-learn</div>', unsafe_allow_html=True)
