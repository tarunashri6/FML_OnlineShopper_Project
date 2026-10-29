"""ShopperAI — Purchase Intention Predictor (Stage 3: Overview + Live Prediction).

Navigation:
  Overview       — dataset stats, experiment leaderboard, research summary
  Live Prediction — interactive ML inference (the main working page)

All prediction logic is in src/predict_purchase.py.  This file handles only UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.predict_purchase import RAW_INPUT_COLUMNS, load_bundle, predict_purchase

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopperAI — Purchase Intention Predictor",
    page_icon="🛒",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths to verified result files
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RES  = ROOT / "results" / "nayantara"
DATA = ROOT / "data" / "processed"
FIGS = ROOT / "figures" / "nayantara"
FIGS_DISHITA = ROOT / "figures" / "dishita"

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Stitch palette + typography
# ─────────────────────────────────────────────────────────────────────────────
STITCH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap');

:root {
    --primary:            #004ac6;
    --primary-container:  #2563eb;
    --on-primary:         #ffffff;
    --surface:            #f8f9ff;
    --surface-low:        #eff4ff;
    --surface-container:  #e5eeff;
    --surface-high:       #dce9ff;
    --surface-highest:    #d3e4fe;
    --surface-bright:     #ffffff;
    --on-surface:         #0b1c30;
    --on-surface-var:     #434655;
    --secondary:          #565e74;
    --outline:            #737686;
    --outline-var:        #c3c6d7;
    --tertiary:           #006242;
    --tertiary-container: #007d55;
    --on-tertiary:        #ffffff;
    --tertiary-fixed:     #6ffbbe;
    --error:              #ba1a1a;
    --error-container:    #ffdad6;
    --on-error-container: #93000a;
    --sidebar-bg:         #0b1c30;
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --font-head:  'Plus Jakarta Sans', system-ui, sans-serif;
    --font-body:  'Inter', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--surface) !important;
    font-family: var(--font-body);
    color: var(--on-surface);
}
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

.sidebar-brand { font-family: var(--font-head); font-size:1.15rem; font-weight:700;
                 letter-spacing:-0.02em; color:#ffffff !important; }
.sidebar-sub   { font-size:0.75rem; color:#94a3b8 !important; }

.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(111,251,190,0.12); border:1px solid rgba(111,251,190,0.25);
    border-radius:var(--radius-md); padding:6px 10px;
    font-size:0.68rem; font-weight:600; letter-spacing:0.06em;
    text-transform:uppercase; color:var(--tertiary-fixed) !important;
}
.ping {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:var(--tertiary-fixed);
    animation:ping 1.4s cubic-bezier(0,0,0.2,1) infinite;
}
@keyframes ping {
    0%  { box-shadow: 0 0 0 0   rgba(111,251,190,0.7); }
    70% { box-shadow: 0 0 0 8px rgba(111,251,190,0);   }
    100%{ box-shadow: 0 0 0 0   rgba(111,251,190,0);   }
}

/* nav buttons styled in sidebar */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width:100%; text-align:left !important; justify-content:flex-start !important;
    background:transparent !important; border:none !important;
    color:#94a3b8 !important; border-radius:var(--radius-md) !important;
    padding:8px 12px !important; font-size:0.875rem !important; font-weight:500 !important;
    box-shadow:none !important; transition:background 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background:rgba(255,255,255,0.07) !important; color:#e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[data-active="true"],
[data-testid="stSidebar"] .nav-active [data-testid="stButton"] > button {
    background:var(--primary-container) !important; color:#ffffff !important;
    font-weight:600 !important;
}

.dataset-bar-label { font-size:0.65rem; font-weight:600; letter-spacing:0.07em;
                     text-transform:uppercase; color:#94a3b8 !important; }
.dataset-bar-value { font-size:0.75rem; font-weight:600; color:var(--tertiary-fixed) !important; }

/* ── Main ── */
.main .block-container { padding: 1rem 2rem 3rem !important; max-width:1440px !important; }

/* ── Custom In-Page Navigation Sidebar ── */
div[class*="st-key-custom_nav_container"] {
    background-color: var(--sidebar-bg) !important;
    border-radius: var(--radius-lg);
    padding: 18px 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
div[class*="st-key-custom_nav_container"] * {
    color: #e2e8f0;
}
div[class*="st-key-custom_nav_container"] [data-testid="stButton"] > button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    border-radius: var(--radius-md) !important;
    padding: 8px 12px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background 0.15s, color 0.15s !important;
}
div[class*="st-key-custom_nav_container"] [data-testid="stButton"] > button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}
div[class*="st-key-custom_nav_container"] .nav-active [data-testid="stButton"] > button {
    background: var(--primary-container) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ── Top Navigation Toggle Button ── */
.nav-toggle-wrap [data-testid="stButton"] > button {
    background: #0b1c30 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 7px 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: all 0.15s ease !important;
}
.nav-toggle-wrap [data-testid="stButton"] > button:hover {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 74, 198, 0.35) !important;
}

/* Hide native Streamlit sidebar and controls so they never conflict */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Generic cards ── */
.page-header {
    background:var(--surface-bright); border-radius:var(--radius-lg);
    padding:20px 24px; margin-bottom:1.5rem;
    box-shadow:0 1px 6px rgba(0,0,0,0.05);
    display:flex; align-items:flex-start; justify-content:space-between;
    gap:1rem; flex-wrap:wrap;
}
.page-header h1 { font-family:var(--font-head); font-size:1.5rem; font-weight:700;
                  letter-spacing:-0.02em; margin:0 0 4px; color:var(--on-surface); }
.subtitle-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(0,98,66,0.08); border:1px solid rgba(0,98,66,0.18);
    color:var(--tertiary); border-radius:999px; padding:3px 10px;
    font-size:0.65rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase;
}
.description { margin-top:8px; font-size:0.875rem; color:var(--secondary); }
.meta-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:var(--surface-low); border-radius:var(--radius-md);
    padding:5px 10px; font-size:0.75rem; font-weight:500;
    color:var(--on-surface-var); white-space:nowrap;
}
.viva-pill {
    background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.2);
    color:var(--primary) !important; font-weight:600;
    letter-spacing:0.04em; text-transform:uppercase; font-size:0.65rem;
}

/* ── Stat cards ── */
.stat-card {
    background:var(--surface-bright); border-radius:var(--radius-lg);
    padding:18px 20px; box-shadow:0 1px 6px rgba(0,0,0,0.05);
    display:flex; flex-direction:column; gap:4px;
}
.stat-label { font-size:0.68rem; font-weight:600; letter-spacing:0.07em;
              text-transform:uppercase; color:var(--secondary); }
.stat-value { font-family:var(--font-head); font-size:1.8rem; font-weight:700;
              letter-spacing:-0.03em; color:var(--on-surface); line-height:1.15; }
.stat-sub   { font-size:0.75rem; color:var(--outline); }

/* ── Section headers ── */
.section-heading {
    font-family:var(--font-head); font-size:1.05rem; font-weight:600;
    color:var(--on-surface); margin:0 0 12px;
    display:flex; align-items:center; gap:8px;
}

/* ── Leaderboard rows ── */
.lb-card {
    background:var(--surface-bright); border-radius:var(--radius-lg);
    padding:18px 20px; box-shadow:0 1px 6px rgba(0,0,0,0.05);
    margin-bottom:10px;
}
.lb-rank { font-size:0.65rem; font-weight:700; letter-spacing:0.07em;
           text-transform:uppercase; margin-bottom:6px; }
.lb-rank.gold   { color:#b45309; }
.lb-rank.silver { color:#64748b; }
.lb-config { font-family:var(--font-head); font-size:1rem; font-weight:600;
             color:var(--on-surface); margin-bottom:8px; }
.lb-metrics { display:flex; gap:16px; flex-wrap:wrap; }
.lb-metric { display:flex; flex-direction:column; }
.lb-metric-label { font-size:0.62rem; font-weight:600; letter-spacing:0.06em;
                   text-transform:uppercase; color:var(--secondary); }
.lb-metric-value { font-size:1.05rem; font-weight:700; color:var(--primary);
                   font-family:var(--font-head); }
.lb-metric-value.green { color:var(--tertiary); }
.lb-metric-value.amber { color:#b45309; }

/* ── Bar chart custom ── */
.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.bar-row-label { font-size:0.75rem; font-weight:500; color:var(--secondary);
                 min-width:140px; text-align:right; }
.bar-track { flex:1; height:10px; background:var(--surface-container);
             border-radius:999px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:999px; }
.bar-row-val { font-size:0.75rem; font-weight:600; color:var(--on-surface);
               min-width:44px; }

/* ── Distribution donut placeholder ── */
.dist-ring-wrap { display:flex; align-items:center; gap:20px; padding:4px 0; }
.dist-legend { display:flex; flex-direction:column; gap:8px; }
.dist-legend-item { display:flex; align-items:center; gap:8px; font-size:0.82rem; }
.dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }

/* ── Research cards ── */
.research-card {
    background:var(--surface-low); border-radius:var(--radius-md);
    padding:14px 16px; margin-bottom:8px;
}
.research-card-title { font-size:0.8rem; font-weight:600; color:var(--primary);
                       text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }
.research-card-body  { font-size:0.82rem; color:var(--on-surface-var); line-height:1.5; }

/* ── Inputs card / section groups ── */
.inputs-card {
    background:var(--surface-bright); border-radius:var(--radius-lg);
    padding:20px; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin-bottom:1rem;
}
.inputs-card-header {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:1rem; padding-bottom:8px;
    border-bottom:1px solid var(--surface-container);
}
.inputs-card .card-title { font-family:var(--font-head); font-size:1rem;
                           font-weight:600; color:var(--on-surface); }
.feature-count-badge {
    background:var(--surface-container); color:var(--secondary);
    font-size:0.65rem; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase; padding:3px 8px; border-radius:var(--radius-sm);
}
.section-group { background:var(--surface-low); border-radius:var(--radius-md);
                 padding:14px 16px; margin-bottom:12px; }
.section-group-title { font-size:0.7rem; font-weight:700; letter-spacing:0.07em;
                       text-transform:uppercase; color:var(--primary); margin-bottom:10px; }
.preset-label { font-size:0.65rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.07em; color:var(--secondary); }

/* ── Widget label overrides ── */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stCheckbox"] label {
    font-size:0.72rem !important; font-weight:600 !important;
    color:var(--secondary) !important; letter-spacing:0.03em;
}

/* ── Primary button ── */
[data-testid="stFormSubmitButton"] button,
[data-testid="stButton"] > button[kind="primary"] {
    background:var(--primary) !important; color:var(--on-primary) !important;
    border:none !important; border-radius:var(--radius-md) !important;
    font-family:var(--font-body) !important; font-size:0.9rem !important;
    font-weight:600 !important; padding:12px 24px !important;
    box-shadow:0 2px 8px rgba(0,74,198,0.25) !important;
    transition:filter 0.15s, box-shadow 0.15s !important;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stButton"] > button[kind="primary"]:hover {
    filter:brightness(1.08) !important;
    box-shadow:0 4px 16px rgba(0,74,198,0.35) !important;
}

/* ── Result banner ── */
.result-banner { border-radius:var(--radius-lg); padding:20px 22px;
                 display:flex; flex-direction:column; gap:14px;
                 margin-bottom:1rem; transition:background 0.4s; }
.result-banner.purchase    { background:rgba(0,98,66,0.08);   border:1px solid rgba(0,98,66,0.2);   }
.result-banner.no-purchase { background:rgba(186,26,26,0.07); border:1px solid rgba(186,26,26,0.18); }
.outcome-row { display:flex; align-items:center; justify-content:space-between;
               gap:12px; flex-wrap:wrap; }
.outcome-icon { width:48px; height:48px; border-radius:var(--radius-md);
                display:flex; align-items:center; justify-content:center;
                font-size:1.5rem; flex-shrink:0; }
.outcome-icon.purchase    { background:var(--tertiary); }
.outcome-icon.no-purchase { background:var(--error); }
.outcome-label { font-size:0.65rem; font-weight:700; letter-spacing:0.07em;
                 text-transform:uppercase; color:var(--secondary); margin-bottom:2px; }
.outcome-title { font-family:var(--font-head); font-size:1.3rem; font-weight:700;
                 letter-spacing:-0.02em; line-height:1.2; }
.outcome-title.purchase    { color:var(--tertiary); }
.outcome-title.no-purchase { color:var(--error);    }
.prob-col { text-align:right; flex-shrink:0; }
.prob-label { font-size:0.65rem; font-weight:600; letter-spacing:0.06em;
              text-transform:uppercase; color:var(--secondary); }
.prob-value { font-family:var(--font-head); font-size:2.2rem; font-weight:700;
              letter-spacing:-0.03em; line-height:1.1; }
.prob-value.purchase    { color:var(--tertiary); }
.prob-value.no-purchase { color:var(--error);    }
.gauge-section  { display:flex; flex-direction:column; gap:6px; }
.gauge-labels   { display:flex; justify-content:space-between;
                  font-size:0.68rem; color:var(--secondary); font-weight:500; }
.gauge-track    { position:relative; width:100%; height:10px;
                  background:var(--surface-container); border-radius:999px; overflow:hidden; }
.gauge-fill     { height:100%; border-radius:999px; transition:width 0.5s ease-out; }
.gauge-fill.purchase    { background:var(--tertiary); }
.gauge-fill.no-purchase { background:var(--error);    }
.gauge-cutoff-labels { display:flex; justify-content:space-between;
                       font-size:0.62rem; color:var(--outline); font-family:monospace; }
.confidence-chip { display:inline-block; background:var(--surface-highest);
                   color:var(--on-surface-var); padding:2px 8px;
                   border-radius:var(--radius-sm); font-size:0.65rem; font-weight:500; }
.result-card { background:var(--surface-bright); border-radius:var(--radius-lg);
               padding:20px; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin-bottom:1rem; }
.result-card-title { font-family:var(--font-head); font-size:0.95rem; font-weight:600;
                     color:var(--on-surface); margin-bottom:12px; }

/* ── Confusion Matrix ── */
.cm-card {
    background:var(--surface-bright); border-radius:var(--radius-lg);
    padding:20px; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin-bottom:1rem;
}
.cm-grid-header {
    background:var(--surface-container-low); padding:6px 8px; border-radius:var(--radius-sm);
    font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;
    color:var(--secondary); text-align:center;
}
.cm-cell {
    border-radius:var(--radius-md); padding:12px 14px; display:flex;
    flex-direction:column; justify-content:space-between; min-height:96px;
}
.cm-cell.tn { background:rgba(0,98,66,0.08); border:1px solid rgba(0,98,66,0.22); }
.cm-cell.fp, .cm-cell.fn { background:rgba(186,26,26,0.07); border:1px solid rgba(186,26,26,0.2); }
.cm-cell.tp { background:rgba(0,74,198,0.08); border:1px solid rgba(0,74,198,0.22); }
.cm-cell-top { display:flex; justify-content:space-between; align-items:center; }
.cm-cell-name { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
.cm-cell.tn .cm-cell-name { color:var(--tertiary); }
.cm-cell.fp .cm-cell-name, .cm-cell.fn .cm-cell-name { color:var(--error); }
.cm-cell.tp .cm-cell-name { color:var(--primary); }
.cm-tag { font-size:0.6rem; font-weight:700; padding:2px 6px; border-radius:4px; }
.cm-cell.tn .cm-tag { background:var(--tertiary); color:#fff; }
.cm-cell.fp .cm-tag, .cm-cell.fn .cm-tag { background:var(--error); color:#fff; }
.cm-cell.tp .cm-tag { background:var(--primary); color:#fff; }
.cm-count { font-family:var(--font-head); font-size:1.6rem; font-weight:700; color:var(--on-surface); line-height:1.2; margin:4px 0; }
.cm-rate { font-size:0.7rem; color:var(--secondary); display:flex; justify-content:space-between; align-items:center; }
.cm-rate-val { font-weight:700; }
.cm-cell.tn .cm-rate-val { color:var(--tertiary); }
.cm-cell.fp .cm-rate-val, .cm-cell.fn .cm-rate-val { color:var(--error); }
.cm-cell.tp .cm-rate-val { color:var(--primary); }
.cm-summary-bar {
    background:var(--surface-container-low); border-radius:var(--radius-md);
    padding:10px 14px; display:flex; justify-content:space-around; align-items:center;
    flex-wrap:wrap; gap:8px; margin-top:12px;
}
.cm-summary-item { font-size:0.75rem; color:var(--secondary); font-weight:500; }
.cm-summary-val { font-family:var(--font-head); font-weight:700; margin-left:4px; }

/* ── Grouped Bar Compare Item ── */
.compare-model-row {
    background:var(--surface-low); border-radius:var(--radius-md);
    padding:12px 14px; margin-bottom:10px; transition:background 0.15s;
}
.compare-model-row.champion {
    background:rgba(0,74,198,0.06); border:1px solid rgba(0,74,198,0.2);
}
.compare-model-head {
    display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;
}
.compare-model-name {
    font-size:0.85rem; font-weight:700; color:var(--on-surface);
    display:flex; align-items:center; gap:6px;
}
.compare-model-sub { font-size:0.7rem; color:var(--secondary); }
.metric-grid-4 {
    display:grid; grid-template-columns:repeat(4, 1fr); gap:8px;
}
.metric-bar-item { display:flex; flex-direction:column; gap:3px; }
.metric-bar-header { display:flex; justify-content:space-between; font-size:0.65rem; color:var(--secondary); font-weight:600; }
.metric-bar-track { width:100%; height:7px; background:var(--surface-container); border-radius:999px; overflow:hidden; }
.metric-bar-fill { height:100%; border-radius:999px; }
</style>
"""
st.markdown(STITCH_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model bundle…")
def get_bundle():
    return load_bundle()

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (cached, read-only)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_experiment_results() -> pd.DataFrame:
    return pd.read_csv(RES / "full_experiment_results.csv")

@st.cache_data
def load_experiment_summary() -> dict:
    return json.loads((RES / "full_experiment_summary.json").read_text())

@st.cache_data
def load_dataset_stats() -> dict:
    train = pd.read_csv(DATA / "train_raw.csv")
    test  = pd.read_csv(DATA / "test_raw.csv")
    total     = len(train) + len(test)
    purchases = int(train["Revenue"].sum() + test["Revenue"].sum())
    return {
        "total": total,
        "train": len(train),
        "test":  len(test),
        "purchases":   purchases,
        "no_purchase": total - purchases,
        "purchase_rate": purchases / total,
        "columns": len(train.columns),  # includes Revenue
    }

@st.cache_data
def load_cv_metrics() -> pd.DataFrame:
    return pd.read_csv(RES / "repeated_cv_metrics.csv")

@st.cache_data
def load_shap_summary() -> dict:
    return json.loads((RES / "best_f1_shap_summary.json").read_text())

@st.cache_resource(show_spinner="Initializing SHAP TreeExplainer…")
def get_shap_explainer():
    import shap
    bundle = get_bundle()
    return shap.TreeExplainer(bundle["model"])

def clean_feature_name(name: str) -> str:
    name = name.replace("numerical__", "").replace("categorical__", "")
    if "_" in name and not name.endswith("_Duration"):
        parts = name.split("_", 1)
        return f"{parts[0]}: {parts[1].replace('_', ' ')}"
    return name

def explain_session(raw_session: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from src.predict_purchase import _normalise_input_row
    bundle = get_bundle()
    pipeline = bundle["preprocessing_pipeline"]
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    processed_names = bundle["processed_feature_names"]

    session_df = _normalise_input_row(raw_session)
    processed = pipeline.transform(session_df)
    dense_processed = processed.toarray() if hasattr(processed, "toarray") else np.asarray(processed)
    processed_df = pd.DataFrame(dense_processed, columns=processed_names)
    selected_df = processed_df.loc[:, selected_features]
    feature_array = selected_df.to_numpy()

    probabilities = model.predict_proba(feature_array)[0]
    purchase_prob = float(probabilities[1])
    pred_class = bool(model.predict(feature_array)[0])

    explainer = get_shap_explainer()
    raw_shap = explainer.shap_values(feature_array, check_additivity=False)
    if isinstance(raw_shap, list):
        p_shap = raw_shap[1][0]
        base_val = explainer.expected_value[1]
    elif hasattr(raw_shap, "values"):
        p_shap = raw_shap.values[0, :, 1] if raw_shap.values.ndim == 3 else raw_shap.values[0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        p_shap = raw_shap[0, :, 1] if raw_shap.ndim == 3 else raw_shap[0]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    df_contribs = pd.DataFrame({
        "raw_feature": selected_features,
        "clean_feature": [clean_feature_name(f) for f in selected_features],
        "shap_value": p_shap
    })
    
    pos_drivers = df_contribs[df_contribs["shap_value"] > 0.0001].sort_values("shap_value", ascending=False)
    neg_drivers = df_contribs[df_contribs["shap_value"] < -0.0001].sort_values("shap_value", ascending=True)

    return {
        "purchase_probability": purchase_prob,
        "predicted_class": pred_class,
        "pos_drivers": pos_drivers,
        "neg_drivers": neg_drivers,
        "base_value": float(base_val)
    }

compute_individual_shap = explain_session

# ─────────────────────────────────────────────────────────────────────────────
# Navigation & Welcome Intro state
# ─────────────────────────────────────────────────────────────────────────────
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if "page" not in st.session_state:
    st.session_state.page = "overview"

if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = True

def _nav(page: str) -> None:
    st.session_state.page = page
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Welcome Screen (Jitter/Lottie Intro Animation)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_intro_animation_data() -> str:
    path = ROOT / "assets" / "Scene.json"
    if not path.exists():
        path = ROOT / "assets" / "shopperai_intro.json"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "{}"

def render_intro_screen() -> None:
    if st.query_params.get("entered") == "1":
        st.session_state.intro_shown = True
        st.session_state.page = "overview"
        if "entered" in st.query_params:
            del st.query_params["entered"]
        st.rerun()

    # Pure white minimal style across entire viewport
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] > .main {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 100vh !important;
    background: #ffffff !important;
    padding: 0 !important;
}
.block-container {
    padding: 0 !important;
    margin: auto !important;
    max-width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    background: #ffffff !important;
}
iframe {
    border: none !important;
    background: #ffffff !important;
}
div[data-testid="stButton"],
.stButton {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 2147483647 !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: auto !important;
}
div[data-testid="stButton"] button,
button[data-testid*="baseButton"],
.stButton button {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 2147483647 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

    anim_json = load_intro_animation_data()

    lottie_local_js = ""
    local_js_path = ROOT / "assets" / "lottie.min.js"
    if local_js_path.exists():
        lottie_local_js = local_js_path.read_text(encoding="utf-8")

    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .anim-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    max-width: 640px;
    max-height: 480px;
    aspect-ratio: 4 / 3;
    cursor: pointer;
    text-decoration: none;
    outline: none;
    border: none;
    background: #ffffff;
    margin: auto;
  }
  .anim-link:focus-visible {
    outline: 2px solid #2563eb;
    outline-offset: 4px;
    border-radius: 6px;
  }
  #lottie-box {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #ffffff;
  }
  #lottie-box svg {
    width: 100% !important;
    height: 100% !important;
    display: block;
    pointer-events: none;
  }
</style>
</head>
<body>
  <a class="anim-link" id="anim-click-target" href="?entered=1" target="_top" role="button" tabindex="0" aria-label="Enter ShopperAI">
    <div id="lottie-box"></div>
  </a>

  <script id="anim-data" type="application/json">
__ANIM_DATA__
  </script>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js"></script>
  <script>
    if (typeof lottie === 'undefined') {
      __LOCAL_LOTTIE_JS__
    }

    var target = document.getElementById('anim-click-target');

    function triggerEnter(e) {
      if (e) {
        try { e.preventDefault(); } catch (err) {}
      }
      try {
        window.top.location.search = '?entered=1';
        return;
      } catch (err) {}
      try {
        window.parent.location.search = '?entered=1';
        return;
      } catch (err) {}
      window.top.location.href = window.top.location.pathname + '?entered=1';
    }

    target.addEventListener('click', triggerEnter);
    target.addEventListener('keydown', function(evt) {
      if (evt.key === 'Enter' || evt.key === ' ') {
        triggerEnter(evt);
      }
    });

    try {
      var dataElem = document.getElementById('anim-data');
      var animData = JSON.parse(dataElem.textContent);
      var container = document.getElementById('lottie-box');
      
      lottie.loadAnimation({
        container: container,
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: animData
      });
    } catch (e) {
      console.error('Lottie load failed', e);
    }
  </script>
</body>
</html>"""

    lottie_html = html_template.replace("__ANIM_DATA__", anim_json).replace("__LOCAL_LOTTIE_JS__", lottie_local_js)

    comp_col1, comp_col2, comp_col3 = st.columns([1, 6, 1])
    with comp_col2:
        components.html(lottie_html, height=500, scrolling=False)

    if st.button("Enter", key="intro_screen_enter_trigger", help="Click anywhere to enter ShopperAI"):
        st.session_state.intro_shown = True
        st.session_state.page = "overview"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — branding + navigation (Custom In-Page Component)
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    ("overview",         "📊 Overview",          True),
    ("live_prediction",  "🔮 Live Prediction",   True),
    ("model_performance","📈 Model Performance",  True),
    ("feature_importance","🌲 Feature Importance", True),
    ("class_imbalance",  "⚖️ Class Imbalance",    True),
]

def render_nav_content() -> None:
    st.markdown("""
<div style="padding:4px 0 16px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
    <div style="width:36px;height:36px;border-radius:10px;background:#004ac6;
                display:flex;align-items:center;justify-content:center;
                font-size:18px;flex-shrink:0;">🛒</div>
    <div>
      <div class="sidebar-brand">ShopperAI</div>
      <div class="sidebar-sub">Purchase Intention ML Research</div>
    </div>
  </div>
  <div class="status-pill">
    <span class="ping"></span>
    Inference Ready &nbsp;·&nbsp; Random Forest
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:4px;'>", unsafe_allow_html=True)
    for page_key, label, available in PAGES:
        is_active = st.session_state.page == page_key
        if available:
            # Style active nav item via a wrapper class
            wrapper_cls = "nav-active" if is_active else ""
            st.markdown(f'<div class="{wrapper_cls}">', unsafe_allow_html=True)
            if st.button(("▶ " if is_active else "   ") + label,
                         key=f"nav_{page_key}", use_container_width=True):
                _nav(page_key)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Coming-soon — visually muted, not clickable
            st.markdown(
                f'<div style="padding:8px 12px;border-radius:10px;font-size:0.875rem;'
                f'color:#475569;display:flex;justify-content:space-between;align-items:center;">'
                f'<span>{label}</span>'
                f'<span style="font-size:0.6rem;background:rgba(255,255,255,0.06);'
                f'padding:2px 6px;border-radius:4px;letter-spacing:0.05em;color:#475569;">SOON</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px;margin-top:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span class="dataset-bar-label">Dataset Split</span>
    <span class="dataset-bar-value">80/20 Stratified</span>
  </div>
  <div style="width:100%;height:4px;background:rgba(255,255,255,0.1);border-radius:99px;
              overflow:hidden;margin-bottom:6px;">
    <div style="width:80%;height:100%;background:#6ffbbe;border-radius:99px;"></div>
  </div>
  <div class="sidebar-sub" style="font-size:0.68rem;">12,330 Total Sessions · 17 Features</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    if st.button("↺ Replay Welcome Animation", key="replay_intro_nav_btn", use_container_width=True, help="Watch the opening welcome animation again"):
        st.session_state.intro_shown = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Shared page header helper
# ─────────────────────────────────────────────────────────────────────────────
def render_page_header(title: str, badge: str, desc: str) -> None:
    st.markdown(f"""
<div class="page-header">
  <div>
    <h1>{title}</h1>
    <div style="margin-bottom:6px;"><span class="subtitle-badge">{badge}</span></div>
    <p class="description">{desc}</p>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
    <span class="meta-pill viva-pill">🎓 University Project Viva Demo</span>
    <span class="meta-pill">📊 UCI Online Shoppers (12,330 Sessions)</span>
    <span class="meta-pill">🌲 Random Forest Classifier</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def render_overview() -> None:
    render_page_header(
        title="Project Overview — Online Shopper Purchase Intention",
        badge="📚 FML Research Project · UCI Dataset",
        desc=(
            "Binary classification study: can we predict whether an online shopping session "
            "will result in a purchase? 48 model configurations were benchmarked across "
            "4 classifiers, 4 feature-selection strategies, and 3 class-imbalance methods."
        ),
    )

    stats  = load_dataset_stats()
    df_exp = load_experiment_results()
    cv     = load_cv_metrics()

    # ── Row 1: KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Sessions",   f"{stats['total']:,}",                "80/20 stratified split"),
        ("Raw Features",     "18",                                  "17 features + Revenue label"),
        ("Purchase Rate",    f"{stats['purchase_rate']:.1%}",      f"{stats['purchases']:,} purchase sessions"),
        ("Experiments Run",  "48",                                  "4 models × 4 features × 3 imbalance"),
        ("Best F1 Score",    "0.664",                               "RF + Tree Importance + SMOTE"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
<div class="stat-card">
  <div class="stat-label">{lbl}</div>
  <div class="stat-value">{val}</div>
  <div class="stat-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Distribution chart (left) + Research objectives (right) ────────
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown("""<div class="result-card">
<div class="section-heading">📊 Purchase Distribution</div>""", unsafe_allow_html=True)

        no_p_pct = stats["no_purchase"] / stats["total"] * 100
        p_pct    = stats["purchase_rate"] * 100

        # Simple horizontal bars as distribution chart
        for label, pct, color, count in [
            ("No Purchase (Revenue=False)", no_p_pct, "#e5eeff", stats["no_purchase"]),
            ("Purchase (Revenue=True)",     p_pct,    "#006242", stats["purchases"]),
        ]:
            st.markdown(f"""
<div class="bar-row" style="margin-bottom:14px;">
  <span class="bar-row-label">{label}</span>
  <div class="bar-track" style="height:18px;">
    <div class="bar-fill" style="width:{pct:.1f}%;background:{color};height:100%;border-radius:999px;"></div>
  </div>
  <span class="bar-row-val">{pct:.1f}%</span>
</div>
<div style="font-size:0.72rem;color:var(--secondary);margin-bottom:12px;
            text-align:right;margin-top:-8px;">{count:,} sessions</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:var(--surface-low);border-radius:var(--radius-md);
            padding:10px 14px;font-size:0.78rem;color:var(--on-surface-var);margin-top:4px;">
  ⚠️ <strong>Class imbalance:</strong> Only 15.5% of sessions result in a purchase.
  This motivated SMOTE and class-weighting strategies in the experiment grid.
</div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""<div class="result-card">
<div class="section-heading">🎯 Research Objectives</div>""", unsafe_allow_html=True)

        objectives = [
            ("Binary Classification Task",
             "Predict Revenue ∈ {True, False} from 17 raw session attributes — "
             "page counts, durations, bounce/exit rates, visitor type, and temporal context."),
            ("Feature Selection",
             "Compare four strategies: All features (baseline), Correlation-based pruning, "
             "Recursive Feature Elimination (RFE), and Tree-Importance ranking."),
            ("Class Imbalance Handling",
             "Evaluate three approaches: No correction (baseline), Class-weight adjustment, "
             "and SMOTE synthetic oversampling — all applied strictly inside CV folds."),
            ("Interpretability via SHAP",
             "Use SHAP values to explain individual predictions and confirm PageValues "
             "as the dominant feature, with ExitRates and ProductRelated_Duration as secondary drivers."),
            ("Validation Protocol",
             "10-fold Repeated Stratified Cross-Validation with mean ± 95% CI reported "
             "for F1, Recall, and ROC-AUC — avoiding single-split overfitting."),
        ]
        for title, body in objectives:
            st.markdown(f"""
<div class="research-card">
  <div class="research-card-title">{title}</div>
  <div class="research-card-body">{body}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Row 3: Experiment leaderboard ─────────────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-bottom:16px;">🏆 Top Configurations — Experiment Leaderboard</div>', unsafe_allow_html=True)

    lb_left, lb_right = st.columns(2, gap="large")

    # Best F1
    with lb_left:
        st.markdown("""<div style="font-size:0.78rem;font-weight:600;color:var(--secondary);
                              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
  🥇 Best F1 Score (Balanced Detection)
</div>""", unsafe_allow_html=True)

        top_f1 = df_exp.sort_values("f1", ascending=False).head(5)
        medals = ["🥇", "🥈", "🥉", "4.", "5."]
        for i, (_, row) in enumerate(top_f1.iterrows()):
            model_disp  = row["model"].replace("_", " ").title()
            feat_disp   = row["feature_configuration"].replace("_", " ").title()
            imb_disp    = row["imbalance_strategy"].replace("_", " ").title()
            st.markdown(f"""
<div class="lb-card">
  <div class="lb-rank {'gold' if i==0 else 'silver'}">{medals[i]} Rank {i+1}</div>
  <div class="lb-config">{model_disp} · {feat_disp} · {imb_disp}</div>
  <div class="lb-metrics">
    <div class="lb-metric">
      <span class="lb-metric-label">F1</span>
      <span class="lb-metric-value {'green' if i==0 else ''}">{row['f1']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Recall</span>
      <span class="lb-metric-value">{row['recall']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Precision</span>
      <span class="lb-metric-value">{row['precision']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">ROC-AUC</span>
      <span class="lb-metric-value">{row['roc_auc']:.4f}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Best Recall
    with lb_right:
        st.markdown("""<div style="font-size:0.78rem;font-weight:600;color:var(--secondary);
                              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
  🎯 Best Recall (Catch More Buyers)
</div>""", unsafe_allow_html=True)

        top_rec = df_exp.sort_values("recall", ascending=False).head(5)
        for i, (_, row) in enumerate(top_rec.iterrows()):
            model_disp = row["model"].replace("_", " ").title()
            feat_disp  = row["feature_configuration"].replace("_", " ").title()
            imb_disp   = row["imbalance_strategy"].replace("_", " ").title()
            st.markdown(f"""
<div class="lb-card">
  <div class="lb-rank {'gold' if i==0 else 'silver'}">{medals[i]} Rank {i+1}</div>
  <div class="lb-config">{model_disp} · {feat_disp} · {imb_disp}</div>
  <div class="lb-metrics">
    <div class="lb-metric">
      <span class="lb-metric-label">Recall</span>
      <span class="lb-metric-value {'green' if i==0 else 'amber'}">{row['recall']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">F1</span>
      <span class="lb-metric-value">{row['f1']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Precision</span>
      <span class="lb-metric-value">{row['precision']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">ROC-AUC</span>
      <span class="lb-metric-value">{row['roc_auc']:.4f}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Row 4: Cross-validation summary & F1 by model ─────────────────────────
    row4_left, row4_right = st.columns([6, 6], gap="large")

    with row4_left:
        st.markdown('<div class="result-card"><div class="section-heading">📉 10-Fold Repeated CV Highlights</div>', unsafe_allow_html=True)

        cv_display = cv[["display_name", "f1_mean", "f1_std", "recall_mean", "roc_auc_mean"]].copy()
        # show top 4 by f1_mean
        cv_top = cv_display.sort_values("f1_mean", ascending=False).head(4)
        for _, row in cv_top.iterrows():
            name     = row["display_name"]
            f1_m     = row["f1_mean"]
            f1_s     = row["f1_std"]
            rec_m    = row["recall_mean"]
            auc_m    = row["roc_auc_mean"]
            bar_w    = f1_m * 100
            st.markdown(f"""
<div style="margin-bottom:14px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-size:0.78rem;font-weight:600;color:var(--on-surface);">{name}</span>
    <span style="font-size:0.78rem;font-weight:700;color:var(--primary);">F1 {f1_m:.3f} ± {f1_s:.3f}</span>
  </div>
  <div class="bar-track" style="height:8px;margin-bottom:4px;">
    <div class="bar-fill" style="width:{bar_w:.1f}%;background:var(--primary);"></div>
  </div>
  <div style="font-size:0.68rem;color:var(--secondary);">
    Recall {rec_m:.3f} &nbsp;·&nbsp; ROC-AUC {auc_m:.3f}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with row4_right:
        st.markdown('<div class="result-card"><div class="section-heading">📊 Mean F1 by Model Class</div>', unsafe_allow_html=True)

        model_agg = (
            df_exp.groupby("model")["f1"]
            .agg(mean="mean", max="max")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        model_labels = {
            "random_forest":      "Random Forest",
            "svm":                "SVM",
            "logistic_regression":"Logistic Regression",
            "decision_tree":      "Decision Tree",
        }
        colors = ["#004ac6", "#2563eb", "#64748b", "#94a3b8"]
        for i, (_, row) in enumerate(model_agg.iterrows()):
            label = model_labels.get(row["model"], row["model"])
            mean  = row["mean"]
            mx    = row["max"]
            bar_w = mean / 0.70 * 100   # scale to 70% as max for visual spread
            st.markdown(f"""
<div class="bar-row" style="margin-bottom:12px;align-items:flex-start;">
  <span class="bar-row-label" style="line-height:1.4;">{label}</span>
  <div style="flex:1;">
    <div class="bar-track" style="height:10px;margin-bottom:3px;">
      <div class="bar-fill" style="width:{min(bar_w,100):.1f}%;background:{colors[i]};"></div>
    </div>
    <div style="font-size:0.68rem;color:var(--secondary);">max {mx:.4f}</div>
  </div>
  <span class="bar-row-val">{mean:.4f}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── CTA to Live Prediction ─────────────────────────────────────────────────
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    cta_c = st.columns([3, 2, 3])[1]
    with cta_c:
        if st.button("🔮 Try the Live Predictor →", type="primary", use_container_width=True):
            _nav("live_prediction")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
def render_live_prediction() -> None:
    render_page_header(
        title="Live Purchase Intention Prediction",
        badge="⚡ Trained RF (n_estimators=100) | Inference: <15ms",
        desc="Interactive Real-Time Inference with Trained Random Forest Classifier (UCI Online Shoppers Dataset)",
    )

    # ── Preset state helpers ──────────────────────────────────────────────────
    PRESETS = {
        "high_intent": dict(
            administrative=4, administrative_duration=95.0,
            informational=2, informational_duration=52.0,
            product_related=35, product_related_duration=1420.0,
            page_values=28.5, bounce_rates=0.005, exit_rates=0.015,
            special_day=0.0, month="Nov", visitor_type="Returning_Visitor",
            weekend=True, operating_systems=2, browser=2, region=1, traffic_type=2,
        ),
        "window_shopper": dict(
            administrative=0, administrative_duration=0.0,
            informational=0, informational_duration=0.0,
            product_related=3, product_related_duration=45.0,
            page_values=0.0, bounce_rates=0.08, exit_rates=0.12,
            special_day=0.0, month="Mar", visitor_type="New_Visitor",
            weekend=False, operating_systems=2, browser=2, region=1, traffic_type=2,
        ),
        "returning_customer": dict(
            administrative=2, administrative_duration=40.0,
            informational=1, informational_duration=12.0,
            product_related=18, product_related_duration=620.0,
            page_values=14.2, bounce_rates=0.010, exit_rates=0.022,
            special_day=0.0, month="Dec", visitor_type="Returning_Visitor",
            weekend=True, operating_systems=2, browser=2, region=1, traffic_type=2,
        ),
    }

    def _load_preset(key: str) -> None:
        for field, val in PRESETS[key].items():
            st.session_state[f"inp_{field}"] = val

    def _get(field, default):
        return st.session_state.get(f"inp_{field}", default)

    left_col, right_col = st.columns([5, 7], gap="large")

    # ── LEFT — input form ─────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
<div class="inputs-card" style="margin-bottom:1rem;">
  <div class="inputs-card-header">
    <span class="card-title">⚙️ Visitor Feature Inputs</span>
    <span class="feature-count-badge">17 Features</span>
  </div>
  <div class="preset-label" style="margin-bottom:6px;">Quick Presets</div>
""", unsafe_allow_html=True)

        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            if st.button("🛍️ High-Intent Buyer", key="preset_high", use_container_width=True):
                _load_preset("high_intent"); st.rerun()
        with pcol2:
            if st.button("👁️ Window Shopper", key="preset_window", use_container_width=True):
                _load_preset("window_shopper"); st.rerun()
        with pcol3:
            if st.button("🔄 Returning Customer", key="preset_return", use_container_width=True):
                _load_preset("returning_customer"); st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("prediction_form"):

            st.markdown('<div class="section-group"><div class="section-group-title">📂 1. Browsing Behaviour</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                administrative = st.number_input(
                    "Admin Pages", min_value=0, max_value=27,
                    value=_get("administrative", 2), step=1, key="f_admin",
                    help="Administrative pages visited.",
                )
                informational = st.number_input(
                    "Info Pages", min_value=0, max_value=24,
                    value=_get("informational", 0), step=1, key="f_info",
                    help="Informational pages visited.",
                )
                product_related = st.number_input(
                    "Product Pages", min_value=0, max_value=705,
                    value=_get("product_related", 18), step=1, key="f_prod",
                    help="Product-related pages visited.",
                )
            with c2:
                administrative_duration = st.number_input(
                    "Admin Duration (s)", min_value=0.0, max_value=3398.75,
                    value=float(_get("administrative_duration", 0.0)),
                    step=1.0, format="%.1f", key="f_admin_dur",
                )
                informational_duration = st.number_input(
                    "Info Duration (s)", min_value=0.0, max_value=2549.375,
                    value=float(_get("informational_duration", 0.0)),
                    step=1.0, format="%.1f", key="f_info_dur",
                )
                product_related_duration = st.number_input(
                    "Product Duration (s)", min_value=0.0, max_value=63973.522,
                    value=float(_get("product_related_duration", 597.0)),
                    step=1.0, format="%.1f", key="f_prod_dur",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-group"><div class="section-group-title">📊 2. Web Analytics Telemetry</div>', unsafe_allow_html=True)
            page_values = st.slider(
                "PageValues (GA Key Metric)", min_value=0.0, max_value=361.764,
                value=float(_get("page_values", 0.0)), step=0.05, format="%.2f", key="f_pv",
                help="Average value of pages viewed prior to transaction.",
            )
            c3, c4 = st.columns(2)
            with c3:
                bounce_rates = st.number_input(
                    "Bounce Rate (0–0.2)", min_value=0.0, max_value=0.2,
                    value=float(_get("bounce_rates", 0.02)), step=0.001, format="%.3f", key="f_bounce",
                )
                special_day = st.select_slider(
                    "Special Day Proximity",
                    options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    value=_get("special_day", 0.0), key="f_special",
                    help="0 = far from special day, 1 = the day itself.",
                )
            with c4:
                exit_rates = st.number_input(
                    "Exit Rate (0–0.2)", min_value=0.0, max_value=0.2,
                    value=float(_get("exit_rates", 0.04)), step=0.001, format="%.3f", key="f_exit",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-group"><div class="section-group-title">👤 3. Session Demographics &amp; Context</div>', unsafe_allow_html=True)
            MONTH_OPTS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            VT_OPTS    = ["Returning_Visitor", "New_Visitor", "Other"]
            c5, c6 = st.columns(2)
            with c5:
                _m_def = _get("month", "May")
                month = st.selectbox(
                    "Month", options=MONTH_OPTS,
                    index=MONTH_OPTS.index(_m_def) if _m_def in MONTH_OPTS else 2,
                    key="f_month",
                )
                _vt_def = _get("visitor_type", "Returning_Visitor")
                visitor_type = st.selectbox(
                    "Visitor Type", options=VT_OPTS,
                    index=VT_OPTS.index(_vt_def) if _vt_def in VT_OPTS else 0,
                    key="f_vt",
                )
                weekend = st.checkbox(
                    "Weekend Session",
                    value=bool(_get("weekend", False)), key="f_weekend",
                )
            with c6:
                operating_systems = st.selectbox(
                    "Operating System", options=list(range(1, 9)),
                    index=_get("operating_systems", 2) - 1, key="f_os",
                )
                browser = st.selectbox(
                    "Browser", options=list(range(1, 14)),
                    index=_get("browser", 2) - 1, key="f_browser",
                )
                region = st.selectbox(
                    "Region", options=list(range(1, 10)),
                    index=_get("region", 1) - 1, key="f_region",
                )
                traffic_type = st.selectbox(
                    "Traffic Type", options=list(range(1, 21)),
                    index=_get("traffic_type", 2) - 1, key="f_traffic",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "🔮  Predict Purchase Intention",
                use_container_width=True, type="primary",
            )

    # ── RIGHT — results panel ─────────────────────────────────────────────────
    with right_col:
        if not submitted:
            st.markdown("""
<div class="result-card" style="min-height:220px;display:flex;flex-direction:column;
     align-items:center;justify-content:center;text-align:center;gap:10px;">
  <div style="font-size:2.5rem;">🛒</div>
  <div style="font-family:var(--font-head);font-size:1.1rem;font-weight:600;color:var(--on-surface);">
    Awaiting Prediction
  </div>
  <div style="font-size:0.85rem;color:var(--secondary);max-width:320px;">
    Fill in visitor session attributes on the left and click
    <strong>Predict Purchase Intention</strong> to see the model's decision.
  </div>
</div>""", unsafe_allow_html=True)
        else:
            raw_session: dict = {
                "Administrative":          int(administrative),
                "Administrative_Duration": float(administrative_duration),
                "Informational":           int(informational),
                "Informational_Duration":  float(informational_duration),
                "ProductRelated":          int(product_related),
                "ProductRelated_Duration": float(product_related_duration),
                "BounceRates":             float(bounce_rates),
                "ExitRates":               float(exit_rates),
                "PageValues":              float(page_values),
                "SpecialDay":              float(special_day),
                "Month":                   str(month),
                "OperatingSystems":        int(operating_systems),
                "Browser":                 int(browser),
                "Region":                  int(region),
                "TrafficType":             int(traffic_type),
                "VisitorType":             str(visitor_type),
                "Weekend":                 bool(weekend),
            }
            assert "Revenue" not in raw_session
            assert sorted(raw_session.keys()) == sorted(RAW_INPUT_COLUMNS)

            try:
                bundle = get_bundle()
                result = predict_purchase(raw_session, bundle=bundle)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

            predicted_purchase: bool     = result["predicted_purchase"]
            purchase_probability: float | None = result["purchase_probability"]
            prob_pct     = (purchase_probability * 100) if purchase_probability is not None else 0.0
            outcome_cls  = "purchase" if predicted_purchase else "no-purchase"
            outcome_icon = "✅" if predicted_purchase else "❌"
            outcome_text = "PREDICTED OUTCOME: PURCHASE" if predicted_purchase else "PREDICTED OUTCOME: NO PURCHASE"
            prob_display = f"{prob_pct:.1f}%" if purchase_probability is not None else "N/A"
            confidence   = abs(prob_pct - 50.0)

            st.markdown(f"""
<div class="result-banner {outcome_cls}">
  <div class="outcome-row">
    <div style="display:flex;align-items:center;gap:14px;">
      <div class="outcome-icon {outcome_cls}">{outcome_icon}</div>
      <div>
        <div class="outcome-label">Predicted Outcome</div>
        <div class="outcome-title {outcome_cls}">{outcome_text}</div>
      </div>
    </div>
    <div class="prob-col">
      <div class="prob-label">Posterior Probability</div>
      <div class="prob-value {outcome_cls}">{prob_display}</div>
    </div>
  </div>
  <div class="gauge-section">
    <div class="gauge-labels">
      <span>0% (No Purchase)</span>
      <span style="color:var(--primary);font-weight:600;">Cutoff τ = 0.50</span>
      <span>100% (Purchase)</span>
    </div>
    <div class="gauge-track">
      <div class="gauge-fill {outcome_cls}" style="width:{min(prob_pct,100):.1f}%;"></div>
    </div>
    <div class="gauge-cutoff-labels">
      <span>Class 0 [Revenue=False]</span>
      <span class="confidence-chip">Confidence Margin: {'+' if predicted_purchase else '-'}{confidence:.1f}%</span>
      <span>Class 1 [Revenue=True]</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # Local SHAP Prediction Attribution (Top 3 Positive & Top 3 Negative Drivers)
            try:
                explanation = compute_individual_shap(raw_session)
                pos_top3 = explanation["pos_drivers"].head(3)
                neg_top3 = explanation["neg_drivers"].head(3)
                base_prior = explanation["base_value"] * 100
                shap_ready = True
            except Exception as shap_err:
                shap_ready = False

            if shap_ready:
                st.markdown(f"""
<div style="background:var(--surface-bright);border-radius:var(--radius-lg);padding:16px 20px;margin-top:14px;box-shadow:0 1px 6px rgba(0,0,0,0.06);border:1px solid var(--outline-var);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">
        Prediction Drivers (SHAP Local Attribution)
      </div>
      <div style="font-size:0.75rem;color:var(--secondary);">
        Additive feature contributions displacing probability from model prior baseline ({base_prior:.1f}%)
      </div>
    </div>
    <span class="viva-pill" style="font-size:0.62rem;background:rgba(0,74,198,0.1);color:var(--primary) !important;">Tree SHAP Attribution</span>
  </div>
""", unsafe_allow_html=True)
                sc_pos, sc_neg = st.columns(2, gap="medium")
                with sc_pos:
                    st.markdown("""
<div style="background:rgba(0,98,66,0.05);border-radius:var(--radius-md);padding:12px 14px;border:1px solid rgba(0,98,66,0.15);height:100%;">
  <div style="font-family:var(--font-head);font-size:0.82rem;font-weight:700;color:var(--tertiary);margin-bottom:8px;display:flex;justify-content:space-between;">
    <span>▲ Top 3 Positive Drivers</span>
    <span style="font-size:0.65rem;background:rgba(0,98,66,0.12);padding:1px 6px;border-radius:3px;">+Δ Toward Purchase</span>
  </div>
""", unsafe_allow_html=True)
                    if len(pos_top3) > 0:
                        for _, r in pos_top3.iterrows():
                            val_shift = r['shap_value'] * 100
                            st.markdown(f"""
  <div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:2px;">
      <span style="font-weight:600;color:var(--on-surface);">{r['clean_feature']}</span>
      <span style="font-family:var(--font-head);font-weight:700;color:var(--tertiary);">+{val_shift:.1f}%</span>
    </div>
    <div class="bar-track" style="height:5px;">
      <div class="bar-fill" style="width:{min(val_shift*3, 100):.1f}%;background:var(--tertiary);"></div>
    </div>
  </div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:0.73rem;color:var(--secondary);'>No positive driver displaced probability for this session.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with sc_neg:
                    st.markdown("""
<div style="background:rgba(186,26,26,0.05);border-radius:var(--radius-md);padding:12px 14px;border:1px solid rgba(186,26,26,0.15);height:100%;">
  <div style="font-family:var(--font-head);font-size:0.82rem;font-weight:700;color:var(--error);margin-bottom:8px;display:flex;justify-content:space-between;">
    <span>▼ Top 3 Negative Drivers</span>
    <span style="font-size:0.65rem;background:rgba(186,26,26,0.12);padding:1px 6px;border-radius:3px;">-Δ Suppressing Intent</span>
  </div>
""", unsafe_allow_html=True)
                    if len(neg_top3) > 0:
                        for _, r in neg_top3.iterrows():
                            val_shift = abs(r['shap_value']) * 100
                            st.markdown(f"""
  <div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:2px;">
      <span style="font-weight:600;color:var(--on-surface);">{r['clean_feature']}</span>
      <span style="font-family:var(--font-head);font-weight:700;color:var(--error);">-{val_shift:.1f}%</span>
    </div>
    <div class="bar-track" style="height:5px;">
      <div class="bar-fill" style="width:{min(val_shift*3, 100):.1f}%;background:var(--error);"></div>
    </div>
  </div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:0.73rem;color:var(--secondary);'>No negative driver displaced probability for this session.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 14px;margin-top:12px;font-size:0.73rem;color:var(--on-surface-var);line-height:1.45;">
    ℹ️ <strong>Statistical Interpretation:</strong> SHAP (SHapley Additive exPlanations) values quantify each feature's marginal displacement of the predicted probability away from the base value ({base_prior:.1f}%), derived from cooperative game theory. Positive values push the probability toward purchase, while negative values suppress it. These reflect the model's learned associations and do not represent causal effects in customer psychology.
  </div>
</div>
""", unsafe_allow_html=True)

            with st.expander("🔬 Technical Inference Details (Examiner Deep-Dive)"):
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown("**Feature Vector Shape**")
                    st.code(f"[1, {len(raw_session)}] Encoded Float")
                with d2:
                    st.markdown("**Model Type**")
                    st.code(type(get_bundle()["model"]).__name__)
                with d3:
                    st.markdown("**Decision Rule**")
                    st.code(f"P ≥ 0.50 → Purchase\nP(session) = {prob_pct:.1f}%")
                st.markdown("**Raw Input Tensor (UCI-17 Schema)**")
                st.code(str([raw_session[col] for col in RAW_INPUT_COLUMNS]))


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE & BENCHMARKS
# ═════════════════════════════════════════════════════════════════════════════
def render_model_performance() -> None:
    render_page_header(
        title="Model Performance & Benchmark Evaluation",
        badge="📈 Research Experimentation & Benchmarks · K-Fold CV Metrics",
        desc="Empirical Comparison of 4 Supervised Classifiers on 10-Fold Stratified Cross-Validation & Held-out Test Set"
    )

    df = load_experiment_results()
    cv_df = load_cv_metrics()

    # Dynamic champion identification
    best_f1_row = df.sort_values("f1", ascending=False).iloc[0]
    best_rec_row = df.sort_values("recall", ascending=False).iloc[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Top Champion Cards (2 Columns)
    # ─────────────────────────────────────────────────────────────────────────
    top_c1, top_c2 = st.columns(2)

    with top_c1:
        st.markdown(f"""
<div class="lb-card" style="border-left:4px solid var(--primary);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
        <span class="viva-pill" style="background:var(--primary);color:#fff !important;padding:2px 8px;font-size:0.6rem;">SELECTED CHAMPION</span>
        <span class="lb-rank gold">★ Best Overall F1-Score</span>
      </div>
      <div class="lb-config" style="font-size:1.2rem;margin-top:2px;">Random Forest (Tree-Imp + SMOTE)</div>
      <div style="font-size:0.75rem;color:var(--secondary);margin-bottom:12px;">
        Optimal balance between capturing intent and suppressing expensive false alarms in production.
      </div>
    </div>
    <div style="width:42px;height:42px;border-radius:10px;background:rgba(0,74,198,0.1);display:flex;align-items:center;justify-content:center;font-size:22px;">🏆</div>
  </div>
  <div class="lb-metrics" style="background:var(--surface-low);padding:10px 14px;border-radius:var(--radius-md);">
    <div class="lb-metric">
      <span class="lb-metric-label">Optimized F1</span>
      <span class="lb-metric-value" style="font-size:1.35rem;">{best_f1_row['f1']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Test Recall</span>
      <span class="lb-metric-value green">{best_f1_row['recall']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Precision</span>
      <span class="lb-metric-value">{best_f1_row['precision']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Accuracy</span>
      <span class="lb-metric-value">{best_f1_row['accuracy']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">ROC-AUC</span>
      <span class="lb-metric-value green">{best_f1_row['roc_auc']:.4f}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with top_c2:
        st.markdown(f"""
<div class="lb-card" style="border-left:4px solid var(--tertiary);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
        <span class="viva-pill" style="background:var(--tertiary);color:#fff !important;padding:2px 8px;font-size:0.6rem;">HIGH SENSITIVITY</span>
        <span class="lb-rank green" style="color:var(--tertiary);">★ Highest Purchase Recall</span>
      </div>
      <div class="lb-config" style="font-size:1.2rem;margin-top:2px;">SVM (RFE + Class Weight)</div>
      <div style="font-size:0.75rem;color:var(--secondary);margin-bottom:12px;">
        Maximizes retrieval of prospective buyers to minimize lost customer conversion opportunities.
      </div>
    </div>
    <div style="width:42px;height:42px;border-radius:10px;background:rgba(0,98,66,0.1);display:flex;align-items:center;justify-content:center;font-size:22px;">🎯</div>
  </div>
  <div class="lb-metrics" style="background:var(--surface-low);padding:10px 14px;border-radius:var(--radius-md);">
    <div class="lb-metric">
      <span class="lb-metric-label">Peak Recall</span>
      <span class="lb-metric-value green" style="font-size:1.35rem;">{best_rec_row['recall']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">F1-Score</span>
      <span class="lb-metric-value">{best_rec_row['f1']:.4f}</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Precision</span>
      <span class="lb-metric-value amber">{best_rec_row['precision']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">Accuracy</span>
      <span class="lb-metric-value">{best_rec_row['accuracy']*100:.1f}%</span>
    </div>
    <div class="lb-metric">
      <span class="lb-metric-label">ROC-AUC</span>
      <span class="lb-metric-value green">{best_rec_row['roc_auc']:.4f}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 1: Baseline Architecture Comparison & ROC-AUC Ranking
    # ─────────────────────────────────────────────────────────────────────────
    col_comp, col_auc = st.columns([7, 5])

    baseline_models = [
        ("Random Forest", "random_forest", "Top Balance Ensemble (Baseline)", True),
        ("Support Vector Machine", "svm", "RBF Kernel (C=1.0, Gamma=Scale)", False),
        ("Logistic Regression", "logistic_regression", "Linear Baseline (L2 Regularized)", False),
        ("Decision Tree", "decision_tree", "CART Tree (Max Depth = 8 Pruned)", False),
    ]

    with col_comp:
        st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Architectural Baseline Comparison</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Held-out test set performance using all 17 features without sampling adjustments</div>
    </div>
    <div style="display:flex;gap:10px;font-size:0.65rem;font-weight:600;">
      <span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:var(--secondary);border-radius:2px;"></span>ACC</span>
      <span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:#3b82f6;border-radius:2px;"></span>PREC</span>
      <span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:var(--tertiary);border-radius:2px;"></span>REC</span>
      <span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:var(--primary);border-radius:2px;"></span>F1</span>
    </div>
  </div>
""", unsafe_allow_html=True)

        for name, key, subtitle, is_champ in baseline_models:
            m_row = df[(df["model"] == key) & (df["feature_configuration"] == "all") & (df["imbalance_strategy"] == "none")].iloc[0]
            champ_cls = "champion" if is_champ else ""
            badge_html = '<span class="viva-pill" style="font-size:0.6rem;padding:1px 6px;">Top Baseline</span>' if is_champ else ""
            st.markdown(f"""
<div class="compare-model-row {champ_cls}">
  <div class="compare-model-head">
    <div class="compare-model-name">
      {name} {badge_html}
    </div>
    <div class="compare-model-sub">{subtitle}</div>
  </div>
  <div class="metric-grid-4">
    <div class="metric-bar-item">
      <div class="metric-bar-header"><span>ACC</span><span>{m_row['accuracy']*100:.1f}%</span></div>
      <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{m_row['accuracy']*100:.1f}%;background:var(--secondary);"></div></div>
    </div>
    <div class="metric-bar-item">
      <div class="metric-bar-header"><span>PREC</span><span>{m_row['precision']*100:.1f}%</span></div>
      <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{m_row['precision']*100:.1f}%;background:#3b82f6;"></div></div>
    </div>
    <div class="metric-bar-item">
      <div class="metric-bar-header"><span>REC</span><span>{m_row['recall']*100:.1f}%</span></div>
      <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{m_row['recall']*100:.1f}%;background:var(--tertiary);"></div></div>
    </div>
    <div class="metric-bar-item">
      <div class="metric-bar-header"><span style="color:var(--primary);font-weight:700;">F1</span><span style="color:var(--primary);font-weight:700;">{m_row['f1']*100:.1f}%</span></div>
      <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{m_row['f1']*100:.1f}%;background:var(--primary);"></div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_auc:
        st.markdown("""
<div class="result-card">
  <div class="result-card-title" style="margin-bottom:2px;">ROC-AUC Discriminative Power</div>
  <div style="font-size:0.75rem;color:var(--secondary);margin-bottom:14px;">Area under Receiver Operating Characteristic curve on held-out test split</div>
""", unsafe_allow_html=True)

        auc_list = []
        for name, key, subtitle, is_champ in baseline_models:
            m_row = df[(df["model"] == key) & (df["feature_configuration"] == "all") & (df["imbalance_strategy"] == "none")].iloc[0]
            auc_list.append((name, m_row["roc_auc"], is_champ))
        auc_list.sort(key=lambda x: x[1], reverse=True)

        for name, auc_val, is_champ in auc_list:
            color = "var(--primary)" if is_champ else "var(--tertiary)" if auc_val >= 0.88 else "var(--secondary)"
            bold_style = "font-weight:700;color:var(--primary);" if is_champ else "color:var(--on-surface);"
            st.markdown(f"""
<div style="margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
    <span style="{bold_style}">{name}</span>
    <span style="font-family:var(--font-head);font-weight:700;{bold_style}">{auc_val:.4f}</span>
  </div>
  <div class="bar-track" style="height:8px;">
    <div class="bar-fill" style="width:{auc_val*100:.1f}%;background:{color};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        roc_fig_path = FIGS / "roc_auc_by_model.png"
        if roc_fig_path.exists():
            st.image(str(roc_fig_path), caption="ROC-AUC Curves by Model (Generated by experiments)", use_container_width=True)
        st.markdown("""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 12px;margin-top:14px;font-size:0.73rem;color:var(--on-surface-var);line-height:1.45;">
    💡 <strong>Examiner Note:</strong> ROC-AUC measures ranking discrimination across all classification thresholds τ ∈ [0, 1]. Both Random Forest (0.916) and Logistic Regression (0.883) achieve excellent separation, proving strong underlying posterior probability calibration before threshold cutoffs.
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 2: Confusion Matrix & Research Interpretation
    # ─────────────────────────────────────────────────────────────────────────
    col_cm, col_interp = st.columns([6, 6])

    with col_cm:
        # Champion selector
        cm_choice = st.radio(
            "Select Evaluation Configuration:",
            ["🥇 Best F1 Champion (Random Forest)", "🎯 Best Recall Champion (SVM)"],
            horizontal=True,
            key="mp_cm_selector"
        )
        use_best_f1_cm = "Best F1" in cm_choice
        cm_row = best_f1_row if use_best_f1_cm else best_rec_row
        cm_label = "RF + Tree-Importance + SMOTE" if use_best_f1_cm else "SVM + RFE + Class Weight"
        tn = int(cm_row["tn"])
        fp = int(cm_row["fp"])
        fn = int(cm_row["fn"])
        tp = int(cm_row["tp"])
        total_test = tn + fp + fn + tp
        total_neg = tn + fp
        total_pos = fn + tp
        tnr = (tn / total_neg) * 100
        fpr = (fp / total_neg) * 100
        fnr = (fn / total_pos) * 100
        tpr = (tp / total_pos) * 100

        st.markdown(f"""
<div class="cm-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <span class="viva-pill" style="font-size:0.6rem;padding:2px 6px;">Evaluation Sample</span>
      <div class="result-card-title" style="margin-top:4px;margin-bottom:2px;">Confusion Matrix — Champion Model</div>
      <div style="font-size:0.75rem;color:var(--secondary);">{cm_label} (Test split n = {total_test:,}, τ = 0.50)</div>
    </div>
    <span class="meta-pill" style="font-size:0.7rem;font-weight:600;">Accuracy {cm_row['accuracy']*100:.1f}%</span>
  </div>

  <!-- Header row -->
  <div style="display:grid;grid-template-columns:100px 1fr 1fr;gap:6px;margin-bottom:6px;">
    <div></div>
    <div class="cm-grid-header">Pred: No Purchase</div>
    <div class="cm-grid-header">Pred: Purchase</div>
  </div>

  <!-- Row 1: Actual No Purchase -->
  <div style="display:grid;grid-template-columns:100px 1fr 1fr;gap:6px;margin-bottom:6px;">
    <div class="cm-grid-header" style="display:flex;align-items:center;justify-content:center;text-align:center;line-height:1.2;">
      Actual: No Purchase
    </div>
    <div class="cm-cell tn">
      <div class="cm-cell-top">
        <span class="cm-cell-name">True Neg (TN)</span>
        <span class="cm-tag">TN</span>
      </div>
      <div class="cm-count">{tn:,}</div>
      <div class="cm-rate">
        <span>Specificity (TNR)</span>
        <span class="cm-rate-val">{tnr:.1f}%</span>
      </div>
    </div>
    <div class="cm-cell fp">
      <div class="cm-cell-top">
        <span class="cm-cell-name">False Pos (FP)</span>
        <span class="cm-tag">FP</span>
      </div>
      <div class="cm-count">{fp:,}</div>
      <div class="cm-rate">
        <span>False Alarm Rate</span>
        <span class="cm-rate-val">{fpr:.1f}%</span>
      </div>
    </div>
  </div>

  <!-- Row 2: Actual Purchase -->
  <div style="display:grid;grid-template-columns:100px 1fr 1fr;gap:6px;">
    <div class="cm-grid-header" style="display:flex;align-items:center;justify-content:center;text-align:center;line-height:1.2;">
      Actual: Purchase
    </div>
    <div class="cm-cell fn">
      <div class="cm-cell-top">
        <span class="cm-cell-name">False Neg (FN)</span>
        <span class="cm-tag">FN</span>
      </div>
      <div class="cm-count">{fn:,}</div>
      <div class="cm-rate">
        <span>Missed Intent (FNR)</span>
        <span class="cm-rate-val">{fnr:.1f}%</span>
      </div>
    </div>
    <div class="cm-cell tp">
      <div class="cm-cell-top">
        <span class="cm-cell-name">True Pos (TP)</span>
        <span class="cm-tag">TP</span>
      </div>
      <div class="cm-count">{tp:,}</div>
      <div class="cm-rate">
        <span>Recall (TPR)</span>
        <span class="cm-rate-val">{tpr:.1f}%</span>
      </div>
    </div>
  </div>

  <!-- Bottom statistical summary bar -->
  <div class="cm-summary-bar">
    <div class="cm-summary-item">Specificity: <span class="cm-summary-val" style="color:var(--tertiary);">{tnr:.1f}%</span></div>
    <span style="color:var(--outline);">|</span>
    <div class="cm-summary-item">Recall: <span class="cm-summary-val" style="color:var(--primary);">{tpr:.1f}%</span></div>
    <span style="color:var(--outline);">|</span>
    <div class="cm-summary-item">Precision: <span class="cm-summary-val" style="color:#2563eb;">{cm_row['precision']*100:.1f}%</span></div>
    <span style="color:var(--outline);">|</span>
    <div class="cm-summary-item">F1 Score: <span class="cm-summary-val" style="color:var(--primary);">{cm_row['f1']:.4f}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
        if use_best_f1_cm:
            cm_fig_path = FIGS / "best_f1_confusion_matrix.png"
            if cm_fig_path.exists():
                with st.expander("🖼️ View Verified Matplotlib Confusion Matrix Heatmap"):
                    st.image(str(cm_fig_path), caption="Best F1 Confusion Matrix (Generated by src/explain_models.py)", use_container_width=True)

    with col_interp:
        st.markdown("""
<div class="result-card">
  <div class="result-card-title">Empirical Research Findings & Architecture Trade-offs</div>
  
  <div class="research-card">
    <div class="research-card-title">1. Precision-Recall Trade-off & Business Objective</div>
    <div class="research-card-body">
      A critical research takeaway is that no single model dominates across every metric. 
      <strong>Random Forest (Tree-Imp + SMOTE)</strong> achieves the optimal harmonic balance (F1 = 0.6641) with 64.6% Precision and 68.3% Recall. 
      Meanwhile, <strong>SVM (RFE + Class Weight)</strong> reaches the highest Recall (80.63%), catching 4 out of 5 purchasers, but sacrifices Precision down to 51.60% (generating 289 false alarms vs 143 for RF).
    </div>
  </div>

  <div class="research-card">
    <div class="research-card-title">2. Feature Selection Impact (Tree Importance vs All)</div>
    <div class="research-card-body">
      Filtering to tree-importance features improved Random Forest F1 from 0.5897 to 0.6641. 
      Behavioral features (<code>PageValues</code>, <code>ExitRates</code>, <code>ProductRelated_Duration</code>) carry dominant discriminative signal. Removing noisy categorical columns (Browser, OperatingSystems, Region) reduces variance without sacrificing predictive capacity.
    </div>
  </div>

  <div class="research-card">
    <div class="research-card-title">3. Class Imbalance Remedy (SMOTE vs Class Weight)</div>
    <div class="research-card-body">
      Due to the 84.5% non-purchase baseline, unweighted models heavily favor Class 0 (baseline LR recall was only 35.86%). 
      <strong>SMOTE</strong> synthesizes minority-class feature neighborhoods, improving Random Forest recall by +20.2% points. 
      <strong>Cost-sensitive weighting</strong> forces linear/margin boundaries away from the minority class, maximizing sensitivity for SVM.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 3: 10-Fold Stratified Cross-Validation Benchmarks
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">10-Fold Stratified Cross-Validation Benchmark Stability</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Validation across 10 stratified folds on the training partition (n = 9,864 sessions) with 95% confidence intervals</div>
    </div>
    <span class="viva-pill">Statistical Generalisation</span>
  </div>
""", unsafe_allow_html=True)

    # Prepare nicely formatted CV table
    cv_display = cv_df.copy()
    cv_display["F1 Score (CV Mean ± Std)"] = cv_display.apply(lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.3f}", axis=1)
    cv_display["F1 95% Conf. Interval"] = cv_display.apply(lambda r: f"[{r['f1_ci95_low']:.4f}, {r['f1_ci95_high']:.4f}]", axis=1)
    cv_display["Test F1 (Holdout)"] = cv_display["test_f1"].apply(lambda v: f"{v:.4f}")
    cv_display["Recall (CV Mean)"] = cv_display["recall_mean"].apply(lambda v: f"{v*100:.1f}%")
    cv_display["Test Recall"] = cv_display["test_recall"].apply(lambda v: f"{v*100:.1f}%")
    cv_display["ROC-AUC (CV Mean)"] = cv_display["roc_auc_mean"].apply(lambda v: f"{v:.4f}")
    cv_display["Accuracy (CV Mean)"] = cv_display["accuracy_mean"].apply(lambda v: f"{v*100:.1f}%")

    table_cols = [
        "display_name", "F1 Score (CV Mean ± Std)", "F1 95% Conf. Interval",
        "Test F1 (Holdout)", "Recall (CV Mean)", "Test Recall", "ROC-AUC (CV Mean)", "Accuracy (CV Mean)"
    ]
    renamed_df = cv_display[table_cols].rename(columns={"display_name": "Model Architecture & Strategy"})

    cv_ci_fig = FIGS / "repeated_cv_confidence_intervals.png"
    if cv_ci_fig.exists():
        st.image(str(cv_ci_fig), caption="10-Fold Repeated CV — 95% Confidence Intervals by Model Configuration", use_container_width=True)

    st.dataframe(renamed_df, use_container_width=True, hide_index=True)

    st.markdown("""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 14px;margin-top:10px;font-size:0.73rem;color:var(--on-surface-var);line-height:1.45;">
    ℹ️ <strong>Academic Methodology Note:</strong> 
    Cross-validation metrics evaluate internal training fold stability and variance, while test metrics measure performance on the fixed, held-out test split (n = 2,466). 
    Notice that the Random Forest Champion achieves a CV Mean F1 of <strong>0.6762</strong> (95% CI: [0.6661, 0.6863]) and a held-out test F1 of <strong>0.6641</strong>, confirming high consistency with no severe overfitting.
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 4: 48-Experiment Results Explorer
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">48-Experiment Grid Search Results Explorer</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Full empirical factorial grid: 4 Classifiers × 4 Feature Subsets × 3 Imbalance Techniques</div>
    </div>
    <span class="meta-pill">n = 48 Experiments</span>
  </div>
""", unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        model_opts = ["All Models", "Random Forest", "Support Vector Machine", "Logistic Regression", "Decision Tree"]
        sel_model = st.selectbox("Filter by Model Architecture", model_opts, key="mp_filter_model")
    with f_col2:
        feat_opts = ["All Configurations", "tree_importance", "rfe", "correlation", "all"]
        sel_feat = st.selectbox("Filter by Feature Subset", feat_opts, key="mp_filter_feat")
    with f_col3:
        imb_opts = ["All Strategies", "smote", "class_weight", "none"]
        sel_imb = st.selectbox("Filter by Imbalance Handling", imb_opts, key="mp_filter_imb")

    model_map = {
        "Random Forest": "random_forest",
        "Support Vector Machine": "svm",
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree"
    }

    filtered_df = df.copy()
    if sel_model != "All Models":
        filtered_df = filtered_df[filtered_df["model"] == model_map[sel_model]]
    if sel_feat != "All Configurations":
        filtered_df = filtered_df[filtered_df["feature_configuration"] == sel_feat]
    if sel_imb != "All Strategies":
        filtered_df = filtered_df[filtered_df["imbalance_strategy"] == sel_imb]

    filtered_df = filtered_df.sort_values("f1", ascending=False)

    exp_view = pd.DataFrame({
        "Model": filtered_df["model"].apply(lambda m: m.replace("_", " ").title()),
        "Features": filtered_df["feature_configuration"],
        "Imbalance": filtered_df["imbalance_strategy"],
        "Selected Features": filtered_df["selected_feature_count"],
        "F1 Score": filtered_df["f1"].apply(lambda v: f"{v:.4f}"),
        "Recall": filtered_df["recall"].apply(lambda v: f"{v*100:.1f}%"),
        "Precision": filtered_df["precision"].apply(lambda v: f"{v*100:.1f}%"),
        "ROC-AUC": filtered_df["roc_auc"].apply(lambda v: f"{v:.4f}"),
        "Accuracy": filtered_df["accuracy"].apply(lambda v: f"{v*100:.1f}%"),
    })

    st.dataframe(exp_view, use_container_width=True, hide_index=True)
    st.markdown(f"<div style='font-size:0.7rem;color:var(--secondary);margin-top:4px;'>Showing {len(exp_view)} of 48 experimental runs (sorted by F1-Score descending).</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 5: Collapsible Viva / Metric Definitions Guide
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 Viva Guide: Evaluation Metrics & Plain-Language Definitions (Examiner Deep-Dive)"):
        st.markdown("""
### Plain-Language Definitions & Contextual Relevance

* **Accuracy ($$\\frac{TP + TN}{Total}$$)**: 
  The proportion of all sessions classified correctly. 
  <br>⚠️ *Critical Examiner Warning:* In an imbalanced dataset where 84.5% of sessions do not purchase, a naive "dummy" classifier predicting *No Purchase* for 100% of sessions achieves **84.5% accuracy** while offering **0% utility** (detecting zero buyers). Accuracy must never be used in isolation.

* **Precision ($$\\frac{TP}{TP + FP}$$)**: 
  When the model sounds the alarm ("Purchase Intention"), how often is the visitor truly a buyer? 
  <br>🎯 *Business Cost of False Positive:* Wasting marketing spend, such as distributing expensive 20% discount checkout vouchers to users who would have purchased anyway or window shoppers who still won't buy.

* **Recall / Sensitivity ($$\\frac{TP}{TP + FN}$$)**: 
  Out of all shoppers who actually purchased, what fraction did the model successfully identify?
  <br>🎯 *Business Cost of False Negative:* Missed sales conversion. If a high-intent shopper leaves without an automated incentive, potential revenue is lost forever.

* **F1-Score ($$2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$)**: 
  The harmonic mean of Precision and Recall. F1 heavily penalizes extreme disparities (e.g. 99% precision with 2% recall). It is our primary optimization criterion because e-commerce systems require both reliable alerts and high buyer coverage.

* **ROC-AUC (Area Under ROC Curve)**: 
  The probability that the classifier ranks a randomly chosen positive session higher than a randomly chosen negative session across all decision thresholds $\\tau \\in [0, 1]$. An AUC $\\ge 0.90$ represents outstanding discrimination.
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Row 6: Best F1 vs Best Recall Trade-off Figure
    # ─────────────────────────────────────────────────────────────────────────
    tradeoff_fig = FIGS / "best_f1_vs_best_recall.png"
    if tradeoff_fig.exists():
        st.markdown("""
<div class="result-card" style="margin-top:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Champion Trade-Off Comparison: Best F1 vs Best Recall</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Side-by-side metric comparison between the two research champion configurations on the held-out test split</div>
    </div>
    <span class="viva-pill">Research Benchmark</span>
  </div>
""", unsafe_allow_html=True)
        st.image(str(tradeoff_fig), caption="Best F1 (RF + SMOTE) vs Best Recall (SVM + Class Weight) — Visualised metric comparison", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: FEATURE IMPORTANCE & SHAP INTERPRETABILITY
# ═════════════════════════════════════════════════════════════════════════════
def render_feature_importance() -> None:
    render_page_header(
        title="Feature Importance & SHAP Interpretability",
        badge="🧠 Explainable AI (XAI) · Tree MDI & Shapley Values",
        desc="Global feature relevance and local prediction attribution analysis evaluated on 12,330 UCI e-commerce sessions."
    )

    bundle = get_bundle()
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    importances = model.feature_importances_

    # Create MDI ranking
    df_mdi = pd.DataFrame({
        "raw_feature": selected_features,
        "clean_feature": [clean_feature_name(f) for f in selected_features],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_mdi = df_mdi.head(12)
    pv_share = df_mdi.loc[df_mdi["clean_feature"] == "PageValues", "importance"].iloc[0] * 100
    exit_share = df_mdi.loc[df_mdi["clean_feature"] == "ExitRates", "importance"].iloc[0] * 100
    ratio = pv_share / exit_share

    # ─────────────────────────────────────────────────────────────────────────
    # Top Telemetry & Dominance Highlight Banner
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="result-card" style="background:rgba(0,74,198,0.04);border:1px solid rgba(0,74,198,0.18);margin-bottom:1.25rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div style="display:flex;align-items:flex-start;gap:12px;max-width:850px;">
      <div style="width:40px;height:40px;border-radius:10px;background:var(--primary);color:#fff;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">
        ⚡
      </div>
      <div>
        <div style="font-family:var(--font-head);font-size:1.05rem;font-weight:700;color:var(--primary);margin-bottom:2px;">
          Empirical Asymmetry Detected: PageValues Dominates Model Capacity
        </div>
        <div style="font-size:0.8rem;color:var(--on-surface-var);line-height:1.5;">
          <strong>PageValues</strong> accounts for <strong>{pv_share:.1f}%</strong> of total Gini impurity reduction across 100 decision trees in the champion Random Forest model. 
          Its concentration ratio is <strong>{ratio:.2f}×</strong> higher than the second-ranked feature (<code>ExitRates</code> at {exit_share:.1f}%). 
          In Google Analytics, PageValues measures user interaction with high-converting URLs (such as active carts or promotions), creating an empirical proxy for commercial intent.
        </div>
      </div>
    </div>
    <div style="background:var(--surface-bright);padding:10px 16px;border-radius:var(--radius-md);box-shadow:0 1px 4px rgba(0,0,0,0.06);display:flex;flex-direction:column;align-items:center;">
      <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:var(--secondary);letter-spacing:0.05em;">Concentration Ratio</span>
      <span style="font-family:var(--font-head);font-size:1.5rem;font-weight:700;color:var(--primary);line-height:1.2;">{ratio:.2f}×</span>
      <span style="font-size:0.68rem;color:var(--outline);">vs #2 ExitRates</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section A & B: Global Feature Importance (MDI) & Global SHAP Summary
    # ─────────────────────────────────────────────────────────────────────────
    col_mdi, col_shap = st.columns([6, 6], gap="large")

    with col_mdi:
        st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Top 12 Global Features (Random Forest MDI Gini Importance)</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Mean Decrease in Impurity across 100 decision trees on the training split</div>
    </div>
    <span class="viva-pill" style="font-size:0.62rem;">Gini Impurity</span>
  </div>
""", unsafe_allow_html=True)

        for i, row in top_mdi.iterrows():
            pct = row["importance"] * 100
            is_pv = row["clean_feature"] == "PageValues"
            bar_color = "var(--primary)" if is_pv else "#2563eb" if i < 3 else "var(--secondary)"
            tag_badge = '<span class="viva-pill" style="padding:1px 6px;font-size:0.55rem;background:rgba(0,74,198,0.1);color:var(--primary) !important;">Primary Driver</span>' if is_pv else ""
            bold_style = "font-weight:700;color:var(--primary);" if is_pv else "font-weight:500;color:var(--on-surface);"

            st.markdown(f"""
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.78rem;margin-bottom:3px;">
    <div style="display:flex;align-items:center;gap:6px;">
      <span style="width:18px;font-family:monospace;font-size:0.7rem;color:var(--outline);">{i+1}.</span>
      <span style="{bold_style}">{row['clean_feature']}</span>
      {tag_badge}
    </div>
    <span style="font-family:var(--font-head);font-weight:700;{bold_style}">{pct:.1f}%</span>
  </div>
  <div class="bar-track" style="height:7px;">
    <div class="bar-fill" style="width:{min(pct, 100):.1f}%;background:{bar_color};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 12px;margin-top:14px;font-size:0.72rem;color:var(--on-surface-var);line-height:1.45;">
    ℹ️ <strong>MDI Metric Definition:</strong> Mean Decrease in Impurity calculates each feature's contribution to splitting tree nodes. While robust for overall model pruning, MDI is population-wide, does not indicate directionality, and can over-favor continuous variables with many cut points.
  </div>
</div>
""", unsafe_allow_html=True)

    with col_shap:
        shap_summary_data = load_shap_summary()
        top_shap_records = shap_summary_data.get("top_features", [])

        st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Global SHAP Summary (Test Set Impact Magnitude)</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Mean absolute Shapley displacement |SHAP| toward Purchase (Class 1)</div>
    </div>
    <span class="viva-pill" style="font-size:0.62rem;background:rgba(0,98,66,0.08);color:var(--tertiary) !important;">Game Theory</span>
  </div>
""", unsafe_allow_html=True)

        max_shap = top_shap_records[0]["mean_abs_shap"] if top_shap_records else 0.2
        for i, item in enumerate(top_shap_records):
            c_name = clean_feature_name(item["feature"])
            s_val = item["mean_abs_shap"]
            bar_w = (s_val / max_shap) * 100
            is_pv = c_name == "PageValues"
            bar_color = "var(--tertiary)" if is_pv else "#059669" if i < 3 else "#64748b"

            st.markdown(f"""
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.78rem;margin-bottom:3px;">
    <div style="display:flex;align-items:center;gap:6px;">
      <span style="width:18px;font-family:monospace;font-size:0.7rem;color:var(--outline);">{i+1}.</span>
      <span style="font-weight:{'700' if is_pv else '500'};color:{'var(--tertiary)' if is_pv else 'var(--on-surface)'};">{c_name}</span>
    </div>
    <span style="font-family:var(--font-head);font-weight:700;color:{'var(--tertiary)' if is_pv else 'var(--on-surface)'};">{s_val:.4f}</span>
  </div>
  <div class="bar-track" style="height:7px;">
    <div class="bar-fill" style="width:{min(bar_w, 100):.1f}%;background:{bar_color};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        global_fig_path = FIGS / "best_f1_shap_global_summary.png"
        if global_fig_path.exists():
            st.image(str(global_fig_path), caption="Global SHAP Importance — Best F1 Model (RF + Tree-Importance + SMOTE, n=2,466 test sessions)", use_container_width=True)

        # Local SHAP examples for both champions
        with st.expander("🖼️ Research Figures: Local SHAP Examples for Both Champions"):
            le_col1, le_col2 = st.columns(2)
            with le_col1:
                f1_local = FIGS / "best_f1_shap_local_example.png"
                if f1_local.exists():
                    st.image(str(f1_local), caption="Local SHAP Attribution — Best F1 Champion (Single Session)", use_container_width=True)
            with le_col2:
                rec_local = FIGS / "best_recall_shap_local_example.png"
                if rec_local.exists():
                    st.image(str(rec_local), caption="Local SHAP Attribution — Best Recall Champion (Single Session)", use_container_width=True)
            rec_global = FIGS / "best_recall_shap_global_summary.png"
            if rec_global.exists():
                st.image(str(rec_global), caption="Global SHAP Importance — Best Recall Model (SVM + RFE + Class Weight)", use_container_width=True)

        st.markdown("""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 12px;margin-top:14px;font-size:0.72rem;color:var(--on-surface-var);line-height:1.45;">
    💡 <strong>SHAP vs MDI Insight:</strong> Notice how <code>Administrative_Duration</code> ranks #2 under SHAP (0.0387) while ranking #6 under MDI (4.7%). This occurs because high administrative duration strongly shifts individual buyer probabilities, which SHAP captures locally even if fewer total tree splits were dedicated to it globally.
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section C: Conceptual Rigor — Global Relevance vs Local Explanation (Viva Anchor)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="result-card" style="background:#0b1c30;color:#e2e8f0;margin-bottom:1.5rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:34px;height:34px;border-radius:8px;background:rgba(111,251,190,0.15);color:var(--tertiary-fixed);display:flex;align-items:center;justify-content:center;font-size:18px;">
        🎓
      </div>
      <div>
        <div style="font-family:var(--font-head);font-size:1.05rem;font-weight:700;color:#ffffff;">
          Viva Defense Anchor: Conceptual Rigor in Explainable AI
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;">Distinguishing dataset-wide feature relevance from instance-level prediction attribution</div>
      </div>
    </div>
    <span class="status-pill" style="font-size:0.62rem;background:rgba(255,255,255,0.08);color:#e2e8f0 !important;border-color:rgba(255,255,255,0.15);">XAI Taxonomical Rigor</span>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:14px;">
    <!-- Box 1: Global Relevance -->
    <div style="background:rgba(255,255,255,0.04);border-left:4px solid var(--primary);border-radius:var(--radius-md);padding:14px 16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:#ffffff;">Global Feature Relevance</span>
        <span style="font-size:0.6rem;background:rgba(0,74,198,0.25);color:#b4c5ff;padding:2px 6px;border-radius:4px;font-weight:600;">Population (N=12,330)</span>
      </div>
      <p style="font-size:0.78rem;color:#cbd5e1;line-height:1.45;margin-bottom:8px;">
        Answers: <em>"Which behavioral signals does the model consistently prioritize across the entire population?"</em>
      </p>
      <ul style="font-size:0.73rem;color:#94a3b8;margin:0;padding-left:16px;line-height:1.5;">
        <li>Evaluated via <strong>Mean Decrease in Impurity (MDI)</strong> across all tree splits.</li>
        <li>Invariant static ranking: <code>PageValues</code> (37.0%) &gt; <code>ExitRates</code> (7.3%) &gt; <code>Administrative</code> (6.5%).</li>
        <li>Primary utility: Feature selection, ETL optimization, dimensionality reduction.</li>
      </ul>
    </div>

    <!-- Box 2: Local Attribution -->
    <div style="background:rgba(255,255,255,0.04);border-left:4px solid var(--tertiary-fixed);border-radius:var(--radius-md);padding:14px 16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:#ffffff;">Local Prediction Attribution</span>
        <span style="font-size:0.6rem;background:rgba(111,251,190,0.2);color:var(--tertiary-fixed);padding:2px 6px;border-radius:4px;font-weight:600;">Instance-Level (Session)</span>
      </div>
      <p style="font-size:0.78rem;color:#cbd5e1;line-height:1.45;margin-bottom:8px;">
        Answers: <em>"Why did this specific shopper receive a predicted probability right now?"</em>
      </p>
      <ul style="font-size:0.73rem;color:#94a3b8;margin:0;padding-left:16px;line-height:1.5;">
        <li>Evaluated via <strong>Tree SHAP</strong> cooperative game theory Shapley values.</li>
        <li>Additive guarantee: $P(\\text{{Purchase}}) = \\text{{Base Value}} + \\sum_{{i}} \\text{{SHAP}}_i$.</li>
        <li>Primary utility: Auditable regulatory compliance, explainable intervention triggers, customer edge-case analysis.</li>
      </ul>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section D: Interactive Individual Prediction Explanation (Local SHAP)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Interactive Session Explanation (Local Tree SHAP)</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Inspect how individual feature values push the probability toward or away from purchase for any session</div>
    </div>
    <span class="meta-pill">Real-Time Shapley Decomposition</span>
  </div>
""", unsafe_allow_html=True)

    col_pre, col_custom = st.columns([4, 8], gap="medium")
    with col_pre:
        st.markdown("<span style='font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--secondary);'>Select Session Profile</span>", unsafe_allow_html=True)
        profile_choice = st.selectbox(
            "Choose a session preset to explain:",
            ["High Intent Buyer (Cart Active)", "Window Shopper (Low Engagement)", "Returning Customer (Moderate Signal)", "Custom Parameters"],
            key="fi_preset_choice"
        )

    if profile_choice == "High Intent Buyer (Cart Active)":
        raw_sess = dict(
            Administrative=4, Administrative_Duration=95.0, Informational=2, Informational_Duration=52.0,
            ProductRelated=35, ProductRelated_Duration=1420.0, BounceRates=0.005, ExitRates=0.015,
            PageValues=28.5, SpecialDay=0.0, Month="Nov", OperatingSystems=2, Browser=2, Region=1,
            TrafficType=2, VisitorType="Returning_Visitor", Weekend=True
        )
    elif profile_choice == "Window Shopper (Low Engagement)":
        raw_sess = dict(
            Administrative=0, Administrative_Duration=0.0, Informational=0, Informational_Duration=0.0,
            ProductRelated=3, ProductRelated_Duration=45.0, BounceRates=0.08, ExitRates=0.12,
            PageValues=0.0, SpecialDay=0.0, Month="Mar", OperatingSystems=2, Browser=2, Region=1,
            TrafficType=2, VisitorType="New_Visitor", Weekend=False
        )
    elif profile_choice == "Returning Customer (Moderate Signal)":
        raw_sess = dict(
            Administrative=2, Administrative_Duration=40.0, Informational=1, Informational_Duration=12.0,
            ProductRelated=18, ProductRelated_Duration=620.0, BounceRates=0.010, ExitRates=0.022,
            PageValues=14.2, SpecialDay=0.0, Month="Dec", OperatingSystems=2, Browser=2, Region=1,
            TrafficType=2, VisitorType="Returning_Visitor", Weekend=True
        )
    else:
        with col_custom:
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                pv = st.number_input("PageValues", min_value=0.0, max_value=300.0, value=10.0, step=1.0, key="fi_cust_pv")
                er = st.slider("ExitRates", 0.0, 0.2, 0.03, step=0.005, key="fi_cust_er")
            with c_b:
                prd = st.number_input("ProductRelated_Duration (s)", 0.0, 10000.0, 500.0, step=50.0, key="fi_cust_prd")
                mo = st.selectbox("Month", ["Nov", "May", "Dec", "Mar", "Oct", "Sep", "Jul", "Aug", "June", "Feb"], index=0, key="fi_cust_mo")
            with c_c:
                vt = st.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor", "Other"], index=0, key="fi_cust_vt")
                wk = st.checkbox("Weekend", value=True, key="fi_cust_wk")
        raw_sess = dict(
            Administrative=2, Administrative_Duration=30.0, Informational=0, Informational_Duration=0.0,
            ProductRelated=15, ProductRelated_Duration=prd, BounceRates=er/2, ExitRates=er,
            PageValues=pv, SpecialDay=0.0, Month=mo, OperatingSystems=2, Browser=2, Region=1,
            TrafficType=2, VisitorType=vt, Weekend=wk
        )

    with st.spinner("Computing local SHAP attribution decomposition…"):
        explanation = explain_session(raw_sess)

    prob = explanation["purchase_probability"]
    pred = explanation["predicted_class"]
    base_val = explanation["base_value"]
    pos_drivers = explanation["pos_drivers"].head(5)
    neg_drivers = explanation["neg_drivers"].head(5)

    out_color = "var(--tertiary)" if pred else "var(--error)"
    out_bg = "rgba(0,98,66,0.08)" if pred else "rgba(186,26,26,0.07)"
    out_border = "rgba(0,98,66,0.22)" if pred else "rgba(186,26,26,0.2)"
    out_label = "Purchase Intention Detected" if pred else "No Purchase Predicted"
    out_icon = "🛍️" if pred else "🛒"

    st.markdown(f"""
<div style="background:{out_bg};border:1px solid {out_border};border-radius:var(--radius-md);padding:14px 18px;margin:12px 0 16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="font-size:24px;">{out_icon}</div>
    <div>
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:var(--secondary);letter-spacing:0.05em;">Session Inference Outcome</div>
      <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:{out_color};">{out_label}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    <div style="text-align:right;">
      <div style="font-size:0.62rem;font-weight:600;text-transform:uppercase;color:var(--secondary);">Model Prior (Base Value)</div>
      <div style="font-family:var(--font-head);font-size:1rem;font-weight:700;color:var(--on-surface);">{base_val*100:.1f}%</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:var(--secondary);">Final P(Purchase)</div>
      <div style="font-family:var(--font-head);font-size:1.6rem;font-weight:700;color:{out_color};line-height:1;">{prob*100:.1f}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_pos, col_neg = st.columns(2, gap="large")

    with col_pos:
        st.markdown("""
<div style="background:var(--surface-low);border-radius:var(--radius-md);padding:14px 16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-family:var(--font-head);font-size:0.88rem;font-weight:700;color:var(--tertiary);">
      ▲ Top Positive Drivers (Pushing toward Purchase)
    </span>
    <span style="font-size:0.62rem;background:rgba(0,98,66,0.1);color:var(--tertiary);padding:2px 6px;border-radius:4px;font-weight:600;">+Δ Probability</span>
  </div>
""", unsafe_allow_html=True)

        if len(pos_drivers) == 0:
            st.markdown("<div style='font-size:0.75rem;color:var(--secondary);'>No positive feature displacement for this session.</div>", unsafe_allow_html=True)
        else:
            for _, r in pos_drivers.iterrows():
                val_raw = raw_sess.get(r['raw_feature'].replace('numerical__', '').replace('categorical__', '').split('_')[0], 'Active')
                pct_shift = r['shap_value'] * 100
                st.markdown(f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:2px;">
    <span><strong>{r['clean_feature']}</strong> <span style="color:var(--outline);font-size:0.7rem;">(value: {val_raw})</span></span>
    <span style="font-family:var(--font-head);font-weight:700;color:var(--tertiary);">+{pct_shift:.1f}%</span>
  </div>
  <div class="bar-track" style="height:6px;">
    <div class="bar-fill" style="width:{min(pct_shift*2.5, 100):.1f}%;background:var(--tertiary);"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_neg:
        st.markdown("""
<div style="background:var(--surface-low);border-radius:var(--radius-md);padding:14px 16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span style="font-family:var(--font-head);font-size:0.88rem;font-weight:700;color:var(--error);">
      ▼ Top Negative Drivers (Suppressing Purchase Intent)
    </span>
    <span style="font-size:0.62rem;background:rgba(186,26,26,0.1);color:var(--error);padding:2px 6px;border-radius:4px;font-weight:600;">-Δ Probability</span>
  </div>
""", unsafe_allow_html=True)

        if len(neg_drivers) == 0:
            st.markdown("<div style='font-size:0.75rem;color:var(--secondary);'>No negative feature displacement for this session.</div>", unsafe_allow_html=True)
        else:
            for _, r in neg_drivers.iterrows():
                val_raw = raw_sess.get(r['raw_feature'].replace('numerical__', '').replace('categorical__', '').split('_')[0], 'Active')
                pct_shift = abs(r['shap_value']) * 100
                st.markdown(f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:2px;">
    <span><strong>{r['clean_feature']}</strong> <span style="color:var(--outline);font-size:0.7rem;">(value: {val_raw})</span></span>
    <span style="font-family:var(--font-head);font-weight:700;color:var(--error);">-{pct_shift:.1f}%</span>
  </div>
  <div class="bar-track" style="height:6px;">
    <div class="bar-fill" style="width:{min(pct_shift*2.5, 100):.1f}%;background:var(--error);"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:10px 14px;margin-top:12px;font-size:0.72rem;color:var(--secondary);line-height:1.45;">
    ⚠️ <strong>Examiner Integrity Notice:</strong> 
    SHAP contributions reflect local linear statistical decomposition of the trained tree ensemble's internal decision logic. 
    They demonstrate <em>associational attribution</em> within the model, not <em>causal real-world intervention effects</em> (e.g. arbitrarily forcing PageValues higher does not guarantee a customer will buy).
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section E: Feature Selection Strategy Comparison & PageValues Ablation
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
    <div>
      <div class="result-card-title" style="margin-bottom:2px;">Comparison of 4 Feature Selection Strategies</div>
      <div style="font-size:0.75rem;color:var(--secondary);">Experimental results testing information retention vs noise reduction across 48 runs</div>
    </div>
    <span class="viva-pill">Ablation Study</span>
  </div>

  <div style="grid-template-columns:repeat(auto-fit, minmax(230px, 1fr));gap:12px;margin-bottom:16px;display:grid;">
    <!-- Strat 1 -->
    <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:var(--secondary);font-weight:700;text-transform:uppercase;margin-bottom:4px;">
          <span>Baseline</span><span>17 Raw / 75 Encoded</span>
        </div>
        <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">All Features</div>
        <div style="font-size:0.72rem;color:var(--on-surface-var);line-height:1.4;margin:6px 0;">Preserves all visitor attributes without pruning low-variance metadata.</div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;margin-bottom:2px;"><span>F1 Score</span><span>0.5897</span></div>
        <div class="bar-track" style="height:5px;"><div class="bar-fill" style="width:59.0%;background:var(--secondary);"></div></div>
      </div>
    </div>

    <!-- Strat 2 -->
    <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:var(--secondary);font-weight:700;text-transform:uppercase;margin-bottom:4px;">
          <span>Filter</span><span>Correlation Cutoff</span>
        </div>
        <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">Correlation Filter</div>
        <div style="font-size:0.72rem;color:var(--on-surface-var);line-height:1.4;margin:6px 0;">Prunes collinear web telemetry (r &gt; 0.15) but discards non-linear interactions.</div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;margin-bottom:2px;"><span>F1 Score</span><span>0.6563</span></div>
        <div class="bar-track" style="height:5px;"><div class="bar-fill" style="width:65.6%;background:#2563eb;"></div></div>
      </div>
    </div>

    <!-- Strat 3 -->
    <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:var(--secondary);font-weight:700;text-transform:uppercase;margin-bottom:4px;">
          <span>Iterative</span><span>Backward Elimination</span>
        </div>
        <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">Recursive Elimination (RFE)</div>
        <div style="font-size:0.72rem;color:var(--on-surface-var);line-height:1.4;margin:6px 0;">Iteratively prunes least impactful features; best for linear margin models (SVM).</div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;margin-bottom:2px;"><span>F1 Score</span><span>0.6292</span></div>
        <div class="bar-track" style="height:5px;"><div class="bar-fill" style="width:62.9%;background:#64748b;"></div></div>
      </div>
    </div>

    <!-- Strat 4 (BEST) -->
    <div style="background:rgba(0,74,198,0.06);border:1px solid rgba(0,74,198,0.25);border-radius:var(--radius-md);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:var(--primary);font-weight:700;text-transform:uppercase;margin-bottom:4px;">
          <span>Top Strategy</span><span>MDI Ranked (40 Feats)</span>
        </div>
        <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--primary);">Tree-Based Importance</div>
        <div style="font-size:0.72rem;color:var(--on-surface-var);line-height:1.4;margin:6px 0;">Focuses capacity on non-linear behavioral signals while shedding categorical noise.</div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;color:var(--primary);margin-bottom:2px;"><span>Champion F1</span><span>0.6641</span></div>
        <div class="bar-track" style="height:5px;"><div class="bar-fill" style="width:66.4%;background:var(--primary);"></div></div>
      </div>
    </div>
  </div>

  <!-- Deep Dive: Real-world limitations & PageValues Ablation Findings -->
  <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:14px 16px;">
    <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      🔬 Target Leakage & Pure Browsing Behavior Ablation Study
    </div>
    <div style="font-size:0.78rem;color:var(--on-surface-var);line-height:1.5;">
      In real-world e-commerce deployments, <code>PageValues</code> may not be available early in a browsing session (cold-start visitors who have not yet navigated to checkout or high-intent promotional pages). 
      To test model robustness against target leakage, our project conducted a systematic ablation study:
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:10px;margin-top:10px;">
      <div style="background:var(--surface-bright);padding:10px 12px;border-radius:var(--radius-sm);border:1px solid var(--outline-var);">
        <div style="font-size:0.7rem;font-weight:700;color:var(--error);text-transform:uppercase;">Unweighted Baselines Collapse</div>
        <div style="font-size:0.75rem;color:var(--on-surface);margin-top:2px;">
          Removing PageValues causes baseline SVM recall to collapse from <strong>46.9% to 0.26%</strong>, and Logistic Regression from <strong>35.9% to 3.4%</strong>. Unweighted models relied almost entirely on PageValues.
        </div>
      </div>
      <div style="background:var(--surface-bright);padding:10px 12px;border-radius:var(--radius-sm);border:1px solid var(--outline-var);">
        <div style="font-size:0.7rem;font-weight:700;color:var(--tertiary);text-transform:uppercase;">Imbalance Mitigation Recovers Signal</div>
        <div style="font-size:0.75rem;color:var(--on-surface);margin-top:2px;">
          With class weighting and SMOTE, pure clickstream features (<code>ExitRates</code>, <code>ProductRelated_Duration</code>, <code>Administrative_Duration</code>) achieve <strong>ROC-AUC = 0.75</strong> and recall of <strong>68.9%–74.9%</strong>.
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section F: Viva Defense Guide (Collapsible Expander)
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 Viva Guide: Feature Attribution & Interpretability FAQs (Examiner Deep-Dive)"):
        st.markdown("""
### Academic & Industrial Defense Questions

* **Q: Why is `PageValues` so much more important than raw duration or page count?**
  * *Answer:* `ProductRelated_Duration` reflects passive browsing time, which window shoppers and buyers alike can accumulate. In contrast, `PageValues` requires navigating through URLs that have historically yielded completed transactions. It is a direct signal of commercial commitment rather than casual browsing.

* **Q: Why did Tree-Based Importance outperform Recursive Feature Elimination (RFE)?**
  * *Answer:* RFE iteratively eliminates features based on linear coefficients or single-model weights. Random Forest's MDI feature importance measures interactions across 100 randomly subsampled feature spaces. It retained non-linear threshold combinations (such as high `PageValues` in `Month=Nov` during Black Friday sales) that linear filtering discarded.

* **Q: Why do SHAP values sometimes rank features differently from Gini Importance (MDI)?**
  * *Answer:* Gini importance (MDI) measures node purity gain across all splits in training, inherently favoring continuous variables with many unique values. SHAP calculates exact marginal displacements on the final probability output via Shapley game theory. A feature like `Administrative_Duration` may be split on less frequently overall, but when an individual spends 95s on administrative pages, its local displacement on purchase likelihood is massive.

* **Q: Can e-commerce businesses use these SHAP scores to make causal interventions?**
  * *Answer:* No. Machine learning models identify associational correlations, not causal relationships. Artificially inflating PageValues (e.g. by forcing visitors to visit cart pages prematurely) will not increase conversions and could irritate users. SHAP explains *what the model learned*, not the underlying customer psychology.
""")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: CLASS IMBALANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def render_class_imbalance() -> None:
    render_page_header(
        title="Class Imbalance Problem & Resampling Solutions",
        badge="⚖️ Empirical Resampling & Asymmetric Cost Optimization",
        desc="Diagnostic Evaluation of 84.5% vs 15.5% Class Skew: Comparing SMOTE, Class Weighting, and Baseline Unweighted Learning"
    )

    stats = load_dataset_stats()
    df = load_experiment_results()
    cv_df = load_cv_metrics()

    total_sessions = stats["total"]
    no_purchase = stats["no_purchase"]
    purchases = stats["purchases"]
    no_purchase_pct = (no_purchase / total_sessions) * 100
    purchase_pct = (purchases / total_sessions) * 100
    imbalance_ratio = no_purchase / purchases

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A: Ground Truth Distribution & The Accuracy Paradox
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span>📊 Section A: Ground Truth Distribution & The Accuracy Paradox</span></div>', unsafe_allow_html=True)

    sec_a_col1, sec_a_col2 = st.columns([1.1, 1.1], gap="medium")

    with sec_a_col1:
        st.markdown(f"""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <span class="subtitle-badge" style="background:rgba(0,74,198,0.08);color:var(--primary);border-color:rgba(0,74,198,0.2);">GROUND TRUTH SKEW</span>
      <span style="font-size:0.75rem;color:var(--secondary);font-weight:600;">Target: Revenue (Boolean)</span>
    </div>
    <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      The 84.5% vs 15.5% Imbalance Dilemma
    </div>
    <p style="font-size:0.82rem;color:var(--secondary);line-height:1.5;margin-bottom:16px;">
      In online retail, purchase conversions are rare positive events. Out of <strong>{total_sessions:,}</strong> total recorded sessions, <strong>{no_purchase:,}</strong> visitors browsed without purchasing, while only <strong>{purchases:,}</strong> completed a purchase.
    </p>

    <!-- Visual Donut & Breakdown Row -->
    <div style="display:flex;align-items:center;gap:20px;background:var(--surface-low);padding:16px;border-radius:var(--radius-md);margin-bottom:14px;">
      <!-- SVG Donut Chart -->
      <div style="position:relative;width:125px;height:125px;flex-shrink:0;">
        <svg viewBox="0 0 100 100" style="width:100%;height:100%;transform:rotate(-90deg);">
          <!-- Negative Class (84.53%): 251.32 * 0.845255 = 212.4 -->
          <circle cx="50" cy="50" r="40" fill="transparent" stroke="var(--secondary-container)" stroke-width="12" stroke-dasharray="212.4 251.3" stroke-linecap="round"/>
          <!-- Positive Class (15.47%): 251.32 * 0.154745 = 38.9 -->
          <circle cx="50" cy="50" r="40" fill="transparent" stroke="var(--primary-container)" stroke-width="12" stroke-dasharray="38.9 251.3" stroke-dashoffset="-212.4" stroke-linecap="round"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
          <span style="font-family:var(--font-head);font-size:1.2rem;font-weight:700;color:var(--on-surface);line-height:1;">{imbalance_ratio:.1f}:1</span>
          <span style="font-size:0.6rem;text-transform:uppercase;color:var(--secondary);font-weight:700;letter-spacing:0.05em;">Ratio</span>
        </div>
      </div>

      <!-- Class Breakdown Stats -->
      <div style="flex:1;display:flex;flex-direction:column;gap:8px;">
        <div style="background:var(--surface-bright);padding:9px 12px;border-radius:var(--radius-sm);box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.75rem;color:var(--on-surface);font-weight:600;display:flex;align-items:center;gap:6px;">
              <span style="width:8px;height:8px;border-radius:50%;background:var(--secondary-container);display:inline-block;"></span>
              Negative (No Revenue)
            </span>
            <span style="font-family:var(--font-head);font-size:0.88rem;font-weight:700;color:var(--secondary);">{no_purchase_pct:.2f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--secondary);margin-top:2px;">
            <span>{no_purchase:,} sessions</span>
            <span style="background:var(--surface-container);padding:1px 6px;border-radius:4px;font-size:0.62rem;font-weight:600;">Majority Class</span>
          </div>
        </div>

        <div style="background:var(--surface-bright);padding:9px 12px;border-radius:var(--radius-sm);box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.75rem;color:var(--primary);font-weight:700;display:flex;align-items:center;gap:6px;">
              <span style="width:8px;height:8px;border-radius:50%;background:var(--primary-container);display:inline-block;"></span>
              Positive (Revenue True)
            </span>
            <span style="font-family:var(--font-head);font-size:0.88rem;font-weight:700;color:var(--primary);">{purchase_pct:.2f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--secondary);margin-top:2px;">
            <span>{purchases:,} sessions</span>
            <span style="background:rgba(37,99,235,0.1);color:var(--primary);padding:1px 6px;border-radius:4px;font-size:0.62rem;font-weight:700;">Minority Class</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div style="font-size:0.75rem;color:var(--secondary);background:var(--surface-container-low);padding:8px 12px;border-radius:var(--radius-sm);">
    📌 <strong>Imbalance Context:</strong> For every single paying customer, ~5.5 non-buying visitors visit the platform. Models trained on raw data suffer severe majority-class bias.
  </div>
</div>
""", unsafe_allow_html=True)

    with sec_a_col2:
        st.markdown(f"""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <span class="subtitle-badge" style="background:rgba(186,26,26,0.08);color:var(--error);border-color:rgba(186,26,26,0.2);">CRITICAL FAILURE MODE</span>
      <span style="font-size:0.75rem;color:var(--error);font-weight:700;">⚠️ The Accuracy Paradox</span>
    </div>
    <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      Deceptive Telemetry on Imbalanced Data
    </div>
    <p style="font-size:0.82rem;color:var(--secondary);line-height:1.5;margin-bottom:14px;">
      Standard loss functions minimize aggregate misclassification count. When 84.5% of samples belong to the negative class, a trivial model predicting <em>No Purchase</em> unconditionally achieves deceptively high accuracy.
    </p>

    <!-- Naive Simulation Grid -->
    <div style="background:var(--surface-low);border-radius:var(--radius-md);padding:14px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="font-size:0.75rem;font-weight:700;color:var(--on-surface);">Naive Zero-Rule Classifier (Predicts "No" 100%)</span>
        <span style="font-size:0.65rem;font-weight:700;color:var(--error);background:rgba(186,26,26,0.1);padding:2px 6px;border-radius:4px;">ZERO COMMERCIAL VALUE</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;text-align:center;">
        <div style="background:var(--surface-bright);padding:10px 8px;border-radius:var(--radius-sm);">
          <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:var(--secondary);display:block;margin-bottom:2px;">Apparent Accuracy</span>
          <span style="font-family:var(--font-head);font-size:1.25rem;font-weight:700;color:var(--primary);">{no_purchase_pct:.1f}%</span>
          <span style="font-size:0.65rem;color:var(--secondary);display:block;margin-top:2px;">Superficially High</span>
        </div>
        <div style="background:var(--surface-bright);padding:10px 8px;border-radius:var(--radius-sm);">
          <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:var(--secondary);display:block;margin-bottom:2px;">Actual Recall</span>
          <span style="font-family:var(--font-head);font-size:1.25rem;font-weight:700;color:var(--error);">0.0%</span>
          <span style="font-size:0.65rem;color:var(--error);font-weight:600;display:block;margin-top:2px;">Misses 100% Buyers</span>
        </div>
        <div style="background:var(--surface-bright);padding:10px 8px;border-radius:var(--radius-sm);">
          <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:var(--secondary);display:block;margin-bottom:2px;">Revenue Captured</span>
          <span style="font-family:var(--font-head);font-size:1.25rem;font-weight:700;color:var(--error);">$0.00</span>
          <span style="font-size:0.65rem;color:var(--error);font-weight:600;display:block;margin-top:2px;">No Business Utility</span>
        </div>
      </div>
    </div>
  </div>

  <div style="font-size:0.75rem;color:var(--on-surface-var);background:rgba(0,74,198,0.06);border:1px solid rgba(0,74,198,0.15);padding:10px 12px;border-radius:var(--radius-sm);">
    💡 <strong>Methodological Takeaway:</strong> Accuracy must never be used alone to evaluate purchase prediction. Evaluation must rely on <strong>Recall</strong>, <strong>Precision</strong>, <strong>F1-Score</strong>, and <strong>ROC-AUC</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B: Imbalance Strategy Comparison (Ablation Overview)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span>🔬 Section B: Imbalance Handling Strategies in the 48-Experiment Grid</span></div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.83rem;color:var(--secondary);margin-bottom:14px;line-height:1.5;">
  The project systematically tested <strong>3 imbalance handling treatments</strong> across 4 supervised algorithms (<em>Random Forest, SVM, Logistic Regression, Decision Tree</em>) and 4 feature selection subsets (<em>All 75, Correlation 75, RFE 40, Tree-Based 25</em>), totaling 48 controlled trials:
</div>
""", unsafe_allow_html=True)

    strat_col1, strat_col2, strat_col3 = st.columns(3, gap="medium")

    with strat_col1:
        st.markdown("""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;border-top:3px solid var(--secondary);">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span class="viva-pill" style="background:var(--surface-container);color:var(--secondary) !important;">STRATEGY 01</span>
      <span style="font-size:0.7rem;color:var(--secondary);font-weight:600;">Data: <code>none</code></span>
    </div>
    <div style="font-family:var(--font-head);font-size:1.05rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      No Resampling (Baseline)
    </div>
    <p style="font-size:0.78rem;color:var(--secondary);line-height:1.45;margin-bottom:12px;">
      Models learn directly from the raw 84.5% / 15.5% training distribution with unweighted cross-entropy / Gini loss.
    </p>
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);margin-bottom:12px;font-size:0.75rem;">
      <div style="color:var(--on-surface);font-weight:600;margin-bottom:4px;">Characteristics:</div>
      <div style="color:var(--secondary);">• High Precision (74% - 76%)</div>
      <div style="color:var(--secondary);">• Severely depressed Recall (35% - 51%)</div>
      <div style="color:var(--secondary);">• Conservative decision boundaries</div>
    </div>
  </div>
  <div style="background:rgba(186,26,26,0.06);border:1px solid rgba(186,26,26,0.15);padding:8px 10px;border-radius:var(--radius-sm);font-size:0.72rem;color:var(--error);">
    <strong>Major Pathology:</strong> Misses over half of all purchasing visitors (48% - 64% false negative rate).
  </div>
</div>
""", unsafe_allow_html=True)

    with strat_col2:
        st.markdown("""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;border-top:3px solid var(--primary);">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span class="viva-pill" style="background:rgba(0,74,198,0.1);color:var(--primary) !important;">STRATEGY 02</span>
      <span style="font-size:0.7rem;color:var(--primary);font-weight:600;">Algorithmic Penalty</span>
    </div>
    <div style="font-family:var(--font-head);font-size:1.05rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      Cost-Sensitive Class Weighting
    </div>
    <p style="font-size:0.78rem;color:var(--secondary);line-height:1.45;margin-bottom:12px;">
      Configured via <code>class_weight='balanced'</code>. Scales minority loss gradients inversely with class frequencies (~3.23× higher penalty for false negatives).
    </p>
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);margin-bottom:12px;font-size:0.75rem;">
      <div style="color:var(--on-surface);font-weight:600;margin-bottom:4px;">Characteristics:</div>
      <div style="color:var(--secondary);">• Zero synthetic data injection</div>
      <div style="color:var(--secondary);">• Strongest Recall boost on SVM & LR (up to 80.6%)</div>
      <div style="color:var(--secondary);">• No change to training geometry</div>
    </div>
  </div>
  <div style="background:rgba(0,74,198,0.06);border:1px solid rgba(0,74,198,0.15);padding:8px 10px;border-radius:var(--radius-sm);font-size:0.72rem;color:var(--primary);">
    <strong>Key Strength:</strong> Shifts the decision boundary without generating artificial samples, yielding the highest buyer sensitivity.
  </div>
</div>
""", unsafe_allow_html=True)

    with strat_col3:
        st.markdown("""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;border-top:3px solid var(--tertiary);">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span class="viva-pill" style="background:rgba(0,98,66,0.1);color:var(--tertiary) !important;">STRATEGY 03 · CHAMPION</span>
      <span style="font-size:0.7rem;color:var(--tertiary);font-weight:700;">Generative Over-sampling</span>
    </div>
    <div style="font-family:var(--font-head);font-size:1.05rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
      SMOTE Resampling
    </div>
    <p style="font-size:0.78rem;color:var(--secondary);line-height:1.45;margin-bottom:12px;">
      Synthetic Minority Over-sampling Technique. Synthesizes artificial minority vectors in continuous feature space via k-NN interpolation inside training folds.
    </p>
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);margin-bottom:12px;font-size:0.75rem;">
      <div style="color:var(--on-surface);font-weight:600;margin-bottom:4px;">Characteristics:</div>
      <div style="color:var(--secondary);">• Applied strictly to training folds</div>
      <div style="color:var(--secondary);">• Highest balanced F1-Score (0.6641)</div>
      <div style="color:var(--secondary);">• Balances Recall (68.3%) & Precision (64.6%)</div>
    </div>
  </div>
  <div style="background:rgba(0,98,66,0.06);border:1px solid rgba(0,98,66,0.18);padding:8px 10px;border-radius:var(--radius-sm);font-size:0.72rem;color:var(--tertiary);">
    <strong>Champion Balance:</strong> Achieves optimal harmonic trade-off, minimizing costly false alarms while capturing 68.3% of buyers.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C: Interactive Performance Comparison Across Strategies
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span>⚙️ Section C: Interactive Model & Feature Configuration Explorer</span></div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.82rem;color:var(--secondary);margin-bottom:12px;">
  Select any model architecture and feature selection method to inspect how the 3 imbalance strategies perform on the held-out test set (2,466 sessions).
</div>
""", unsafe_allow_html=True)

    ctrl_c1, ctrl_c2 = st.columns(2)
    with ctrl_c1:
        model_name_map = {
            "Random Forest": "random_forest",
            "Support Vector Machine (SVM)": "svm",
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
        }
        sel_model_label = st.selectbox(
            "Select Classifier Architecture:",
            list(model_name_map.keys()),
            index=0,
            key="imbalance_model_select"
        )
        sel_model_code = model_name_map[sel_model_label]

    with ctrl_c2:
        feat_name_map = {
            "Tree-Based Importance (Top 25 Features)": "tree_importance",
            "Recursive Feature Elimination (RFE - Top 40 Features)": "rfe",
            "Correlation Filtering (75 Features)": "correlation",
            "All Engineered Features (75 Features)": "all",
        }
        sel_feat_label = st.selectbox(
            "Select Feature Configuration:",
            list(feat_name_map.keys()),
            index=0,
            key="imbalance_feat_select"
        )
        sel_feat_code = feat_name_map[sel_feat_label]

    # Filter experiment results for selected model and feature configuration
    sub = df[(df["model"] == sel_model_code) & (df["feature_configuration"] == sel_feat_code)].copy()
    strat_order = ["none", "class_weight", "smote"]
    sub["strat_order"] = sub["imbalance_strategy"].map(lambda s: strat_order.index(s) if s in strat_order else 99)
    sub = sub.sort_values("strat_order").reset_index(drop=True)

    strat_display_map = {
        "none": ("None (Baseline)", "Unweighted standard loss", "var(--secondary)", "#64748b"),
        "class_weight": ("Class Weighting", "Balanced cost penalty", "var(--primary)", "#004ac6"),
        "smote": ("SMOTE Resampling", "Synthetic oversampling", "var(--tertiary)", "#006242"),
    }

    # Render dynamic strategy cards for selected configuration
    card_cols = st.columns(len(sub), gap="medium")
    for i, (_, row) in enumerate(sub.iterrows()):
        st_code = row["imbalance_strategy"]
        disp_title, disp_sub, disp_color, hex_color = strat_display_map.get(st_code, (st_code.title(), "", "var(--primary)", "#004ac6"))
        with card_cols[i]:
            st.markdown(f"""
<div class="result-card" style="height:100%;padding:16px;border-top:3px solid {disp_color};">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
    <div>
      <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">{disp_title}</div>
      <div style="font-size:0.68rem;color:var(--secondary);">{disp_sub}</div>
    </div>
    <span style="font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;background:var(--surface-container);color:var(--secondary);">
      {row['test_rows']} Test Rows
    </span>
  </div>

  <div style="display:flex;flex-direction:column;gap:8px;margin:12px 0;">
    <div>
      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px;">
        <span style="color:var(--secondary);font-weight:600;">Recall (Sensitivity)</span>
        <span style="font-family:var(--font-head);font-weight:700;color:{disp_color};">{row['recall']*100:.1f}%</span>
      </div>
      <div style="width:100%;height:6px;background:var(--surface-container);border-radius:999px;overflow:hidden;">
        <div style="width:{min(100, row['recall']*100):.1f}%;height:100%;background:{disp_color};border-radius:999px;"></div>
      </div>
    </div>

    <div>
      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px;">
        <span style="color:var(--secondary);font-weight:600;">Precision</span>
        <span style="font-family:var(--font-head);font-weight:700;color:var(--on-surface);">{row['precision']*100:.1f}%</span>
      </div>
      <div style="width:100%;height:6px;background:var(--surface-container);border-radius:999px;overflow:hidden;">
        <div style="width:{min(100, row['precision']*100):.1f}%;height:100%;background:var(--secondary);border-radius:999px;"></div>
      </div>
    </div>

    <div>
      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px;">
        <span style="color:var(--secondary);font-weight:600;">F1-Score</span>
        <span style="font-family:var(--font-head);font-weight:700;color:{disp_color};">{row['f1']:.4f}</span>
      </div>
      <div style="width:100%;height:6px;background:var(--surface-container);border-radius:999px;overflow:hidden;">
        <div style="width:{min(100, row['f1']*100):.1f}%;height:100%;background:{disp_color};border-radius:999px;"></div>
      </div>
    </div>

    <div>
      <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:2px;">
        <span style="color:var(--secondary);font-weight:600;">ROC-AUC</span>
        <span style="font-family:var(--font-head);font-weight:700;color:var(--on-surface);">{row['roc_auc']:.4f}</span>
      </div>
      <div style="width:100%;height:6px;background:var(--surface-container);border-radius:999px;overflow:hidden;">
        <div style="width:{min(100, row['roc_auc']*100):.1f}%;height:100%;background:var(--outline);border-radius:999px;"></div>
      </div>
    </div>
  </div>

  <div style="background:var(--surface-low);padding:8px 10px;border-radius:var(--radius-sm);font-size:0.68rem;color:var(--secondary);">
    <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
      <span>True Positives (Buyers):</span>
      <strong style="color:var(--on-surface);">{int(row['tp'])} / 382</strong>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
      <span>False Alarms (FP):</span>
      <strong style="color:var(--error);">{int(row['fp'])}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span>Missed Buyers (FN):</span>
      <strong style="color:var(--error);">{int(row['fn'])}</strong>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Summary empirical table for this slice
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    table_rows_html = ""
    for _, r in sub.iterrows():
        st_name = strat_display_map.get(r['imbalance_strategy'], (r['imbalance_strategy'],))[0]
        table_rows_html += f"""
<tr style="border-bottom:1px solid var(--surface-container);font-size:0.8rem;">
  <td style="padding:10px 12px;font-weight:600;color:var(--on-surface);">{st_name}</td>
  <td style="padding:10px 12px;font-family:var(--font-head);font-weight:700;color:var(--primary);">{r['f1']:.4f}</td>
  <td style="padding:10px 12px;font-family:var(--font-head);font-weight:700;color:var(--tertiary);">{r['recall']*100:.2f}%</td>
  <td style="padding:10px 12px;font-family:var(--font-head);font-weight:600;">{r['precision']*100:.2f}%</td>
  <td style="padding:10px 12px;font-family:var(--font-head);font-weight:600;">{r['roc_auc']:.4f}</td>
  <td style="padding:10px 12px;color:var(--secondary);">{r['accuracy']*100:.2f}%</td>
  <td style="padding:10px 12px;font-weight:600;color:var(--on-surface);">{int(r['tp'])}</td>
  <td style="padding:10px 12px;color:var(--error);font-weight:600;">{int(r['fp'])}</td>
  <td style="padding:10px 12px;color:var(--error);">{int(r['fn'])}</td>
</tr>
"""

    st.markdown(f"""
<div style="background:var(--surface-bright);border-radius:var(--radius-lg);padding:14px 18px;box-shadow:0 1px 6px rgba(0,0,0,0.05);overflow-x:auto;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:0.75rem;font-weight:700;color:var(--secondary);text-transform:uppercase;letter-spacing:0.05em;">
      Ablation Comparison Table · {sel_model_label} ({sel_feat_label})
    </span>
    <span style="font-size:0.68rem;color:var(--outline);font-weight:600;">Protocol: 80/20 Stratified Test Split (N=2,466)</span>
  </div>
  <table style="width:100%;border-collapse:collapse;text-align:left;">
    <thead>
      <tr style="border-bottom:2px solid var(--surface-container);font-size:0.7rem;color:var(--secondary);text-transform:uppercase;letter-spacing:0.05em;">
        <th style="padding:8px 12px;">Imbalance Strategy</th>
        <th style="padding:8px 12px;">F1-Score</th>
        <th style="padding:8px 12px;">Recall</th>
        <th style="padding:8px 12px;">Precision</th>
        <th style="padding:8px 12px;">ROC-AUC</th>
        <th style="padding:8px 12px;">Accuracy</th>
        <th style="padding:8px 12px;">TP</th>
        <th style="padding:8px 12px;">FP</th>
        <th style="padding:8px 12px;">FN</th>
      </tr>
    </thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>
  <div style="margin-top:10px;font-size:0.72rem;color:var(--secondary);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <span>* Positive Class = <code>Revenue=True</code> (382 actual test buyers). Negative Class = <code>Revenue=False</code> (2,084 test non-buyers).</span>
    <span>Source: <code>results/nayantara/full_experiment_results.csv</code></span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D: F1 vs Recall Trade-Off (Verified Champions)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span>⚖️ Section D: F1 vs Recall Trade-Off Analysis</span></div>', unsafe_allow_html=True)

    # Verified Champion identification
    best_f1_idx = df["f1"].idxmax()
    best_f1_row = df.loc[best_f1_idx]

    best_rec_idx = df["recall"].idxmax()
    best_rec_row = df.loc[best_rec_idx]

    top_d1, top_d2 = st.columns(2, gap="medium")

    with top_d1:
        st.markdown(f"""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;border-left:4px solid var(--tertiary);">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
      <div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <span class="viva-pill" style="background:var(--tertiary);color:#ffffff !important;font-size:0.6rem;">SELECTED OVERALL CHAMPION</span>
          <span style="font-size:0.72rem;font-weight:700;color:var(--tertiary);">★ Best F1-Score</span>
        </div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);">
          Random Forest + Tree Importance + SMOTE
        </div>
        <div style="font-size:0.75rem;color:var(--secondary);margin-top:2px;">
          Optimized for minimal false alarms while recovering 68.3% of buying visitors.
        </div>
      </div>
      <div style="width:40px;height:40px;border-radius:10px;background:rgba(0,98,66,0.1);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">🏆</div>
    </div>

    <div style="background:var(--surface-low);padding:12px;border-radius:var(--radius-md);margin:12px 0;display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;text-align:center;">
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">F1-Score</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--primary);">{best_f1_row['f1']:.4f}</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">Recall</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--tertiary);">{best_f1_row['recall']*100:.1f}%</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">Precision</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);">{best_f1_row['precision']*100:.1f}%</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">ROC-AUC</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--outline);">{best_f1_row['roc_auc']:.4f}</div>
      </div>
    </div>

    <div style="font-size:0.75rem;color:var(--secondary);line-height:1.45;margin-bottom:12px;">
      • <strong>Confusion Matrix:</strong> TP = {int(best_f1_row['tp'])}, FP = {int(best_f1_row['fp'])}, FN = {int(best_f1_row['fn'])}, TN = {int(best_f1_row['tn'])}.<br>
      • <strong>Commercial Strength:</strong> With only 143 false alarms, promotional discounts are not wasted on non-buyers.
    </div>
  </div>

  <div style="font-size:0.72rem;background:rgba(0,98,66,0.06);border:1px solid rgba(0,98,66,0.18);padding:8px 10px;border-radius:var(--radius-sm);color:var(--tertiary);">
    <strong>Decision Objective:</strong> Ideal when customer retention offers (e.g. 20% discount vouchers) have non-trivial marginal costs.
  </div>
</div>
""", unsafe_allow_html=True)

    with top_d2:
        st.markdown(f"""
<div class="result-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;border-left:4px solid #b45309;">
  <div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
      <div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <span class="viva-pill" style="background:#b45309;color:#ffffff !important;font-size:0.6rem;">MAXIMUM INTENT DETECTION</span>
          <span style="font-size:0.72rem;font-weight:700;color:#b45309;">★ Best Recall</span>
        </div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);">
          SVM + RFE (40 Features) + Class Weight
        </div>
        <div style="font-size:0.75rem;color:var(--secondary);margin-top:2px;">
          Optimized to capture the absolute maximum number of converting shoppers.
        </div>
      </div>
      <div style="width:40px;height:40px;border-radius:10px;background:rgba(180,83,9,0.1);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">🎯</div>
    </div>

    <div style="background:var(--surface-low);padding:12px;border-radius:var(--radius-md);margin:12px 0;display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;text-align:center;">
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">Recall</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:#b45309;">{best_rec_row['recall']*100:.1f}%</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">F1-Score</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--primary);">{best_rec_row['f1']:.4f}</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">Precision</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--on-surface);">{best_rec_row['precision']*100:.1f}%</div>
      </div>
      <div>
        <div style="font-size:0.62rem;font-weight:700;color:var(--secondary);text-transform:uppercase;">ROC-AUC</div>
        <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--outline);">{best_rec_row['roc_auc']:.4f}</div>
      </div>
    </div>

    <div style="font-size:0.75rem;color:var(--secondary);line-height:1.45;margin-bottom:12px;">
      • <strong>Confusion Matrix:</strong> TP = {int(best_rec_row['tp'])}, FP = {int(best_rec_row['fp'])}, FN = {int(best_rec_row['fn'])}, TN = {int(best_rec_row['tn'])}.<br>
      • <strong>Trade-Off Dilemma:</strong> Captures 47 more buyers (+18.0%), but generates 146 additional false alarms (+102.1%).
    </div>
  </div>

  <div style="font-size:0.72rem;background:rgba(180,83,9,0.06);border:1px solid rgba(180,83,9,0.2);padding:8px 10px;border-radius:var(--radius-sm);color:#b45309;">
    <strong>Decision Objective:</strong> Ideal when marketing intervention is costless (e.g. subtle automated reminders) and missing high-value buyers is unacceptable.
  </div>
</div>
""", unsafe_allow_html=True)

    # Operational Trade-off Deep Dive Callout
    tp_diff = int(best_rec_row['tp'] - best_f1_row['tp'])
    fp_diff = int(best_rec_row['fp'] - best_f1_row['fp'])
    fp_pct_increase = (fp_diff / best_f1_row['fp']) * 100

    st.markdown(f"""
<div style="background:var(--surface-bright);border-radius:var(--radius-lg);padding:16px 20px;box-shadow:0 1px 6px rgba(0,0,0,0.05);margin-top:12px;">
  <div style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);margin-bottom:6px;">
    Business Decision Rule: Asymmetric Cost Function
  </div>
  <p style="font-size:0.8rem;color:var(--secondary);line-height:1.5;margin-bottom:10px;">
    The difference between the two configurations illustrates the classical precision-recall trade-off:
  </p>
  <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;margin-bottom:10px;">
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:var(--secondary);">Additional Buyers Caught</div>
      <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--tertiary);">+{tp_diff} Sessions (+18.0%)</div>
      <div style="font-size:0.68rem;color:var(--secondary);">From 261 to 308 true positives</div>
    </div>
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:var(--secondary);">False Alarms Incurred</div>
      <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--error);">+{fp_diff} False Positives (+{fp_pct_increase:.1f}%)</div>
      <div style="font-size:0.68rem;color:var(--secondary);">From 143 to 289 false alarms</div>
    </div>
    <div style="background:var(--surface-low);padding:10px;border-radius:var(--radius-sm);">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:var(--secondary);">Exchange Ratio</div>
      <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:var(--primary);">{fp_diff / tp_diff:.1f} : 1</div>
      <div style="font-size:0.68rem;color:var(--secondary);">~3.1 false alarms per extra buyer</div>
    </div>
  </div>
  <div style="font-size:0.75rem;color:var(--on-surface-var);line-height:1.5;">
    • If each missed buyer costs $100 and a false alarm voucher costs $5, <strong>SVM + Class Weight</strong> wins (net benefit: +$4,700 - $730 = +$3,970).<br>
    • If offering a free gift or live agent contact costs $40 per false alarm, <strong>Random Forest + SMOTE</strong> wins (net benefit: +$3,970 - $5,840 = -$1,870 loss for SVM).
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E: Research Interpretation
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span>📚 Section E: Methodological Research Interpretations</span></div>', unsafe_allow_html=True)

    interp_col1, interp_col2 = st.columns(2, gap="medium")

    with interp_col1:
        st.markdown("""
<div class="result-card" style="height:100%;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="viva-pill" style="background:rgba(0,74,198,0.1);color:var(--primary) !important;">INSIGHT 01</span>
    <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">Why Accuracy is Insufficient</span>
  </div>
  <p style="font-size:0.8rem;color:var(--secondary);line-height:1.5;">
    Accuracy weights every sample equally: \(\\text{Accuracy} = \\frac{TP + TN}{N}\). In a dataset with 84.5% non-buyers, predicting the majority class unconditionally yields 84.5% accuracy with zero predictive utility. Accuracy fails because it masks severe minority error rates. In contrast, F1-Score uses the harmonic mean of Precision and Recall, which guarantees that poor performance on either metric penalizes the aggregate score.
  </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="result-card" style="height:100%;margin-top:12px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="viva-pill" style="background:rgba(0,98,66,0.1);color:var(--tertiary) !important;">INSIGHT 02</span>
    <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">SMOTE vs Class Weighting: Algorithmic Distinction</span>
  </div>
  <p style="font-size:0.8rem;color:var(--secondary);line-height:1.5;">
    <strong>SMOTE operates in data space:</strong> It creates synthetic continuous instances along the line segments connecting k-nearest minority neighbors, increasing minority instance density before training.
    <br><br>
    <strong>Class Weighting operates in loss space:</strong> It leaves training points unchanged, but multiplies the loss penalty for minority misclassifications by \(w_1 = \\frac{N}{2 \\cdot N_1}\). This forces optimization algorithms (gradient descent, quadratic programming in SVM) to shift decision boundaries toward the majority class without creating synthetic data.
  </p>
</div>
""", unsafe_allow_html=True)

    with interp_col2:
        st.markdown("""
<div class="result-card" style="height:100%;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="viva-pill" style="background:rgba(180,83,9,0.1);color:#b45309 !important;">INSIGHT 03</span>
    <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">The Recall vs False Positive Inevitability</span>
  </div>
  <p style="font-size:0.8rem;color:var(--secondary);line-height:1.5;">
    In real-world web traffic, the feature distributions of high-intent window shoppers and actual buyers overlap significantly. To increase recall from 48% to 80%, the classifier must move its decision boundary deeper into the feature space. While this captures borderline buyers who were previously missed, it inevitably captures non-buyers who exhibited similar browsing behaviors, causing precision to decline from 74% to 51%.
  </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="result-card" style="height:100%;margin-top:12px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="viva-pill" style="background:rgba(0,74,198,0.1);color:var(--primary) !important;">INSIGHT 04</span>
    <span style="font-family:var(--font-head);font-size:0.95rem;font-weight:700;color:var(--on-surface);">Objective-Driven Strategy Selection</span>
  </div>
  <p style="font-size:0.8rem;color:var(--secondary);line-height:1.5;">
    Neither SMOTE nor Class Weighting is universally superior across all models. For Random Forest, SMOTE proved best (F1 = 0.6641) because synthetic vectors create richer sub-partitions for tree splits. For SVM, Class Weighting proved best (Recall = 80.6%, F1 = 0.6292) because scaling the C regularization parameter directly shifts the maximum-margin hyperplane without distorting support vector distances.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION F: Viva Questions (Examiner Deep-Dive)
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 Viva Guide: Class Imbalance & Asymmetric Learning FAQs (Examiner Deep-Dive)"):
        st.markdown("""
### Core Academic Questions & Viva Defenses

* **Q1: What is class imbalance and why is it prevalent in online shopping datasets?**
  * *Answer:* Class imbalance occurs when the target categories are not equally distributed in the dataset. In e-commerce, casual browsing is the default human behavior—visitors explore products, check prices, or abandon carts. Only a small fraction (here, 15.47%) complete transactions. This natural behavioral skew creates an asymmetric 5.46:1 ratio where standard loss minimization naturally ignores the rare buying class.

* **Q2: What is SMOTE and how does it generate synthetic training instances?**
  * *Answer:* SMOTE (Synthetic Minority Over-sampling Technique) operates in the continuous feature space of the minority class. For every minority sample \(x_i\), it finds its \(k\)-nearest minority neighbors. It then randomly selects one neighbor \(x_{zi}\) and generates a synthetic instance:
    $$x_{\\text{new}} = x_i + \\lambda (x_{zi} - x_i), \\quad \\lambda \\sim \\text{Uniform}(0, 1)$$
    Crucially, SMOTE was applied **only to the training partition** in each cross-validation fold, completely preventing data leakage into the evaluation test folds.

* **Q3: What is class weighting and how does it modify the loss function?**
  * *Answer:* Class weighting modifies the loss function directly during gradient descent or convex optimization. Under `class_weight='balanced'`, scikit-learn assigns weights inversely proportional to class frequencies:
    $$w_c = \\frac{N}{K \\times N_c}$$
    For the minority class (\(N_1 = 1,908\)), the weight is \(w_1 \\approx 3.23\), while the majority class (\(N_0 = 10,422\)) receives \(w_0 \\approx 0.59\). Misclassifying a positive instance is penalized ~5.5× more severely than misclassifying a negative instance, forcing the classifier to respect minority decision boundaries.

* **Q4: Why is overall classification accuracy insufficient or misleading for this problem?**
  * *Answer:* The Accuracy Paradox states that on imbalanced datasets, high accuracy can be achieved by trivial, non-informative models. A naive baseline that outputs `False` for every single visitor achieves **84.53% accuracy**, yet has **0.0% recall** and captures **$0.00 in revenue**. Accuracy cannot differentiate between a model that genuinely predicts buyers and one that simply memorizes the majority prior.

* **Q5: What is the fundamental difference between precision and recall in an e-commerce deployment?**
  * *Answer:* 
    * **Precision (\(\\frac{TP}{TP + FP}\))** measures the quality of positive alerts: when the system predicts a visitor will buy, how often do they actually buy? Low precision leads to wasted promotional spend (e.g. distributing $15 checkout coupons to window shoppers).
    * **Recall (\(\\frac{TP}{TP + FN}\))** measures the completeness of buyer detection: of all visitors who would have bought, how many did the system identify? Low recall leads to missed sales opportunities.

* **Q6: Why can the best F1 configuration differ from the best recall configuration?**
  * *Answer:* F1-score is the harmonic mean of precision and recall (\(2 \\times \\frac{P \\times R}{P + R}\)). It reaches its maximum when precision and recall are both reasonably high and well-balanced. Our champion F1 model (**Random Forest + Tree Importance + SMOTE**) achieves **F1 = 0.6641** with 64.6% precision and 68.3% recall, producing only 143 false alarms. In contrast, our best recall model (**SVM + RFE + Class Weight**) maximizes recall at **80.63%**, but pushes precision down to 51.59%, producing 289 false alarms. Maximizing recall requires accepting high false alarm rates, whereas maximizing F1 balances both error types.
""")


# ═════════════════════════════════════════════════════════════════════════════
#  Router and Dynamic Navigation Layout
# ═════════════════════════════════════════════════════════════════════════════
def render_current_page(page_name: str) -> None:
    if page_name == "overview":
        render_overview()
    elif page_name == "live_prediction":
        render_live_prediction()
    elif page_name == "model_performance":
        render_model_performance()
    elif page_name == "feature_importance":
        render_feature_importance()
    elif page_name == "class_imbalance":
        render_class_imbalance()
    else:
        st.info(f"Page '{page_name}' is coming soon.")

# Intercept with Welcome Screen on initial load
if not st.session_state.get("intro_shown", False):
    render_intro_screen()
    st.stop()

# Top persistent toggle bar near top-left of main dashboard content
top_col1, top_col2 = st.columns([0.22, 0.78])
with top_col1:
    st.markdown('<div class="nav-toggle-wrap">', unsafe_allow_html=True)
    is_exp = st.session_state.sidebar_expanded
    toggle_label = "☰ Hide Sidebar" if is_exp else "☰ Show Sidebar"
    toggle_help = "Collapse sidebar navigation panel" if is_exp else "Expand sidebar navigation panel"
    if st.button(toggle_label, key="nav_toggle_btn", help=toggle_help):
        st.session_state.sidebar_expanded = not is_exp
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    page_breadcrumbs = {
        "overview": ("📊", "Overview Dashboard"),
        "live_prediction": ("🔮", "Live Prediction Engine"),
        "model_performance": ("📈", "Model Performance & Benchmarks"),
        "feature_importance": ("🌲", "Feature Importance & SHAP"),
        "class_imbalance": ("⚖️", "Class Imbalance Analysis"),
    }
    icon, ptitle = page_breadcrumbs.get(st.session_state.page, ("📄", st.session_state.page.title()))
    st.markdown(f"""
<div style="display:flex;align-items:center;height:100%;gap:8px;font-size:0.8rem;color:var(--secondary);padding-top:4px;">
  <span>{icon} {ptitle}</span>
  <span style="color:var(--outline-var);">·</span>
  <span style="display:inline-flex;align-items:center;gap:4px;color:var(--tertiary);font-weight:600;font-size:0.75rem;">
    <span style="width:6px;height:6px;border-radius:50%;background:var(--tertiary);display:inline-block;"></span> Active Page
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# Main layout dispatch
current_page = st.session_state.get("page", "overview")

if st.session_state.sidebar_expanded:
    col_nav, col_main = st.columns([2.6, 9.4], gap="large")
    with col_nav:
        with st.container(key="custom_nav_container"):
            render_nav_content()
    with col_main:
        render_current_page(current_page)
else:
    render_current_page(current_page)


