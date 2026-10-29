"""ShopperAI — Purchase Intention Predictor (Stage 3: Overview + Live Prediction).

Navigation:
  Overview       — dataset stats, experiment leaderboard, research summary
  Live Prediction — interactive ML inference (the main working page)

All prediction logic is in src/predict_purchase.py.  This file handles only UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
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

/* ── Live Prediction quick presets ── */
div[class*="st-key-preset_high"] [data-testid="stButton"] > button,
div[class*="st-key-preset_window"] [data-testid="stButton"] > button,
div[class*="st-key-preset_return"] [data-testid="stButton"] > button {
    width:100% !important; min-height:78px !important; height:auto !important;
    padding:12px 8px !important; white-space:normal !important;
    line-height:1.25 !important; font-size:0.88rem !important; font-weight:700 !important;
    color:var(--primary) !important; background:var(--surface-low) !important;
    border:1px solid var(--outline-var) !important; border-radius:var(--radius-md) !important;
    box-shadow:0 1px 4px rgba(11,28,48,0.08) !important;
}
div[class*="st-key-preset_high"] [data-testid="stButton"] > button:hover,
div[class*="st-key-preset_window"] [data-testid="stButton"] > button:hover,
div[class*="st-key-preset_return"] [data-testid="stButton"] > button:hover {
    color:var(--primary) !important; background:var(--surface-container) !important;
    border-color:var(--primary) !important;
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
    df = load_experiment_results()
    cv_df = load_cv_metrics()
    dataset_stats = load_dataset_stats()
    render_page_header(
        title="Model Performance",
        badge="📈 Test set + cross-validation",
        desc=f"{len(df)} model runs · {df['model'].nunique()} classifiers · {dataset_stats['total']:,} sessions"
    )
    best_f1_row = df.loc[df["f1"].idxmax()]
    best_rec_row = df.loc[df["recall"].idxmax()]

    def model_label(row: pd.Series) -> str:
        model = str(row["model"]).replace("_", " ").title()
        features = str(row["feature_configuration"]).replace("_", " ").title()
        imbalance = str(row["imbalance_strategy"]).replace("_", " ").title()
        return f"{model} · {features} · {imbalance}"

    # Five-metric radar comparison for the two data-selected champions.
    import math
    radar_metrics = [
        ("F1", "f1"), ("Recall", "recall"), ("Precision", "precision"),
        ("Accuracy", "accuracy"), ("ROC-AUC", "roc_auc"),
    ]
    radar_cx, radar_cy, radar_radius = 350, 265, 170
    radar_angles = [-math.pi / 2 + i * 2 * math.pi / len(radar_metrics) for i in range(len(radar_metrics))]

    def radar_points(row: pd.Series, scale: float = 1.0) -> str:
        return " ".join(
            f"{radar_cx + radar_radius * scale * float(row[key]) * math.cos(angle):.1f},"
            f"{radar_cy + radar_radius * scale * float(row[key]) * math.sin(angle):.1f}"
            for (_, key), angle in zip(radar_metrics, radar_angles)
        )

    radar_grid = "".join(
        f'<polygon points="{radar_points(pd.Series({key: 1.0 for _, key in radar_metrics}), level)}" '
        f'fill="none" stroke="#dce3f1" stroke-width="2" />'
        for level in (.25, .5, .75, 1.0)
    )
    radar_axes = "".join(
        f'<line x1="{radar_cx}" y1="{radar_cy}" '
        f'x2="{radar_cx + radar_radius * math.cos(angle):.1f}" '
        f'y2="{radar_cy + radar_radius * math.sin(angle):.1f}" '
        f'stroke="#dce3f1" stroke-width="2" />'
        for angle in radar_angles
    )
    radar_labels = "".join(
        f'<text x="{radar_cx + (radar_radius + 34) * math.cos(angle):.1f}" '
        f'y="{radar_cy + (radar_radius + 34) * math.sin(angle):.1f}" '
        f'text-anchor="middle" dominant-baseline="middle" '
        f'font-family="Inter, sans-serif" font-size="22" font-weight="600" fill="#586579">{label}</text>'
        for (label, _), angle in zip(radar_metrics, radar_angles)
    )
    radar_svg = f"""
<svg viewBox="0 0 700 540" role="img" aria-label="Best F1 and best recall across five evaluation metrics" style="width:100%;height:auto;max-height:570px">
 <g>{radar_grid}{radar_axes}</g>
 <polygon points="{radar_points(best_f1_row)}" fill="#1d4ed8" fill-opacity=".18" stroke="#1d4ed8" stroke-width="4" stroke-linejoin="round" />
 <polygon points="{radar_points(best_rec_row)}" fill="#1f6b47" fill-opacity=".16" stroke="#1f6b47" stroke-width="4" stroke-linejoin="round" />
 {''.join(f'<circle cx="{radar_cx + radar_radius * float(row[key]) * math.cos(angle):.1f}" cy="{radar_cy + radar_radius * float(row[key]) * math.sin(angle):.1f}" r="7" fill="{color}" />' for row, color in ((best_f1_row, '#1d4ed8'), (best_rec_row, '#1f6b47')) for (_, key), angle in zip(radar_metrics, radar_angles))}
 {radar_labels}
</svg>"""

    def champion_card(title: str, row: pd.Series, metric: str, color: str) -> str:
        value = row[metric] * 100
        label = "Recall" if metric == "recall" else "F1 score"
        return f"""
<div class="lb-card" style="border-left:6px solid {color};padding:24px 26px;">
  <div style="font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{color};">{title}</div>
  <div style="font:700 2rem/1.2 var(--font-head);margin:8px 0 4px;color:var(--on-surface);">{value:.1f}%</div>
  <div style="font-size:1rem;font-weight:600;margin-bottom:14px;">{model_label(row)}</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;background:var(--surface-low);border-radius:var(--radius-md);padding:12px 16px;">
    <div><span class="lb-metric-label" style="font-size:.78rem">{label}</span><br><strong style="font-size:1.3rem;color:{color};">{value:.1f}%</strong></div>
    <div><span class="lb-metric-label" style="font-size:.78rem">Precision</span><br><strong style="font-size:1.3rem;">{row['precision']*100:.1f}%</strong></div>
    <div><span class="lb-metric-label" style="font-size:.78rem">ROC-AUC</span><br><strong style="font-size:1.3rem;">{row['roc_auc']:.3f}</strong></div>
    <div><span class="lb-metric-label" style="font-size:.78rem">Accuracy</span><br><strong style="font-size:1.3rem;">{row['accuracy']*100:.1f}%</strong></div>
  </div>
</div>"""

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(champion_card("Best F1", best_f1_row, "f1", "var(--primary)"), unsafe_allow_html=True)
    with c2:
        st.markdown(champion_card("Best purchase recall", best_rec_row, "recall", "var(--tertiary)"), unsafe_allow_html=True)

    # Compact model overview and ROC-AUC comparison.
    baseline_models = [
        ("Random Forest", "random_forest"), ("Support Vector Machine", "svm"),
        ("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
    ]
    col_comp, col_auc = st.columns([1, 1])
    with col_comp:
        st.markdown('<div class="result-card"><div class="result-card-title" style="font-size:1.2rem;">Baseline metrics</div><div style="color:var(--secondary);font-size:.9rem;margin-bottom:14px;">Held-out test · all features</div>', unsafe_allow_html=True)
        for name, key in baseline_models:
            row = df[(df["model"] == key) & (df["feature_configuration"] == "all") & (df["imbalance_strategy"] == "none")].iloc[0]
            st.markdown(f"""
<div class="compare-model-row" style="padding:14px 16px;">
 <div class="compare-model-head"><div class="compare-model-name" style="font-size:1rem;">{name}</div></div>
 <div class="metric-grid-4">
  <div class="metric-bar-item"><div class="metric-bar-header"><span>Accuracy</span><span>{row['accuracy']*100:.1f}%</span></div><div class="metric-bar-track" style="height:10px"><div class="metric-bar-fill" style="width:{row['accuracy']*100:.1f}%;background:var(--secondary)"></div></div></div>
  <div class="metric-bar-item"><div class="metric-bar-header"><span>Precision</span><span>{row['precision']*100:.1f}%</span></div><div class="metric-bar-track" style="height:10px"><div class="metric-bar-fill" style="width:{row['precision']*100:.1f}%;background:#3b82f6"></div></div></div>
  <div class="metric-bar-item"><div class="metric-bar-header"><span>Recall</span><span>{row['recall']*100:.1f}%</span></div><div class="metric-bar-track" style="height:10px"><div class="metric-bar-fill" style="width:{row['recall']*100:.1f}%;background:var(--tertiary)"></div></div></div>
  <div class="metric-bar-item"><div class="metric-bar-header"><span>F1</span><span>{row['f1']*100:.1f}%</span></div><div class="metric-bar-track" style="height:10px"><div class="metric-bar-fill" style="width:{row['f1']*100:.1f}%;background:var(--primary)"></div></div></div>
 </div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_auc:
        st.markdown('<div class="result-card"><div class="result-card-title" style="font-size:1.2rem;">ROC-AUC by model</div><div style="color:var(--secondary);font-size:.9rem;margin-bottom:18px;">Baseline · held-out test</div>', unsafe_allow_html=True)
        auc_rows = []
        for name, key in baseline_models:
            row = df[(df["model"] == key) & (df["feature_configuration"] == "all") & (df["imbalance_strategy"] == "none")].iloc[0]
            auc_rows.append((name, row["roc_auc"]))
        for name, score in sorted(auc_rows, key=lambda item: item[1], reverse=True):
            color = "var(--primary)" if name == "Random Forest" else "var(--tertiary)" if score >= .88 else "#c2571a"
            st.markdown(f'<div class="bar-row" style="margin:18px 0"><div class="bar-row-label" style="font-size:.9rem;min-width:170px">{name}</div><div class="bar-track" style="height:16px"><div class="bar-fill" style="width:{score*100:.1f}%;background:{color}"></div></div><div class="bar-row-val" style="font-size:1rem;min-width:55px">{score:.3f}</div></div>', unsafe_allow_html=True)
        roc_fig_path = FIGS / "roc_auc_by_model.png"
        if roc_fig_path.exists():
            st.image(str(roc_fig_path), caption="ROC-AUC comparison", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Confusion matrix: keep the existing four-cell visual, with clearer labels.
    cm_choice = st.selectbox("Confusion matrix", ["Best F1", "Best recall"], key="mp_cm_selector")
    use_best_f1_cm = cm_choice == "Best F1"
    cm_row = best_f1_row if use_best_f1_cm else best_rec_row
    tn, fp, fn, tp = (int(cm_row[k]) for k in ("tn", "fp", "fn", "tp"))
    total_test, total_neg, total_pos = tn + fp + fn + tp, tn + fp, fn + tp
    tnr, fpr = tn / total_neg * 100, fp / total_neg * 100
    fnr, tpr = fn / total_pos * 100, tp / total_pos * 100
    st.markdown(f"""
<div class="cm-card" style="padding:26px;margin-bottom:1.25rem;">
 <div class="result-card-title" style="font-size:1.25rem;">Confusion matrix · {'Best F1' if use_best_f1_cm else 'Best recall'}</div>
 <div style="font-size:.9rem;color:var(--secondary);margin-bottom:16px">{model_label(cm_row)} · {total_test:,} test sessions</div>
 <div style="display:grid;grid-template-columns:130px 1fr 1fr;gap:8px;margin-bottom:8px"><div></div><div class="cm-grid-header" style="font-size:.82rem;padding:10px">Predicted · no purchase</div><div class="cm-grid-header" style="font-size:.82rem;padding:10px">Predicted · purchase</div></div>
 <div style="display:grid;grid-template-columns:130px 1fr 1fr;gap:8px;margin-bottom:8px"><div class="cm-grid-header" style="display:grid;place-items:center;font-size:.82rem">Actual · no purchase</div><div class="cm-cell tn" style="min-height:130px;padding:18px"><div class="cm-cell-name">True negative</div><div class="cm-count" style="font-size:2.2rem">{tn:,}</div><div class="cm-rate">Specificity <b>{tnr:.1f}%</b></div></div><div class="cm-cell fp" style="min-height:130px;padding:18px"><div class="cm-cell-name">False positive</div><div class="cm-count" style="font-size:2.2rem">{fp:,}</div><div class="cm-rate">False alarm rate <b>{fpr:.1f}%</b></div></div></div>
 <div style="display:grid;grid-template-columns:130px 1fr 1fr;gap:8px"><div class="cm-grid-header" style="display:grid;place-items:center;font-size:.82rem">Actual · purchase</div><div class="cm-cell fn" style="min-height:130px;padding:18px"><div class="cm-cell-name">False negative</div><div class="cm-count" style="font-size:2.2rem">{fn:,}</div><div class="cm-rate">Miss rate <b>{fnr:.1f}%</b></div></div><div class="cm-cell tp" style="min-height:130px;padding:18px"><div class="cm-cell-name">True positive</div><div class="cm-count" style="font-size:2.2rem">{tp:,}</div><div class="cm-rate">Recall <b>{tpr:.1f}%</b></div></div></div>
 <div class="cm-summary-bar" style="font-size:.9rem;padding:14px"><span>Accuracy <b>{cm_row['accuracy']*100:.1f}%</b></span><span>Precision <b>{cm_row['precision']*100:.1f}%</b></span><span>F1 <b>{cm_row['f1']:.3f}</b></span></div>
</div>""", unsafe_allow_html=True)
    cm_fig_path = FIGS / "best_f1_confusion_matrix.png"
    if use_best_f1_cm and cm_fig_path.exists():
        with st.expander("View confusion matrix heatmap"):
            st.image(str(cm_fig_path), use_container_width=True)

    # Cross-validation intervals and the original champion trade-off chart.
    st.markdown('<div class="result-card"><div class="result-card-title" style="font-size:1.25rem;">Cross-validation · 95% confidence intervals</div><div style="color:var(--secondary);font-size:.9rem;margin-bottom:12px">10-fold validation on the training split</div>', unsafe_allow_html=True)
    cv_ci_fig = FIGS / "repeated_cv_confidence_intervals.png"
    if cv_ci_fig.exists():
        st.image(str(cv_ci_fig), caption="Cross-validation confidence intervals", use_container_width=True)
    cv_show = cv_df[["display_name", "f1_mean", "f1_ci95_low", "f1_ci95_high", "test_f1", "recall_mean", "roc_auc_mean"]].copy()
    cv_show.columns = ["Model / setup", "CV F1", "CI low", "CI high", "Test F1", "CV recall", "CV ROC-AUC"]
    with st.expander("CV metrics table"):
        st.dataframe(cv_show, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="result-card"><div class="result-card-title" style="font-size:1.45rem;">Best F1 vs best recall</div><div style="color:var(--secondary);font-size:1rem;margin-bottom:16px">The two champions across all five metrics</div>', unsafe_allow_html=True)
    st.markdown(radar_svg, unsafe_allow_html=True)
    f1_fp = int(best_f1_row["fp"])
    recall_fp = int(best_rec_row["fp"])
    recall_champ_pct = float(best_rec_row["recall"]) * 100
    f1_recommendation = "Balanced choice · fewer false alarms" if f1_fp < recall_fp else "Balanced choice · strongest F1 score"
    def short_champion_label(row: pd.Series) -> str:
        model = str(row["model"]).replace("_", " ").title()
        strategy = str(row["imbalance_strategy"]).replace("_", " ").title()
        return f"{model} + {strategy}"
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin:10px 0 4px">
 <div style="background:#eef3ff;border-left:6px solid #1d4ed8;border-radius:18px;padding:22px 24px">
  <div style="font-size:1.5rem;font-weight:800;color:#1d4ed8">{short_champion_label(best_f1_row)}</div>
  <div style="font-size:1.1rem;font-weight:600;color:#586579;margin-top:4px">{f1_recommendation} · F1 {float(best_f1_row['f1']):.3f} · {f1_fp:,} false alarms</div>
 </div>
 <div style="background:#eef3ff;border-left:6px solid #1f6b47;border-radius:18px;padding:22px 24px">
  <div style="font-size:1.5rem;font-weight:800;color:#1f6b47">{short_champion_label(best_rec_row)}</div>
  <div style="font-size:1.1rem;font-weight:600;color:#586579;margin-top:4px">Higher buyer coverage · recall {recall_champ_pct:.1f}% · {recall_fp:,} false alarms</div>
 </div>
</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    tradeoff_fig = FIGS / "best_f1_vs_best_recall.png"
    if tradeoff_fig.exists():
        with st.expander("Champion metrics · held-out test", expanded=False):
            st.image(str(tradeoff_fig), use_container_width=True)

    # Keep full experiment access available without dominating the page.
    with st.expander("Explore all experiment results"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_model = st.selectbox("Model", ["All Models", "Random Forest", "Support Vector Machine", "Logistic Regression", "Decision Tree"], key="mp_filter_model")
        with f_col2:
            sel_feat = st.selectbox("Features", ["All Configurations", "tree_importance", "rfe", "correlation", "all"], key="mp_filter_feat")
        with f_col3:
            sel_imb = st.selectbox("Imbalance strategy", ["All Strategies", "smote", "class_weight", "none"], key="mp_filter_imb")
        model_map = {"Random Forest": "random_forest", "Support Vector Machine": "svm", "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree"}
        filtered_df = df.copy()
        if sel_model != "All Models":
            filtered_df = filtered_df[filtered_df["model"] == model_map[sel_model]]
        if sel_feat != "All Configurations":
            filtered_df = filtered_df[filtered_df["feature_configuration"] == sel_feat]
        if sel_imb != "All Strategies":
            filtered_df = filtered_df[filtered_df["imbalance_strategy"] == sel_imb]
        filtered_df = filtered_df.sort_values("f1", ascending=False)
        exp_view = pd.DataFrame({
            "Model": filtered_df["model"].str.replace("_", " ").str.title(),
            "Features": filtered_df["feature_configuration"], "Imbalance": filtered_df["imbalance_strategy"],
            "F1": filtered_df["f1"].map(lambda v: f"{v:.3f}"), "Recall": filtered_df["recall"].map(lambda v: f"{v*100:.1f}%"),
            "Precision": filtered_df["precision"].map(lambda v: f"{v*100:.1f}%"), "ROC-AUC": filtered_df["roc_auc"].map(lambda v: f"{v:.3f}"),
        })
        st.dataframe(exp_view, use_container_width=True, hide_index=True)

    with st.expander("Metric guide"):
        st.markdown("**F1** balances precision and recall. **Recall** measures captured purchases. **ROC-AUC** measures ranking quality across thresholds. **95% CI** shows the cross-validation uncertainty range.")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: FEATURE IMPORTANCE & SHAP INTERPRETABILITY
# ═════════════════════════════════════════════════════════════════════════════
def render_feature_importance() -> None:
        # ── 1. Clean Header (Concise title & subtitle, no decorative badges) ────
    st.markdown("""
<div class="page-header" style="padding:16px 20px;margin-bottom:1.25rem;">
  <div>
    <h1 style="font-size:1.45rem;font-weight:700;margin:0 0 4px;color:var(--on-surface);">Feature Importance &amp; SHAP Interpretability</h1>
    <p style="margin:0;font-size:0.85rem;color:var(--secondary);">Global feature relevance and local prediction attribution analysis evaluated on 12,330 UCI e-commerce sessions.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    bundle = get_bundle()
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    importances = model.feature_importances_

    # Create MDI ranking
    df_mdi = pd.DataFrame({
        "feature": [clean_feature_name(f) for f in selected_features],
        "importance_pct": importances * 100,
    }).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    top_10_mdi = df_mdi.head(10).copy()

    # Load verified SHAP summary
    shap_summary_data = load_shap_summary()
    top_shap_records = shap_summary_data.get("top_features", [])
    df_shap = pd.DataFrame([
        {
            "feature": clean_feature_name(item["feature"]),
            "mean_abs_shap": item["mean_abs_shap"],
        }
        for item in top_shap_records
    ]).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_10_shap = df_shap.head(10).copy()
    # ── 2. Feature Importance Visuals: MDI vs SHAP Bar Charts ──────────────
    col_mdi, col_shap = st.columns(2, gap="large")

    with col_mdi:
        st.markdown('<div class="section-heading" style="margin-bottom:4px;">🌲 Top 10 Features by MDI Importance</div>', unsafe_allow_html=True)
        st.caption("Mean Decrease in Impurity across 100 decision trees in the champion Random Forest.")

        chart_mdi = alt.Chart(top_10_mdi).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("importance_pct:Q", title="Gini Importance (%)", axis=alt.Axis(format=".1f", grid=True)),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelFontSize=12)),
            color=alt.condition(
                alt.datum.feature == "PageValues",
                alt.value("#004ac6"),
                alt.value("#2563eb")
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance_pct:Q", format=".2f", title="Importance (%)")
            ]
        ).properties(height=340)
        st.altair_chart(chart_mdi, use_container_width=True)

    with col_shap:
        st.markdown('<div class="section-heading" style="margin-bottom:4px;">🧠 Top 10 Features by Global SHAP Value</div>', unsafe_allow_html=True)
        st.caption("Mean absolute Shapley displacement |SHAP| on purchase probability (held-out test set).")

        chart_shap = alt.Chart(top_10_shap).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP Value|", axis=alt.Axis(format=".3f", grid=True)),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelFontSize=12)),
            color=alt.condition(
                alt.datum.feature == "PageValues",
                alt.value("#006242"),
                alt.value("#059669")
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("mean_abs_shap:Q", format=".4f", title="Mean |SHAP|")
            ]
        ).properties(height=340)
        st.altair_chart(chart_shap, use_container_width=True)

    # ── 3. Compact Viva Defense Comparison (MDI vs SHAP) ───────────────────
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    viva_col1, viva_col2 = st.columns(2, gap="large")
    with viva_col1:
        st.markdown("""<div class="result-card" style="padding:14px 16px;border-left:4px solid var(--primary);height:100%;">
  <div style="font-weight:700;font-size:0.9rem;color:var(--primary);margin-bottom:4px;">MDI: Population-Level Relevance</div>
  <div style="font-size:0.78rem;color:var(--on-surface-var);line-height:1.45;">
    Answers <em>which features matter most across the forest</em> by tracking cumulative Gini impurity reduction across all 100 trees. PageValues dominates globally with 37.0% importance.
   </div>
</div>""", unsafe_allow_html=True)

    with viva_col2:
        st.markdown("""<div class="result-card" style="padding:14px 16px;border-left:4px solid var(--tertiary);height:100%;">
  <div style="font-weight:700;font-size:0.9rem;color:var(--tertiary);margin-bottom:4px;">SHAP: Instance-Level Attribution</div>
  <div style="font-size:0.78rem;color:var(--on-surface-var);line-height:1.45;">
    Answers <em>which features contribute most to individual predictions</em> using game-theoretic Shapley values. Captures local non-linear effects such as session-specific duration shifts.
    </div>
</div>""", unsafe_allow_html=True)

    # ── 4. Interactive Local SHAP Explanation ──────────────────────────────
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🔍 Interactive Session Prediction &amp; SHAP Attribution</div>', unsafe_allow_html=True)

    col_sel, col_custom = st.columns([4, 8], gap="medium")
    with col_sel:
        profile_choice = st.selectbox(
            "Select session Profile:",
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

    explanation = explain_session(raw_sess)
     
    prob = explanation["purchase_probability"]
    pred = explanation["predicted_class"]
    base_val = explanation["base_value"]
    
    top3_pos = explanation["pos_drivers"].head(3).copy()
    top3_neg = explanation["neg_drivers"].head(3).copy()
    
    out_label = "Purchase Intended (Class 1)" if pred else "No Purchase (Class 0)"

    out_color = "var(--tertiary)" if pred else "var(--error)"
    out_bg = "rgba(0,98,66,0.08)" if pred else "rgba(186,26,26,0.06)"
    out_border = "rgba(0,98,66,0.2)" if pred else "rgba(186,26,26,0.2)"
    st.markdown(f"""<div style="background:{out_bg};border:1px solid {out_border};border-radius:var(--radius-md);padding:12px 18px;margin:8px 0 16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
  <div>
    <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;color:var(--secondary);letter-spacing:0.05em;">Predicted Outcome</div>
    <div style="font-family:var(--font-head);font-size:1.15rem;font-weight:700;color:{out_color};">{out_label}</div>
  </div>
  <div style="display:flex;align-items:center;gap:24px;">
    <div>
      div style="font-size:0.68rem;color:var(--secondary);">Baseline Prior</div>
      <div style="font-weight:700;font-size:0.95rem;color:var(--on-surface);">{base_val*100:.1f}%</div>

  <div style="display:flex;align-items:center;gap:20px;">
    <div style="text-align:right;">
      <div style="font-size:0.62rem;font-weight:600;text-transform:uppercase;color:var(--secondary);">Model Prior (Base Value)</div>
      <div style="font-family:var(--font-head);font-size:1rem;font-weight:700;color:var(--on-surface);">{base_val*100:.1f}%</div>
       <div>
      <div style="font-size:0.68rem;color:var(--secondary);font-weight:600;">Predicted Probability</div>
      <div style="font-family:var(--font-head);font-size:1.4rem;font-weight:700;color:{out_color};line-height:1;">{prob*100:.1f}%</div> 
    </div>
""", unsafe_allow_html=True)

    col_p, col_n = st.columns(2, gap="large")

    with col_p:
        st.markdown('<div style="font-weight:600;font-size:0.85rem;color:var(--tertiary);margin-bottom:6px;">▲ Top 3 Positive Contributors (+Δ Probability)</div>', unsafe_allow_html=True)
        if len(top3_pos) > 0:
            top3_pos["contrib_pct"] = top3_pos["shap_value"] * 100
            chart_p = alt.Chart(top3_pos).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#006242").encode(
                x=alt.X("contrib_pct:Q", title="+Δ Probability (%)", axis=alt.Axis(format=".1f", grid=True)),
                y=alt.Y("clean_feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180, labelFontSize=12)),
                tooltip=[
                    alt.Tooltip("clean_feature:N", title="Feature"),
                    alt.Tooltip("contrib_pct:Q", format="+.2f", title="+Δ Prob (%)")
                ]
            ).properties(height=140)
            st.altair_chart(chart_p, use_container_width=True)
        else:
            st.info("No positive SHAP displacement for this session.")

    with col_n:
        st.markdown('<div style="font-weight:600;font-size:0.85rem;color:var(--error);margin-bottom:6px;">▼ Top 3 Negative Contributors (-Δ Probability)</div>', unsafe_allow_html=True)
        if len(top3_neg) > 0:
            top3_neg["contrib_pct"] = top3_neg["shap_value"] * 100
            chart_n = alt.Chart(top3_neg).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#ba1a1a").encode(
                x=alt.X("contrib_pct:Q", title="-Δ Probability (%)", axis=alt.Axis(format=".1f", grid=True)),
                y=alt.Y("clean_feature:N", sort="x", title=None, axis=alt.Axis(labelLimit=180, labelFontSize=12)),
                tooltip=[
                    alt.Tooltip("clean_feature:N", title="Feature"),
                    alt.Tooltip("contrib_pct:Q", format="+.2f", title="Δ Prob (%)")
                ]
            ).properties(height=140)
            st.altair_chart(chart_n, use_container_width=True)
        else:
            st.info("No negative SHAP displacement for this session.")

    # ── 5. Feature Selection Results & Target Leakage Ablation Study ────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Feature Selection Strategy Comparison &amp; Ablation Study</div>', unsafe_allow_html=True)  

    col_fs_chart, col_fs_ablation = st.columns([5, 7], gap="large")

    with col_fs_chart:
        st.markdown('<div style="font-weight:600;font-size:0.85rem;margin-bottom:4px;">F1-Score Across 4 Selection Strategies</div>', unsafe_allow_html=True)
        st.caption("Benchmark on Random Forest + SMOTE across 48 experimental trials.")

        df_fs = pd.DataFrame([
            {"Strategy": "Tree-Based (MDI)", "F1_Score": 0.6641, "Features": "Top 25 MDI Features", "Highlight": "Champion"},
            {"Strategy": "Correlation Filter", "F1_Score": 0.6563, "Features": "Pruned (|r| > 0.15)", "Highlight": "Standard"},
            {"Strategy": "All Features (Baseline)", "F1_Score": 0.6563, "Features": "All 75 Encoded Features", "Highlight": "Standard"},
            {"Strategy": "Recursive Elimin. (RFE)", "F1_Score": 0.6406, "Features": "Top 40 Iterative Features", "Highlight": "Standard"}
        ])

        bars_fs = alt.Chart(df_fs).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("F1_Score:Q", title="Minority Class F1-Score", scale=alt.Scale(domain=[0, 0.82]), axis=alt.Axis(format=".2f", grid=True)),
            y=alt.Y("Strategy:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180, labelFontSize=12)),
            color=alt.condition(
                alt.datum.Highlight == "Champion",
                alt.value("#004ac6"),
                alt.value("#94a3b8")
            ),
            tooltip=[
                alt.Tooltip("Strategy:N", title="Strategy"),
                alt.Tooltip("Features:N", title="Features Retained"),
                alt.Tooltip("F1_Score:Q", format=".4f", title="F1-Score")
            ]
        )
        text_fs = bars_fs.mark_text(
            align="left",
            baseline="middle",
            dx=5,
            fontSize=11,
            fontWeight=600
        ).encode(
            text=alt.Text("F1_Score:Q", format=".4f")
        )
        chart_fs = (bars_fs + text_fs).properties(height=200)
        st.altair_chart(chart_fs, use_container_width=True)

    with col_fs_ablation:
        st.markdown('<div style="font-weight:600;font-size:0.85rem;margin-bottom:4px;">🔬 Target Leakage &amp; PageValues Ablation Finding</div>', unsafe_allow_html=True)
        st.markdown("""<div class="result-card" style="padding:12px 14px;font-size:0.78rem;line-height:1.45;color:var(--on-surface-var);margin-bottom:8px;">
  <strong>Key Finding:</strong> Removing <code>PageValues</code> causes baseline SVM recall to collapse from <strong>46.9% to 0.26%</strong>, confirming that unweighted models heavily rely on it. However, with SMOTE and class weighting, pure clickstream features (<code>ExitRates</code>, <code>ProductRelated_Duration</code>) achieve <strong>ROC-AUC = 0.75</strong> and recall of <strong>68.9%–74.9%</strong>.
</div>""", unsafe_allow_html=True)

        ablation_fig = FIGS / "pagevalues_ablation_performance.png"
        if ablation_fig.exists():
            st.image(str(ablation_fig), caption="PageValues Ablation: Performance Comparison Across Classifiers", use_container_width=True)

    # ── 6. Technical Details (Compact Expander) ────────────────────────────
    with st.expander("ℹ️ Technical & Methodology Details"):
        st.markdown("""
- **Tree-SHAP Calculation:** Uses path-dependent tree traversal to calculate exact Shapley values in polynomial time $O(TLD^2)$, where $T$ is the number of trees (100) and $D$ is maximum tree depth.
- **MDI vs SHAP Distinction:** Mean Decrease in Impurity (MDI) measures global node purity gain during training splits, while SHAP calculates exact additive marginal displacements on the final probability output via cooperative game theory.
- **Ablation Protocol:** Evaluated across all 4 supervised classifiers on held-out test splits ($n=2,466$) with both balanced resampling and unweighted baselines.
""")
        global_fig_path = FIGS / "best_f1_shap_global_summary.png"
        local_fig_path = FIGS / "best_f1_shap_local_example.png"
        if global_fig_path.exists() or local_fig_path.exists():
            st.markdown('<div style="font-size:0.75rem;font-weight:600;color:var(--secondary);margin-top:10px;">Research Figure Artifacts:</div>', unsafe_allow_html=True)
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                if global_fig_path.exists():
                    st.image(str(global_fig_path), caption="Matplotlib Global SHAP Summary Plot", use_container_width=True)
            with r_c2:
                if local_fig_path.exists():
                    st.image(str(local_fig_path), caption="Matplotlib Local SHAP Explanation Example", use_container_width=True)


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

  
    <div style="display:flex;align-items:center;gap:20px;background:var(--surface-low);padding:16px;border-radius:var(--radius-md);margin-bottom:14px;">
      <div style="position:relative;width:125px;height:125px;flex-shrink:0;">
        <svg viewBox="0 0 100 100" style="width:100%;height:100%;transform:rotate(-90deg);">
          <circle cx="50" cy="50" r="40" fill="transparent" stroke="var(--secondary-container)" stroke-width="12" stroke-dasharray="212.4 251.3" stroke-linecap="round"/>
          <circle cx="50" cy="50" r="40" fill="transparent" stroke="var(--primary-container)" stroke-width="12" stroke-dasharray="38.9 251.3" stroke-dashoffset="-212.4" stroke-linecap="round"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
          <span style="font-family:var(--font-head);font-size:1.2rem;font-weight:700;color:var(--on-surface);line-height:1;">{imbalance_ratio:.1f}:1</span>
          <span style="font-size:0.6rem;text-transform:uppercase;color:var(--secondary);font-weight:700;letter-spacing:0.05em;">Ratio</span>
        </div>
      </div>

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


