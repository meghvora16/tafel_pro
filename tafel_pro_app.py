"""
Polarization Curve Fitter — Publication-Grade Streamlit App
(Robust local Tafel overlays + Tafel intersection i_corr + Original UI)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import io, zipfile, warnings, traceback, re
from itertools import groupby
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter, find_peaks
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, HRFlowable, KeepTogether)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Polarization Curve Fitter", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

if "results" not in st.session_state: st.session_state.results = []
if "figures" not in st.session_state: st.session_state.figures = []

st.markdown("""
<style>
  .main-header{font-size:2rem;font-weight:700;color:#1a3a5c;
    border-bottom:3px solid #2e86de;padding-bottom:8px;margin-bottom:1rem}
  div[data-testid="metric-container"]{
    background:#f0f4ff;border-left:3px solid #2e86de;
    border-radius:6px;padding:8px 12px}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MATH
# ─────────────────────────────────────────────────────────────────────────────
TINY = 1e-30
PALETTE = ["#2e86de","#e84393","#27ae60","#e67e22","#8e44ad","#16a085","#c0392b"]
REGION_COLORS = {"cathodic": ("#6baed6", 0.14), "active": ("#fd8d3c", 0.22), "passive": ("#74c476", 0.16), "transpassive": ("#9e9ac8", 0.18)}
MATERIALS = {"Carbon Steel / Iron": (27.92, 7.87), "304 Stainless Steel": (25.10, 7.90), "316 Stainless Steel": (25.56, 8.00), "Copper": (31.77, 8.96), "Aluminum": (8.99, 2.70)}

def slog(x): return np.log10(np.maximum(np.abs(x), TINY))
def sig(x, k=40.0):
    xk = np.clip(k * x, -60, 60)
    return np.where(xk >= 0, 1.0 / (1.0 + np.exp(-xk)), np.exp(xk) / (1.0 + np.exp(xk)))

def _theil_sen(x, y):
    m = len(x)
    if m < 2: return 0.0, float(np.median(y))
    i_idx, j_idx = np.triu_indices(m, k=1)
    dx = x[j_idx] - x[i_idx]
    valid = np.abs(dx) > 1e-15
    slopes = (y[j_idx][valid] - y[i_idx][valid]) / dx[valid]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    return slope, float(np.median(y - slope * x))

def r2_score(yt, yp):
    sr = np.sum((yt - yp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    return float(max(0.0, 1.0 - sr / st)) if st > 1e-30 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST SLIDING REGRESSION & BRANCH FITTING
# ─────────────────────────────────────────────────────────────────────────────
def _sliding_regress_full(x, y, min_len, max_len=25):
    n = len(x)
    if n < min_len: return (np.array([], int), np.array([], int), np.array([]), np.array([]), np.array([]))
    Sx, Sy, Sxx, Sxy, Syy = np.cumsum(x), np.cumsum(y), np.cumsum(x*x), np.cumsum(x*y), np.cumsum(y*y)
    starts, ends, slopes, inters, r2s = [], [], [], [], []
    for w in range(min_len, min(max_len, n)+1):
        i0 = np.arange(0, n - w + 1); i1 = i0 + w - 1
        def segsum(csum): return csum[i1] - np.concatenate(([0.0], csum[i0[:-1]]))
        sx, sy, sxx, sxy, syy = segsum(Sx), segsum(Sy), segsum(Sxx), segsum(Sxy), segsum(Syy)
        mx, my = sx / w, sy / w
        denom = sxx - w * mx * mx
        slp = np.where(np.abs(denom) > 1e-18, (sxy - w * mx * my) / denom, 0.0)
        itp = my - slp * mx
        sse = (syy - 2*itp*sy - 2*slp*sxy + (itp**2)*w + 2*itp*slp*sx + (slp**2)*sxx)
        sst = syy - w * my * my
        r2 = np.clip(np.where(sst > 1e-18, 1.0 - sse/sst, 0.0), 0.0, 1.0)
        starts.append(i0); ends.append(i1+1); slopes.append(slp); inters.append(itp); r2s.append(r2)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(slopes), np.concatenate(inters), np.concatenate(r2s)

def fit_cathodic(E, i, Ecorr, cfg):
    cat = i < 0
    if np.sum(cat) < cfg["min_w_cat"]: return dict(bc=0.12, icorr=1e-9, r2=0)
    Ec, lgi = E[cat], slog(i[cat])
    si = np.argsort(Ec); Ec, lgi = Ec[si], lgi[si]
    mask = Ec < (Ecorr - cfg["cath_guard"])
    Ex, Yx = Ec[mask], lgi[mask]
    if len(Ex) < cfg["min_w_cat"]: Ex, Yx = Ec, lgi
    s_idx, e_idx, slps, itps, r2s = _sliding_regress_full(Ex, Yx, cfg["min_w_cat"])
    score = r2s * np.exp(-np.abs(Ex[e_idx-1] - Ecorr) / 0.15)
    score[slps >= 0] = 0
    best = np.argmax(score)
    sl_ref, b_ref = _theil_sen(Ex[s_idx[best]:e_idx[best]], Yx[s_idx[best]:e_idx[best]])
    return dict(bc=abs(1.0/sl_ref), icorr=10**(b_ref + sl_ref*Ecorr), slope_c=sl_ref, intercept_c=b_ref, win_c=(Ex[s_idx[best]], Ex[e_idx[best]-1]), r2=r2s[best], E_cat=Ec, lgi_cat=lgi)

def fit_anodic(E, i, Ecorr, cfg):
    ano = i > 0
    if np.sum(ano) < cfg["min_w_ano"]: return dict(ba=0.06, r2=0)
    Ea, lgia = E[ano], slog(i[ano])
    si = np.argsort(Ea); Ea, lgia = Ea[si], lgia[si]
    mask = Ea > (Ecorr + cfg["anod_guard"])
    Ex, Yx = Ea[mask], lgia[mask]
    if len(Ex) < cfg["min_w_ano"]: Ex, Yx = Ea, lgia
    s_idx, e_idx, slps, itps, r2s = _sliding_regress_full(Ex, Yx, cfg["min_w_ano"])
    score = r2s * np.exp(-np.abs(Ex[s_idx] - Ecorr) / 0.15)
    score[slps <= 0] = 0
    best = np.argmax(score)
    sl_ref, b_ref = _theil_sen(Ex[s_idx[best]:e_idx[best]], Yx[s_idx[best]:e_idx[best]])
    return dict(ba=abs(1.0/sl_ref), slope_a=sl_ref, intercept_a=b_ref, win_a=(Ex[s_idx[best]], Ex[e_idx[best]-1]), r2=r2s[best], E_an=Ea, lgi_an=lgia, has_passive=False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (RESTORED ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    material = st.selectbox("Material (for CR calculation)", list(MATERIALS.keys()), index=0)
    ew_mat, rho_mat = MATERIALS[material]
    st.divider()
    st.markdown("**Data Import**")
    area = st.number_input("Electrode area (cm²)", 0.001, 1000.0, 1.0)
    unit_fac = st.selectbox("Current unit", ["A/cm²","mA/cm²","µA/cm²"], index=0)
    fac = {"A/cm²":1.0, "mA/cm²":1e-3, "µA/cm²":1e-6}[unit_fac]
    st.divider()
    st.markdown("**Plotting**")
    show_regs = st.toggle("Shade regions", True)
    pub_dpi = st.slider("Export DPI", 150, 600, 300)
    with st.expander("Advanced (Detection Guards)"):
        adv_cfg = {
            "anod_guard": st.number_input("Anodic Guard (V)", 0.0, 0.1, 0.015, 0.005),
            "cath_guard": st.number_input("Cathodic Guard (V)", 0.0, 0.15, 0.035, 0.005),
            "min_w_ano": st.number_input("Min Anodic Pts", 3, 20, 6),
            "min_w_cat": st.number_input("Min Cathodic Pts", 3, 20, 8)
        }
    if st.button("🗑 Clear all", use_container_width=True):
        st.session_state.results, st.session_state.figures = [], []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI & TABS (RESTORED ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Polarization Curve Fitter</div>', unsafe_allow_html=True)
tab_fit, tab_res, tab_cmp = st.tabs(["📂 Upload & Fit", "📊 Results", "📋 Compare"])

with tab_fit:
    c1, c2 = st.columns([1.2, 0.8])
    uploaded_files = c1.file_uploader("CSV Data", accept_multiple_files=True)
    sample_name = c2.text_input("Sample label", "Sample 1")

    if uploaded_files:
        for idx, uf in enumerate(uploaded_files):
            df_raw = pd.read_csv(uf)
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            ec1, ec2 = st.columns(2)
            e_col = ec1.selectbox(f"E column [{uf.name}]", num_cols, key=f"e_{idx}")
            i_col = ec2.selectbox(f"i column [{uf.name}]", num_cols, key=f"i_{idx}")
            
            E = df_raw[e_col].values
            i_obs = df_raw[i_col].values * fac / area

            if st.button(f"🚀 Run Pipeline: {uf.name}", key=f"btn_{idx}", type="primary"):
                # 1. Detect Ecorr
                sc = np.where(np.diff(np.sign(i_obs)))[0]
                Ecorr = E[sc[0]] if len(sc)>0 else E[np.argmin(np.abs(i_obs))]
                
                # 2. Branch Fits (using robust logic)
                cat_res = fit_cathodic(E, i_obs, Ecorr, adv_cfg)
                an_res = fit_anodic(E, i_obs, Ecorr, adv_cfg)
                
                # 3. Final Display Params
                p = [Ecorr, cat_res['icorr'], an_res['ba'], cat_res['bc']]
                
                # 4. Figure (GridSpec)
                fig = plt.figure(figsize=(12, 8), dpi=pub_dpi)
                gs = GridSpec(2, 2, figure=fig)
                ax1 = fig.add_subplot(gs[0, :])
                ax1.scatter(E, slog(i_obs), s=10, alpha=0.4, label="Data")
                # Visual Tafel Extensions
                E_ext_c = np.linspace(cat_res['win_c'][0], Ecorr, 50)
                ax1.plot(E_ext_c, cat_res['slope_c']*E_ext_c + cat_res['intercept_c'], 'r--', alpha=0.7)
                E_ext_a = np.linspace(Ecorr, an_res['win_a'][1], 50)
                ax1.plot(E_ext_a, an_res['slope_a']*E_ext_a + an_res['intercept_a'], 'g--', alpha=0.7)
                ax1.set_title(f"Evans Diagram: {sample_name}")
                st.pyplot(fig)
                
                # 5. Session Save
                res_rec = {"name": sample_name, "params": p, "ba": p[2], "bc": p[3], "icorr": p[1], "Ecorr": p[0], "r2": cat_res['r2']}
                st.session_state.results.append(res_rec)
                
                # 6. Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("E_corr (V)", f"{p[0]:.4f}")
                m2.metric("i_corr (A/cm²)", f"{p[1]:.2e}")
                m3.metric("CR (mm/yr)", f"{(p[1]*3.27*ew_mat/rho_mat):.4f}")

with tab_res:
    if st.session_state.results:
        st.dataframe(pd.DataFrame(st.session_state.results))
    else:
        st.info("No results yet.")

with tab_cmp:
    if len(st.session_state.results) > 1:
        fig_cmp, ax_cmp = plt.subplots()
        for r in st.session_state.results:
            ax_cmp.bar(r['name'], r['icorr'])
        ax_cmp.set_yscale('log')
        st.pyplot(fig_cmp)
