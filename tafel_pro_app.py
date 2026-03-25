"""
Polarization Curve Fitter — Publication-Grade Streamlit App
Robust local Tafel overlays + Tafel intersection i_corr
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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Polarization Curve Fitter", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TINY    = 1e-30
PALETTE = ["#2e86de","#e84393","#27ae60","#e67e22","#8e44ad","#16a085","#c0392b"]

REGION_COLORS = {
    "cathodic":     ("#6baed6", 0.14),
    "active":       ("#fd8d3c", 0.22),
    "passive":      ("#74c476", 0.16),
    "transpassive": ("#9e9ac8", 0.18),
}

MATERIALS = {
    "Carbon Steel / Iron":   (27.92, 7.87),
    "304 Stainless Steel":   (25.10, 7.90),
    "316 Stainless Steel":   (25.56, 8.00),
    "Copper":                (31.77, 8.96),
    "Aluminum":              ( 8.99, 2.70),
}

CFG = dict(
    cath_guard=0.030,      # V — Distance from Ecorr to avoid rounding
    anod_guard=0.015,      
    curvature_max=35.0,    
    lin_frac=0.75,         
    min_w_cat=8,           # More points for stability
    min_w_ano=6,           
    beta_min=0.020,        
    beta_max_c=0.450,      
    beta_max_a=0.400       
)

# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def slog(x): return np.log10(np.maximum(np.abs(x), TINY))

def sig(x, k=40.0):
    xk = np.clip(k * x, -60, 60)
    return np.where(xk >= 0, 1.0 / (1.0 + np.exp(-xk)),
                    np.exp(xk) / (1.0 + np.exp(xk)))

def r2_score(yt, yp):
    sr = np.sum((yt - yp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    return float(max(0.0, 1.0 - sr / st)) if st > 1e-30 else 0.0

def _theil_sen(x, y):
    m = len(x)
    if m < 2: return 0.0, float(np.median(y))
    i_idx, j_idx = np.triu_indices(m, k=1)
    dx = x[j_idx] - x[i_idx]
    valid = np.abs(dx) > 1e-15
    slopes = (y[j_idx][valid] - y[i_idx][valid]) / dx[valid]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept

# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED SLIDING REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def _sliding_regress_full(x, y, min_len=4, max_len=25):
    n = len(x)
    if n < min_len: return (np.array([], int), np.array([], int), np.array([]), np.array([]), np.array([]))
    Sx = np.cumsum(x); Sy = np.cumsum(y)
    Sxx = np.cumsum(x*x); Sxy = np.cumsum(x*y); Syy = np.cumsum(y*y)
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
        starts.append(i0); ends.append(i1 + 1); slopes.append(slp); inters.append(itp); r2s.append(r2)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(slopes), np.concatenate(inters), np.concatenate(r2s)

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH FITTING (FIXED LOGIC)
# ─────────────────────────────────────────────────────────────────────────────
def fit_cathodic(E, i, Ecorr):
    cat = i < 0
    if np.sum(cat) < CFG["min_w_cat"]: return dict(bc=0.12, icorr=1e-9, r2=0)
    Ec, lgi = E[cat], slog(i[cat])
    si = np.argsort(Ec); Ec, lgi = Ec[si], lgi[si]
    mask = Ec < (Ecorr - CFG["cath_guard"])
    Ex, Yx = Ec[mask], lgi[mask]
    if len(Ex) < CFG["min_w_cat"]: Ex, Yx = Ec, lgi
    
    s_idx, e_idx, slps, itps, r2s = _sliding_regress_full(Ex, Yx, CFG["min_w_cat"], 25)
    if len(slps) == 0: return dict(bc=0.12, icorr=1e-9, r2=0)

    # Scoring: High R2 + Negative slope + Proximity to Ecorr
    invm = 1.0 / (np.abs(slps) + 1e-20)
    beta_ok = (invm > CFG["beta_min"]) & (invm < CFG["beta_max_c"])
    dist_score = np.exp(-np.abs(Ex[e_idx-1] - Ecorr) / 0.15)
    score = r2s * dist_score
    score[(slps >= 0) | (~beta_ok)] = 0
    
    best = np.argmax(score)
    sl_ref, b_ref = _theil_sen(Ex[s_idx[best]:e_idx[best]], Yx[s_idx[best]:e_idx[best]])
    
    return dict(bc=abs(1.0/sl_ref), icorr=10**(b_ref + sl_ref*Ecorr), 
                slope_c=sl_ref, intercept_c=b_ref, win_c=(Ex[s_idx[best]], Ex[e_idx[best]-1]),
                r2=r2s[best], E_cat=Ec, lgi_cat=lgi, has_diff=False, iL=1e-2)

def fit_anodic(E, i, Ecorr):
    ano = i > 0
    if np.sum(ano) < CFG["min_w_ano"]: return dict(ba=0.06, r2=0)
    Ea, lgia = E[ano], slog(i[ano])
    si = np.argsort(Ea); Ea, lgia = Ea[si], lgia[si]
    mask = Ea > (Ecorr + CFG["anod_guard"])
    Ex, Yx = Ea[mask], lgia[mask]
    if len(Ex) < CFG["min_w_ano"]: Ex, Yx = Ea, lgia
    
    s_idx, e_idx, slps, itps, r2s = _sliding_regress_full(Ex, Yx, CFG["min_w_ano"], 25)
    if len(slps) == 0: return dict(ba=0.06, r2=0)

    invm = 1.0 / (np.abs(slps) + 1e-20)
    beta_ok = (invm > CFG["beta_min"]) & (invm < CFG["beta_max_a"])
    dist_score = np.exp(-np.abs(Ex[s_idx] - Ecorr) / 0.15)
    score = r2s * dist_score
    score[(slps <= 0) | (~beta_ok)] = 0
    
    best = np.argmax(score)
    sl_ref, b_ref = _theil_sen(Ex[s_idx[best]:e_idx[best]], Yx[s_idx[best]:e_idx[best]])
    
    # Passive detection (simple threshold on gradient)
    has_p = False; Ep = None; ip = 1e-6
    if len(Ea) > 10:
        grad = np.abs(np.gradient(savgol_filter(lgia, 7, 3), Ea))
        flats = np.where(grad < 1.0)[0]
        if len(flats) > 4:
            has_p = True; Ep = Ea[flats[0]]; ip = 10**lgia[flats[len(flats)//2]]

    return dict(ba=abs(1.0/sl_ref), slope_a=sl_ref, intercept_a=b_ref, 
                win_a=(Ex[s_idx[best]], Ex[e_idx[best]-1]), r2=r2s[best],
                E_an=Ea, lgi_an=lgia, has_passive=has_p, Epass=Ep, ip=ip, has_trans=False)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS MODEL & GLOBAL FIT
# ─────────────────────────────────────────────────────────────────────────────
def pol_model(E, p, ct="A"):
    Ec, ic, ba, bc = p[0], p[1], max(p[2],1e-4), max(p[3],1e-4)
    eta = E - Ec
    i_cat = ic * np.exp(-2.303 * eta / bc)
    i_act = ic * np.exp(2.303 * eta / ba)
    if ct == "A": return i_act - i_cat
    # Simple Passive model
    Ep, ip = p[4], p[6]
    w_p = sig(E - Ep, 1.0/max(p[5],0.001))
    return ((1.0 - w_p)*i_act + w_p*ip) - i_cat

def global_polish(E, i_obs, p0, ct):
    ld = slog(i_obs)
    def obj(x):
        # x: [Ecorr, logIcorr, ba, bc, ...]
        p = x.copy(); p[1] = 10**x[1]
        if ct == "P": p[6] = 10**x[6]
        pred = slog(pol_model(E, p, ct))
        return np.sum((ld - pred)**2)
    
    x0 = p0.copy(); x0[1] = np.log10(max(p0[1], TINY))
    if ct == "P": x0[6] = np.log10(max(p0[6], TINY))
    
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000})
    pf = res.x.copy(); pf[1] = 10**pf[1]
    if ct == "P": pf[6] = 10**pf[6]
    return pf, r2_score(ld, slog(pol_model(E, pf, ct)))

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Pro Polarization Fitter")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    cols = df.columns.tolist()
    e_col = st.selectbox("E Column", cols, index=0)
    i_col = st.selectbox("i Column", cols, index=1)
    
    E = df[e_col].values; i = df[i_col].values
    
    if st.button("Run Fit"):
        # 1. Ecorr
        sc = np.where(np.diff(np.sign(i)))[0]
        Ecorr = E[sc[0]] if len(sc)>0 else E[np.argmin(np.abs(i))]
        
        # 2. Branch Fits
        cat_res = fit_cathodic(E, i, Ecorr)
        an_res = fit_anodic(E, i, Ecorr)
        
        # 3. Global
        ct = "P" if an_res['has_passive'] else "A"
        p0 = [Ecorr, cat_res['icorr'], an_res['ba'], cat_res['bc'], 
              an_res.get('Epass', Ecorr+0.2), 0.01, an_res.get('ip', 1e-6)]
        
        best_p, r2_val = global_polish(E, i, p0, ct)
        
        # 4. Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(E, slog(i), s=10, alpha=0.3, label="Data")
        E_den = np.linspace(E.min(), E.max(), 500)
        ax.plot(E_den, slog(pol_model(E_den, best_p, ct)), 'r', lw=2, label=f"Fit (R2={r2_val:.4f})")
        
        # Local Overlays
        ax.plot(cat_res['win_c'], cat_res['slope_c']*np.array(cat_res['win_c'])+cat_res['intercept_c'], 'b--', lw=2, label="Local Cathodic")
        ax.plot(an_res['win_a'], an_res['slope_a']*np.array(an_res['win_a'])+an_res['intercept_a'], 'orange', ls='--', lw=2, label="Local Anodic")
        
        ax.set_xlabel("E (V)"); ax.set_ylabel("log|i|")
        ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        
        st.write(f"**Results:** icorr={best_p[1]:.2e} A/cm², ba={best_p[2]*1000:.1f} mV, bc={best_p[3]*1000:.1f} mV")
