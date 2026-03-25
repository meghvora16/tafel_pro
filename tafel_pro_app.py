"""
Polarization Curve Fitter — Publication-Grade Streamlit App (Optimized)
========================================================================
Key updates:
- Vectorized sliding-window regressions for Tafel branch detection (fast & robust)
- Adaptive dense sampling, rasterized scatters, and optional downsampling for faster plotting
- Lighter, early-exit global optimization with fewer stages where possible
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
from scipy.stats import linregress
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Polarization Curve Fitter", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .main-header{font-size:2rem;font-weight:700;color:#1a3a5c;
    border-bottom:3px solid #2e86de;padding-bottom:8px;margin-bottom:1rem}
  div[data-testid="metric-container"]{
    background:#f0f4ff;border-left:3px solid #2e86de;
    border-radius:6px;padding:8px 12px}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
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
    "Nickel":                (29.36, 8.91),
    "Titanium":              (11.99, 4.51),
    "Zinc":                  (32.69, 7.14),
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER MATH
# ══════════════════════════════════════════════════════════════════════════════

def slog(x):
    """Safe log10 of absolute value."""
    return np.log10(np.maximum(np.abs(x), TINY))

def sig(x, k=40.0):
    """Numerically stable logistic sigmoid."""
    xk = np.clip(k * x, -60, 60)
    return np.where(xk >= 0, 1.0 / (1.0 + np.exp(-xk)),
                    np.exp(xk) / (1.0 + np.exp(xk)))

def sm(y, w=11, p=3):
    """Savitzky-Golay smooth with auto window."""
    n = len(y)
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(5, w) if w >= 5 else n
    if w > n or w < 5: return y.copy()
    return savgol_filter(y, w, min(p, w - 1), mode="interp")

def r2_score(yt, yp):
    sr = np.sum((yt - yp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    return float(max(0.0, 1.0 - sr / st)) if st > 1e-30 else 0.0

def aicc(n, k, sse):
    """Corrected AIC."""
    if n <= k + 1 or sse <= 0: return 1e30
    return n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)

def downsample_uniform(x, y, max_pts=400):
    """Simple uniform downsample to cap points for secondary panels."""
    if len(x) <= max_pts:
        return x, y
    idx = np.linspace(0, len(x)-1, max_pts).astype(int)
    return x[idx], y[idx]

# ══════════════════════════════════════════════════════════════════════════════
# FAST SLIDING REGRESSION (VECTORIZED)
# ══════════════════════════════════════════════════════════════════════════════

def _sliding_regress_full(x, y, min_len=4, max_len=25):
    """
    Vectorized sliding-window linear regression on (x,y) sorted by x.
    Returns concatenated arrays: start_idx, end_idx (exclusive), slope, intercept, r2.
    """
    n = len(x)
    if n < min_len:
        return (np.array([], int), np.array([], int),
                np.array([]), np.array([]), np.array([]))
    Sx  = np.cumsum(x)
    Sy  = np.cumsum(y)
    Sxx = np.cumsum(x*x)
    Sxy = np.cumsum(x*y)
    Syy = np.cumsum(y*y)

    starts_all = []
    ends_all = []
    slopes_all = []
    inters_all = []
    r2_all = []

    wmax = min(max_len, n)
    for w in range(min_len, wmax+1):
        i0 = np.arange(0, n - w + 1)
        i1 = i0 + w - 1
        def segsum(csum):
            return csum[i1] - np.concatenate(([0.0], csum[i0[:-1]]))
        sum_x  = segsum(Sx)
        sum_y  = segsum(Sy)
        sum_xx = segsum(Sxx)
        sum_xy = segsum(Sxy)
        sum_yy = segsum(Syy)

        w_f = float(w)
        mx = sum_x / w_f
        my = sum_y / w_f

        denom = sum_xx - w_f * mx * mx
        slope = np.where(np.abs(denom) > 1e-18, (sum_xy - w_f * mx * my) / denom, 0.0)
        intercept = my - slope * mx

        # SSE via sums
        SSE = (sum_yy
               - 2.0*intercept*sum_y
               - 2.0*slope*sum_xy
               + (intercept*intercept)*w_f
               + 2.0*intercept*slope*sum_x
               + (slope*slope)*sum_xx)
        SST = sum_yy - w_f * my * my
        R2 = np.where(SST > 1e-18, 1.0 - SSE / SST, 0.0)
        R2 = np.clip(R2, 0.0, 1.0)

        starts_all.append(i0)
        ends_all.append(i1 + 1)
        slopes_all.append(slope)
        inters_all.append(intercept)
        r2_all.append(R2)

    starts = np.concatenate(starts_all) if starts_all else np.array([], int)
    ends   = np.concatenate(ends_all) if ends_all else np.array([], int)
    slopes = np.concatenate(slopes_all) if slopes_all else np.array([])
    inters = np.concatenate(inters_all) if inters_all else np.array([])
    r2s    = np.concatenate(r2_all) if r2_all else np.array([])
    return starts, ends, slopes, inters, r2s

# ══════════════════════════════════════════════════════════════════════════════
# CURVE TYPE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Parameters:  [0]Ecorr  [1]icorr  [2]ba  [3]bc
#              [4]Epass  [5]k_pass [6]ip
#              [7]Etrans [8]k_trans[9]itrans
#              [10]iL (diffusion limit)

PARAM_NAMES = ["Ecorr","icorr","ba","bc",
               "Epass","k_pass","ip",
               "Etrans","k_trans","itrans",
               "iL"]
NP = 11

class CT:
    A  = "A"   # Active only
    AD = "AD"  # Active + diffusion limit cathodic
    P  = "P"   # Active + Passive
    PT = "PT"  # Active + Passive + Transpassive
    F  = "F"   # Full (PT + diffusion)

    INFO = {
        "A":  ("Active",                    [0,1,2,3],             4),
        "AD": ("Active + Diffusion",        [0,1,2,3,10],          5),
        "P":  ("Active–Passive",            [0,1,2,3,4,5,6],       7),
        "PT": ("Active–Passive–Transpassive",[0,1,2,3,4,5,6,7,8,9],10),
        "F":  ("Full (PT+Diffusion)",       list(range(NP)),       11),
    }
    ALL    = ["A","AD","P","PT","F"]
    SIMPLE = ["A","AD"]
    PASS   = ["P","PT","F"]
    TRANS  = ["PT","F"]

    @staticmethod
    def idx(ct):   return CT.INFO.get(ct, CT.INFO["A"])[1]
    @staticmethod
    def nfree(ct): return CT.INFO.get(ct, CT.INFO["A"])[2]
    @staticmethod
    def name(ct):  return CT.INFO.get(ct, ("?", [], 0))[0]

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS MODEL
# ══════════════════════════════════════════════════════════════════════════════

def pol_model(E, p, ct="PT"):
    """
    Full polarisation model (signed output: cathodic < 0, anodic > 0).
    p = [Ecorr, icorr, ba, bc, Epass, k_pass, ip, Etrans, k_trans, itrans, iL]
    """
    E    = np.asarray(E, float)
    Ec   = p[0]; ic = p[1]; ba = max(p[2], 1e-6); bc = max(p[3], 1e-6)
    Ep   = p[4]; kp = max(p[5], 0.001); ip = p[6]
    Et   = p[7]; kt = max(p[8], 0.001); it = p[9]
    iL   = max(p[10], 1e-30)

    eta  = E - Ec

    # Cathodic partial — with optional diffusion limit
    ik_cat = ic * np.exp(np.clip(-2.303 * eta / bc, -60, 60))
    if ct in ("AD", "F"):
        i_cat = ik_cat / (1.0 + ik_cat / iL)
    else:
        i_cat = ik_cat

    # Anodic: active Tafel
    i_act = ic * np.exp(np.clip(2.303 * eta / ba, -60, 60))

    if ct in CT.SIMPLE:
        return i_act - i_cat

    # Active → passive transition
    w_p   = sig(E - Ep, 1.0 / kp)
    i_ano = (1.0 - w_p) * i_act + w_p * ip

    if ct == "P":
        return i_ano - i_cat

    # Passive → transpassive
    w_t   = sig(E - Et, 1.0 / kt)
    i_tp  = ip + it * np.exp(np.clip(2.303 * (E - Et) / ba, -60, 60))
    i_ano = (1.0 - w_t) * i_ano + w_t * i_tp

    return i_ano - i_cat

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — E_corr DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_ecorr(E, i):
    """
    Robust Ecorr detection — most cathodic cathodic→anodic crossing
    with linear interpolation; fallback to minimum |i|.
    """
    E = np.asarray(E, float); i = np.asarray(i, float)
    si = np.argsort(E); Es = E[si]; is_ = i[si]
    sc = np.where(np.diff(np.sign(is_)))[0]
    if len(sc) == 0:
        idx = int(np.argmin(np.abs(is_)))
        return float(Es[idx]), int(si[idx])

    crossings = []
    for k in sc:
        denom = is_[k+1] - is_[k]
        if abs(denom) < TINY: continue
        Ec = float(Es[k] - is_[k] * (Es[k+1] - Es[k]) / denom)
        goes_anodic = (is_[k] < 0 and is_[k+1] > 0)
        crossings.append((Ec, int(si[k]), goes_anodic))

    if not crossings:
        idx = int(np.argmin(np.abs(is_)))
        return float(Es[idx]), int(si[idx])

    anodic = [(Ec, idx) for Ec, idx, ga in crossings if ga]
    if anodic:
        return min(anodic, key=lambda x: x[0])
    best = min(crossings, key=lambda x: x[0])
    return best[0], best[1]

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CATHODIC BRANCH FIT (FAST)
# ══════════════════════════════════════════════════════════════════════════════

def fit_cathodic(E, i, Ecorr):
    """
    Vectorized sliding regression on log|i| vs E for cathodic branch.
    Scores windows by R2 × proximity to Ecorr; returns bc, icorr, iL, has_diff.
    """
    cat = i < 0
    if np.sum(cat) < 4:
        return dict(bc=0.120, icorr=1e-8, iL=1e-2, has_diff=False, r2=0.0)

    Ec  = E[cat]; lgi = np.log10(np.maximum(np.abs(i[cat]), TINY))
    si  = np.argsort(Ec); Ec, lgi = Ec[si], lgi[si]

    # Guard near Ecorr
    TAFEL_GUARD = 0.020
    mask = Ec < (Ecorr - TAFEL_GUARD)
    if np.sum(mask) < 4:
        mask = np.ones_like(Ec, bool)

    Ex = Ec[mask]; Yx = lgi[mask]
    s_idx, e_idx, slope, intercept, R2 = _sliding_regress_full(Ex, Yx, min_len=4, max_len=25)
    if len(slope) == 0:
        sl, b, r2_best = -3.0, -8.0, 0.0
    else:
        invm = np.where(np.abs(slope) > 1e-12, 1.0/np.abs(slope), np.inf)
        ok   = (slope < 0) & (invm > 0.02) & (invm < 0.35) & (R2 > 0.90)
        if not np.any(ok):
            ok = np.ones_like(R2, bool)

        # Prefer windows ending closest to Ecorr
        E_end = Ex[e_idx - 1]
        clos  = np.exp(-np.abs(E_end - Ecorr)/0.20)
        score = R2 * clos
        score[~ok] *= 0.2

        k_best = int(np.argmax(score))
        sl, b, r2_best = float(slope[k_best]), float(intercept[k_best]), float(R2[k_best])

    bc = min(abs(1.0 / sl), 0.400) if abs(sl) > 1e-6 else 0.120
    icorr_tafel = float(10 ** (b + sl * Ecorr)) if abs(sl) > 1e-6 else 1e-12

    # Robust icorr using near-Ecorr window
    near = np.abs(Ec - Ecorr) < 0.050
    if np.any(near):
        icorr_near = float(np.percentile(np.abs(i[cat][si][near]), 10))
    else:
        icorr_near = icorr_tafel
    icorr = max(min(icorr_tafel, icorr_near * 10), 1e-15)

    # Diffusion limit detection (light smoothing)
    iL, has_diff = None, False
    if len(Ec) > 8:
        lgi_sm  = savgol_filter(lgi, max(5, min(9, len(Ec)//2*2-1)), 3, mode="interp")
        dlg     = np.abs(np.gradient(lgi_sm, Ec))
        flat    = dlg < max(np.percentile(dlg, 30), 0.5)
        runs    = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            idxs = [s[0] for s in max(runs, key=lambda x: len(x[1]))[1]]
            if len(idxs) >= 3 and abs(Ec[idxs[-1]] - Ec[idxs[0]]) > 0.03:
                iL = float(np.median(np.abs(i[cat][si][idxs])))
                has_diff = True
    if iL is None:
        iL = icorr * 1e4

    return dict(bc=bc, icorr=icorr, iL=iL, has_diff=has_diff,
                r2=r2_best, E_cat=Ec, lgi_cat=lgi)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANODIC BRANCH FIT (FAST)
# ══════════════════════════════════════════════════════════════════════════════

def fit_anodic(E, i, Ecorr):
    """
    Vectorized sliding regression for anodic active Tafel; passive/transpassive detection.
    """
    ano = i > 0
    if np.sum(ano) < 4:
        return dict(ba=0.060, has_passive=False, Epass=None, ip=1e-6,
                    has_trans=False, Etrans=None, r2=0.0)

    Ea  = E[ano]; lgia = np.log10(np.maximum(np.abs(i[ano]), TINY))
    si  = np.argsort(Ea); Ea, lgia = Ea[si], lgia[si]

    TAFEL_GUARD_A = 0.010
    mask = Ea > (Ecorr + TAFEL_GUARD_A)
    if np.sum(mask) < 4:
        mask = np.ones_like(Ea, bool)

    Ex = Ea[mask]; Yx = lgia[mask]
    s_idx, e_idx, slope, intercept, R2 = _sliding_regress_full(Ex, Yx, min_len=3, max_len=20)
    if len(slope) == 0:
        sl, r2_best = 25.0, 0.0
    else:
        invm = np.where(np.abs(slope) > 1e-12, 1.0/np.abs(slope), np.inf)
        ok   = (slope > 0) & (invm > 0.01) & (invm < 0.40) & (R2 > 0.85)
        if not np.any(ok):
            ok = np.ones_like(R2, bool)

        E_start = Ex[s_idx]
        clos    = np.exp(-np.abs(E_start - Ecorr)/0.10)
        score   = R2 * clos
        score[~ok] *= 0.2

        k_best  = int(np.argmax(score))
        sl, r2_best = float(slope[k_best]), float(R2[k_best])

    ba = min(abs(1.0 / sl), 0.250) if abs(sl) > 1e-6 else 0.040

    # Active peak detection
    Epeak_detected = None
    if len(Ea) > 4:
        pks, _ = find_peaks(lgia, prominence=0.3, distance=2)
        if len(pks) > 0:
            pk = int(np.argmin(np.abs(Ea[pks] - Ecorr)))
            Epeak_detected = float(Ea[pks[pk]])

    # Passive / Transpassive detection
    has_passive = False; Epass = None; ip = 1e-6
    has_trans   = False; Etrans = None
    if len(Ea) > 8:
        lgia_sm = savgol_filter(lgia, max(5, min(11, len(Ea)//2*2-1)), 3, mode="interp")
        dlg     = np.gradient(lgia_sm, Ea)
        adlg    = np.abs(dlg)
        thr  = max(np.percentile(adlg, 30), 0.8)
        flat = adlg < thr

        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        for _, ri in runs:
            idxs = [s[0] for s in ri]
            span = abs(Ea[idxs[-1]] - Ea[idxs[0]])
            if len(idxs) >= 4 and span > 0.06:
                ip_cand = float(np.median(np.abs(i[ano][si][idxs])))
                if ip_cand < float(np.max(np.abs(i[ano]))) * 0.7:
                    has_passive = True
                    Epass       = float(Ea[idxs[0]])
                    ip          = ip_cand
                    post = Ea > Ea[idxs[-1]]
                    if np.sum(post) > 3:
                        post_dlg = dlg[np.where(post)[0]]
                        post_E   = Ea[np.where(post)[0]]
                        rising   = np.where(post_dlg > 3.0)[0]
                        if len(rising) > 0:
                            has_trans = True
                            Etrans    = float(post_E[rising[0]])
                    break

    return dict(ba=ba, has_passive=has_passive, Epass=Epass, ip=ip,
                has_trans=has_trans, Etrans=Etrans, Epeak=Epeak_detected,
                r2=float(r2_best), E_an=Ea, lgi_an=lgia)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — CLASSIFY CURVE TYPE
# ══════════════════════════════════════════════════════════════════════════════

def classify_curve(cat_res, an_res):
    hp = an_res["has_passive"]
    ht = an_res["has_trans"]
    hd = cat_res["has_diff"]
    if hp and ht:  return CT.F  if hd else CT.PT
    if hp:         return CT.P
    if hd:         return CT.AD
    return CT.A

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ASSEMBLE P0 & GLOBAL POLISH (LEANER)
# ══════════════════════════════════════════════════════════════════════════════

def _make_p0(Ecorr, cat, an, ct, E_max):
    ic  = cat["icorr"]
    bc  = cat["bc"]
    ba  = an["ba"]
    iL  = cat["iL"]

    if an.get("Epeak") is not None:
        Ep = an["Epeak"]
    elif an["has_passive"] and an["Epass"] is not None:
        Ep = an["Epass"]
    else:
        Ep = E_max + 5.0

    ip = an["ip"] if an["has_passive"] else ic * 0.01

    Et = an["Etrans"] if an["has_trans"] else E_max + 5.0
    it = ip * 0.5    if an["has_trans"] else ic * 0.001

    kpass_p0 = 0.010
    return np.array([Ecorr, ic, ba, bc, Ep, kpass_p0, ip, Et, 0.015, it, iL])

def _build_bounds(Ecorr, cat, an, ct, E_min, E_max, E_span):
    ic   = max(cat["icorr"], 1e-14)
    iL   = max(cat["iL"],    1e-10)
    ba_fit = float(an["ba"])
    bc_fit = float(cat["bc"])
    ba_lo  = max(ba_fit * 0.30, 0.010); ba_hi  = min(ba_fit * 3.00, 0.250)
    bc_lo  = max(bc_fit * 0.30, 0.010); bc_hi  = min(bc_fit * 3.00, 0.400)

    lo = np.array([
        E_min,                  # Ecorr
        max(ic * 1e-5, 1e-15),  # icorr
        ba_lo,                  # ba
        bc_lo,                  # bc
        Ecorr + 0.005,          # Epass
        0.002,                  # k_pass
        max(ic * 1e-6, 1e-16),  # ip
        Ecorr + 0.05*E_span,    # Etrans
        0.002,                  # k_trans
        max(ic * 1e-7, 1e-16),  # itrans
        max(ic * 0.5, 1e-12),   # iL
    ])
    hi = np.array([
        E_max,
        min(ic * 1e6, 1.0),
        ba_hi,
        bc_hi,
        E_max,
        0.120,
        min(ic * 1e5, 1.0),
        E_max + 0.1,
        0.120,
        min(ic * 1e7, 10.0),
        min(iL * 1000, 1.0),
    ])
    lo = np.minimum(lo, hi - 1e-12)
    return lo, hi

LOG_IDX = {1, 6, 9, 10}

def _pack(p, fidx):
    return np.array([np.log10(max(p[j], TINY)) if j in LOG_IDX else p[j]
                     for j in fidx])

def _unpack(x, fidx, p_base, lo=None, hi=None):
    p = p_base.copy()
    for k, j in enumerate(fidx):
        val = 10.0**x[k] if j in LOG_IDX else x[k]
        if lo is not None:
            val = float(np.clip(val, lo[j], hi[j]))
        p[j] = val
    return p

def _pbounds(lo, hi, fidx):
    return [(np.log10(max(lo[j], TINY)), np.log10(max(hi[j], TINY)))
            if j in LOG_IDX else (lo[j], hi[j])
            for j in fidx]

def global_polish(E, i, p0, ct, lo, hi):
    """
    Leaner 3-stage optimization with early exit:
    DE (lighter) → L-BFGS-B; if needed Powell.
    Weighted error near Ecorr.
    """
    ld    = np.log10(np.maximum(np.abs(i), TINY))
    fidx  = CT.idx(ct)
    bnds  = _pbounds(lo, hi, fidx)
    n, nf = len(E), len(fidx)

    Ecorr_p0 = float(p0[0])
    w_base   = 1.0 + 7.0 * np.exp(-np.abs(E - Ecorr_p0) / 0.120)
    w_base  /= w_base.mean()

    def obj(x):
        p = _unpack(x, fidx, p0.copy(), lo, hi)
        pred = pol_model(E, p, ct)
        return float(np.sum(w_base * (ld - np.log10(np.maximum(np.abs(pred), TINY))) ** 2))

    best_x   = _pack(p0, fidx)
    best_val = obj(best_x)

    def update(x):
        nonlocal best_x, best_val
        v = obj(x)
        if v < best_val - 1e-12:
            best_x, best_val = x.copy(), v
        return v

    # Stage 1: Differential Evolution (lighter config)
    try:
        ps  = max(12, min(16, nf * 3))
        mi  = max(150, min(400, nf * 50))
        res = differential_evolution(
            obj, bnds, seed=42, maxiter=mi, popsize=ps,
            tol=1e-9, mutation=(0.5, 1.7), recombination=0.85,
            polish=False, workers=1, strategy="best1bin"
        )
        update(res.x)
    except Exception:
        pass

    # Stage 2: L-BFGS-B (main polish)
    try:
        r = minimize(obj, best_x, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": 15000, "ftol": 1e-12, "gtol": 1e-10})
        update(r.x)
    except Exception:
        pass

    # Early-exit if excellent
    p_tmp = _unpack(best_x, fidx, p0.copy(), lo, hi)
    pred  = pol_model(E, p_tmp, ct)
    r2_now = r2_score(ld, np.log10(np.maximum(np.abs(pred), TINY)))
    if r2_now < 0.995:
        # Stage 3: Powell only if needed
        try:
            r = minimize(obj, best_x, method="Powell",
                         options={"maxiter": 8000, "xtol": 1e-12, "ftol": 1e-12})
            update(r.x)
        except Exception:
            pass

    best_p = _unpack(best_x, fidx, p0.copy(), lo, hi)
    pred   = pol_model(E, best_p, ct)
    log_p  = np.log10(np.maximum(np.abs(pred), TINY))
    sse    = float(np.sum((ld - log_p) ** 2))
    r2     = r2_score(ld, log_p)
    aic    = aicc(n, nf, sse)

    return best_p, r2, aic, sse

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

COL_SIG = [
    (r"we.*potential|ewe|potential/v|e/v|^e$|e \(v\)|e_v|^vf$",
     r"we.*current|<i>/ma|i/ma|current/a|i/a|^i$|i \(a\)|i_a|^im$",  "A"),
    (r"potential|volt|^e$",
     r"current.*ma|ima",  "mA"),
]
UNIT_PAT = {r"\(a\)|_a$|/a$|a/cm²?": 1.0,
            r"\(ma\)|_ma$|/ma$|ma/cm²?": 1e-3,
            r"\(ua\)|_ua$|/ua$|ua/cm²?": 1e-6}

def _auto_cols(df):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cl  = {c: c.lower().strip() for c in df.columns}
    for Ep, Ip, uf in COL_SIG:
        em = [c for c, v in cl.items() if re.search(Ep, v) and c in num]
        im = [c for c, v in cl.items() if re.search(Ip, v) and c in num and c not in em]
        if em and im:
            ec = sorted(em, key=lambda c: 0 if "we" in cl[c] else 1)[0]
            ic = im[0]
            f  = 1e-3 if uf == "mA" else 1.0
            for pat, fv in UNIT_PAT.items():
                if re.search(pat, cl[ic]): f = fv; break
            return ec, ic, f
    if len(num) >= 2:
        return num[0], num[1], 1.0
    raise ValueError("Cannot detect E/i columns automatically.")

def load_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    content = uploaded.getvalue().decode("utf-8", errors="replace")
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep,
                             engine="python", comment="#")
            if df.shape[1] >= 2 and df.shape[0] > 4:
                return df.dropna(axis=1, how="all")
        except:
            continue
    raise ValueError("Cannot parse file.")

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION FIGURE (ADAPTIVE & FAST)
# ══════════════════════════════════════════════════════════════════════════════

PLT_RC = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 8, "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc", "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "#fafbff",
}

def make_figure(E, i_obs, best_p, ct, sample_name,
                cat_res, an_res, Ecorr, show_regions=True, dpi=150):
    """
    Publication-quality 4-panel figure. X=E (V), Y=log10|i| with adaptive dense sampling,
    rasterized scatters, and robust y-limits.
    """
    Ecorr = float(best_p[0])   # use fitted Ecorr

    ba    = max(float(best_p[2]), 1e-9)
    bc    = max(float(best_p[3]), 1e-9)
    icorr = float(best_p[1])
    logIc = float(np.log10(max(icorr, TINY)))

    E_lo, E_hi = float(E.min()), float(E.max())
    span = max(E_hi - E_lo, 1e-6)
    n_dense = int(np.clip(200 * span, 800, 2500))
    E_dense = np.linspace(E_lo, E_hi, n_dense)

    i_dense    = pol_model(E_dense, best_p, ct)
    i_fit_E    = pol_model(E, best_p, ct)
    log_obs    = np.log10(np.maximum(np.abs(i_obs), TINY))
    log_den    = np.log10(np.maximum(np.abs(i_dense), TINY))
    log_fitE   = np.log10(np.maximum(np.abs(i_fit_E), TINY))
    residuals  = log_obs - log_fitE
    r2v  = r2_score(log_obs, log_fitE)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    fin  = log_obs[np.isfinite(log_obs)]
    y_lo = float(np.nanmin(fin)) - 0.2

    if ct in CT.PASS:
        Epass_4y = float(best_p[4])
        act_mask = (E >= Ecorr - 0.02) & (E <= Epass_4y + 0.05)
        if np.sum(act_mask) >= 2:
            log_act = slog(i_obs[act_mask])
            log_act = log_act[np.isfinite(log_act)]
            active_peak_log = float(np.max(log_act)) if len(log_act) > 0 else logIc
        else:
            active_peak_log = logIc + 2.303 * (Epass_4y - Ecorr) / ba
        logIp_val = float(np.log10(max(float(best_p[6]), TINY)))
        y_hi = max(active_peak_log + 1.5, logIp_val + 2.0)
    else:
        y_hi = float(np.percentile(fin, 90)) + 1.0

    # tighten upper limit to avoid extreme compression by outliers
    y_hi = min(y_hi, float(np.percentile(fin, 99)) + 1.8)

    # Partial current Tafel lines
    logI_cat = np.log10(np.clip(icorr * np.exp(2.303*(Ecorr-E_dense)/bc), TINY, None))
    logI_ano = np.log10(np.clip(icorr * np.exp(2.303*(E_dense-Ecorr)/ba), TINY, None))

    msk_cat = (logI_cat >= y_lo - 0.05) & (logI_cat <= y_hi + 0.05) & (E_dense <= Ecorr + 0.005)

    if ct in CT.PASS:
        Epass_fit = float(best_p[4])
        msk_ano_E = (E_dense >= Ecorr) & (E_dense <= Epass_fit)
    else:
        msk_ano_E = E_dense >= Ecorr
    msk_ano = msk_ano_E & (logI_ano >= y_lo - 0.05) & (logI_ano <= y_hi + 0.05)

    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        gs  = GridSpec(2, 3, figure=fig,
                       hspace=0.44, wspace=0.36,
                       left=0.07, right=0.97, top=0.93, bottom=0.08)
        ax_ev  = fig.add_subplot(gs[0, :])
        ax_br  = fig.add_subplot(gs[1, 0])
        ax_lin = fig.add_subplot(gs[1, 1])
        ax_res = fig.add_subplot(gs[1, 2])

        # ══ PANEL A — Evans Diagram ═══════════════════════════════════════════
        ax = ax_ev

        if show_regions:
            def vband(e0, e1, key, lbl):
                c, a = REGION_COLORS[key]
                e0c = float(np.clip(e0, E_lo, E_hi))
                e1c = float(np.clip(e1, E_lo, E_hi))
                if e1c > e0c:
                    ax.axvspan(e0c, e1c, color=c, alpha=a, lw=0, label=lbl, zorder=1)
            vband(E_lo, Ecorr, "cathodic", "Cathodic region")
            if ct in CT.SIMPLE:
                vband(Ecorr, E_hi, "active", "Anodic (active)")
            elif ct in CT.PASS:
                Ep = float(best_p[4])
                Et = float(best_p[7]) if ct in CT.TRANS else E_hi + 1
                vband(Ecorr, min(Ep, E_hi), "active", "Active dissolution")
                vband(min(Ep, E_hi), min(Et, E_hi), "passive", "Passive region")
                if ct in CT.TRANS and Et < E_hi:
                    vband(Et, E_hi, "transpassive", "Transpassive / pitting")

        ax.scatter(E, log_obs, s=12, color="#4a7fa8", alpha=0.60,
                   zorder=2, label="Experimental data", linewidths=0, rasterized=True)
        ax.plot(E_dense, log_den, color="#1a3a5c", lw=2.0, zorder=5,
                label=f"Global fit  (R\u00b2={r2v:.5f})")

        if msk_cat.any():
            ax.plot(E_dense[msk_cat], logI_cat[msk_cat],
                    "--", color="#8e44ad", lw=2.2, zorder=6,
                    label=f"\u03b2c = {bc*1000:.0f} mV/dec")

        if msk_ano.any():
            ax.plot(E_dense[msk_ano], logI_ano[msk_ano],
                    "--", color="#e67e22", lw=2.2, zorder=6,
                    label=f"\u03b2a = {ba*1000:.0f} mV/dec")

        # Crossing point + drop-lines
        ax.plot(Ecorr, logIc, "x", color="#e84393", ms=12, mew=2.5, zorder=8)
        ax.plot([Ecorr, Ecorr], [y_lo, logIc],
                ":", color="#e84393", lw=1.2, alpha=0.85, zorder=3)
        ax.plot([E_lo, Ecorr], [logIc, logIc],
                ":", color="#e84393", lw=1.2, alpha=0.85, zorder=3)

        # Annotations
        y_span = y_hi - y_lo
        ax.annotate(f"E\u1d9c\u1d52\u02b3\u02b3 = {Ecorr:.4f} V",
                    xy=(Ecorr, y_lo),
                    xytext=(Ecorr + 0.01*(E_hi-E_lo), y_lo + 0.04*y_span),
                    fontsize=8.5, color="#e84393", fontweight="bold", ha="left")
        ax.annotate(f"i\u1d9c\u1d52\u02b3\u02b3 = {icorr:.2e} A/cm\u00b2",
                    xy=(E_lo, logIc),
                    xytext=(E_lo + 0.01*(E_hi-E_lo), logIc + 0.03*y_span),
                    fontsize=8.5, color="#e84393", fontweight="bold")

        if ct in CT.PASS:
            ip_val = float(best_p[6])
            ax.axhline(np.log10(max(ip_val, TINY)), color="#27ae60",
                       ls=":", lw=1.1, alpha=0.75, zorder=3,
                       label=f"i_pass = {ip_val:.2e} A/cm\u00b2")

        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("E vs. Reference (V)", fontsize=10)
        ax.set_ylabel("log\u2081\u2080 |i| (A cm\u207b\u00b2)", fontsize=10)
        ax.set_title(f"Evans Diagram \u2014 {sample_name}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.45)
        ax.grid(True, which="minor", ls=":", alpha=0.18)
        ax.legend(loc="lower right", ncol=4, fontsize=7.5,
                  framealpha=0.95, edgecolor="#cccccc")
        r2c = "#27ae60" if r2v > 0.99 else "#e67e22" if r2v > 0.95 else "#e84393"
        ax.text(0.01, 0.97,
                f"R\u00b2={r2v:.5f}  RMSE={rmse:.4f}  Model: {CT.name(ct)}",
                transform=ax.transAxes, fontsize=8.5, color=r2c,
                fontweight="bold", va="top",
                bbox=dict(fc="white", ec=r2c, alpha=0.88, pad=3,
                          boxstyle="round,pad=0.3"))

        # ══ PANEL B — Branch Fits ═════════════════════════════════════════════
        ax = ax_br
        E_ds, log_obs_ds = downsample_uniform(E, log_obs, 400)
        ax.scatter(E_ds, log_obs_ds, s=5, color="#aab4c4", alpha=0.28, zorder=1, linewidths=0, rasterized=True)
        if "E_cat" in cat_res:
            ax.scatter(cat_res["E_cat"], cat_res["lgi_cat"],
                       s=18, color="#6baed6", alpha=0.80, zorder=3,
                       label="Cathodic data", linewidths=0, rasterized=True)
            msk_c_br = msk_cat & (E_dense <= Ecorr)
            if msk_c_br.any():
                ax.plot(E_dense[msk_c_br], logI_cat[msk_c_br],
                        "--", color="#3182bd", lw=1.6, zorder=4,
                        label=f"\u03b2c={bc*1000:.0f} mV/dec")
        if "E_an" in an_res:
            ax.scatter(an_res["E_an"], an_res["lgi_an"],
                       s=18, color="#fd8d3c", alpha=0.80, zorder=3,
                       label="Anodic data", linewidths=0, rasterized=True)
            if msk_ano.any():
                ax.plot(E_dense[msk_ano], logI_ano[msk_ano],
                        "--", color="#e6550d", lw=1.6, zorder=4,
                        label=f"\u03b2a={ba*1000:.0f} mV/dec")
            if an_res["has_passive"] and an_res["Epass"] is not None:
                ax.axvline(float(an_res["Epass"]), color="#27ae60",
                           ls="-.", lw=1.0, alpha=0.85,
                           label=f"E_pass={an_res['Epass']:.3f}V")
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.axhline(logIc, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("E (V)", fontsize=9)
        ax.set_ylabel("log\u2081\u2080 |i|", fontsize=9)
        ax.set_title("Branch Fits (Stage 2\u20133)", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=7.5, loc="upper right")

        # ══ PANEL C — Linear i vs E ═══════════════════════════════════════════
        ax = ax_lin
        i_p95 = float(np.percentile(np.abs(i_obs), 95))
        if i_p95 < 1e-6:
            uscale, ulbl = 1e9,  "nA/cm\u00b2"
        elif i_p95 < 1e-3:
            uscale, ulbl = 1e6,  "\u03bcA/cm\u00b2"
        else:
            uscale, ulbl = 1e3,  "mA/cm\u00b2"
        ylim_lin = max(i_p95 * uscale * 1.20, 1e-12)
        E_lin_ds, i_obs_lin_ds = downsample_uniform(E, i_obs, 800)
        i_fit_cl = np.clip(i_dense * uscale, -ylim_lin * 3, ylim_lin * 3)
        ax.scatter(E_lin_ds, i_obs_lin_ds * uscale, s=9, color="#4a7fa8",
                   alpha=0.65, zorder=2, label="Data", linewidths=0, rasterized=True)
        ax.plot(E_dense, i_fit_cl, color="#1a3a5c", lw=2.0, zorder=5, label="Fit")
        ax.axhline(0, color="#888", lw=0.7, zorder=1)
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(-ylim_lin, ylim_lin)
        ax.set_xlabel("E (V)", fontsize=9)
        ax.set_ylabel(f"i ({ulbl})", fontsize=9)
        ax.set_title("Linear Scale", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        # ══ PANEL D — Residuals ═══════════════════════════════════════════════
        ax = ax_res
        ax.fill_between([E_lo, E_hi], -0.1, 0.1, color="#e84393", alpha=0.07, zorder=1)
        ax.scatter(E, residuals, s=10, color="#2e86de", alpha=0.65, zorder=3, linewidths=0, rasterized=True)
        ax.axhline(0,    color="#333",    lw=0.9, zorder=2)
        ax.axhline( 0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(-0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7, label="\u00b10.1 log")
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlim(E_lo, E_hi)
        ax.set_xlabel("E (V)", fontsize=9)
        ax.set_ylabel("\u0394 log\u2081\u2080 |i|", fontsize=9)
        ax.set_title(f"Residuals   R\u00b2={r2v:.5f}", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        fig.suptitle("Polarisation Curve Analysis", fontsize=12,
                     fontweight="bold", color="#1a3a5c", y=0.98)

    return fig, r2v, rmse

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_excel(results_list):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Results"
    H_FILL = PatternFill("solid", fgColor="1A3A5C")
    H_FONT = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    BRD    = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"),  bottom=Side(style="thin"))
    ALT    = PatternFill("solid", fgColor="EEF2FF")
    GRN    = Font(color="1E8449", bold=True, name="Arial", size=10)
    RED    = Font(color="C0392B", bold=True, name="Arial", size=10)

    hdrs = ["Sample","Model","E_corr (V)","i_corr (A/cm²)",
            "βa (mV/dec)","βc (mV/dec)","B (V)",
            "CR (mm/yr)","i_pass (A/cm²)","E_pass (V)",
            "E_trans (V)","R²","RMSE (log)","Status"]
    for c, h in enumerate(hdrs, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = H_FILL; cell.font = H_FONT; cell.border = BRD
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for ri, res in enumerate(results_list, 2):
        p  = res["params"]; ct = res["ct"]
        ok = res.get("r2", 0) > 0.95
        fill = ALT if ri % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        B_val = (p[2]*p[3]) / (2.303*(p[2]+p[3])) if p[2]>0 and p[3]>0 else 0
        ew, rho = res.get("material", (27.92, 7.87))
        CR = p[1] * 3.27 * ew / rho
        vals = [
            res.get("name","?"),
            CT.name(ct),
            round(p[0], 5),
            f"{p[1]:.4e}",
            round(p[2]*1000, 2),
            round(p[3]*1000, 2),
            round(B_val, 5),
            round(CR, 5),
            f"{p[6]:.3e}" if ct in CT.PASS else "—",
            round(p[4], 5) if ct in CT.PASS else "—",
            round(p[7], 5) if ct in CT.TRANS else "—",
            round(res.get("r2", 0), 6),
            round(res.get("rmse", 0), 6),
            "Good" if ok else "Check",
        ]
        for c, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=c, value=val)
            cell.fill = fill; cell.border = BRD
            cell.alignment = Alignment(horizontal="center")
            if c == 14: cell.font = GRN if ok else RED

    for col in ws.columns:
        w = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(w+3, 22)
    ws.freeze_panes = "A2"
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf

def export_pdf(results_list, png_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                         textColor=rl_colors.HexColor("#1A3A5C"), spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                          textColor=rl_colors.HexColor("#2e86de"),
                          spaceBefore=10, spaceAfter=3)
    bs = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=14)

    tbl_s = TableStyle([
        ("BACKGROUND", (0,0),(-1,0), rl_colors.HexColor("#1A3A5C")),
        ("TEXTCOLOR",  (0,0),(-1,0), rl_colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [rl_colors.HexColor("#EEF2FF"), rl_colors.white]),
        ("GRID",       (0,0),(-1,-1), 0.4, rl_colors.HexColor("#BBBBBB")),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",      (2,1),(-1,-1), "RIGHT"),
        ("TOPPADDING", (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ])

    story = []
    story.append(Paragraph("Polarisation Curve Analysis Report", ts))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=rl_colors.HexColor("#2e86de"), spaceAfter=4))
    story.append(Paragraph(
        f"Polarization Curve Fitter  |  "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", bs))
    story.append(Spacer(1, 0.5*cm))

    for idx, (res, png) in enumerate(zip(results_list, png_list)):
        p  = res["params"]; ct = res["ct"]
        nm = res.get("name", f"Sample {idx+1}")
        B_val = (p[2]*p[3])/(2.303*(p[2]+p[3])) if p[2]>0 and p[3]>0 else 0
        ew, rho = res.get("material", (27.92, 7.87))
        CR = p[1] * 3.27 * ew / rho

        story.append(Paragraph(f"Sample {idx+1}: {nm}", h2))
        rows = [["Parameter","Symbol","Value","Unit"],
                ["Corrosion potential","E_corr",f"{p[0]:.5f}","V"],
                ["Corrosion current density","i_corr",f"{p[1]:.4e}","A cm-2"],
                ["Anodic Tafel slope","ba",f"{p[2]*1000:.2f}","mV dec-1"],
                ["Cathodic Tafel slope","bc",f"{p[3]*1000:.2f}","mV dec-1"],
                ["Stern-Geary constant","B",f"{B_val:.5f}","V"],
                ["Corrosion rate","CR",f"{CR:.5f}","mm yr-1"],
                ]
        if ct in CT.PASS:
            rows += [["Passive current density","i_pass",f"{p[6]:.4e}","A cm-2"],
                     ["Passivation potential","E_pass",f"{p[4]:.5f}","V"]]
        if ct in CT.TRANS:
            rows.append(["Transpassive potential","E_trans",f"{p[7]:.5f}","V"])
        rows += [["R² (log-domain)","R2",f"{res.get('r2',0):.6f}","—"],
                 ["RMSE (log-domain)","RMSE",f"{res.get('rmse',0):.6f}","log-units"],
                 ["Model","—",CT.name(ct),"—"],
                 ["Fit status","—","Converged" if res.get("success") else "Check","—"]]
        tbl = Table(rows, colWidths=[6*cm, 2.2*cm, 3.2*cm, 2.6*cm])
        tbl.setStyle(tbl_s)
        story.append(KeepTogether([tbl, Spacer(1, 0.3*cm)]))
        if png:
            story.append(RLImage(io.BytesIO(png), width=15.5*cm, height=11.0*cm))
        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=rl_colors.HexColor("#CCCCCC"), spaceAfter=4))

    doc.build(story)
    buf.seek(0); return buf

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k in ("results", "figures"):
    if _k not in st.session_state:
        st.session_state[_k] = []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    st.markdown("**Material**")
    material = st.selectbox("Material (for CR calculation)",
                            list(MATERIALS.keys()), index=0)
    ew_mat, rho_mat = MATERIALS[material]

    st.markdown("**Data Import**")
    skip_rows  = st.number_input("Skip header rows", 0, 30, 0)
    delimiter  = st.selectbox("CSV delimiter", ["auto",",",";","\t"," "])
    e_col_name = st.text_input("E column (blank = auto)", "")
    i_col_name = st.text_input("i column (blank = auto)", "")
    i_unit     = st.selectbox("Current unit in file",
                              ["A/cm²","mA/cm²","µA/cm²","A/m²"])
    unit_fac   = {"A/cm²":1.0,"mA/cm²":1e-3,"µA/cm²":1e-6,"A/m²":1e-4}[i_unit]
    area       = st.number_input("Electrode area (cm²)", 0.001, 10000.0, 1.0, format="%.4f")

    st.markdown("**Fitting**")
    force_ct   = st.selectbox("Force model (auto = best AICc)",
                              ["auto","A","AD","P","PT","F"])
    show_regs  = st.toggle("Shade regions", True)
    smooth_pre = st.toggle("Pre-smooth (Savitzky-Golay)", False)
    pub_dpi    = st.slider("Export DPI", 150, 600, 300, 50)

    st.divider()
    if st.button("🗑 Clear all", use_container_width=True):
        st.session_state.results = []
        st.session_state.figures = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">⚡ Polarization Curve Fitter</div>',
            unsafe_allow_html=True)

tab_fit, tab_res, tab_cmp, tab_help = st.tabs(
    ["📂 Upload & Fit", "📊 Results & Export", "📋 Compare", "ℹ️ Help"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD & FIT
# ─────────────────────────────────────────────────────────────────────────────
with tab_fit:
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("### 📁 Upload Data")
        uploaded_files = st.file_uploader(
            "CSV / TXT / XLSX  (signed current: cathodic < 0, anodic > 0)",
            type=["csv","txt","xlsx","xls"],
            accept_multiple_files=True)
    with c2:
        st.markdown("### 🏷️ Sample")
        sample_name = st.text_input("Sample label", "Sample 1")

    if uploaded_files:
        for uf in uploaded_files:
            st.markdown(f"---\n#### 📄 `{uf.name}`")
            with st.container():

                # Load
                try:
                    df_raw = load_file(uf)
                except Exception as ex:
                    st.error(f"Load error: {ex}"); continue

                # Column selection
                try:
                    ec_auto, ic_auto, fac_auto = _auto_cols(df_raw)
                    auto_ok = True
                except:
                    ec_auto = ic_auto = None; fac_auto = 1.0; auto_ok = False

                num_cols = [c for c in df_raw.columns
                            if pd.api.types.is_numeric_dtype(df_raw[c])]
                cc1, cc2 = st.columns(2)
                with cc1:
                    e_sel = st.selectbox(f"E column [{uf.name}]", num_cols,
                        index=num_cols.index(ec_auto) if auto_ok and ec_auto in num_cols else 0,
                        key=f"ec_{uf.name}")
                with cc2:
                    i_sel = st.selectbox(f"i column [{uf.name}]", num_cols,
                        index=num_cols.index(ic_auto) if auto_ok and ic_auto in num_cols else min(1,len(num_cols)-1),
                        key=f"ic_{uf.name}")

                # Build arrays
                E_raw = df_raw[e_sel].values.astype(float)
                i_raw = df_raw[i_sel].values.astype(float) * unit_fac / area
                ok_mask = np.isfinite(E_raw) & np.isfinite(i_raw)
                E_raw, i_raw = E_raw[ok_mask], i_raw[ok_mask]
                srt = np.argsort(E_raw); E, i = E_raw[srt], i_raw[srt]

                # Preview
                p1, p2 = st.columns([1, 1.6])
                sc = np.where(np.diff(np.sign(i)))[0]
                with p1:
                    st.markdown(f"**{len(E)} pts** | "
                                f"E: [{E.min():.4f}, {E.max():.4f}] V")
                    if len(sc) > 0:
                        Ec_pre = E[sc[0]] - i[sc[0]]*(E[sc[0]+1]-E[sc[0]])/(i[sc[0]+1]-i[sc[0]])
                        st.success(f"✓ Zero-crossing at E ≈ {Ec_pre:.4f} V  "
                                   f"({np.sum(i<0)} cat / {np.sum(i>0)} ano)")
                    else:
                        st.warning("No sign change — verify sign convention")
                    st.dataframe(df_raw.head(5), use_container_width=True, height=160)

                with p2:
                    with plt.rc_context(PLT_RC):
                        fp, ap = plt.subplots(figsize=(6, 3.8))
                        ap.scatter(E, slog(i), s=7, color="#5a7fa8", alpha=0.65, rasterized=True)
                        ap.set_xlabel("E (V)"); ap.set_ylabel("log|i|")
                        ap.set_title("Raw Data Preview", fontsize=10)
                        ap.grid(True, ls="--", alpha=0.4)
                        if len(sc) > 0:
                            ap.axvline(Ec_pre, color="#e84393", ls="--", lw=1,
                                       label=f"E_corr≈{Ec_pre:.4f}V")
                            ap.legend(fontsize=8)
                        fp.tight_layout()
                    st.pyplot(fp, use_container_width=True)
                    plt.close(fp)

                # ── FIT BUTTON ──────────────────────────────────────────────
                if st.button(f"🚀 Run Full Pipeline · {uf.name}",
                             key=f"btn_{uf.name}", type="primary",
                             use_container_width=True):
                    import time; t0 = time.time()

                    if smooth_pre:
                        w = min(9, len(i)//2*2-1)
                        if w >= 5:
                            i = savgol_filter(i, w, 3, mode="interp")

                    # ── Pipeline progress ────────────────────────────────────
                    prog     = st.progress(0, text="Stage 1 — Detecting E_corr…")
                    log_area = st.empty()
                    log_lines = []
                    def log(msg):
                        log_lines.append(msg)
                        log_area.markdown("  \n".join(log_lines))

                    # Stage 1 — E_corr
                    Ecorr, _ = detect_ecorr(E, i)
                    log(f"✅ **Stage 1** — E_corr = `{Ecorr:.5f}` V")
                    prog.progress(15, text="Stage 2 — Cathodic branch fit…")

                    # Stage 2 — Cathodic
                    cat_res = fit_cathodic(E, i, Ecorr)
                    log(f"✅ **Stage 2** — bc = `{cat_res['bc']*1000:.0f}` mV/dec  "
                        f"i_corr = `{cat_res['icorr']:.2e}`  "
                        f"R²_cat = `{cat_res['r2']:.4f}`  "
                        f"diff_limit = `{'yes' if cat_res['has_diff'] else 'no'}`")
                    prog.progress(30, text="Stage 3 — Anodic branch fit…")

                    # Stage 3 — Anodic
                    an_res = fit_anodic(E, i, Ecorr)
                    log(f"✅ **Stage 3** — ba = `{an_res['ba']*1000:.0f}` mV/dec  "
                        f"passive = `{'yes  E_pass=%.4f V' % an_res['Epass'] if an_res['has_passive'] else 'no'}`  "
                        f"transpassive = `{'yes' if an_res['has_trans'] else 'no'}`")
                    prog.progress(45, text="Stage 4 — Classifying curve type…")

                    # Stage 4 — Classify
                    ct_detected = classify_curve(cat_res, an_res)
                    log(f"🔍 **Stage 4** — Detected: **{CT.name(ct_detected)}**  "
                        f"({CT.nfree(ct_detected)} free parameters)")
                    prog.progress(50, text="Stage 5 — Global optimisation…")

                    # Stage 5 — Global optimisation
                    E_lo = float(E.min()); E_hi = float(E.max()); E_sp = E_hi - E_lo

                    # Adaptive candidate set
                    if force_ct != "auto":
                        candidates = [force_ct]
                    else:
                        candidates = [ct_detected]
                        if ct_detected in (CT.A, CT.AD):
                            candidates += [CT.P]  # try passive if plausible
                        if an_res["has_passive"]:
                            candidates += [CT.P, CT.PT]
                        if cat_res["has_diff"]:
                            candidates += [CT.AD]
                        # ensure uniqueness and valid order
                        seen = set(); uniq = []
                        for c in candidates:
                            if c not in seen:
                                uniq.append(c); seen.add(c)
                        candidates = uniq

                    all_res = []
                    n_cand  = len(candidates)
                    for k_c, ct_try in enumerate(candidates):
                        prog.progress(50 + int(40 * k_c / max(n_cand,1)),
                                      text=f"Optimising: {CT.name(ct_try)}…")
                        p0 = _make_p0(Ecorr, cat_res, an_res, ct_try, E_hi)
                        lo, hi = _build_bounds(Ecorr, cat_res, an_res,
                                               ct_try, E_lo, E_hi, E_sp)
                        p0 = np.clip(p0, lo, hi)
                        bp, r2v, aic_v, sse = global_polish(E, i, p0, ct_try, lo, hi)
                        all_res.append(dict(ct=ct_try, r2=r2v, aicc=aic_v,
                                            params=bp, success=r2v > 0.90))
                        log(f"  · {CT.name(ct_try):35s} R² = `{r2v:.6f}`  AICc = `{aic_v:.1f}`")

                    # AICc selection — parsimony: prefer simpler if ΔR² < 0.002
                    all_res.sort(key=lambda x: x["aicc"])
                    best_r = all_res[0]
                    for r in all_res:
                        if (CT.nfree(r["ct"]) < CT.nfree(best_r["ct"])
                                and best_r["r2"] - r["r2"] < 0.002):
                            best_r = r; break

                    best_p  = best_r["params"]
                    best_ct = best_r["ct"]
                    r2_fin  = best_r["r2"]

                    log(f"🏆 **Stage 5 complete** — Best model: **{CT.name(best_ct)}**  "
                        f"R² = `{r2_fin:.6f}`  "
                        f"(elapsed `{time.time()-t0:.1f}` s)")
                    prog.progress(95, text="Building figure…")

                    # ── Stage 6: Figure ──────────────────────────────────────
                    try:
                        fig, r2_fig, rmse_fig = make_figure(
                            E, i, best_p, best_ct, sample_name or uf.name,
                            cat_res, an_res, Ecorr, show_regs, dpi=pub_dpi)
                        fig.tight_layout(rect=[0, 0, 1, 0.97])

                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, dpi=pub_dpi, bbox_inches="tight",
                                    facecolor="white"); buf_png.seek(0)
                        png_bytes = buf_png.read()

                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight",
                                    facecolor="white"); buf_svg.seek(0)
                        svg_bytes = buf_svg.read()

                        # Store result
                        res_rec = dict(
                            name=sample_name or uf.name,
                            params=best_p, ct=best_ct,
                            r2=r2_fin, rmse=rmse_fig,
                            success=r2_fin > 0.90,
                            material=(ew_mat, rho_mat),
                            all_candidates=all_res,
                        )
                        st.session_state.results.append(res_rec)
                        st.session_state.figures.append({
                            "png": png_bytes, "svg": svg_bytes,
                            "name": res_rec["name"]})

                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as ex:
                        st.error(f"Figure error: {ex}")
                        st.code(traceback.format_exc())

                    # ── Metrics ──────────────────────────────────────────────
                    p = best_p
                    B_val = (p[2]*p[3])/(2.303*(p[2]+p[3])) if p[2]>0 and p[3]>0 else 0
                    CR    = p[1] * 3.27 * ew_mat / rho_mat

                    st.markdown("#### 📐 Fitted Parameters")
                    mc = st.columns(5)
                    mc[0].metric("E_corr (V)",       f"{p[0]:.5f}")
                    mc[1].metric("i_corr (A/cm²)",   f"{p[1]:.4e}")
                    mc[2].metric("βa (mV/dec)",       f"{p[2]*1000:.1f}")
                    mc[3].metric("βc (mV/dec)",       f"{p[3]*1000:.1f}")
                    mc[4].metric("B (V)",             f"{B_val:.5f}")

                    mc2 = st.columns(5)
                    mc2[0].metric("CR (mm/yr)",      f"{CR:.5f}")
                    mc2[1].metric("R²",              f"{r2_fin:.5f}",
                                  "Excellent" if r2_fin>0.99 else
                                  "Good" if r2_fin>0.95 else "⚠ Check")
                    if best_ct in CT.PASS:
                        mc2[2].metric("i_pass (A/cm²)", f"{p[6]:.4e}")
                        mc2[3].metric("E_pass (V)",     f"{p[4]:.5f}")
                    if best_ct in CT.TRANS:
                        mc2[4].metric("E_trans (V)",    f"{p[7]:.5f}")

                    if len(all_res) > 1:
                        st.markdown("**🏆 Model Comparison (AICc)**")
                        cmp_rows = [{
                            "Model": CT.name(r["ct"]),
                            "Free params": CT.nfree(r["ct"]),
                            "R²": f"{r['r2']:.6f}",
                            "AICc": f"{r['aicc']:.1f}",
                            "Selected": "✅" if r["ct"]==best_ct else ""
                        } for r in sorted(all_res, key=lambda x: x["aicc"])]
                        st.dataframe(pd.DataFrame(cmp_rows),
                                     use_container_width=True, hide_index=True)

                    d1, d2 = st.columns(2)
                    d1.download_button("⬇ PNG", png_bytes,
                                       f"{sample_name}.png", "image/png")
                    d2.download_button("⬇ SVG", svg_bytes,
                                       f"{sample_name}.svg", "image/svg+xml")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RESULTS & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_res:
    if not st.session_state.results:
        st.info("No results yet.")
    else:
        rows = []
        for r in st.session_state.results:
            p = r["params"]; ct = r["ct"]
            B  = (p[2]*p[3])/(2.303*(p[2]+p[3])) if p[2]>0 and p[3]>0 else 0
            ew, rho = r.get("material",(27.92,7.87))
            CR = p[1] * 3.27 * ew / rho
            rows.append({"Sample":r.get("name","?"),
                          "Model":CT.name(ct),
                          "E_corr (V)":f"{p[0]:.5f}",
                          "i_corr (A/cm²)":f"{p[1]:.4e}",
                          "βa (mV/dec)":f"{p[2]*1000:.1f}",
                          "βc (mV/dec)":f"{p[3]*1000:.1f}",
                          "B (V)":f"{B:.5f}",
                          "CR (mm/yr)":f"{CR:.5f}",
                          "i_pass":f"{p[6]:.3e}" if ct in CT.PASS else "—",
                          "E_pass (V)":f"{p[4]:.5f}" if ct in CT.PASS else "—",
                          "E_trans (V)":f"{p[7]:.5f}" if ct in CT.TRANS else "—",
                          "R²":f"{r.get('r2',0):.5f}",
                          "RMSE":f"{r.get('rmse',0):.5f}",
                          "Status":"✓ Good" if r.get("r2",0)>0.95 else "⚠ Check"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.divider()

        ec1, ec2, ec3 = st.columns(3)
        ec1.download_button("📥 Excel (.xlsx)",
            data=export_excel(st.session_state.results),
            file_name="polarization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
        ec2.download_button("📥 PDF Report",
            data=export_pdf(st.session_state.results,
                            [f["png"] for f in st.session_state.figures]),
            file_name="polarization_report.pdf", mime="application/pdf",
            use_container_width=True)
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as zf:
            for fd in st.session_state.figures:
                zf.writestr(f"{fd['name']}.png", fd["png"])
                zf.writestr(f"{fd['name']}.svg", fd["svg"])
        zb.seek(0)
        ec3.download_button("📥 Figures (.zip)", data=zb,
            file_name="polarization_figures.zip", mime="application/zip",
            use_container_width=True)

        st.markdown("### Fitted Figures")
        for fd in st.session_state.figures:
            st.markdown(f"**{fd['name']}**")
            st.image(fd["png"], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    if len(st.session_state.results) < 2:
        st.info("Fit ≥ 2 samples to enable comparison.")
    else:
        with plt.rc_context(PLT_RC):
            fig_c, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
            names = [r.get("name","?") for r in st.session_state.results]
            for idx, res in enumerate(st.session_state.results):
                col = PALETTE[idx % len(PALETTE)]
                p   = res["params"]; ct = res["ct"]
                lbl = res.get("name", f"S{idx+1}")
                E_pl = np.linspace(min(p[0]-1.5, -1.5), max(p[0]+1.5, 1.5), 3000)
                try:
                    i_pl = pol_model(E_pl, p, ct)
                    axes[0].plot(E_pl, slog(i_pl), color=col, lw=2, label=lbl)
                    axes[0].axvline(p[0], color=col, ls=":", lw=0.9, alpha=0.6)
                except:
                    pass
                axes[1].bar(idx, p[1], color=col, alpha=0.85)
                axes[2].bar(idx-0.2, p[2]*1000, 0.38, color=col, alpha=0.85)
                axes[2].bar(idx+0.2, p[3]*1000, 0.38, color=col, alpha=0.45, hatch="//")

            axes[0].set_xlabel("E (V)"); axes[0].set_ylabel("log|i| (A/cm²)")
            axes[0].set_title("Evans Diagram Overlay", fontweight="bold")
            axes[0].legend(fontsize=8); axes[0].grid(True, ls="--", alpha=0.35)
            axes[0].set_facecolor("#fafbff")

            for ax, ttl, yl in zip(axes[1:],
                ["i_corr Comparison", "Tafel Slopes (filled=βa, hatch=βc)"],
                ["i_corr (A/cm²)", "Tafel slope (mV/dec)"]):
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=18, ha="right")
                ax.set_ylabel(yl); ax.set_title(ttl, fontweight="bold")
                ax.grid(True, axis="y", ls="--", alpha=0.35)
                ax.set_facecolor("#fafbff")
            axes[1].set_yscale("log")
            fig_c.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HELP
# ─────────────────────────────────────────────────────────────────────────────
with tab_help:
    st.markdown("""
### Fitting Pipeline

| Stage | What happens |
|---|---|
| **1 — E_corr detection** | Interpolated zero-crossing of signed current |
| **2 — Cathodic fit** | Vectorized sliding regression on log\\|i\\| vs E; best Tafel region; diffusion limit detection |
| **3 — Anodic fit** | Same for anodic; passive plateau via flat log\\|i\\| region; transpassive detection |
| **4 — Classification** | Curve type inferred from detected features |
| **5 — Global polish** | Physics-informed p₀ → DE (light) → L-BFGS-B (Powell if needed) |
| **6 — AICc selection** | Multiple candidate models compared; parsimony-penalised |

---
### Model Parameters

| Symbol | Meaning | Typical range |
|---|---|---|
| E_corr | Corrosion potential | — |
| i_corr | Corrosion current density | 1e-10 … 1e-2 A/cm² |
| βa | Anodic Tafel slope | 40–200 mV/dec |
| βc | Cathodic Tafel slope | 40–200 mV/dec |
| B | Stern-Geary constant = βaβc / 2.303(βa+βc) | — |
| E_pass | Passivation onset potential | — |
| i_pass | Passive current density | — |
| E_trans | Transpassive / pitting potential | — |
| i_L | Cathodic diffusion limiting current | — |

---
### Data Format
- Column 1: **E (V vs reference)** — any reference
- Column 2: **Signed current density** — cathodic **must** be negative, anodic positive
- Autolab/NOVA `.txt` / `.csv` exports accepted (auto column detection)
- Skip-row and column selection in sidebar

### Fit Quality
| R² | Quality |
|---|---|
| > 0.99 | Excellent — publication-ready |
| 0.95–0.99 | Good |
| < 0.95 | Review: try different model or check data |
""")
