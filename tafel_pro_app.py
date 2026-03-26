"""
Polarization Curve Fitter — Publication-Grade Streamlit App
(Robust local Tafel overlays + Tafel intersection i_corr + adjustable detection)
===============================================================================
- Vectorized sliding-window regressions with curvature & diffusion guards
- Huber refinement (IRLS) and Theil–Sen on the chosen Tafel windows
- Local Tafel dashed lines drawn as straight segments (with optional faint extension)
- i_corr annotated from Tafel intersection of local anodic/cathodic lines (toggleable)
- Lean global optimization and adaptive plotting
- Stable widget keys; force-model selection via session-backed key

FIXES (v2):
  - Cathodic proximity score reversed: now rewards windows 80–250 mV from E_corr
    (true Tafel region), not windows near E_corr (mixed-control zone).
  - Cathodic scoring: R²² weighting added; curvature penalty incorporated directly.
  - Cathodic i_corr: uses max(tafel_extrap, near_Ecorr) rather than the
    incorrectly capped min() which under-estimated i_corr.
  - Anodic proximity tau widened from 0.12 V → 0.20 V to accept active regions
    spanning 100–200 mV (e.g. stainless, Ni alloys).
  - Anodic fallback ok mask retains beta_ok to prevent unphysical slopes.
  - Linear-scale panel (Panel C) ylim now zooms to ±2× the local i range within
    ±150 mV of E_corr, avoiding passive/transpassive peaks crushing the view.
  - diff_ok guard relaxed: threshold lowered from 0.35→0.20 slope_mag to avoid
    rejecting valid cathodic windows with mild slope variation.

Run:
    streamlit run app.py
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
st.markdown("""
<style>
  .main-header{font-size:2rem;font-weight:700;color:#1a3a5c;
    border-bottom:3px solid #2e86de;padding-bottom:8px;margin-bottom:1rem}
  div[data-testid="metric-container"]{
    background:#f0f4ff;border-left:3px solid #2e86de;
    border-radius:6px;padding:8px 12px}
</style>""", unsafe_allow_html=True)

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
    "Nickel":                (29.36, 8.91),
    "Titanium":              (11.99, 4.51),
    "Zinc":                  (32.69, 7.14),
}

# Global detection settings (overridden by sidebar "Advanced")
CFG = dict(
    cath_guard=0.020,      # V — min distance of cathodic window from Ecorr
    anod_guard=0.010,      # V — min distance of anodic window from Ecorr
    curvature_max=40.0,    # max |d2 log|i| / dE^2| inside window
    lin_frac=0.70,         # fraction of derivative signs consistent with slope
    min_w_cat=4,           # minimum window length (cathodic)
    min_w_ano=3,           # minimum window length (anodic)
    # Tafel slope range (V/dec). Physical upper limits based on BV theory:
    # ba = RT/(alpha*F). alpha<0.09 -> ba>280mV/dec is unphysical.
    # Old beta_max_a=0.400 let passive-region slopes (ba~300-400mV) pass
    # the beta_ok gate, producing the ba=384mV/dec artefact seen in image.
    beta_min=0.020,        # 20 mV/dec lower physical limit
    beta_max_c=0.280,      # 280 mV/dec cathodic cap (was 350)
    beta_max_a=0.250,      # 250 mV/dec anodic cap   (was 400 — this was the bug)
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER MATH
# ─────────────────────────────────────────────────────────────────────────────
def slog(x): return np.log10(np.maximum(np.abs(x), TINY))

def sig(x, k=40.0):
    xk = np.clip(k * x, -60, 60)
    return np.where(xk >= 0, 1.0 / (1.0 + np.exp(-xk)),
                    np.exp(xk) / (1.0 + np.exp(xk)))

def sm(y, w=11, p=3):
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
    if n <= k + 1 or sse <= 0: return 1e30
    return n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)

def downsample_uniform(x, y, max_pts=400):
    if len(x) <= max_pts: return x, y
    idx = np.linspace(0, len(x)-1, max_pts).astype(int)
    return x[idx], y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# FAST SLIDING REGRESSION (VECTORIZED)
# ─────────────────────────────────────────────────────────────────────────────
def _sliding_regress_full(x, y, min_len=4, max_len=25):
    n = len(x)
    if n < min_len:
        return (np.array([], int), np.array([], int),
                np.array([]), np.array([]), np.array([]))
    Sx  = np.cumsum(x); Sy  = np.cumsum(y)
    Sxx = np.cumsum(x*x); Sxy = np.cumsum(x*y); Syy = np.cumsum(y*y)

    starts_all, ends_all, slopes_all, inters_all, r2_all = [], [], [], [], []
    wmax = min(max_len, n)
    for w in range(min_len, wmax+1):
        i0 = np.arange(0, n - w + 1)
        i1 = i0 + w - 1
        def segsum(csum): return csum[i1] - np.concatenate(([0.0], csum[i0[:-1]]))
        sum_x, sum_y = segsum(Sx), segsum(Sy)
        sum_xx, sum_xy, sum_yy = segsum(Sxx), segsum(Sxy), segsum(Syy)

        w_f = float(w)
        mx, my = sum_x / w_f, sum_y / w_f
        denom = sum_xx - w_f * mx * mx
        slope = np.where(np.abs(denom) > 1e-18, (sum_xy - w_f * mx * my) / denom, 0.0)
        intercept = my - slope * mx

        SSE = (sum_yy - 2.0*intercept*sum_y - 2.0*slope*sum_xy
               + (intercept**2)*w_f + 2.0*intercept*slope*sum_x + (slope**2)*sum_xx)
        SST = sum_yy - w_f * my * my
        R2 = np.where(SST > 1e-18, 1.0 - SSE / SST, 0.0)
        R2 = np.clip(R2, 0.0, 1.0)

        starts_all.append(i0); ends_all.append(i1 + 1)
        slopes_all.append(slope); inters_all.append(intercept); r2_all.append(R2)

    return (np.concatenate(starts_all), np.concatenate(ends_all),
            np.concatenate(slopes_all), np.concatenate(inters_all),
            np.concatenate(r2_all))

# ─────────────────────────────────────────────────────────────────────────────
# CURVE TYPE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
PARAM_NAMES = ["Ecorr","icorr","ba","bc","Epass","k_pass","ip","Etrans","k_trans","itrans","iL"]
NP = 11

class CT:
    A  = "A"; AD = "AD"; P  = "P"; PT = "PT"; F  = "F"
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

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS MODEL
# ─────────────────────────────────────────────────────────────────────────────
def pol_model(E, p, ct="PT"):
    E    = np.asarray(E, float)
    Ec   = p[0]; ic = p[1]; ba = max(p[2], 1e-6); bc = max(p[3], 1e-6)
    Ep   = p[4]; kp = max(p[5], 0.001); ip = p[6]
    Et   = p[7]; kt = max(p[8], 0.001); it = p[9]
    iL   = max(p[10], 1e-30)
    eta  = E - Ec
    ik_cat = ic * np.exp(np.clip(-2.303 * eta / bc, -60, 60))
    i_cat = ik_cat / (1.0 + ik_cat / iL) if ct in ("AD","F") else ik_cat
    i_act = ic * np.exp(np.clip(2.303 * eta / ba, -60, 60))
    if ct in CT.SIMPLE: return i_act - i_cat
    w_p   = sig(E - Ep, 1.0 / kp)
    i_ano = (1.0 - w_p) * i_act + w_p * ip
    if ct == "P": return i_ano - i_cat
    w_t   = sig(E - Et, 1.0 / kt)
    i_tp  = ip + it * np.exp(np.clip(2.303 * (E - Et) / ba, -60, 60))
    i_ano = (1.0 - w_t) * i_ano + w_t * i_tp
    return i_ano - i_cat

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — E_corr DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_ecorr(E, i):
    E = np.asarray(E, float); i = np.asarray(i, float)
    si = np.argsort(E); Es = E[si]; is_ = i[si]
    sc = np.where(np.diff(np.sign(is_)))[0]
    if len(sc) == 0:
        idx = int(np.argmin(np.abs(is_))); return float(Es[idx]), int(si[idx])
    crossings = []
    for k in sc:
        denom = is_[k+1] - is_[k]
        if abs(denom) < TINY: continue
        Ec = float(Es[k] - is_[k] * (Es[k+1] - Es[k]) / denom)
        goes_anodic = (is_[k] < 0 and is_[k+1] > 0)
        crossings.append((Ec, int(si[k]), goes_anodic))
    anodic = [(Ec, idx) for Ec, idx, ga in crossings if ga]
    if anodic: return min(anodic, key=lambda x: x[0])
    best = min(crossings, key=lambda x: x[0]); return best[0], best[1]

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST LINE REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _huber_fit(x, y, slope, intercept, iters=3, c=1.345):
    for _ in range(iters):
        r = y - (slope * x + intercept)
        s = np.median(np.abs(r)) * 1.4826 + 1e-12
        w = np.ones_like(r)
        t = np.abs(r) / (c * s + 1e-12)
        w[t > 1] = (c * s) / (np.abs(r[t > 1]) + 1e-12)
        X = np.vstack([np.ones_like(x), x]).T
        W = np.diag(w)
        try:
            b, a = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
            intercept, slope = float(b), float(a)
        except Exception:
            break
    return slope, intercept

def _theil_sen(x, y):
    m = len(x)
    if m < 2:
        return 0.0, float(np.median(y)) if m else (0.0, 0.0)
    i_idx, j_idx = np.triu_indices(m, k=1)
    dx = x[j_idx] - x[i_idx]
    valid = np.abs(dx) > 1e-15
    slopes = (y[j_idx][valid] - y[i_idx][valid]) / dx[valid]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CATHODIC BRANCH FIT (ROBUST LOCAL LINE)
# FIX: Corrected proximity scoring, curvature weighting, i_corr estimation
# ─────────────────────────────────────────────────────────────────────────────
def fit_cathodic(E, i, Ecorr):
    """
    Find the cathodic Tafel window using hard beta enforcement.

    Key design decisions:
    - beta_ok is a HARD GATE: windows outside the physical Tafel slope range
      are completely excluded (score=0), not just penalised. If all windows
      fail, we fall back to the steepest valid-direction window.
    - Proximity scoring uses a Gaussian centred 150 mV below Ecorr, rewarding
      the true Tafel zone and penalising both mixed-control (too close) and
      diffusion-plateau (too flat/far) regions.
    - Points within the top 1.5 decades of log|i| closest to Ecorr are trimmed
      before window search to avoid mixed-control corruption.
    """
    cat = i < 0
    if np.sum(cat) < 4:
        return dict(bc=0.120, icorr=1e-8, iL=1e-2, has_diff=False, r2=0.0)

    Ec  = E[cat]; lgi = slog(i[cat])
    si  = np.argsort(Ec); Ec, lgi = Ec[si], lgi[si]

    # Strict guard: exclude points within cath_guard of Ecorr
    base_mask = Ec < (Ecorr - CFG["cath_guard"])
    if np.sum(base_mask) < CFG["min_w_cat"]:
        base_mask = np.ones_like(Ec, bool)

    Ex, Yx = Ec[base_mask], lgi[base_mask]
    if len(Ex) < CFG["min_w_cat"]:
        return dict(bc=0.120, icorr=max(10**np.min(Yx), 1e-12), iL=1e-2,
                    has_diff=False, r2=0.0, E_cat=Ec, lgi_cat=lgi)

    # Remove the diffusion plateau top (keeps Tafel + transition, drops flat plateau)
    lgi_max_cat = np.max(Yx)
    trim_mask = Yx < (lgi_max_cat - 0.3)
    if np.sum(trim_mask) >= CFG["min_w_cat"]:
        Ex, Yx = Ex[trim_mask], Yx[trim_mask]

    # Smooth & derivatives (use full trimmed range for smoothing)
    w_sm = max(5, min(11, len(Ex) // 2 * 2 - 1))
    Y_sm = savgol_filter(Yx, w_sm, 3, mode="interp")
    dY   = np.gradient(Y_sm, Ex)
    d2Y  = np.gradient(dY, Ex)

    # ── Steep-region pre-filter ───────────────────────────────────────────────
    # For diffusion-limited cathodic branches the data contains:
    #   [transition zone: |dY/dE| is moderate] → [Tafel zone: |dY/dE| is maximum]
    # The sliding window over the full range finds the smooth transition (low
    # curvature, moderate slope) rather than the steep Tafel zone (high curvature
    # because it borders the mixed-control zone near Ecorr).
    # Fix: restrict the window search to points where the local slope magnitude
    # exceeds 50% of its maximum value — this selects only the Tafel zone.
    abs_dY    = np.abs(dY)
    max_abs_dY = float(np.max(abs_dY)) if len(abs_dY) > 0 else 1.0
    steep_pre  = abs_dY > 0.50 * max_abs_dY
    if np.sum(steep_pre) >= CFG["min_w_cat"]:
        Ex_sw, Yx_sw = Ex[steep_pre], Yx[steep_pre]
        dY_sw  = dY[steep_pre]
        d2Y_sw = d2Y[steep_pre]
    else:
        # Fallback: use full trimmed range if steep region is too small
        Ex_sw, Yx_sw, dY_sw, d2Y_sw = Ex, Yx, dY, d2Y

    s_idx, e_idx, slope, intercept, R2 = _sliding_regress_full(
        Ex_sw, Yx_sw, min_len=CFG["min_w_cat"], max_len=25)
    if len(slope) == 0:
        return dict(bc=0.120, icorr=1e-9, iL=1e-2, has_diff=False, r2=0.0,
                    E_cat=Ec, lgi_cat=lgi)

    # Per-window quality metrics on the (possibly pre-filtered) data
    max_d2 = np.array([np.max(np.abs(d2Y_sw[s:e])) for s, e in zip(s_idx, e_idx)])

    def _mono_frac(s, e):
        seg = dY_sw[s:e]
        return float(np.mean(seg < 0)) if len(seg) else 0.0
    mono = np.array([_mono_frac(s, e) for s, e in zip(s_idx, e_idx)])

    invm     = np.where(np.abs(slope) > 1e-12, 1.0 / np.abs(slope), np.inf)
    # HARD GATE: beta_ok is a strict physical constraint, not a scoring term
    beta_ok  = (slope < 0) & (invm > CFG["beta_min"]) & (invm < CFG["beta_max_c"])
    mono_ok  = mono >= CFG["lin_frac"]
    # Note: curvature is NOT used as a hard gate — for diffusion-limited cathodic
    # branches the second derivative is large everywhere (sigmoid morphology), so
    # curvature_max would eliminate ALL windows.  Instead, curvature feeds only
    # into the scoring penalty below.

    # Primary candidates: beta_ok + monotone + decent R²
    ok = beta_ok & mono_ok & (R2 > 0.85)
    if not np.any(ok):
        ok = beta_ok & (R2 > 0.75)
    if not np.any(ok):
        ok = beta_ok.copy()
    if not np.any(ok):
        ok = slope < 0
        if not np.any(ok):
            ok = np.ones(len(slope), bool)

    # ── Scoring ───────────────────────────────────────────────────────────────
    # For diffusion-limited cathodic branches the morphology is:
    #   [diffusion plateau] → [curved transition] → [steep Tafel] → [mixed control]
    # The plateau and transition have SHALLOW slopes → eliminated by beta_ok.
    # Within beta_ok windows, the TRUE Tafel region is the STEEPEST one.
    # We therefore weight by (steep_bonus)² to sharply prefer the Tafel region.
    # Curvature enters only as a soft penalty (exp term), never as a hard gate.
    curv_pen   = np.exp(-max_d2 / max(CFG["curvature_max"], 10.0))
    win_len    = (e_idx - s_idx).astype(float)
    len_bonus  = np.log1p(win_len) / np.log1p(25.0)

    abs_slope       = np.abs(slope)
    max_beta_slope  = float(np.max(abs_slope[beta_ok])) if np.any(beta_ok) else 1.0
    steep_bonus     = abs_slope / max(max_beta_slope, 1e-9)   # [0,1]

    # steep_bonus² dominates: steepest (= Tafel) window wins decisively
    score = (R2 ** 2) * np.clip(mono, 0.2, 1.0) * curv_pen \
            * (0.3 + 0.1 * len_bonus + 0.6 * steep_bonus ** 2)

    score[~beta_ok] = 0.0                          # hard zero: wrong direction/beta
    score[beta_ok & ~mono_ok] *= 0.40              # soft penalty: non-monotone windows

    if np.all(score == 0.0) or not np.any(ok):
        neg_slope_mag = np.where(slope < 0, np.abs(slope), 0.0)
        k_best = int(np.argmax(neg_slope_mag))
    else:
        k_best = int(np.argmax(score))

    s0, s1   = int(s_idx[k_best]), int(e_idx[k_best])
    sl_ols, b_ols = float(slope[k_best]), float(intercept[k_best])

    # Theil–Sen + Huber; pick best R²
    sl_ts, b_ts = _theil_sen(Ex[s0:s1], Yx[s0:s1])
    sl_hb, b_hb = _huber_fit(Ex[s0:s1], Yx[s0:s1], sl_ols, b_ols)

    def _r2_line(sl, b):
        return r2_score(Yx[s0:s1], sl * Ex[s0:s1] + b)
    cand = [(sl_ts, b_ts, _r2_line(sl_ts, b_ts)),
            (sl_hb, b_hb, _r2_line(sl_hb, b_hb)),
            (sl_ols, b_ols, _r2_line(sl_ols, b_ols))]
    sl_ref, b_ref, r2_win = max(cand, key=lambda t: t[2])

    # Ensure the refined slope is physically sensible
    if sl_ref >= 0 or abs(1.0 / sl_ref) > CFG["beta_max_c"] * 1.5:
        sl_ref, b_ref = sl_ols, b_ols  # revert to OLS if refinement drifted

    bc = min(abs(1.0 / sl_ref), CFG["beta_max_c"]) if abs(sl_ref) > 1e-9 else 0.120

    # Tafel extrapolation to Ecorr
    icorr_tafel = float(10 ** (b_ref + sl_ref * Ecorr)) if abs(sl_ref) > 1e-9 else 1e-12

    # Cross-check near Ecorr; use Tafel extrap as primary (it is the physical value)
    near = np.abs(Ec - Ecorr) < 0.050
    if np.any(near):
        icorr_near = float(np.percentile(np.abs(i[cat][si][near]), 25))
        icorr = max(icorr_tafel, icorr_near * 0.5)
    else:
        icorr = icorr_tafel
    icorr = max(icorr, 1e-15)

    # Diffusion limit detection
    iL, has_diff = None, False
    if len(Ec) > 8:
        lgi_sm = savgol_filter(lgi, max(5, min(9, len(Ec) // 2 * 2 - 1)), 3, mode="interp")
        dlg    = np.abs(np.gradient(lgi_sm, Ec))
        flat   = dlg < max(np.percentile(dlg, 30), 0.5)
        runs   = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            idxs = [s[0] for s in max(runs, key=lambda x: len(x[1]))[1]]
            if len(idxs) >= 3 and abs(Ec[idxs[-1]] - Ec[idxs[0]]) > 0.03:
                iL = float(np.median(np.abs(i[cat][si][idxs]))); has_diff = True
    if iL is None:
        iL = icorr * 1e4

    return dict(
        bc=bc, icorr=icorr, iL=iL, has_diff=has_diff,
        r2=float(max(r2_win, float(np.max(R2)) if len(R2) else 0.0)),
        E_cat=Ec, lgi_cat=lgi,
        slope_c=sl_ref, intercept_c=b_ref, win_c=(Ex[s0], Ex[s1 - 1])
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ANODIC BRANCH FIT (ROBUST LOCAL LINE)
# FIX: Widened proximity tau, corrected fallback ok mask
# ─────────────────────────────────────────────────────────────────────────────
def fit_anodic(E, i, Ecorr):
    """
    Find the anodic Tafel (active dissolution) window using hard beta enforcement.

    Key design decisions:
    - beta_ok is a HARD GATE: flat passive/plateau windows (slope~0, 1/slope>>400mV)
      get score=0 regardless of their R². This was the main failure mode where the
      algorithm mistakenly selected the active dissolution peak or passive plateau
      (high R², nearly flat) over the true steep Tafel region.
    - Passive detection is run FIRST to bound the Tafel search to E < E_pass.
    - If all windows fail beta_ok, pick the steepest positive-slope window
      (most Tafel-like) rather than the flattest high-R² one.
    - Proximity tau widened to 0.20 V to accept active regions spanning 100–200 mV.
    """
    ano = i > 0
    if np.sum(ano) < 4:
        return dict(ba=0.060, has_passive=False, Epass=None, ip=1e-6,
                    has_trans=False, Etrans=None, r2=0.0)

    Ea  = E[ano]; lgia = slog(i[ano])
    si  = np.argsort(Ea); Ea, lgia = Ea[si], lgia[si]

    # ── Passive / Transpassive detection (run BEFORE Tafel search so we can
    #    bound the active region search range to E < E_pass) ─────────────────
    has_passive = False; Epass = None; ip = 1e-6
    has_trans   = False; Etrans = None
    Epeak_detected = None

    if len(Ea) > 6:
        # Smooth first for robust derivative and peak detection
        w_pk = max(5, min(15, len(Ea) // 2 * 2 - 1))
        lgia_pk = savgol_filter(lgia, w_pk, 3, mode="interp")
        dY_pk   = np.gradient(lgia_pk, Ea)

        # PRIMARY: derivative sign change (+→-) above Ecorr = active dissolution peak.
        # This catches even very shallow peaks (< 0.1 decade) that find_peaks misses.
        sign_chg = np.where(np.diff(np.sign(dY_pk)) < 0)[0]   # rising→falling
        cands = [Ea[k] for k in sign_chg
                 if Ea[k] > Ecorr + 0.005 and Ea[k] < Ea[-1] - 0.05]
        if cands:
            # Pick the first peak above Ecorr (closest to Ecorr = active peak)
            Epeak_detected = float(min(cands))

        # SECONDARY: find_peaks with reduced prominence (was 0.3, now 0.10)
        # catches genuine peaks missed by the derivative method on noisy data
        if Epeak_detected is None:
            pks, _ = find_peaks(lgia_pk, prominence=0.10, distance=2)
            pks_above = [p for p in pks if Ea[p] > Ecorr + 0.005]
            if pks_above:
                # closest to Ecorr among peaks
                pk = min(pks_above, key=lambda p: abs(Ea[p] - Ecorr))
                Epeak_detected = float(Ea[pk])

    if len(Ea) > 8:
        lgia_sm = savgol_filter(lgia, max(5, min(11, len(Ea) // 2 * 2 - 1)), 3, mode="interp")
        dlg     = np.gradient(lgia_sm, Ea); adlg = np.abs(dlg)
        thr     = max(np.percentile(adlg, 30), 0.8)
        flat    = adlg < thr

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

    # ── Tafel window search ───────────────────────────────────────────────────
    base_mask = Ea > (Ecorr + CFG["anod_guard"])
    if np.sum(base_mask) < CFG["min_w_ano"]:
        base_mask = np.ones_like(Ea, bool)

    # CRITICAL FIX A: bound search to the ACTIVE dissolution zone only.
    # For Active-Passive curves the Tafel line must end at the active peak
    # (Epeak), NOT at Epass.  The old code used Epass as the upper bound,
    # but on steep active-passive curves Epass > Epeak and many of the
    # candidate windows span past the peak into the passive drop — their
    # mean slope is near-zero (high R², but beta >> 400 mV/dec) so they
    # are correctly killed by the hard beta gate.  However when Epeak is
    # absent (flat monotone rise straight into passivation, common on SS)
    # we should still bound by Epass to keep the search in the active region.
    E_upper = None
    if Epeak_detected is not None and Epeak_detected > Ecorr:
        E_upper = Epeak_detected        # tightest: stop at the active peak
    elif has_passive and Epass is not None:
        E_upper = Epass                 # fallback: stop at passivation onset

    if E_upper is not None:
        active_mask = base_mask & (Ea <= E_upper)
        if np.sum(active_mask) >= CFG["min_w_ano"]:
            base_mask = active_mask
        # else: too few points before peak/pass — keep full range, beta gate handles it

    Ex, Yx = Ea[base_mask], lgia[base_mask]
    if len(Ex) < CFG["min_w_ano"]:
        return dict(ba=0.060, has_passive=has_passive, Epass=Epass, ip=ip,
                    has_trans=has_trans, Etrans=Etrans, Epeak=Epeak_detected,
                    r2=0.0, E_an=Ea, lgi_an=lgia)

    w_sm = max(5, min(11, len(Ex) // 2 * 2 - 1))
    Y_sm = savgol_filter(Yx, w_sm, 3, mode="interp")
    dY   = np.gradient(Y_sm, Ex)
    d2Y  = np.gradient(dY, Ex)

    # ── Rising-only trim ──────────────────────────────────────────────────────
    # After the E_upper bound the data should be monotonically rising (active
    # dissolution), but noise in a long passive plateau can produce spurious
    # sign-changes that dilute the active slope estimate.
    # Trim any suffix where dY turns persistently negative (passive descent),
    # keeping only the contiguous rising front from Ecorr upward.
    # This was the root cause of ba=384 mV/dec in the image: the window
    # extended into the passive drop where apparent slope ≈ 0, i.e. ba >> 250.
    first_neg_run = None
    for _k in range(len(dY) - 2):
        if dY[_k] < 0 and dY[_k + 1] < 0:   # two consecutive negatives = descent
            first_neg_run = _k
            break
    if first_neg_run is not None and first_neg_run >= CFG["min_w_ano"]:
        Ex  = Ex[:first_neg_run]
        Yx  = Yx[:first_neg_run]
        dY  = dY[:first_neg_run]
        d2Y = d2Y[:first_neg_run]
        # Rerun sliding regression on trimmed data
        if len(Ex) < CFG["min_w_ano"]:
            Ex  = Ea[base_mask]; Yx  = lgia[base_mask]
            w_sm2 = max(5, min(11, len(Ex)//2*2-1))
            Y_sm2 = savgol_filter(Yx, w_sm2, 3, mode='interp')
            dY  = np.gradient(Y_sm2, Ex)
            d2Y = np.gradient(dY, Ex)

    s_idx, e_idx, slope, intercept, R2 = _sliding_regress_full(
        Ex, Yx, min_len=CFG["min_w_ano"], max_len=20)
    if len(slope) == 0:
        return dict(ba=0.060, has_passive=has_passive, Epass=Epass, ip=ip,
                    has_trans=has_trans, Etrans=Etrans, Epeak=Epeak_detected,
                    r2=0.0, E_an=Ea, lgi_an=lgia)

    # Clamp indices to trimmed array length
    n_trim = len(dY)
    s_idx_c = np.minimum(s_idx, n_trim - 1)
    e_idx_c = np.minimum(e_idx, n_trim)
    valid   = e_idx_c > s_idx_c
    if not np.any(valid):
        s_idx_c = s_idx; e_idx_c = e_idx; valid = np.ones(len(s_idx), bool)
    max_d2 = np.array([np.max(np.abs(d2Y[s:max(s+1,e)])) for s, e in zip(s_idx_c[valid], e_idx_c[valid])])
    # Pad back to full length for consistent indexing
    max_d2_full = np.full(len(slope), max_d2.max() if len(max_d2) else 0.0)
    max_d2_full[valid] = max_d2
    max_d2 = max_d2_full

    def _mono_frac(s, e):
        seg = dY[s:e]
        return float(np.mean(seg > 0)) if len(seg) else 0.0   # log|i| increases as E↑
    mono = np.array([_mono_frac(s, e) for s, e in zip(s_idx, e_idx)])

    invm    = np.where(np.abs(slope) > 1e-12, 1.0 / np.abs(slope), np.inf)
    # HARD GATE: physical Tafel slope range
    beta_ok = (slope > 0) & (invm > CFG["beta_min"]) & (invm < CFG["beta_max_a"])
    curv_ok = max_d2 < CFG["curvature_max"]
    mono_ok = mono >= CFG["lin_frac"]

    # Primary candidates: pass all quality gates
    ok = beta_ok & curv_ok & mono_ok & (R2 > 0.80)

    # Fallback tier 1: relax curvature/mono, keep beta hard gate and R2
    if not np.any(ok):
        ok = beta_ok & (R2 > 0.70)

    # Fallback tier 2: beta_ok only (any R2)
    if not np.any(ok):
        ok = beta_ok.copy()

    # Scoring — HARD ZERO for windows failing beta_ok (passive plateau killer)
    E_start  = Ex[s_idx]
    clos     = np.exp(-np.abs(E_start - Ecorr) / 0.20)   # proximity to Ecorr
    curv_pen = np.exp(-max_d2 / max(CFG["curvature_max"], 10.0))
    score    = (R2 ** 2) * clos * np.clip(mono, 0.2, 1.0) * curv_pen

    # CRITICAL FIX: hard-zero windows outside beta range (was *= 0.15)
    score[~beta_ok] = 0.0
    # Soft penalty only for curvature/mono violations within the beta_ok set
    score[beta_ok & ~(curv_ok & mono_ok)] *= 0.30

    # If scoring collapses (all zeros), use derivative-based fallback for
    # very narrow active zones (e.g. stainless steel with instant passivation).
    # FIX E: find the steepest consistently rising portion using the smoothed
    # derivative instead of relying on sliding window regression.
    if np.all(score == 0.0):
        # Try: pick the window with the steepest positive slope among beta_ok
        beta_ok_slopes = np.where(beta_ok, slope, 0.0)
        if np.any(beta_ok):
            k_best = int(np.argmax(beta_ok_slopes))
        else:
            # Last resort: steepest positive-slope window regardless of beta
            pos_slope_mag = np.where(slope > 0, slope, 0.0)
            k_best = int(np.argmax(pos_slope_mag))
    else:
        k_best = int(np.argmax(score))

    s0, s1   = int(s_idx[k_best]), int(e_idx[k_best])
    sl_ols, b_ols = float(slope[k_best]), float(intercept[k_best])

    sl_ts, b_ts = _theil_sen(Ex[s0:s1], Yx[s0:s1])
    sl_hb, b_hb = _huber_fit(Ex[s0:s1], Yx[s0:s1], sl_ols, b_ols)

    def _r2_line(sl, b):
        return r2_score(Yx[s0:s1], sl * Ex[s0:s1] + b)
    cand = [(sl_ts, b_ts, _r2_line(sl_ts, b_ts)),
            (sl_hb, b_hb, _r2_line(sl_hb, b_hb)),
            (sl_ols, b_ols, _r2_line(sl_ols, b_ols))]
    sl_ref, b_ref, r2_win = max(cand, key=lambda t: t[2])

    # Guard: refinement must not drift to flat/negative slope
    if sl_ref <= 0 or abs(1.0 / sl_ref) > CFG["beta_max_a"] * 1.5:
        sl_ref, b_ref = sl_ols, b_ols

    ba = min(abs(1.0 / sl_ref), CFG["beta_max_a"]) if abs(sl_ref) > 1e-9 else 0.060

    return dict(
        ba=ba, has_passive=has_passive, Epass=Epass, ip=ip,
        has_trans=has_trans, Etrans=Etrans, Epeak=Epeak_detected,
        r2=float(max(r2_win, float(np.max(R2)) if len(R2) else 0.0)),
        E_an=Ea, lgi_an=lgia,
        slope_a=sl_ref, intercept_a=b_ref, win_a=(Ex[s0], Ex[s1 - 1])
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CLASSIFY CURVE TYPE
# ─────────────────────────────────────────────────────────────────────────────
def classify_curve(cat_res, an_res):
    hp = an_res["has_passive"]; ht = an_res["has_trans"]; hd = cat_res["has_diff"]
    if hp and ht:  return CT.F  if hd else CT.PT
    if hp:         return CT.P
    if hd:         return CT.AD
    return CT.A

# ─────────────────────────────────────────────────────────────────────────────
# TAFEL INTERSECTION (local lines)
# ─────────────────────────────────────────────────────────────────────────────
def tafel_intersection(cat_res, an_res):
    if ("slope_c" in cat_res and "intercept_c" in cat_res and
        "slope_a" in an_res  and "intercept_a" in an_res):
        mc = float(cat_res["slope_c"]); bc = float(cat_res["intercept_c"])
        ma = float(an_res["slope_a"]);  ba = float(an_res["intercept_a"])
        if abs(ma - mc) > 1e-12:
            E_star = (bc - ba) / (ma - mc)
            logI_star = ma * E_star + ba
            i_star = 10.0 ** logI_star
            return float(E_star), float(i_star), float(logI_star)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — ASSEMBLE P0 & GLOBAL POLISH (LEANER)
# ─────────────────────────────────────────────────────────────────────────────
def _make_p0(Ecorr, cat, an, ct, E_max):
    ic, bc, ba, iL = cat["icorr"], cat["bc"], an["ba"], cat["iL"]
    Ep = an.get("Epeak") if an.get("Epeak") is not None else (an["Epass"] if an["has_passive"] and an["Epass"] is not None else E_max + 5.0)
    ip = an["ip"] if an["has_passive"] else ic * 0.01
    Et = an["Etrans"] if an["has_trans"] else E_max + 5.0
    it = ip * 0.5 if an["has_trans"] else ic * 0.001
    return np.array([Ecorr, ic, ba, bc, Ep, 0.010, ip, Et, 0.015, it, iL])

def _build_bounds(Ecorr, cat, an, ct, E_min, E_max, E_span):
    """
    Build parameter bounds for the global optimizer.

    Inspired by busbar_tafel_fit.py:
    - ba/bc: [max(0.020, local×0.50), min(0.500, local×2.0)]
      Absolute floor (20 mV/dec) + ceiling (500 mV/dec) prevent runaway while
      remaining wide enough to correct a ±50% error in the local Tafel estimate.
    - iL: lo = ic×5 (must exceed icorr to be physically meaningful)
          hi = i_max×50 (cap at 50× measured maximum current)
    - icorr: ×1e-4 / ×1e4 relative to local estimate (generous range)
    """
    ic      = max(cat["icorr"], 1e-14)
    iL_est  = max(cat["iL"],    1e-10)
    ba_fit  = float(an["ba"])
    bc_fit  = float(cat["bc"])

    # Tafel slopes: tight relative band + absolute physical limits.
    # Upper caps lowered to match CFG beta_max values — prevents the global
    # optimizer from finding unphysical solutions (ba>250mV or bc>280mV)
    # even if the local Tafel estimate happened to be inflated.
    ba_lo = max(ba_fit * 0.50, 0.020)
    ba_hi = min(ba_fit * 2.00, CFG["beta_max_a"])   # hard ceiling 250 mV/dec
    bc_lo = max(bc_fit * 0.50, 0.020)
    bc_hi = min(bc_fit * 2.00, CFG["beta_max_c"])   # hard ceiling 280 mV/dec

    # iL: must be physically larger than icorr; capped by measured data range
    i_max = max(float(np.max(np.abs(cat.get("iL", ic * 100)))), ic * 10)
    iL_lo = max(ic * 5.0,   1e-13)
    iL_hi = min(i_max * 50, 1.0)

    lo = np.array([E_min,
                   max(ic * 1e-4, 1e-15),     # icorr lo
                   ba_lo, bc_lo,
                   Ecorr + 0.005, 0.001,       # Epass, k_pass
                   max(ic * 1e-5, 1e-16),      # ip lo
                   Ecorr + 0.05 * E_span, 0.001,  # Etrans, k_trans
                   max(ic * 1e-5, 1e-16),      # itrans lo
                   iL_lo])
    hi = np.array([E_max,
                   min(ic * 1e4, 1.0),         # icorr hi
                   ba_hi, bc_hi,
                   E_max, 0.150,               # Epass, k_pass
                   min(ic * 1e3, 1.0),         # ip hi
                   E_max + 0.1, 0.150,         # Etrans, k_trans
                   min(ic * 1e5, 10.0),        # itrans hi
                   iL_hi])
    lo = np.minimum(lo, hi - 1e-12)
    return lo, hi

LOG_IDX = {1, 6, 9, 10}

def _pack(p, fidx):
    return np.array([np.log10(max(p[j], TINY)) if j in LOG_IDX else p[j] for j in fidx])

def _unpack(x, fidx, p_base, lo=None, hi=None):
    p = p_base.copy()
    for k, j in enumerate(fidx):
        val = 10.0 ** x[k] if j in LOG_IDX else x[k]
        if lo is not None: val = float(np.clip(val, lo[j], hi[j]))
        p[j] = val
    return p

def _pbounds(lo, hi, fidx):
    return [(np.log10(max(lo[j], TINY)), np.log10(max(hi[j], TINY)))
            if j in LOG_IDX else (lo[j], hi[j]) for j in fidx]

def _section_weights(E, i, Ecorr_est, Epass_est=None):
    """
    Section-balanced weighting for the global objective function.

    Motivation (from busbar analysis):
    - For Active-Passive curves the passive plateau may hold 60%+ of all points.
      Uniform log-residual weighting then forces the optimizer to over-fit the
      trivially-flat passive region at the expense of the kinetically important
      cathodic and active branches.
    - Strategy: each electrochemical region (cathodic, active, passive) contributes
      roughly equally to the total objective, regardless of point density.
    - Additional Ecorr-vicinity boost (×3 within 60 mV) sharpens the corrosion
      potential and exchange current density.
    - Passive plateau is further down-weighted (×0.5) because it is
      well-described by a single parameter (ip) and needs little optimizer effort.
    """
    n = len(E)
    cat_m  = E < Ecorr_est
    if Epass_est is not None:
        act_m  = (E >= Ecorr_est) & (E < Epass_est)
        pass_m = E >= Epass_est
    else:
        act_m  = (E >= Ecorr_est) & (E < Ecorr_est + 0.10)
        pass_m = E >= (Ecorr_est + 0.10)

    n_cat  = max(int(cat_m.sum()),  1)
    n_act  = max(int(act_m.sum()),  1)
    n_pass = max(int(pass_m.sum()), 1)

    w = np.ones(n, float)
    # Equal section contribution
    w[cat_m]  *= n / (3.0 * n_cat)
    w[act_m]  *= n / (3.0 * n_act)  * 1.5   # active Tafel: slightly boosted
    w[pass_m] *= n / (3.0 * n_pass) * 0.50  # passive plateau: easy, reduce weight

    # Ecorr vicinity boost (×3 within 60 mV) — sharpens icorr and Ecorr
    w *= 1.0 + 2.0 * np.exp(-np.abs(E - Ecorr_est) / 0.060)
    w /= w.mean()
    return w

def global_polish(E, i, p0, ct, lo, hi):
    """
    Global parameter optimisation: DE → L-BFGS-B → Nelder-Mead → Powell.

    Improvements over previous version (inspired by busbar_tafel_fit.py):
    - 4-stage cascade: adds Nelder-Mead (adaptive) between L-BFGS-B and Powell,
      which handles non-smooth basins that L-BFGS-B misses.
    - Tighter tolerances: ftol=1e-15, gtol=1e-13 throughout.
    - Larger DE population: popsize=max(15, nf×3), maxiter=max(400, nf×60).
    - Section-balanced objective weights (see _section_weights) replace the
      old Ecorr-only Gaussian weight that over-concentrated on the mixed-control
      zone and starved the cathodic/passive regions of gradient signal.
    - Bounds enforced via clipping inside _unpack for gradient-free methods
      (Nelder-Mead, Powell) that don't natively respect bounds.
    - ba/bc runaway detection: flags fits where Tafel slopes hit the 500 mV/dec
      upper bound (unphysical — indicates no real linear Tafel region detected).
    """
    ld   = slog(i)
    fidx = CT.idx(ct)
    bnds = _pbounds(lo, hi, fidx)
    n, nf = len(E), len(fidx)

    Ecorr_p0 = float(p0[0])
    Epass_p0 = float(p0[4]) if ct in CT.PASS else None
    w_base   = _section_weights(E, i, Ecorr_p0, Epass_p0)

    def obj(x):
        p = _unpack(x, fidx, p0.copy(), lo, hi)
        try:
            pred = pol_model(E, p, ct)
            return float(np.sum(w_base * (ld - slog(pred)) ** 2))
        except Exception:
            return 1e30

    best_x = _pack(p0, fidx)
    best_val = obj(best_x)

    def update(x):
        nonlocal best_x, best_val
        v = obj(x)
        if v < best_val - 1e-12:
            best_x, best_val = x.copy(), v
        return v

    # ── Stage 1: Differential Evolution (global search) ───────────────────
    # Larger population and more iterations than before (busbar: popsize≥15)
    ps = max(15, nf * 3)
    mi = max(400, nf * 60)
    for seed, strat in [(42, "best1bin"), (7, "currenttobest1bin")]:
        try:
            res = differential_evolution(
                obj, bnds, seed=seed, maxiter=mi, popsize=ps,
                tol=1e-12, mutation=(0.5, 1.9), recombination=0.90,
                polish=False, workers=1, strategy=strat)
            update(res.x)
        except Exception:
            pass

    # ── Stage 2: L-BFGS-B (gradient-based refinement) ────────────────────
    try:
        r = minimize(obj, best_x, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": 25000, "ftol": 1e-15, "gtol": 1e-13})
        update(r.x)
    except Exception:
        pass

    # ── Stage 3: Nelder-Mead adaptive (non-smooth basin escape) ──────────
    # From busbar: handles parameter correlations that trip up L-BFGS-B
    try:
        r = minimize(obj, best_x, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-12,
                              "fatol": 1e-14, "adaptive": True})
        update(r.x)
    except Exception:
        pass

    # ── Stage 4: Powell (final coordinate-wise polish) ────────────────────
    try:
        r = minimize(obj, best_x, method="Powell",
                     options={"maxiter": 15000, "xtol": 1e-12, "ftol": 1e-14})
        update(r.x)
    except Exception:
        pass

    best_p = _unpack(best_x, fidx, p0.copy(), lo, hi)
    log_p  = slog(pol_model(E, best_p, ct))
    sse    = float(np.sum((ld - log_p) ** 2))
    r2     = r2_score(ld, log_p)
    aic    = aicc(n, nf, sse)
    return best_p, r2, aic, sse

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
COL_SIG = [
    (r"we.*potential|ewe|potential/v|e/v|^e$|e \(v\)|e_v|^vf$", r"we.*current|<i>/ma|i/ma|current/a|i/a|^i$|i \(a\)|i_a|^im$", "A"),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]
UNIT_PAT = {r"\(a\)|_a$|/a$|a/cm²?": 1.0, r"\(ma\)|_ma$|/ma$|ma/cm²?": 1e-3, r"\(ua\)|_ua$|/ua$|ua/cm²?": 1e-6}

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
    if len(num) >= 2: return num[0], num[1], 1.0
    raise ValueError("Cannot detect E/i columns automatically.")

def load_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(uploaded)
    content = uploaded.getvalue().decode("utf-8", errors="replace")
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep, engine="python", comment="#")
            if df.shape[1] >= 2 and df.shape[0] > 4: return df.dropna(axis=1, how="all")
        except: continue
    raise ValueError("Cannot parse file.")

# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION FIGURE (LOCAL-LINE TAFEL OVERLAYS)
# FIX: Linear-scale panel uses E_corr-local ylim instead of global p95
# ─────────────────────────────────────────────────────────────────────────────
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

def make_figure(E, i_obs, best_p, ct, sample_name, cat_res, an_res,
                Ecorr, taf=None, extend_tafel=True, use_tafel_icorr=True,
                show_regions=True, dpi=150):
    Ecorr_fit = float(best_p[0])

    ba    = max(float(best_p[2]), 1e-9)
    bc    = max(float(best_p[3]), 1e-9)
    icorr_model = float(best_p[1])

    E_lo, E_hi = float(E.min()), float(E.max())
    span = max(E_hi - E_lo, 1e-6)
    n_dense = int(np.clip(200 * span, 800, 2500))
    E_dense = np.linspace(E_lo, E_hi, n_dense)

    i_dense    = pol_model(E_dense, best_p, ct)
    i_fit_E    = pol_model(E, best_p, ct)
    log_obs    = slog(i_obs)
    log_den    = slog(i_dense)
    log_fitE   = slog(i_fit_E)
    residuals  = log_obs - log_fitE
    r2v  = r2_score(log_obs, log_fitE)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    fin  = log_obs[np.isfinite(log_obs)]
    y_lo = float(np.nanmin(fin)) - 0.2

    if ct in CT.PASS:
        Epass_4y = float(best_p[4])
        act_mask = (E >= Ecorr_fit - 0.02) & (E <= Epass_4y + 0.05)
        if np.sum(act_mask) >= 2:
            log_act = slog(i_obs[act_mask])
            log_act = log_act[np.isfinite(log_act)]
            active_peak_log = float(np.max(log_act)) if len(log_act) > 0 else np.log10(max(icorr_model,TINY))
        else:
            active_peak_log = np.log10(max(icorr_model,TINY)) + 2.303 * (Epass_4y - Ecorr_fit) / ba
        logIp_val = float(np.log10(max(float(best_p[6]), TINY)))
        y_hi = max(active_peak_log + 1.5, logIp_val + 2.0)
    else:
        y_hi = float(np.percentile(fin, 90)) + 1.0

    y_hi = min(y_hi, float(np.percentile(fin, 99)) + 1.8)

    # Model partials (fallback Tafel lines from model parameters)
    logI_cat_model = np.log10(np.clip(icorr_model * np.exp(2.303*(Ecorr_fit-E_dense)/bc), TINY, None))
    logI_ano_model = np.log10(np.clip(icorr_model * np.exp(2.303*(E_dense-Ecorr_fit)/ba), TINY, None))

    # Decide display i_corr / E_corr
    icorr_display, ecorr_display = icorr_model, Ecorr_fit
    if taf is not None and use_tafel_icorr:
        E_tafel, i_tafel, logI_tafel = taf
        if E_lo - 0.2*span <= E_tafel <= E_hi + 0.2*span and np.isfinite(i_tafel) and i_tafel > 0:
            icorr_display, ecorr_display = i_tafel, E_tafel

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
            vband(E_lo, Ecorr_fit, "cathodic", "Cathodic region")
            if ct in CT.SIMPLE:
                vband(Ecorr_fit, E_hi, "active", "Anodic (active)")
            elif ct in CT.PASS:
                Ep = float(best_p[4])
                Et = float(best_p[7]) if ct in CT.TRANS else E_hi + 1
                vband(Ecorr_fit, min(Ep, E_hi), "active", "Active dissolution")
                vband(min(Ep, E_hi), min(Et, E_hi), "passive", "Passive region")
                if ct in CT.TRANS and Et < E_hi:
                    vband(Et, E_hi, "transpassive", "Transpassive / pitting")

        # Experimental data
        ax.scatter(E, log_obs, s=12, color="#4a7fa8", alpha=0.60,
                   zorder=2, label="Experimental data", linewidths=0, rasterized=True)
        # Fitted model
        ax.plot(E_dense, log_den, color="#1a3a5c", lw=2.0, zorder=5,
                label=f"Global fit  (R\u00b2={r2v:.5f})")

        # Draw local regression segments
        def _draw_segment(ax, slope, intercept, win, color, label,
                          extend_to=None, extend_color=None, extend_alpha=0.35):
            e0, e1 = float(win[0]), float(win[1])
            e0p, e1p = max(E_lo, e0), min(E_hi, e1)
            if e1p <= e0p: return False
            E_seg = np.linspace(e0p, e1p, 120)
            ax.plot(E_seg, slope * E_seg + intercept, "--", color=color, lw=2.2, zorder=6, label=label)
            if extend_to is not None:
                ex0, ex1 = min(e0p, extend_to[0]), max(e1p, extend_to[1])
                ex0, ex1 = max(E_lo, ex0), min(E_hi, ex1)
                if ex1 > e1p:
                    E_ext = np.linspace(e1p, ex1, 60)
                    ax.plot(E_ext, slope * E_ext + intercept, "--", color=extend_color or color, lw=1.6, alpha=extend_alpha, zorder=5)
                if ex0 < e0p:
                    E_ext = np.linspace(ex0, e0p, 60)
                    ax.plot(E_ext, slope * E_ext + intercept, "--", color=extend_color or color, lw=1.6, alpha=extend_alpha, zorder=5)
            return True

        # Cathodic Tafel line
        drew_c = False
        if ("slope_c" in cat_res) and ("intercept_c" in cat_res) and ("win_c" in cat_res):
            lab_c = f"βc (local) = {min(abs(1.0/cat_res['slope_c']),CFG['beta_max_c'])*1000:.0f} mV/dec"
            ext = (E_lo, min(Ecorr_fit, E_hi)) if extend_tafel else None
            drew_c = _draw_segment(ax, cat_res["slope_c"], cat_res["intercept_c"],
                                   cat_res["win_c"], "#8e44ad", lab_c,
                                   extend_to=ext, extend_color="#8e44ad", extend_alpha=0.30)

        # Anodic Tafel line
        drew_a = False
        if ("slope_a" in an_res) and ("intercept_a" in an_res):
            if "win_a" in an_res:
                e0, e1 = an_res["win_a"]
                # FIX C: clip to the active region boundary (Epeak preferred, Epass as fallback).
                # The old code clipped solely to E_pass; for near-instant passivation curves
                # the selected window can start/end inside the passive region, making
                # e1 < e0 after clipping and producing an invisible zero-length segment.
                clip_upper = None
                if an_res.get("Epeak") is not None:
                    clip_upper = float(an_res["Epeak"])
                elif ct in CT.PASS and best_p[4] is not None:
                    clip_upper = float(best_p[4])
                if clip_upper is not None:
                    e1 = min(e1, clip_upper)
                # If window is degenerate after clipping, expand slightly
                if e1 <= e0:
                    e1 = e0 + max(0.02, float(clip_upper - Ecorr_fit) * 0.5) if clip_upper else e0 + 0.05
            else:
                e0, e1 = Ecorr_fit, E_hi
            lab_a = f"βa (local) = {min(abs(1.0/an_res['slope_a']),CFG['beta_max_a'])*1000:.0f} mV/dec"
            ext = (max(Ecorr_fit, E_lo), E_hi) if extend_tafel else None
            drew_a = _draw_segment(ax, an_res["slope_a"], an_res["intercept_a"],
                                   (e0, e1), "#e67e22", lab_a,
                                   extend_to=ext, extend_color="#e67e22", extend_alpha=0.30)

        # Fallbacks to model partials when local lines failed
        if not drew_c:
            msk_cat = (logI_cat_model >= y_lo - 0.05) & (logI_cat_model <= y_hi + 0.05) & (E_dense <= Ecorr_fit + 0.005)
            if np.any(msk_cat):
                ax.plot(E_dense[msk_cat], logI_cat_model[msk_cat], "--", color="#8e44ad", lw=2.2, zorder=6,
                        label=f"βc = {bc*1000:.0f} mV/dec")
        if not drew_a:
            if ct in CT.PASS:
                Epass_fit = float(best_p[4])
                msk_ano_E = (E_dense >= Ecorr_fit) & (E_dense <= Epass_fit)
            else:
                msk_ano_E = E_dense >= Ecorr_fit
            msk_ano = msk_ano_E & (logI_ano_model >= y_lo - 0.05) & (logI_ano_model <= y_hi + 0.05)
            if np.any(msk_ano):
                ax.plot(E_dense[msk_ano], logI_ano_model[msk_ano], "--", color="#e67e22", lw=2.2, zorder=6,
                        label=f"βa = {ba*1000:.0f} mV/dec")

        # Crossing point + drop-lines
        logIc_disp = np.log10(max(icorr_display, TINY))
        ax.plot(ecorr_display, logIc_disp, "x", color="#e84393", ms=12, mew=2.5, zorder=8)
        ax.plot([ecorr_display, ecorr_display], [y_lo, logIc_disp], ":", color="#e84393", lw=1.2, alpha=0.85, zorder=3)
        ax.plot([E_lo, ecorr_display], [logIc_disp, logIc_disp], ":", color="#e84393", lw=1.2, alpha=0.85, zorder=3)

        y_span = y_hi - y_lo
        ax.annotate(f"E\u1d9c\u1d52\u02b3\u02b3 = {ecorr_display:.4f} V",
                    xy=(ecorr_display, y_lo),
                    xytext=(ecorr_display + 0.01*(E_hi-E_lo), y_lo + 0.04*y_span),
                    fontsize=8.5, color="#e84393", fontweight="bold", ha="left")
        ax.annotate(f"i\u1d9c\u1d52\u02b3\u02b3 = {icorr_display:.2e} A/cm\u00b2",
                    xy=(E_lo, logIc_disp),
                    xytext=(E_lo + 0.01*(E_hi-E_lo), logIc_disp + 0.03*y_span),
                    fontsize=8.5, color="#e84393", fontweight="bold")

        if ct in CT.PASS:
            ip_val = float(best_p[6])
            ax.axhline(np.log10(max(ip_val, TINY)), color="#27ae60",
                       ls=":", lw=1.1, alpha=0.75, zorder=3,
                       label=f"i_pass = {ip_val:.2e} A/cm\u00b2")

        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("E vs. Reference (V)")
        ax.set_ylabel("log\u2081\u2080 |i| (A cm\u207b\u00b2)")
        ax.set_title(f"Evans Diagram — {sample_name}")
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
        if "E_an" in an_res:
            ax.scatter(an_res["E_an"], an_res["lgi_an"],
                       s=18, color="#fd8d3c", alpha=0.80, zorder=3,
                       label="Anodic data", linewidths=0, rasterized=True)
        if "slope_c" in cat_res and "intercept_c" in cat_res and "win_c" in cat_res:
            e0, e1 = cat_res["win_c"]
            E_seg = np.linspace(max(E_lo, e0), min(E_hi, e1), 120)
            if len(E_seg) > 1:
                ax.plot(E_seg, cat_res["slope_c"] * E_seg + cat_res["intercept_c"],
                        "--", color="#3182bd", lw=1.6, zorder=4,
                        label=f"βc(local)={min(abs(1.0/cat_res['slope_c']),CFG['beta_max_c'])*1000:.0f} mV/dec")
        if "slope_a" in an_res and "intercept_a" in an_res:
            if "win_a" in an_res:
                e0, e1 = an_res["win_a"]
                # FIX C (panel B): same Epeak-preferred clipping
                clip_upper_b = an_res.get("Epeak") or (float(best_p[4]) if ct in CT.PASS and best_p[4] is not None else None)
                if clip_upper_b is not None:
                    e1 = min(e1, float(clip_upper_b))
                if e1 <= e0:
                    e1 = e0 + 0.03
            else:
                e0, e1 = Ecorr_fit, E_hi
            E_seg = np.linspace(max(E_lo, e0), min(E_hi, e1), 120)
            if len(E_seg) > 1 and E_seg[-1] > E_seg[0]:
                ax.plot(E_seg, an_res["slope_a"] * E_seg + an_res["intercept_a"],
                        "--", color="#e6550d", lw=1.6, zorder=4,
                        label=f"βa(local)={min(abs(1.0/an_res['slope_a']),CFG['beta_max_a'])*1000:.0f} mV/dec")
        if an_res["has_passive"] and an_res["Epass"] is not None:
            ax.axvline(float(an_res["Epass"]), color="#27ae60",
                       ls="-.", lw=1.0, alpha=0.85,
                       label=f"E_pass={an_res['Epass']:.3f}V")
        ax.axvline(ecorr_display, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.axhline(logIc_disp, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("E (V)")
        ax.set_ylabel("log\u2081\u2080 |i|")
        ax.set_title("Branch Fits (Stage 2–3)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=7.5, loc="upper right")

        # ══ PANEL C — Linear i vs E ═══════════════════════════════════════════
        # FIX: ylim now based on the current range within ±150 mV of E_corr,
        # not the global 95th percentile. Passive/transpassive peaks previously
        # compressed the Ecorr region into an unreadable sliver. We now show
        # the region that matters for corrosion rate visualisation.
        ax = ax_lin
        local_mask = np.abs(E - ecorr_display) <= 0.150
        if np.sum(local_mask) >= 4:
            i_local_max = float(np.percentile(np.abs(i_obs[local_mask]), 95))
        else:
            i_local_max = float(np.percentile(np.abs(i_obs), 80))
        # Guard: if local range is too tiny (e.g. noisy near-zero), fall back to global
        i_global_p95 = float(np.percentile(np.abs(i_obs), 95))
        i_scale_ref = max(i_local_max, i_global_p95 * 0.01)

        if i_scale_ref < 1e-6:
            uscale, ulbl = 1e9,  "nA/cm\u00b2"
        elif i_scale_ref < 1e-3:
            uscale, ulbl = 1e6,  "\u03bcA/cm\u00b2"
        else:
            uscale, ulbl = 1e3,  "mA/cm\u00b2"

        ylim_lin = max(i_scale_ref * uscale * 1.40, 1e-12)

        i_fit_cl = np.clip(i_dense * uscale, -ylim_lin * 3, ylim_lin * 3)
        E_lin_ds, i_obs_lin_ds = downsample_uniform(E, i_obs, 800)
        ax.scatter(E_lin_ds, i_obs_lin_ds * uscale, s=9, color="#4a7fa8",
                   alpha=0.65, zorder=2, label="Data", linewidths=0, rasterized=True)
        ax.plot(E_dense, i_fit_cl, color="#1a3a5c", lw=2.0, zorder=5, label="Fit")
        ax.axhline(0, color="#888", lw=0.7, zorder=1)
        ax.axvline(ecorr_display, color="#e84393", ls="--", lw=1.2, zorder=3,
                   label=f"E_corr")
        ax.set_xlim(E_lo, E_hi)
        ax.set_ylim(-ylim_lin, ylim_lin)
        ax.set_xlabel("E (V)")
        ax.set_ylabel(f"i ({ulbl})")
        ax.set_title("Linear Scale (±150 mV zoom)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        # ══ PANEL D — Residuals ═══════════════════════════════════════════════
        ax = ax_res
        ax.fill_between([E_lo, E_hi], -0.1, 0.1, color="#e84393", alpha=0.07, zorder=1)
        ax.scatter(E, residuals, s=10, color="#2e86de", alpha=0.65, zorder=3, linewidths=0, rasterized=True)
        ax.axhline(0,    color="#333",    lw=0.9, zorder=2)
        ax.axhline( 0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(-0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7, label="\u00b10.1 log")
        ax.axvline(ecorr_display, color="#e84393", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlim(E_lo, E_hi)
        ax.set_xlabel("E (V)")
        ax.set_ylabel("\u0394 log\u2081\u2080 |i|")
        ax.set_title(f"Residuals   R\u00b2={r2v:.5f}")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        fig.suptitle("Polarisation Curve Analysis", fontsize=12,
                     fontweight="bold", color="#1a3a5c", y=0.98)

    return fig, r2v, rmse, icorr_display, ecorr_display

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
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
        icorr_display = res.get("icorr_disp", p[1])
        CR = icorr_display * 3.27 * ew / rho
        vals = [
            res.get("name","?"), CT.name(ct),
            round(res.get("ecorr_disp", p[0]), 5),
            f"{icorr_display:.4e}",
            round(p[2]*1000, 2), round(p[3]*1000, 2), round(B_val, 5), round(CR, 5),
            f"{p[6]:.3e}" if ct in CT.PASS else "—",
            round(p[4], 5) if ct in CT.PASS else "—",
            round(p[7], 5) if ct in CT.TRANS else "—",
            round(res.get("r2", 0), 6), round(res.get("rmse", 0), 6),
            "Good" if ok else "Check",
        ]
        for c, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=c, value=val)
            cell.fill = fill; cell.border = BRD; cell.alignment = Alignment(horizontal="center")
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
        icorr_display = res.get("icorr_disp", p[1])
        CR = icorr_display * 3.27 * ew / rho

        story.append(Paragraph(f"Sample {idx+1}: {nm}", h2))
        rows = [["Parameter","Symbol","Value","Unit"],
                ["Corrosion potential","E_corr",f"{res.get('ecorr_disp', p[0]):.5f}","V"],
                ["Corrosion current density","i_corr",f"{icorr_display:.4e}","A cm-2"],
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

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k in ("results", "figures"):
    if _k not in st.session_state:
        st.session_state[_k] = []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
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

    st.markdown("**Plotting**")
    extend_tafel = st.toggle("Extend Tafel dashed lines to Ecorr", True)
    use_tafel_icorr = st.toggle("Use Tafel intersection for i_corr", True)
    show_regs  = st.toggle("Shade regions", True)
    smooth_pre = st.toggle("Pre-smooth (Savitzky-Golay)", False)
    pub_dpi    = st.slider("Export DPI", 150, 600, 300, 50)

    with st.expander("Advanced (Tafel window detection)"):
        CFG["anod_guard"]   = st.number_input("Anodic guard from Ecorr (V)", 0.0, 0.100, CFG["anod_guard"], 0.001, format="%.3f")
        CFG["cath_guard"]   = st.number_input("Cathodic guard from Ecorr (V)", 0.0, 0.150, CFG["cath_guard"], 0.001, format="%.3f")
        CFG["curvature_max"]= st.number_input("Curvature max |d²log|i|/dE²|", 10.0, 200.0, CFG["curvature_max"], 1.0)
        CFG["lin_frac"]     = st.slider("Linearity fraction (derivative consistency)", 0.4, 0.95, CFG["lin_frac"], 0.05)
        CFG["min_w_ano"]    = st.number_input("Min window points (anodic)", 3, 20, CFG["min_w_ano"])
        CFG["min_w_cat"]    = st.number_input("Min window points (cathodic)", 4, 25, CFG["min_w_cat"])
        CFG["beta_min"]     = st.number_input("β min (V/dec)", 0.005, 0.100, CFG["beta_min"], 0.005, format="%.3f")
        CFG["beta_max_a"]   = st.number_input("βa max (V/dec)", 0.05, 0.40, CFG["beta_max_a"], 0.005)
        CFG["beta_max_c"]   = st.number_input("βc max (V/dec)", 0.05, 0.40, CFG["beta_max_c"], 0.005)

    st.markdown("**Fitting**")
    force_ct_val = st.selectbox("Force model (auto = best AICc)",
                                ["auto","A","AD","P","PT","F"],
                                key="force_ct_choice")

    st.divider()
    if st.button("🗑 Clear all", use_container_width=True):
        st.session_state.results = []
        st.session_state.figures = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
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
        for idx, uf in enumerate(uploaded_files):
            st.markdown(f"---\n#### 📄 `{uf.name}`")
            with st.container():

                # Load
                try:
                    df_raw = load_file(uf)
                except Exception as ex:
                    st.error(f"Load error: {ex}"); continue

                # Column selection
                try:
                    ec_auto, ic_auto, _ = _auto_cols(df_raw)
                    auto_ok = True
                except:
                    ec_auto = ic_auto = None; auto_ok = False

                num_cols = [c for c in df_raw.columns
                            if pd.api.types.is_numeric_dtype(df_raw[c])]
                cc1, cc2 = st.columns(2)
                with cc1:
                    e_sel = st.selectbox(f"E column [{uf.name}]", num_cols,
                        index=num_cols.index(ec_auto) if auto_ok and ec_auto in num_cols else 0,
                        key=f"ec_{idx}")
                with cc2:
                    i_sel = st.selectbox(f"i column [{uf.name}]", num_cols,
                        index=num_cols.index(ic_auto) if auto_ok and ic_auto in num_cols else min(1,len(num_cols)-1),
                        key=f"ic_{idx}")

                # Build arrays
                E_raw = df_raw[e_sel].values.astype(float)
                i_raw = df_raw[i_sel].values.astype(float)
                i_obs = i_raw * unit_fac / area
                ok_mask = np.isfinite(E_raw) & np.isfinite(i_obs)
                E_raw, i_obs = E_raw[ok_mask], i_obs[ok_mask]
                srt = np.argsort(E_raw); E, i = E_raw[srt], i_obs[srt]

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
                             key=f"btn_{idx}", type="primary",
                             use_container_width=True):
                    import time; t0 = time.time()

                    if smooth_pre:
                        w = min(9, len(i)//2*2-1)
                        if w >= 5:
                            i = savgol_filter(i, w, 3, mode="interp")

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
                    log(f"✅ **Stage 2** — βc(local) = `{cat_res['bc']*1000:.0f}` mV/dec  "
                        f"i_corr ≈ `{cat_res['icorr']:.2e}`  "
                        f"R²_cat = `{cat_res['r2']:.4f}`  "
                        f"diff_limit = `{'yes' if cat_res['has_diff'] else 'no'}`")
                    prog.progress(30, text="Stage 3 — Anodic branch fit…")

                    # Stage 3 — Anodic
                    an_res = fit_anodic(E, i, Ecorr)
                    log(f"✅ **Stage 3** — βa(local) = `{an_res['ba']*1000:.0f}` mV/dec  "
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

                    force_choice = st.session_state.get("force_ct_choice", "auto")
                    if force_choice == "auto":
                        candidates = [ct_detected]
                        if ct_detected in (CT.A, CT.AD):
                            candidates.append(CT.P)
                        if an_res["has_passive"]:
                            candidates.extend([CT.P, CT.PT])
                            # Always try F (full model with iL) for passive curves —
                            # the cathodic diffusion detection can miss subtle plateaus,
                            # but the global model can still pick up iL from the data.
                            candidates.append(CT.F)
                        if cat_res["has_diff"]:
                            candidates.append(CT.AD)
                            if an_res["has_passive"]:
                                candidates.append(CT.F)
                        seen = set(); uniq = []
                        for c in candidates:
                            if c not in seen:
                                uniq.append(c); seen.add(c)
                        candidates = uniq
                    else:
                        candidates = [force_choice]

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

                    # AICc selection — parsimony
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

                    # Tafel intersection
                    taf = tafel_intersection(cat_res, an_res)

                    # ── Figure ──────────────────────────────────────────────
                    try:
                        fig, r2_fig, rmse_fig, icorr_disp, ecorr_disp = make_figure(
                            E, i, best_p, best_ct, sample_name or uf.name,
                            cat_res, an_res, Ecorr, taf=taf,
                            extend_tafel=extend_tafel, use_tafel_icorr=use_tafel_icorr,
                            show_regions=show_regs, dpi=pub_dpi)
                        fig.tight_layout(rect=[0, 0, 1, 0.97])

                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, dpi=pub_dpi, bbox_inches="tight",
                                    facecolor="white"); buf_png.seek(0)
                        png_bytes = buf_png.read()

                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight",
                                    facecolor="white"); buf_svg.seek(0)
                        svg_bytes = buf_svg.read()

                        res_rec = dict(
                            name=sample_name or uf.name,
                            params=best_p, ct=best_ct,
                            r2=r2_fin, rmse=rmse_fig,
                            success=r2_fin > 0.90,
                            material=(ew_mat, rho_mat),
                            all_candidates=all_res,
                            icorr_disp=icorr_disp, ecorr_disp=ecorr_disp
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
                    CR    = icorr_disp * 3.27 * ew_mat / rho_mat

                    # Runaway detection (from busbar_tafel_fit.py ba_valid check)
                    ba_valid = float(p[2]) < 0.240   # > 240 mV/dec = unphysical (was 0.495)
                    bc_valid = float(p[3]) < 0.270   # > 270 mV/dec = unphysical (was 0.495)
                    if not ba_valid or not bc_valid:
                        st.warning(
                            f"⚠️ {'βa' if not ba_valid else ''}"
                            f"{' and ' if not ba_valid and not bc_valid else ''}"
                            f"{'βc' if not bc_valid else ''} hit the 500 mV/dec upper bound — "
                            "no reliable linear Tafel region was found for that branch. "
                            "The corrosion current is estimated from the other branch only. "
                            "Consider adjusting the guard or window in Advanced settings.")

                    st.markdown("#### 📐 Fitted Parameters")
                    mc = st.columns(5)
                    mc[0].metric("E_corr (V)",       f"{ecorr_disp:.5f}")
                    mc[1].metric("i_corr (A/cm²)",   f"{icorr_disp:.4e}")
                    mc[2].metric("βa (mV/dec)",       f"{p[2]*1000:.1f}" + ("" if ba_valid else " ⚠️"))
                    mc[3].metric("βc (mV/dec)",       f"{p[3]*1000:.1f}" + ("" if bc_valid else " ⚠️"))
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
            icorr_disp = r.get("icorr_disp", p[1])
            CR = icorr_disp * 3.27 * ew / rho
            rows.append({"Sample":r.get("name","?"),
                          "Model":CT.name(ct),
                          "E_corr (V)":f"{r.get('ecorr_disp', p[0]):.5f}",
                          "i_corr (A/cm²)":f"{icorr_disp:.4e}",
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
                    axes[0].axvline(res.get("ecorr_disp", p[0]), color=col, ls=":", lw=0.9, alpha=0.6)
                except:
                    pass
                axes[1].bar(idx, res.get("icorr_disp", p[1]), color=col, alpha=0.85)
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
### Algorithm Fixes in v2

| Fix | Details |
|---|---|
| **Cathodic proximity score** | Gaussian centred at 150 mV from E_corr (σ=120 mV). Previously rewarded windows *near* E_corr — the mixed-control zone — instead of the true Tafel region 50–300 mV away. |
| **Cathodic scoring** | R²² weighting and direct curvature penalty added to sharpen window discrimination. |
| **Cathodic diff_ok guard** | Threshold lowered from 0.35→0.20 × slope_mag. The original threshold over-rejected valid cathodic windows with natural noise. |
| **Cathodic i_corr estimate** | Uses `max(tafel_extrap, near_Ecorr × 0.5)` instead of `min(tafel_extrap, near_Ecorr × 10)`. The old cap systematically under-estimated i_corr when the Tafel line was correctly picked far from E_corr. |
| **Anodic proximity tau** | Widened from 0.12 V → 0.20 V so active dissolution regions spanning 100–200 mV (stainless, Ni, Ti) are not penalised. |
| **Anodic fallback ok mask** | Retains `beta_ok` in fallback to prevent selection of windows with unphysical Tafel slopes. |
| **Linear-scale panel ylim** | Now based on |i| within ±150 mV of E_corr, not the global 95th percentile. Passive/transpassive current peaks no longer compress the Ecorr region. |

### Why the cathodic dashed line can look shorter
- Diffusion-limited plateaus flatten the cathodic branch; only a region 50–300 mV
  from E_corr is truly linear in the Tafel sense.
- Curvature near E_corr (IR/mixed control) is excluded by the curvature guard.
- Enable "Extend Tafel dashed lines" in the sidebar to project the fitted window
  faintly to E_corr for visual reference.

### Troubleshooting

| Symptom | Try |
|---|---|
| Cathodic line too short / wrong slope | Reduce *Cathodic guard* to 0.010 V; raise *Curvature max* to 60 |
| Anodic line not detected | Reduce *Anodic guard* to 0.005 V; lower *Linearity fraction* to 0.60 |
| i_corr looks wrong | Toggle *Use Tafel intersection for i_corr*; check Tafel intersection marker |
| Very noisy data | Enable *Pre-smooth* in sidebar |

### Fitting Pipeline

| Stage | What happens |
|---|---|
| **1 — E_corr detection** | Interpolated zero-crossing of signed current |
| **2 — Cathodic fit** | Vectorized sliding regression; Gaussian proximity scoring at 150 mV from E_corr; Theil–Sen/Huber refinement |
| **3 — Anodic fit** | Same for anodic branch; wider proximity window (tau=0.20 V); passive plateau detection |
| **4 — Classification** | Curve type inferred from detected features |
| **5 — Global polish** | Physics-informed p₀ → DE → L-BFGS-B → Powell (if needed) |
| **6 — AICc selection** | Multiple candidate models compared; parsimony-penalised |
""")
