"""
Polarization Curve Fitter — Publication-Grade Streamlit App
============================================================
Architecture (inspired by TAFEL-PRO v2.1):
  Stage 1 – Detect E_corr from interpolated zero-crossing
  Stage 2 – Fit cathodic branch independently (sliding-window Tafel + diffusion limit)
  Stage 3 – Fit anodic branch independently (sliding-window Tafel + passive + transpassive)
  Stage 4 – Classify curve type (A / AD / P / PT / Full)
  Stage 5 – Assemble physics-informed p0 → DE + L-BFGS-B + Nelder-Mead global polish
  Stage 6 – AICc model selection
  Stage 7 – Publication figure + Excel + PDF export
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import io, os, zipfile, warnings, traceback, re
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
    "cathodic":     ("#6baed6", 0.12),
    "active":       ("#fd8d3c", 0.12),
    "passive":      ("#74c476", 0.14),
    "transpassive": ("#9e9ac8", 0.16),
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
    return np.where(xk >= 0,
                    1.0 / (1.0 + np.exp(-xk)),
                    np.exp(xk) / (1.0 + np.exp(xk)))

def sm(y, w=11, p=3):
    """Savitzky-Golay smooth with auto window."""
    n = len(y)
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(5, w) if w >= 5 else n
    if w > n or w < 5: return y.copy()
    return savgol_filter(y, w, min(p, w - 1))

def r2_score(yt, yp):
    sr = np.sum((yt - yp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    return float(max(0.0, 1.0 - sr / st)) if st > 1e-30 else 0.0

def aicc(n, k, sse):
    """Corrected AIC."""
    if n <= k + 1 or sse <= 0: return 1e30
    return n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)

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
    """Interpolated zero-crossing of signed current."""
    sc = np.where(np.diff(np.sign(i)))[0]
    if len(sc) == 0:
        idx = int(np.argmin(np.abs(i)))
        return float(E[idx]), idx
    # Prefer cathodic→anodic crossing (negative→positive)
    anodic_cross = [k for k in sc if i[k] < 0 and i[k+1] > 0]
    idx_sc = anodic_cross[0] if anodic_cross else sc[0]
    denom  = i[idx_sc+1] - i[idx_sc]
    Ecorr  = float(E[idx_sc] - i[idx_sc] * (E[idx_sc+1] - E[idx_sc]) / denom) \
             if abs(denom) > TINY else float(E[idx_sc])
    return Ecorr, idx_sc

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CATHODIC BRANCH FIT
# ══════════════════════════════════════════════════════════════════════════════

def fit_cathodic(E, i, Ecorr):
    """
    Sliding-window linear regression on log|i_cat| vs E.
    KEY: prefer the window CLOSEST to Ecorr (true Tafel region, least IR/conc effects).
    Scoring: R2 * closeness_weight, NOT pure R2.
    Returns: bc, icorr, iL, has_diffusion
    """
    cat = i < 0
    if np.sum(cat) < 4:
        return dict(bc=0.120, icorr=1e-8, iL=1e-2, has_diff=False, r2=0.0)

    Ec  = E[cat]; lgi = slog(i[cat])
    si  = np.argsort(Ec); Ec, lgi = Ec[si], lgi[si]

    # ── Sliding-window: score = R2 x closeness-to-Ecorr ──
    # Exclude points within 20 mV of Ecorr (near zero-crossing, not Tafel)
    TAFEL_GUARD = 0.020   # V — minimum distance from Ecorr
    valid = Ec < (Ecorr - TAFEL_GUARD)
    Ec_v  = Ec[valid]; lgi_v = lgi[valid]

    best_score, best_sl, best_b, best_r2 = 0.0, -8.0, -6.0, 0.0
    mp = max(4, len(Ec_v) // 5)
    for s0 in range(len(Ec_v) - mp + 1):
        for s1 in range(s0 + mp, min(s0 + 25, len(Ec_v) + 1)):
            try:
                sl, b, r, *_ = linregress(Ec_v[s0:s1], lgi_v[s0:s1])
                if sl < 0 and r**2 > 0.90 and 0.020 < abs(1/sl) < 0.350:
                    # Windows ending closer to Ecorr (but outside guard) are preferred
                    dist = abs(Ec_v[s1-1] - Ecorr)
                    closeness = np.exp(-dist / 0.20)
                    score = r**2 * closeness
                    if score > best_score:
                        best_score, best_sl, best_b, best_r2 = score, sl, b, r**2
            except:
                continue

    if abs(best_sl) < 0.5:
        try:
            best_sl, best_b, r, *_ = linregress(Ec_v if len(Ec_v)>3 else Ec, lgi_v if len(Ec_v)>3 else lgi)
            best_r2 = r**2
        except:
            best_sl, best_b, best_r2 = -3.0, -4.0, 0.0

    bc    = min(abs(1.0 / best_sl), 0.300) if abs(best_sl) > 0.5 else 0.120
    icorr = max(float(10 ** (best_b + best_sl * Ecorr)), 1e-15)

    # ── Diffusion limit ──
    iL     = None; has_diff = False
    if len(Ec) > 8:
        lgi_sm  = sm(lgi, min(9, (len(Ec) // 2) * 2 - 1))
        dlg     = np.abs(np.gradient(lgi_sm, Ec))
        flat    = dlg < max(np.percentile(dlg, 30), 0.5)
        runs    = [(k, list(g)) for k, g in groupby(enumerate(flat),
                   key=lambda x: x[1]) if k]
        if runs:
            best_run = max(runs, key=lambda x: len(x[1]))
            idxs     = [s[0] for s in best_run[1]]
            if len(idxs) >= 3 and abs(Ec[idxs[-1]] - Ec[idxs[0]]) > 0.03:
                iL       = float(np.median(np.abs(i[cat][si][idxs])))
                has_diff = True
    if iL is None:
        iL = icorr * 1e4

    return dict(bc=bc, icorr=icorr, iL=iL, has_diff=has_diff,
                r2=best_r2, E_cat=Ec, lgi_cat=lgi)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANODIC BRANCH FIT
# ══════════════════════════════════════════════════════════════════════════════

def fit_anodic(E, i, Ecorr):
    """
    Sliding-window Tafel fit for active region, then detect passive/transpassive.
    Returns: ba, has_passive, Epass, ip, has_trans, Etrans
    """
    ano = i > 0
    if np.sum(ano) < 4:
        return dict(ba=0.060, has_passive=False, Epass=None, ip=1e-6,
                    has_trans=False, Etrans=None, r2=0.0)

    Ea  = E[ano]; lgia = slog(i[ano])
    si  = np.argsort(Ea); Ea, lgia = Ea[si], lgia[si]

    # ── Best active Tafel window (near Ecorr) ──
    best_r2, best_sl = 0.0, 8.0
    mp = max(4, len(Ea) // 5)
    for s0 in range(max(0, len(Ea) - mp)):
        for s1 in range(s0 + mp, min(s0 + 20, len(Ea) + 1)):
            try:
                sl, b, r, *_ = linregress(Ea[s0:s1], lgia[s0:s1])
                if sl > 0 and r**2 > best_r2 and 0.020 < abs(1/sl) < 0.600:
                    best_r2, best_sl = r**2, sl
            except:
                continue

    ba = abs(1.0 / best_sl) if abs(best_sl) > 0.5 else 0.060

    # ── Passive plateau: flat region in log|i| vs E ──
    has_passive = False; Epass = None; ip = 1e-6
    has_trans   = False; Etrans = None

    if len(Ea) > 8:
        lgia_sm = sm(lgia, min(11, (len(Ea) // 2) * 2 - 1))
        dlg     = np.gradient(lgia_sm, Ea)
        adlg    = np.abs(dlg)

        # Flat = |d log i / dE| < threshold
        thr  = max(np.percentile(adlg, 30), 0.8)
        flat = adlg < thr

        runs = [(k, list(g)) for k, g in groupby(enumerate(flat),
                key=lambda x: x[1]) if k]
        for _, ri in runs:
            idxs = [s[0] for s in ri]
            span = abs(Ea[idxs[-1]] - Ea[idxs[0]])
            if len(idxs) >= 4 and span > 0.06:
                pre_mask = Ea < Ea[idxs[0]]
                if np.sum(pre_mask) < 1:
                    continue
                ip_cand = float(np.median(np.abs(i[ano][si][idxs])))
                # Confirm passive: current must be < peak current (drop from active)
                if ip_cand < float(np.max(np.abs(i[ano]))) * 0.7:
                    has_passive = True
                    Epass       = float(Ea[idxs[0]])
                    ip          = ip_cand

                    # Transpassive: rising current after passive end
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
                has_trans=has_trans, Etrans=Etrans,
                r2=best_r2, E_an=Ea, lgi_an=lgia)

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
# STAGE 5 — ASSEMBLE P0 & GLOBAL POLISH
# ══════════════════════════════════════════════════════════════════════════════

def _make_p0(Ecorr, cat, an, ct, E_max):
    """Build initial parameter vector from branch fits."""
    ic    = cat["icorr"]
    bc    = cat["bc"]
    ba    = an["ba"]
    iL    = cat["iL"]
    Ep    = an["Epass"]  if an["has_passive"] else E_max + 5.0
    ip    = an["ip"]     if an["has_passive"] else ic * 0.01
    Et    = an["Etrans"] if an["has_trans"]   else E_max + 5.0
    it    = ip * 0.1     if an["has_trans"]   else ic * 0.001

    return np.array([
        Ecorr, ic, ba, bc,
        Ep, 0.020, ip,
        Et, 0.015, it,
        iL,
    ])

def _build_bounds(Ecorr, cat, an, ct, E_min, E_max, E_span):
    ic   = max(cat["icorr"], 1e-14)
    iL   = max(cat["iL"],    1e-10)
    lo = np.array([
        E_min,                  # Ecorr
        max(ic * 1e-4, 1e-14),  # icorr
        0.025,                  # ba  — 25 mV/dec minimum (physical)
        0.025,                  # bc
        Ecorr,                  # Epass
        0.003,                  # k_pass
        max(ic * 1e-5, 1e-14),  # ip
        Ecorr + 0.05*E_span,    # Etrans
        0.003,                  # k_trans
        max(ic * 1e-6, 1e-14),  # itrans
        max(ic * 0.1, 1e-10),   # iL
    ])
    hi = np.array([
        E_max,
        min(ic * 1e5, 1.0),
        0.250,                  # ba  — 250 mV/dec maximum (physical)
        0.250,                  # bc
        E_max,
        0.100,
        min(ic * 1e4, 1.0),
        E_max + 0.1,
        0.100,
        min(ic * 1e6, 10.0),
        min(iL * 100, 10.0),
    ])
    # Clip p0 to bounds
    lo = np.minimum(lo, hi - 1e-12)
    return lo, hi

LOG_IDX = {1, 6, 9, 10}   # indices fitted in log-space

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
    Three-stage optimization:
      1) Differential Evolution  (global)
      2) L-BFGS-B                (local, fast)
      3) Nelder-Mead             (local, robust)
    Returns best_p, r2, aicc_val
    """
    ld    = slog(i)
    fidx  = CT.idx(ct)
    bnds  = _pbounds(lo, hi, fidx)
    n, nf = len(E), len(fidx)

    def obj(x):
        p = _unpack(x, fidx, p0.copy(), lo, hi)
        try:
            pred = pol_model(E, p, ct)
            return float(np.sum((ld - slog(pred)) ** 2))
        except:
            return 1e30

    best_x   = _pack(p0, fidx)
    best_val = obj(best_x)

    def update(x):
        nonlocal best_x, best_val
        v = obj(x)
        if v < best_val:
            best_x, best_val = x.copy(), v

    # 1. DE
    try:
        ps  = max(12, min(20, nf * 3))
        mi  = max(200, min(600, nf * 50))
        res = differential_evolution(obj, bnds, seed=42, maxiter=mi, popsize=ps,
                                     tol=1e-12, mutation=(0.5, 1.7),
                                     recombination=0.85, polish=False, workers=1)
        update(res.x)
    except:
        pass

    # 2. L-BFGS-B
    try:
        r = minimize(obj, best_x, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
        update(r.x)
    except:
        pass

    # 3. Nelder-Mead
    try:
        r = minimize(obj, best_x, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-14,
                              "adaptive": True})
        update(r.x)
    except:
        pass

    best_p = _unpack(best_x, fidx, p0.copy(), lo, hi)

    # Goodness of fit
    pred  = pol_model(E, best_p, ct)
    log_p = slog(pred)
    sse   = float(np.sum((ld - log_p) ** 2))
    r2    = r2_score(ld, log_p)
    aic   = aicc(n, nf, sse)

    return best_p, r2, aic, sse

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

# Column auto-detection signatures
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
# PUBLICATION FIGURE
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
    4-panel publication figure:
      A (wide): Evans diagram — data, fit, region shading, Tafel tangents, markers
      B:        Separate branch fits (cathodic / anodic)
      C:        Linear i vs E
      D:        Log-domain residuals
    """
    ba     = max(best_p[2], 1e-6)
    bc     = max(best_p[3], 1e-6)
    icorr  = best_p[1]
    log_ic = np.log10(icorr + TINY)

    E_lo, E_hi = float(E.min()), float(E.max())
    E_dense    = np.linspace(E_lo - 0.01, E_hi + 0.01, 5000)
    i_dense    = pol_model(E_dense, best_p, ct)
    i_fit_E    = pol_model(E, best_p, ct)

    log_obs  = slog(i_obs)
    log_den  = slog(i_dense)
    log_fit  = slog(i_fit_E)
    residuals = log_obs - log_fit

    r2v  = r2_score(log_obs, log_fit)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        gs  = GridSpec(2, 3, figure=fig,
                       hspace=0.44, wspace=0.34,
                       left=0.06, right=0.97, top=0.93, bottom=0.08)
        ax_ev  = fig.add_subplot(gs[0, :])      # full width Evans diagram
        ax_br  = fig.add_subplot(gs[1, 0])      # branch fits
        ax_lin = fig.add_subplot(gs[1, 1])      # linear scale
        ax_res = fig.add_subplot(gs[1, 2])      # residuals

        # ── Panel A: Evans Diagram ─────────────────────────────────────────
        ax = ax_ev

        # Region shading (CORRECT boundaries from fitted params)
        if show_regions:
            def shade(x0, x1, key, label):
                c, a = REGION_COLORS[key]
                ax.axvspan(x0, x1, color=c, alpha=a, lw=0,
                           label=label, zorder=1)

            shade(E_lo, Ecorr, "cathodic", "Cathodic region")
            if ct in CT.SIMPLE:
                shade(Ecorr, E_hi, "active", "Anodic (active)")
            elif ct in CT.PASS:
                Epass  = best_p[4]
                Etrans = best_p[7] if ct in CT.TRANS else E_hi + 1
                shade(Ecorr,            min(Epass, E_hi),  "active",       "Active dissolution")
                shade(min(Epass, E_hi), min(Etrans, E_hi), "passive",      "Passive region")
                if ct in CT.TRANS and Etrans <= E_hi:
                    shade(min(Etrans, E_hi), E_hi, "transpassive", "Transpassive / pitting")

        # Experimental data
        ax.scatter(E, log_obs, s=14, color="#4a7fa8", alpha=0.60,
                   zorder=2, label="Experimental data", linewidths=0)

        # Fitted curve
        ax.plot(E_dense, log_den, color="#1a3a5c", lw=2.4,
                zorder=5, label=f"Global fit  (R²={r2v:.5f})")

        # Tafel tangent lines ±150 mV from Ecorr (clamped to data range)
        dE = min(0.15, (E_hi - E_lo) * 0.20)
        E_ta = np.linspace(Ecorr,        Ecorr + dE, 300)
        E_tc = np.linspace(Ecorr - dE,   Ecorr,      300)
        # log|i_a| = log(icorr) + (E-Ecorr)*2.303/ba
        # log|i_c| = log(icorr) + (Ecorr-E)*2.303/bc
        ax.plot(E_ta, log_ic + (E_ta - Ecorr) * 2.303 / ba,
                "--", color="#e67e22", lw=1.8, zorder=4,
                label=f"$\\beta_a$ = {ba*1000:.0f} mV dec$^{{-1}}$")
        ax.plot(E_tc, log_ic + (Ecorr - E_tc) * 2.303 / bc,
                "--", color="#8e44ad", lw=1.8, zorder=4,
                label=f"$\\beta_c$ = {bc*1000:.0f} mV dec$^{{-1}}$")

        # E_corr marker
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.5, zorder=3,
                   label=f"$E_{{corr}}$ = {Ecorr:.4f} V")

        # i_corr marker
        ax.axhline(log_ic, color="#e84393", ls=":", lw=1.1, alpha=0.7, zorder=3)
        y_lo = np.nanmin(log_obs[np.isfinite(log_obs)])
        y_hi = np.nanmax(log_obs[np.isfinite(log_obs)])
        ax.annotate(
            f"$i_{{corr}}$ = {icorr:.2e} A cm$^{{-2}}$",
            xy=(Ecorr, log_ic),
            xytext=(E_lo + 0.05*(E_hi-E_lo), log_ic + max(0.3, (y_hi-y_lo)*0.07)),
            fontsize=9, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.9),
        )

        # Passive current marker
        if ct in CT.PASS:
            ip_val = best_p[6]
            ax.axhline(np.log10(ip_val + TINY), color="#27ae60",
                       ls=":", lw=1.1, alpha=0.7, zorder=3,
                       label=f"$i_{{pass}}$ = {ip_val:.2e} A cm$^{{-2}}$")

        ax.set_xlabel("$E$ vs. Reference (V)", fontsize=10)
        ax.set_ylabel("$\\log_{{10}}$ |$i$| (A cm$^{{-2}}$)", fontsize=10)
        ax.set_title(f"Evans Diagram — {sample_name}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.45)
        ax.grid(True, which="minor", ls=":", alpha=0.18)
        ax.legend(loc="lower right", ncol=4, fontsize=7.5,
                  framealpha=0.95, edgecolor="#cccccc")

        # Fit quality badge
        r2c = "#27ae60" if r2v > 0.99 else "#e67e22" if r2v > 0.95 else "#e84393"
        ax.text(0.01, 0.97,
                f"R² = {r2v:.5f}   RMSE = {rmse:.4f} log-units   "
                f"Model: {CT.name(ct)}",
                transform=ax.transAxes, fontsize=8.5,
                color=r2c, fontweight="bold", va="top",
                bbox=dict(fc="white", ec=r2c, alpha=0.88, pad=3,
                          boxstyle="round,pad=0.3"))

        # ── Panel B: Branch fits ───────────────────────────────────────────
        ax = ax_br
        ax.scatter(E, log_obs, s=5, color="#aab4c4", alpha=0.35,
                   zorder=1, label="_", linewidths=0)

        # Cathodic branch data + Tafel tangent
        if "E_cat" in cat_res:
            ax.scatter(cat_res["E_cat"], cat_res["lgi_cat"],
                       s=16, color="#6baed6", alpha=0.75, zorder=3,
                       label="Cathodic data", linewidths=0)
            Elin = np.linspace(Ecorr - 0.55, Ecorr - 0.005, 200)
            ax.plot(Elin, log_ic + (Ecorr - Elin) * 2.303 / bc,
                    "--", color="#6baed6", lw=1.8, zorder=4,
                    label=f"bc={bc*1000:.0f} mV/dec")
        # Anodic branch data + Tafel tangent
        if "E_an" in an_res:
            ax.scatter(an_res["E_an"], an_res["lgi_an"],
                       s=16, color="#fd8d3c", alpha=0.75, zorder=3,
                       label="Anodic data", linewidths=0)
            Ea_end = an_res.get("Epass", Ecorr + 0.3)
            if Ea_end is None: Ea_end = Ecorr + 0.3
            Elin = np.linspace(Ecorr + 0.005, min(float(Ea_end), E_hi), 200)
            ax.plot(Elin, log_ic + (Elin - Ecorr) * 2.303 / ba,
                    "--", color="#fd8d3c", lw=1.8, zorder=4,
                    label=f"ba={ba*1000:.0f} mV/dec")
            if an_res["has_passive"] and an_res["Epass"] is not None:
                ax.axvline(an_res["Epass"], color="#27ae60", ls="-.", lw=1.0,
                           alpha=0.85, label=f"E_pass={an_res['Epass']:.3f}V")

        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.axhline(log_ic, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("$E$ (V)", fontsize=9)
        ax.set_ylabel("$\\log_{{10}}$ |$i$|", fontsize=9)
        ax.set_title("Branch Fits (Stage 2–3)", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=7.5)

        # ── Panel C: Linear scale ──────────────────────────────────────────
        ax = ax_lin
        i_scale = np.clip(i_dense, -10*np.max(np.abs(i_obs)),
                                    10*np.max(np.abs(i_obs)))
        ax.scatter(E, i_obs * 1e3, s=9, color="#4a7fa8",
                   alpha=0.60, zorder=2, label="Data", linewidths=0)
        ax.plot(E_dense, i_scale * 1e3, color="#1a3a5c", lw=2.0,
                zorder=5, label="Fit")
        ax.axhline(0, color="#888", lw=0.7, zorder=1)
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.set_xlabel("$E$ (V)", fontsize=9)
        ax.set_ylabel("$i$ (mA cm$^{-2}$)", fontsize=9)
        ax.set_title("Linear Scale", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        # ── Panel D: Residuals ─────────────────────────────────────────────
        ax = ax_res
        ax.scatter(E, residuals, s=10, color="#2e86de",
                   alpha=0.65, zorder=3, linewidths=0)
        ax.axhline(0,    color="#333", lw=0.9, zorder=2)
        ax.axhline( 0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(-0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7,
                   label="±0.1 log-unit")
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel("$E$ (V)", fontsize=9)
        ax.set_ylabel("$\\Delta\\log_{{10}}$ |$i$|", fontsize=9)
        ax.set_title(f"Residuals   R²={r2v:.5f}", fontsize=10)
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
            with st.expander(f"📄 {uf.name}", expanded=True):

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
                        ap.scatter(E, slog(i), s=7, color="#5a7fa8", alpha=0.65)
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
                        w = min(9, len(i)//2*2-1); i = savgol_filter(i, w, 3)

                    stages = st.container()

                    # Stage 1
                    with stages.status("⚡ Stage 1 — Detecting E_corr…") as s1:
                        Ecorr, _ = detect_ecorr(E, i)
                        s1.update(label=f"✅ E_corr = {Ecorr:.5f} V",
                                  state="complete")

                    # Stage 2
                    with stages.status("🔵 Stage 2 — Cathodic branch fit…") as s2:
                        cat_res = fit_cathodic(E, i, Ecorr)
                        s2.update(label=f"✅ bc = {cat_res['bc']*1000:.0f} mV/dec  "
                                  f"i_corr = {cat_res['icorr']:.2e}  "
                                  f"R²_cat = {cat_res['r2']:.4f}",
                                  state="complete")

                    # Stage 3
                    with stages.status("🟠 Stage 3 — Anodic branch fit…") as s3:
                        an_res = fit_anodic(E, i, Ecorr)
                        s3.update(label=f"✅ ba = {an_res['ba']*1000:.0f} mV/dec  "
                                  f"passive={'Yes (E=%.3fV)'%an_res['Epass'] if an_res['has_passive'] else 'No'}",
                                  state="complete")

                    # Stage 4
                    ct_detected = classify_curve(cat_res, an_res)
                    stages.info(f"🔍 Detected curve type: **{CT.name(ct_detected)}** "
                                f"({CT.nfree(ct_detected)} free params)")

                    # Stage 5
                    with stages.status("⚙️ Stage 5 — Global optimization…") as s5:
                        E_lo = float(E.min()); E_hi = float(E.max())
                        E_sp = E_hi - E_lo

                        # Candidates to try
                        candidates = [ct_detected]
                        # Always try simpler model as baseline
                        if CT.A not in candidates:  candidates.append(CT.A)
                        # If passive detected, also try PT
                        if an_res["has_passive"] and CT.PT not in candidates:
                            candidates.append(CT.PT)
                        if force_ct != "auto":
                            candidates = [force_ct]

                        all_res = []
                        for ct_try in candidates:
                            p0 = _make_p0(Ecorr, cat_res, an_res, ct_try, E_hi)
                            lo, hi = _build_bounds(Ecorr, cat_res, an_res,
                                                   ct_try, E_lo, E_hi, E_sp)
                            # Clip p0 to bounds
                            p0 = np.clip(p0, lo, hi)
                            bp, r2v, aic_v, sse = global_polish(E, i, p0, ct_try, lo, hi)
                            all_res.append(dict(ct=ct_try, r2=r2v, aicc=aic_v,
                                                params=bp, success=r2v>0.90))
                            stages.write(f"  {CT.name(ct_try):35s} "
                                         f"R²={r2v:.6f}  AICc={aic_v:.1f}")

                        # AICc selection (parsimony: prefer simpler if ΔR²<0.002)
                        all_res.sort(key=lambda x: x["aicc"])
                        best_r = all_res[0]
                        for r in all_res:
                            if (CT.nfree(r["ct"]) < CT.nfree(best_r["ct"])
                                    and best_r["r2"] - r["r2"] < 0.002):
                                best_r = r; break

                        best_p  = best_r["params"]
                        best_ct = best_r["ct"]
                        r2_fin  = best_r["r2"]

                        s5.update(label=f"✅ Best: {CT.name(best_ct)}  "
                                  f"R² = {r2_fin:.6f}  "
                                  f"(elapsed {time.time()-t0:.1f}s)",
                                  state="complete")

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

                    # Model comparison table
                    if len(all_res) > 1:
                        with st.expander("🏆 Model Comparison (AICc)"):
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
                E_pl = np.linspace(-1.5, 1.5, 3000)
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
| **2 — Cathodic fit** | Sliding-window linear regression on log\|i\| vs E; best Tafel region; diffusion limit detection |
| **3 — Anodic fit** | Same for anodic; passive plateau via flat log\|i\| region; transpassive detection |
| **4 — Classification** | Curve type inferred from detected features |
| **5 — Global polish** | Physics-informed p₀ → DE → L-BFGS-B → Nelder-Mead |
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
