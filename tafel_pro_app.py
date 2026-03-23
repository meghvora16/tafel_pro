"""
Polarization Curve Fitter — Publication-Grade Streamlit App
============================================================
Correct electrochemical physics:
  - Signed current data (cathodic < 0, anodic > 0) with zero-crossing at E_corr
  - Full Evans diagram model: Butler-Volmer + active-passive transition + transpassive
  - Two-stage global fitting: Differential Evolution → Levenberg-Marquardt
  - Log-domain residuals on |i|, with correct handling of sign change near E_corr
  - Auto-detection of all electrochemical regions
  - Publication-quality 300-DPI figures with correct region annotations
  - Export: PDF report, Excel, PNG, SVG
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
import io, os, zipfile, warnings, traceback
from scipy.optimize import differential_evolution, least_squares
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Polarization Curve Fitter",
    page_icon="electrolyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header{font-size:2rem;font-weight:700;color:#1a3a5c;
    border-bottom:3px solid #2e86de;padding-bottom:8px;margin-bottom:1rem}
  div[data-testid="metric-container"]{
    background:#f0f4ff;border-left:3px solid #2e86de;
    border-radius:6px;padding:8px 12px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS & COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#2e86de","#e84393","#27ae60","#e67e22","#8e44ad","#16a085","#c0392b"]
TINY    = 1e-20          # floor to avoid log(0)

REGION_COLORS = {
    "cathodic":     ("#6baed6", 0.13),
    "active":       ("#fd8d3c", 0.13),
    "passive":      ("#74c476", 0.15),
    "transpassive": ("#9e9ac8", 0.18),
}

# ══════════════════════════════════════════════════════════════════════════════
# ELECTROCHEMICAL MODEL  (signed current, full polarisation curve)
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def model_BV(E, p):
    """Pure Butler-Volmer (signed output)."""
    eta = E - p["Ecorr"]
    return p["icorr"] * (np.exp(eta / p["ba"]) - np.exp(-eta / p["bc"]))

def model_passive(E, p):
    """BV with smooth active-to-passive transition on the anodic branch."""
    eta  = E - p["Ecorr"]
    i_cat = -p["icorr"] * np.exp(-eta / p["bc"])

    i_ano_active = p["icorr"] * np.exp(eta / p["ba"])
    k_p  = p.get("k_pass", 0.015)
    w_p  = _sigmoid((E - p["Epass"]) / k_p)
    i_ano = (1 - w_p) * i_ano_active + w_p * p["ip"]
    return i_ano + i_cat

def model_full(E, p):
    """
    Full polarisation curve:
      cathodic : pure BV cathodic partial
      anodic   : active Tafel -> passive plateau -> transpassive rise
    All transitions are smooth sigmoids (differentiable everywhere).
    """
    eta   = E - p["Ecorr"]
    i_cat = -p["icorr"] * np.exp(-eta / p["bc"])

    # Active anodic Tafel
    i_ano_active = p["icorr"] * np.exp(eta / p["ba"])

    # Active -> passive transition at Epass
    k_p   = p.get("k_pass",  0.015)
    w_p   = _sigmoid((E - p["Epass"]) / k_p)
    i_ano_pass = (1 - w_p) * i_ano_active + w_p * p["ip"]

    # Passive -> transpassive transition at Etrans
    k_t   = p.get("k_trans", 0.012)
    w_t   = _sigmoid((E - p["Etrans"]) / k_t)
    i_trans_val = p["ip"] + p["itrans"] * np.exp((E - p["Etrans"]) / p["ba"])
    i_ano = (1 - w_t) * i_ano_pass + w_t * i_trans_val

    return i_ano + i_cat

MODEL_FNS = {
    "butler_volmer": model_BV,
    "passive":       model_passive,
    "full":          model_full,
}

def eval_model(E, params, model_type):
    return MODEL_FNS[model_type](E, params)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def build_bounds(E, i_signed, model_type):
    E_min, E_max = float(E.min()), float(E.max())
    E_span = E_max - E_min
    i_abs  = np.abs(i_signed)

    # Ecorr: search near the sign change or |i| minimum
    sc = np.where(np.diff(np.sign(i_signed)))[0]
    if len(sc) > 0:
        idx_ec   = sc[0]
        Ecorr_lo = max(E_min, float(E[idx_ec]) - 0.10)
        Ecorr_hi = min(E_max, float(E[idx_ec]) + 0.10)
    else:
        Ecorr_lo = E_min + 0.05 * E_span
        Ecorr_hi = E_max - 0.05 * E_span

    # icorr: small absolute value near E_corr
    icorr_est = float(np.percentile(i_abs[i_abs > 0], 3)) if np.any(i_abs > 0) else 1e-6
    icorr_lo  = max(icorr_est * 1e-4, 1e-12)
    icorr_hi  = min(float(np.max(i_abs)) * 5, 1.0)

    bounds = {
        "Ecorr": (Ecorr_lo, Ecorr_hi),
        "icorr": (icorr_lo, icorr_hi),
        "ba":    (0.020, 0.250),   # 20–250 mV/dec — hard physical limits
        "bc":    (0.020, 0.250),
    }

    if model_type in ("passive", "full"):
        bounds["ip"]     = (icorr_lo * 0.001, icorr_hi * 0.3)
        bounds["Epass"]  = (Ecorr_hi, E_max - 0.02 * E_span)
        bounds["k_pass"] = (0.005, 0.060)

    if model_type == "full":
        bounds["Etrans"]  = (Ecorr_hi + 0.10 * E_span, E_max)
        bounds["itrans"]  = (icorr_lo, icorr_hi * 20)
        bounds["k_trans"] = (0.005, 0.060)

    return bounds

# ══════════════════════════════════════════════════════════════════════════════
# FITTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _make_weights(E, i_obs):
    """Upweight Tafel regions; downweight noisy zero-crossing zone."""
    i_abs   = np.abs(i_obs)
    min_idx = int(np.argmin(i_abs))
    dE      = np.abs(E - E[min_idx])
    w       = 1.0 + 2.5 * np.tanh(dE / 0.04)
    return w / w.mean()

def _objective(x, keys, E, i_obs, model_type, weights):
    p        = dict(zip(keys, x))
    i_pred   = eval_model(E, p, model_type)
    log_obs  = np.log10(np.abs(i_obs)  + TINY)
    log_pred = np.log10(np.abs(i_pred) + TINY)
    return weights * (log_pred - log_obs)

def fit_curve(E, i_obs, model_type="full", progress_cb=None):
    """
    Two-stage global fitting:
      Stage 1 – Differential Evolution  (global, bound-constrained)
      Stage 2 – Levenberg-Marquardt     (local polish)
    All in log10|i| space with distance-from-Ecorr weighting.
    """
    bounds_dict = build_bounds(E, i_obs, model_type)
    keys    = list(bounds_dict.keys())
    bounds  = list(bounds_dict.values())
    weights = _make_weights(E, i_obs)

    # Stage 1: Differential Evolution
    de = differential_evolution(
        lambda x: float(np.sum(_objective(x, keys, E, i_obs, model_type, weights)**2)),
        bounds=list(zip([b[0] for b in bounds], [b[1] for b in bounds])),
        maxiter=1500, popsize=20, tol=1e-10, seed=42,
        polish=False, workers=1,
        mutation=(0.5, 1.5), recombination=0.8,
        callback=lambda xk, conv:
            (progress_cb(0.55) if progress_cb else None) or False,
    )
    if progress_cb: progress_cb(0.60)

    # Stage 2: Levenberg-Marquardt
    lm = least_squares(
        _objective, de.x,
        args=(keys, E, i_obs, model_type, weights),
        method="lm",
        ftol=1e-14, xtol=1e-14, gtol=1e-14,
        max_nfev=30000,
    )
    if progress_cb: progress_cb(0.90)

    # Clip to bounds (LM can wander outside for unconstrained variables)
    raw_x = lm.x.copy()
    for ki, k in enumerate(keys):
        lo, hi = bounds_dict[k]
        raw_x[ki] = float(np.clip(raw_x[ki], lo, hi))
    params = dict(zip(keys, raw_x))

    # Goodness of fit in log-domain
    i_fit    = eval_model(E, params, model_type)
    log_obs  = np.log10(np.abs(i_obs)  + TINY)
    log_fit  = np.log10(np.abs(i_fit)  + TINY)
    ss_res   = float(np.sum((log_obs - log_fit)**2))
    ss_tot   = float(np.sum((log_obs - np.mean(log_obs))**2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    rmse     = float(np.sqrt(np.mean((log_obs - log_fit)**2)))

    # Parameter uncertainties via Jacobian covariance
    try:
        J    = lm.jac
        cov  = np.linalg.pinv(J.T @ J) * (ss_res / max(len(E) - len(keys), 1))
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        perr = np.zeros(len(keys))

    if progress_cb: progress_cb(1.0)

    return {
        "params":        params,
        "uncertainties": dict(zip(keys, perr)),
        "r2":            r2,
        "rmse":          rmse,
        "i_fit":         i_fit,
        "success":       bool(lm.success or lm.cost < 1.0),
        "model_type":    model_type,
    }

# ══════════════════════════════════════════════════════════════════════════════
# REGION AUTO-DETECTION  (from raw signed data)
# ══════════════════════════════════════════════════════════════════════════════

def detect_regions(E, i_signed):
    info = {}
    i_abs = np.abs(i_signed)

    # Ecorr: sign change (zero crossing)
    sc = np.where(np.diff(np.sign(i_signed)))[0]
    if len(sc) > 0:
        idx_ec          = int(sc[0])
        info["Ecorr_idx"] = idx_ec
        info["Ecorr"]     = float(E[idx_ec])
    else:
        idx_ec            = int(np.argmin(i_abs))
        info["Ecorr_idx"] = idx_ec
        info["Ecorr"]     = float(E[idx_ec])

    # Anodic side (E > Ecorr, i > 0)
    ano_mask = E > info["Ecorr"]
    if not np.any(ano_mask):
        return info

    E_ano = E[ano_mask]
    i_ano = i_signed[ano_mask]

    # Active peak: local maximum in anodic current
    if len(i_ano) > 6:
        pks, _ = find_peaks(i_ano, prominence=float(np.max(i_ano)) * 0.05)
        if len(pks) > 0:
            pk = pks[int(np.argmax(i_ano[pks]))]
            info["Epeak"]  = float(E_ano[pk])
            info["ipeak"]  = float(i_ano[pk])

    # Passive plateau: low d(log i)/dE after active peak
    if len(i_ano) > 10:
        sm_log = savgol_filter(np.log10(i_ano + TINY),
                               min(9, (len(i_ano)//2)*2 - 1), 3)
        dlogdE = np.gradient(sm_log, E_ano)
        start  = int(np.searchsorted(E_ano, info.get("Epeak", E_ano[0])))
        post   = dlogdE[start:]
        post_E = E_ano[start:]
        flat   = np.where(np.abs(post) < 2.5)[0]
        if len(flat) > 0:
            info["Epass"]     = float(post_E[flat[0]])
            info["Epass_end"] = float(post_E[flat[-1]])

    # Transpassive: current rises sharply after passive end
    if "Epass_end" in info and len(i_ano) > 10:
        ts_idx = int(np.searchsorted(E_ano, info["Epass_end"]))
        if ts_idx < len(E_ano) - 3:
            post2   = dlogdE[ts_idx:]
            post2_E = E_ano[ts_idx:]
            rising  = np.where(post2 > 3.0)[0]
            if len(rising) > 0:
                info["Etrans"] = float(post2_E[rising[0]])

    return info

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(uploaded_file, e_col, i_col, skip_rows, delimiter):
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            content = uploaded_file.read().decode("utf-8", errors="replace")
            uploaded_file.seek(0)
            sep = delimiter if delimiter != "auto" else None
            df  = pd.read_csv(uploaded_file, skiprows=skip_rows,
                              sep=sep, engine="python", comment="#")

        cols  = list(df.columns)
        E_raw = (df[e_col].values if (e_col and e_col in cols)
                 else df.iloc[:, 0].values).astype(float)
        i_raw = (df[i_col].values if (i_col and i_col in cols)
                 else df.iloc[:, 1].values).astype(float)

        mask  = np.isfinite(E_raw) & np.isfinite(i_raw)
        E_raw, i_raw = E_raw[mask], i_raw[mask]
        order = np.argsort(E_raw)
        return E_raw[order], i_raw[order], df, None
    except Exception as ex:
        return None, None, None, str(ex)

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB PUBLICATION STYLE
# ══════════════════════════════════════════════════════════════════════════════
PLT_RC = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 8, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc", "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "#fafbff",
}

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION-QUALITY FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(E, i_obs, fit_res, sample_name, detected, show_regions=True, dpi=150):
    p      = fit_res["params"]
    mt     = fit_res["model_type"]
    Ecorr  = p["Ecorr"]
    icorr  = p["icorr"]
    ba     = p["ba"]
    bc     = p["bc"]

    E_lo, E_hi = float(E.min()), float(E.max())

    # Dense grid for smooth fitted curve
    E_dense      = np.linspace(E_lo - 0.01, E_hi + 0.01, 5000)
    i_dense      = eval_model(E_dense, p, mt)
    i_fit_at_E   = eval_model(E, p, mt)

    log_obs   = np.log10(np.abs(i_obs)   + TINY)
    log_dense = np.log10(np.abs(i_dense) + TINY)
    log_fit   = np.log10(np.abs(i_fit_at_E) + TINY)
    log_icorr = np.log10(icorr + TINY)
    residuals = log_obs - log_fit

    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(13, 8.5), dpi=dpi)
        gs  = GridSpec(2, 2, figure=fig,
                       hspace=0.44, wspace=0.34,
                       left=0.07, right=0.97, top=0.93, bottom=0.09)
        ax_evans = fig.add_subplot(gs[0, :])
        ax_lin   = fig.add_subplot(gs[1, 0])
        ax_res   = fig.add_subplot(gs[1, 1])

        # ── Panel A: Evans / Tafel diagram ────────────────────────────────────
        ax = ax_evans

        # Region shading — CORRECT boundaries
        if show_regions:
            def shade(x0, x1, key, label):
                c, a = REGION_COLORS[key]
                ax.axvspan(x0, x1, color=c, alpha=a, lw=0,
                           label=label, zorder=1)

            shade(E_lo, Ecorr, "cathodic", "Cathodic (HER/ORR)")

            if mt == "butler_volmer":
                shade(Ecorr, E_hi, "active", "Anodic Tafel")
            elif mt == "passive":
                Epass = p.get("Epass", E_hi)
                shade(Ecorr, min(Epass, E_hi), "active", "Active dissolution")
                shade(min(Epass, E_hi), E_hi,  "passive", "Passive region")
            else:  # full
                Epass  = p.get("Epass",  Ecorr + 0.05 * (E_hi - E_lo))
                Etrans = p.get("Etrans", E_hi)
                shade(Ecorr, min(Epass, E_hi),   "active", "Active dissolution")
                shade(min(Epass, E_hi),
                      min(Etrans, E_hi),           "passive", "Passive region")
                if Etrans < E_hi:
                    shade(min(Etrans, E_hi), E_hi, "transpassive",
                          "Transpassive / pitting")

        # Experimental data
        ax.scatter(E, log_obs, s=12, color="#4a7fa8", alpha=0.60,
                   zorder=2, label="Experimental data", linewidths=0)

        # Fitted curve
        ax.plot(E_dense, log_dense, color="#1a3a5c", lw=2.2,
                zorder=5, label="Global fit")

        # Tafel tangent lines — clamped to ±150 mV and to data log-range
        dE_tan = min(0.15, (E_hi - E_lo) * 0.20)
        E_tan_a = np.linspace(Ecorr, Ecorr + dE_tan, 300)
        E_tan_c = np.linspace(Ecorr - dE_tan, Ecorr, 300)
        # Tafel tangents in log|i| space:
        #   Anodic:   log|i_ano| = log(icorr) + (E - Ecorr)/ba   → slope +1/ba
        #   Cathodic: log|i_cat| = log(icorr) + (Ecorr - E)/bc   → slope -1/bc
        log_tan_a = log_icorr + (E_tan_a - Ecorr) / ba
        log_tan_c = log_icorr + (Ecorr  - E_tan_c) / bc
        ax.plot(E_tan_a, log_tan_a, "--", color="#e67e22", lw=1.8,
                zorder=4, label=f"$\\beta_a$ = {ba*1000:.0f} mV dec$^{{-1}}$")
        ax.plot(E_tan_c, log_tan_c, "--", color="#8e44ad", lw=1.8,
                zorder=4, label=f"$\\beta_c$ = {bc*1000:.0f} mV dec$^{{-1}}$")

        # E_corr vertical marker
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.5, zorder=3,
                   label=f"$E_{{corr}}$ = {Ecorr:.4f} V")

        # i_corr horizontal marker + annotation
        ax.axhline(log_icorr, color="#e84393", ls=":", lw=1.1, alpha=0.75, zorder=3)
        # Place annotation in a safe y-position
        y_data_min = np.min(log_obs[np.isfinite(log_obs)])
        y_data_max = np.max(log_obs[np.isfinite(log_obs)])
        ann_y = log_icorr + max(0.25, (y_data_max - y_data_min) * 0.06)
        ax.annotate(
            f"$i_{{corr}}$ = {icorr:.2e} A cm$^{{-2}}$",
            xy=(Ecorr, log_icorr),
            xytext=(E_lo + 0.06*(E_hi-E_lo), ann_y),
            fontsize=9, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.9),
        )

        # Extra markers (detected)
        if show_regions:
            if "Epeak" in detected and mt != "butler_volmer":
                ax.axvline(detected["Epeak"], color="#fd8d3c", ls="-.",
                           lw=1.1, alpha=0.85, zorder=3,
                           label=f"$E_{{peak}}$ = {detected['Epeak']:.4f} V")
            if mt == "full" and "Etrans" in p:
                ax.axvline(p["Etrans"], color="#9e9ac8", ls="-.",
                           lw=1.1, alpha=0.85, zorder=3,
                           label=f"$E_{{trans}}$ = {p['Etrans']:.4f} V")

        ax.set_xlabel("$E$ vs. Reference (V)", fontsize=10)
        ax.set_ylabel("$\\log_{10}$ |$i$| (A cm$^{-2}$)", fontsize=10)
        ax.set_title(f"Evans Diagram — {sample_name}",
                     fontsize=11, fontweight="bold", pad=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(True, which="major", ls="--", alpha=0.5)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        ax.legend(loc="lower right", ncol=3, fontsize=7.8,
                  framealpha=0.95, edgecolor="#cccccc")

        # Fit quality badge
        r2_color = "#27ae60" if fit_res["r2"] > 0.99 else \
                   "#e67e22" if fit_res["r2"] > 0.95 else "#e84393"
        ax.text(0.01, 0.97,
                f"R\u00b2 = {fit_res['r2']:.5f}   RMSE = {fit_res['rmse']:.4f} (log-domain)",
                transform=ax.transAxes, fontsize=8.5,
                color=r2_color, fontweight="bold", va="top",
                bbox=dict(fc="white", ec=r2_color, alpha=0.85, pad=3,
                          boxstyle="round,pad=0.3"))

        # ── Panel B: Linear i vs E ─────────────────────────────────────────────
        ax = ax_lin
        scale_factor = 1e3   # A/cm² → mA/cm²

        # Clip extreme fit values for clean linear plot
        i_dense_lin = np.clip(i_dense, -10 * np.max(np.abs(i_obs)),
                                         10 * np.max(np.abs(i_obs)))

        ax.scatter(E, i_obs * scale_factor, s=9, color="#4a7fa8",
                   alpha=0.60, zorder=2, label="Data", linewidths=0)
        ax.plot(E_dense, i_dense_lin * scale_factor, color="#1a3a5c",
                lw=2.0, zorder=5, label="Fit")
        ax.axhline(0, color="#888", lw=0.7, zorder=1)
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=1.2, zorder=3)
        ax.set_xlabel("$E$ (V)", fontsize=9)
        ax.set_ylabel("$i$ (mA cm$^{-2}$)", fontsize=9)
        ax.set_title("Linear Scale", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

        # ── Panel C: Residuals ─────────────────────────────────────────────────
        ax = ax_res
        ax.scatter(E, residuals, s=10, color="#2e86de",
                   alpha=0.65, zorder=3, linewidths=0)
        ax.axhline(0,    color="#333", lw=0.9, zorder=2)
        ax.axhline( 0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(-0.1, color="#e84393", ls=":", lw=1.0, alpha=0.7,
                   label="±0.1 log-unit")
        ax.axvline(Ecorr, color="#e84393", ls="--", lw=0.9, alpha=0.6, zorder=2)
        ax.set_xlabel("$E$ (V)", fontsize=9)
        ax.set_ylabel("$\\Delta\\log_{10}$ |$i$|", fontsize=9)
        ax.set_title(f"Residuals   R\u00b2 = {fit_res['r2']:.5f}", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

        fig.suptitle("Polarisation Curve Analysis", fontsize=12,
                     fontweight="bold", color="#1a3a5c", y=0.98)

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT: EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def export_excel(results_list):
    wb  = openpyxl.Workbook()
    ws  = wb.active
    ws.title = "Fitting Results"

    H_FILL = PatternFill("solid", fgColor="1A3A5C")
    H_FONT = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    ALT    = PatternFill("solid", fgColor="EEF2FF")
    BRD    = Border(left=Side(style="thin"),  right=Side(style="thin"),
                    top=Side(style="thin"),   bottom=Side(style="thin"))
    GRN    = Font(color="1E8449", bold=True, name="Arial", size=10)
    RED    = Font(color="C0392B", bold=True, name="Arial", size=10)

    headers = ["Sample","Model","E_corr (V)","sigma_Ecorr",
               "i_corr (A/cm2)","sigma_icorr",
               "ba (mV/dec)","sigma_ba","bc (mV/dec)","sigma_bc",
               "i_pass (A/cm2)","E_pass (V)","E_trans (V)",
               "R2","RMSE (log)","Status"]

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = H_FILL; cell.font = H_FONT; cell.border = BRD
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for ri, res in enumerate(results_list, 2):
        p  = res["params"]
        u  = res["uncertainties"]
        ok = res.get("r2", 0) > 0.95
        fill = ALT if ri % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        vals = [
            res.get("name", f"S{ri-1}"),
            res["model_type"].replace("_"," ").title(),
            round(p.get("Ecorr",  np.nan), 5),
            round(u.get("Ecorr",  0),      6),
            f"{p.get('icorr', np.nan):.4e}",
            f"{u.get('icorr', 0):.2e}",
            round(p.get("ba", 0)*1000, 2),
            round(u.get("ba", 0)*1000, 3),
            round(p.get("bc", 0)*1000, 2),
            round(u.get("bc", 0)*1000, 3),
            f"{p.get('ip', np.nan):.3e}" if "ip" in p else "—",
            round(p.get("Epass",  np.nan), 5) if "Epass"  in p else "—",
            round(p.get("Etrans", np.nan), 5) if "Etrans" in p else "—",
            round(res.get("r2",   np.nan), 6),
            round(res.get("rmse", np.nan), 6),
            "Good" if ok else "Check",
        ]
        for c, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=c, value=val)
            cell.fill = fill; cell.border = BRD
            cell.alignment = Alignment(horizontal="center")
            if c == 16:
                cell.font = GRN if ok else RED

    for col in ws.columns:
        w = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(w + 3, 24)
    ws.freeze_panes = "A2"

    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT: PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_pdf(results_list, png_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("CT", parent=styles["Title"],
                               fontSize=20, textColor=rl_colors.HexColor("#1A3A5C"),
                               spaceAfter=4)
    h2_s = ParagraphStyle("H2", parent=styles["Heading2"],
                            fontSize=12, textColor=rl_colors.HexColor("#2e86de"),
                            spaceBefore=10, spaceAfter=3)
    body_s = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=14)
    mono_s = ParagraphStyle("M", parent=styles["Normal"], fontName="Courier",
                              fontSize=8.5, backColor=rl_colors.HexColor("#F0F4FF"),
                              leftIndent=8, leading=15)

    story = []
    story.append(Paragraph("Polarisation Curve Analysis Report", title_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=rl_colors.HexColor("#2e86de"), spaceAfter=4))
    story.append(Paragraph(
        f"Generated by Polarization Curve Fitter  |  "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", body_s))
    story.append(Spacer(1, 0.5*cm))

    tbl_style = TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), rl_colors.HexColor("#1A3A5C")),
        ("TEXTCOLOR",      (0,0),(-1,0), rl_colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1),8),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [rl_colors.HexColor("#EEF2FF"), rl_colors.white]),
        ("GRID",           (0,0),(-1,-1),0.4,rl_colors.HexColor("#BBBBBB")),
        ("VALIGN",         (0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",          (2,1),(-1,-1),"RIGHT"),
        ("TOPPADDING",     (0,0),(-1,-1),3),
        ("BOTTOMPADDING",  (0,0),(-1,-1),3),
    ])

    for idx, (res, png) in enumerate(zip(results_list, png_bytes_list)):
        p  = res["params"]
        u  = res["uncertainties"]
        nm = res.get("name", f"Sample {idx+1}")
        story.append(Paragraph(f"Sample {idx+1}: {nm}", h2_s))

        rows = [["Parameter","Symbol","Value","Uncertainty","Unit"],
                ["Corrosion potential","E_corr",
                 f"{p.get('Ecorr',0):.5f}", f"± {u.get('Ecorr',0):.6f}","V"],
                ["Corrosion current density","i_corr",
                 f"{p.get('icorr',0):.4e}", f"± {u.get('icorr',0):.2e}","A cm-2"],
                ["Anodic Tafel slope","ba",
                 f"{p.get('ba',0)*1000:.2f}", f"± {u.get('ba',0)*1000:.3f}","mV dec-1"],
                ["Cathodic Tafel slope","bc",
                 f"{p.get('bc',0)*1000:.2f}", f"± {u.get('bc',0)*1000:.3f}","mV dec-1"],
                ]
        if "ip" in p:
            rows += [
                ["Passive current density","i_pass",
                 f"{p.get('ip',0):.4e}","—","A cm-2"],
                ["Passivation potential","E_pass",
                 f"{p.get('Epass',0):.5f}","—","V"],
            ]
        if "Etrans" in p:
            rows.append(["Transpassive potential","E_trans",
                          f"{p.get('Etrans',0):.5f}","—","V"])
        rows += [
            ["R-squared (log-domain)","R2",
             f"{res.get('r2',0):.6f}","—","—"],
            ["RMSE (log-domain)","RMSE",
             f"{res.get('rmse',0):.6f}","—","log-units"],
            ["Fit status","—",
             "Converged" if res.get("success") else "Check","—","—"],
        ]
        tbl = Table(rows,
                    colWidths=[5.2*cm, 2.0*cm, 2.8*cm, 3.0*cm, 2.0*cm])
        tbl.setStyle(tbl_style)
        story.append(KeepTogether([tbl, Spacer(1, 0.3*cm)]))

        if png:
            story.append(RLImage(io.BytesIO(png), width=15.5*cm, height=10.2*cm))
        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=rl_colors.HexColor("#CCCCCC"), spaceAfter=4))

    doc.build(story)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k in ("results","figures"):
    if _k not in st.session_state:
        st.session_state[_k] = []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    st.markdown("**Electrochemical Model**")
    model_type = st.selectbox("Model", ["full","passive","butler_volmer"],
        format_func=lambda x: {"full": "Full (BV + Passive + Transpassive)",
                                "passive": "BV + Passive plateau",
                                "butler_volmer": "Butler-Volmer only"}[x])

    st.markdown("**Data Import**")
    skip_rows   = st.number_input("Skip header rows", 0, 30, 0)
    delimiter   = st.selectbox("CSV delimiter", ["auto",",",";","\t"," "])
    e_col_name  = st.text_input("E column name (blank = col 1)", "")
    i_col_name  = st.text_input("i column name (blank = col 2)", "")
    i_unit      = st.selectbox("Current density unit in file",
                               ["A/cm²","mA/cm²","µA/cm²","A/m²"])
    unit_factor = {"A/cm²":1.0,"mA/cm²":1e-3,"µA/cm²":1e-6,"A/m²":1e-4}[i_unit]

    st.markdown("**Figure Options**")
    show_regions = st.toggle("Shade electrochemical regions", True)
    smooth_data  = st.toggle("Pre-smooth data (Savitzky-Golay)", False)
    pub_dpi      = st.slider("Export DPI", 150, 600, 300, 50)

    st.divider()
    if st.button("Clear all results", use_container_width=True):
        st.session_state.results = []
        st.session_state.figures = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">Polarization Curve Fitter</div>',
            unsafe_allow_html=True)

tab_fit, tab_res, tab_cmp, tab_help = st.tabs(
    ["Upload & Fit", "Results & Export", "Compare Samples", "Help"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – UPLOAD & FIT
# ─────────────────────────────────────────────────────────────────────────────
with tab_fit:
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("### Upload Data")
        uploaded_files = st.file_uploader(
            "CSV / TXT / XLSX  (E column + signed i column)",
            type=["csv","txt","xlsx","xls"],
            accept_multiple_files=True,
            help="Current must be signed: cathodic < 0, anodic > 0")
    with c2:
        st.markdown("### Sample Label")
        sample_name = st.text_input("Label", "Sample 1")

    if uploaded_files:
        for uf in uploaded_files:
            with st.expander(f"{uf.name}", expanded=True):
                E, i_raw, df_raw, err = load_data(
                    uf, e_col_name or None, i_col_name or None,
                    skip_rows, delimiter)
                if err:
                    st.error(f"Load error: {err}"); continue

                i_raw = i_raw * unit_factor

                # Preview
                pa, pb = st.columns([1, 1.6])
                with pa:
                    sc = np.where(np.diff(np.sign(i_raw)))[0]
                    st.markdown(f"**Points:** {len(E)}  |  "
                                f"**E:** [{E.min():.4f}, {E.max():.4f}] V")
                    if len(sc) > 0:
                        st.success(f"Sign change at E ≈ {E[sc[0]]:.4f} V")
                    else:
                        st.warning("No sign change detected — verify sign convention")
                    if df_raw is not None:
                        st.dataframe(df_raw.head(6), use_container_width=True, height=170)

                with pb:
                    with plt.rc_context(PLT_RC):
                        fig_p, ax_p = plt.subplots(figsize=(6, 3.8))
                        ax_p.scatter(E, np.log10(np.abs(i_raw)+TINY),
                                     s=7, color="#5a7fa8", alpha=0.65)
                        ax_p.set_xlabel("E (V)")
                        ax_p.set_ylabel("log|i| (A/cm²)")
                        ax_p.set_title("Raw Data Preview", fontsize=10)
                        ax_p.grid(True, ls="--", alpha=0.4)
                        if len(sc) > 0:
                            ax_p.axvline(E[sc[0]], color="#e84393", ls="--", lw=1,
                                         label=f"E_corr≈{E[sc[0]]:.4f}V")
                            ax_p.legend(fontsize=8)
                        fig_p.tight_layout()
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close(fig_p)

                if st.button(f"Fit  ·  {uf.name}",
                             key=f"btn_{uf.name}", type="primary",
                             use_container_width=True):
                    prog = st.progress(0.0, text="Initialising optimizer…")

                    def cb(v):
                        prog.progress(float(v),
                                      text={0.55:"Differential Evolution…",
                                            0.60:"DE complete",
                                            0.90:"LM refinement…",
                                            1.00:"Done"}.get(float(v),"Running…"))

                    try:
                        i_in = i_raw
                        if smooth_data:
                            w = min(9, len(i_raw)//2*2-1)
                            i_in = savgol_filter(i_raw, w, 3)

                        result   = fit_curve(E, i_in, model_type, cb)
                        result["name"] = sample_name or uf.name
                        detected = detect_regions(E, i_raw)

                        fig = make_figure(E, i_raw, result, result["name"],
                                         detected, show_regions, dpi=pub_dpi)
                        fig.tight_layout(rect=[0,0,1,0.97])

                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, dpi=pub_dpi, bbox_inches="tight",
                                    facecolor="white")
                        buf_png.seek(0); png_bytes = buf_png.read()

                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight",
                                    facecolor="white")
                        buf_svg.seek(0); svg_bytes = buf_svg.read()

                        st.session_state.results.append(result)
                        st.session_state.figures.append({
                            "png": png_bytes, "svg": svg_bytes,
                            "name": result["name"]})

                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        p = result["params"]
                        u = result["uncertainties"]
                        st.markdown("#### Fitted Parameters")
                        m1,m2,m3,m4 = st.columns(4)
                        m1.metric("E_corr (V)",       f"{p['Ecorr']:.5f}",
                                   f"σ={u.get('Ecorr',0):.5f}")
                        m2.metric("i_corr (A/cm²)",   f"{p['icorr']:.4e}",
                                   f"σ={u.get('icorr',0):.2e}")
                        m3.metric("βa (mV/dec)",       f"{p['ba']*1000:.1f}",
                                   f"σ={u.get('ba',0)*1000:.2f}")
                        m4.metric("βc (mV/dec)",       f"{p['bc']*1000:.1f}",
                                   f"σ={u.get('bc',0)*1000:.2f}")
                        if "ip" in p:
                            m5,m6,m7,m8 = st.columns(4)
                            m5.metric("i_pass (A/cm²)", f"{p.get('ip',0):.4e}")
                            m6.metric("E_pass (V)",      f"{p.get('Epass',0):.5f}")
                            if "Etrans" in p:
                                m7.metric("E_trans (V)", f"{p.get('Etrans',0):.5f}")
                            m8.metric("R²", f"{result['r2']:.5f}",
                                      "Excellent" if result["r2"]>0.99 else
                                      "Check" if result["r2"]<0.90 else "Good")

                        d1,d2 = st.columns(2)
                        d1.download_button("Download PNG", png_bytes,
                                           f"{result['name']}.png","image/png")
                        d2.download_button("Download SVG", svg_bytes,
                                           f"{result['name']}.svg","image/svg+xml")

                    except Exception as ex:
                        st.error(f"Fitting failed: {ex}")
                        st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – RESULTS & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_res:
    if not st.session_state.results:
        st.info("No results yet.")
    else:
        rows = []
        for r in st.session_state.results:
            p = r["params"]
            rows.append({"Sample":r.get("name","?"),
                          "Model":r["model_type"],
                          "E_corr (V)":f"{p.get('Ecorr',0):.5f}",
                          "i_corr (A/cm²)":f"{p.get('icorr',0):.4e}",
                          "βa (mV/dec)":f"{p.get('ba',0)*1000:.1f}",
                          "βc (mV/dec)":f"{p.get('bc',0)*1000:.1f}",
                          "i_pass":f"{p.get('ip',0):.3e}" if "ip" in p else "—",
                          "E_pass (V)":f"{p.get('Epass',0):.5f}" if "Epass" in p else "—",
                          "E_trans (V)":f"{p.get('Etrans',0):.5f}" if "Etrans" in p else "—",
                          "R²":f"{r.get('r2',0):.5f}",
                          "RMSE":f"{r.get('rmse',0):.5f}",
                          "Status":"Good" if r.get("r2",0)>0.95 else "Check"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.divider()

        ec1,ec2,ec3 = st.columns(3)
        ec1.download_button("Excel (.xlsx)",
            data=export_excel(st.session_state.results),
            file_name="polarization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
        ec2.download_button("PDF Report",
            data=export_pdf(st.session_state.results,
                            [f["png"] for f in st.session_state.figures]),
            file_name="polarization_report.pdf",
            mime="application/pdf",
            use_container_width=True)
        zb = io.BytesIO()
        with zipfile.ZipFile(zb,"w") as zf:
            for fd in st.session_state.figures:
                zf.writestr(f"{fd['name']}.png", fd["png"])
                zf.writestr(f"{fd['name']}.svg", fd["svg"])
        zb.seek(0)
        ec3.download_button("All Figures (.zip)", data=zb,
            file_name="polarization_figures.zip",
            mime="application/zip", use_container_width=True)

        st.markdown("### Figures")
        for fd in st.session_state.figures:
            st.markdown(f"**{fd['name']}**")
            st.image(fd["png"], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    if len(st.session_state.results) < 2:
        st.info("Fit at least 2 samples to enable comparison.")
    else:
        with plt.rc_context(PLT_RC):
            fig_c, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
            for idx, res in enumerate(st.session_state.results):
                col   = PALETTE[idx % len(PALETTE)]
                p     = res["params"]
                label = res.get("name", f"S{idx+1}")
                E_pl  = np.linspace(-1.5, 1.5, 3000)
                i_pl  = eval_model(E_pl, p, res["model_type"])
                axes[0].plot(E_pl, np.log10(np.abs(i_pl)+TINY), color=col, lw=2, label=label)
                axes[0].axvline(p["Ecorr"], color=col, ls=":", lw=0.9, alpha=0.6)
                axes[1].bar(idx, p["icorr"], color=col, alpha=0.85)
                axes[2].bar(idx-0.2, p["ba"]*1000, 0.38, color=col, alpha=0.85)
                axes[2].bar(idx+0.2, p["bc"]*1000, 0.38, color=col, alpha=0.45, hatch="//")

            names = [r.get("name","?") for r in st.session_state.results]
            axes[0].set_xlabel("E (V)"); axes[0].set_ylabel("log|i| (A/cm²)")
            axes[0].set_title("Evans Overlay", fontweight="bold")
            axes[0].legend(fontsize=8); axes[0].grid(True, ls="--", alpha=0.35)
            axes[0].set_facecolor("#fafbff")

            for ax, ttl, yl in zip(axes[1:],
                ["i_corr Comparison","Tafel Slopes (filled=βa, hatch=βc)"],
                ["i_corr (A/cm²)","Tafel slope (mV/dec)"]):
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=20, ha="right")
                ax.set_ylabel(yl); ax.set_title(ttl, fontweight="bold")
                ax.grid(True, axis="y", ls="--", alpha=0.35)
                ax.set_facecolor("#fafbff")
            axes[1].set_yscale("log")
            fig_c.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – HELP
# ─────────────────────────────────────────────────────────────────────────────
with tab_help:
    st.markdown("""
### Model Physics

All models use **signed current** convention: cathodic < 0, anodic > 0.

**Cathodic branch (all models)**
- Pure Butler-Volmer: `i_cat = -i_corr · exp(-(E-Ecorr)/βc)`

**Anodic branch**

| Model | Expression |
|---|---|
| Butler-Volmer | `i_ano = i_corr · exp((E-Ecorr)/βa)` |
| + Passive | Active → passive via logistic weight at E_pass |
| + Transpassive | + second sigmoid transition at E_trans |

---
### Fitting Strategy

1. **Differential Evolution** (global, 1500 iter, popsize=20) — avoids local minima  
2. **Levenberg-Marquardt** polish — convergence to tol = 10⁻¹⁴  
3. **Log₁₀|i| residuals** — equal weight across all current decades  
4. **Distance weighting** — upweights Tafel regions ±50 mV away from Ecorr

---
### Data Format

- Column 1: **E (V vs. reference)** — sorted ascending
- Column 2: **Signed current density** — cathodic **must** be negative
- Supported: CSV, TXT (Autolab/NOVA export), XLSX
- Comment lines beginning with `#` are skipped automatically

---
### Fit Quality

| R² | Quality |
|---|---|
| > 0.99 | Excellent — publication-ready |
| 0.95–0.99 | Good |
| < 0.95 | Review model / data |

RMSE < 0.15 log-units is generally acceptable for corrosion literature.
""")
