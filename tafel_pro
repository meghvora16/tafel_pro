#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  TAFEL-PRO  —  ML-Augmented Global Polarization Curve Fitter       ║
║  Version 2.0 — Production / Conference-Ready                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  3-Stage Pipeline:                                                  ║
║    1. Random-Forest curve-type classifier (self-trained on          ║
║       synthetic data from the physics engine)                       ║
║    2. Neural surrogate for initial parameter estimation             ║
║    3. Physics-constrained hybrid optimizer with AICc model          ║
║       selection and separate anodic/cathodic pre-fitting            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Usage:                                                             ║
║    python tafel_pro.py <root_folder> [options]                      ║
║    python tafel_pro.py . --area 0.5 --material "304 SS"            ║
║    python tafel_pro.py ./data --pattern "**/*.csv" --fit-rs         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter, argrelextrema
from scipy.stats import linregress
from itertools import groupby
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings, re, sys, os, glob, time, argparse, textwrap, json, pickle, hashlib

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ════════════════════════════════════════════════════════════════════
def slog(x):
    """Safe log10 of absolute value."""
    return np.log10(np.maximum(np.abs(x), 1e-30))

def sm(y, w=11, p=3):
    """Savitzky-Golay smooth with auto window capping."""
    n = len(y)
    w = min(w, n if n % 2 == 1 else n - 1)
    return savgol_filter(y, w, min(p, w - 1)) if w >= 5 else y.copy()

def r2sc(yt, yp):
    """R² in log-space."""
    sr = np.sum((yt - yp)**2)
    st = np.sum((yt - yt.mean())**2)
    return float(max(0, 1 - sr / st)) if st > 0 else 0.0

def aicc(n, k, sse):
    """Corrected Akaike Information Criterion."""
    if n <= k + 1 or sse <= 0:
        return 1e30
    ll = -0.5 * n * np.log(sse / n)
    aic = 2 * k - 2 * ll
    correction = (2 * k * (k + 1)) / max(n - k - 1, 1)
    return aic + correction

def sig(x, k=1.0):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -50, 50)))


# ════════════════════════════════════════════════════════════════════
#  CURVE TYPES & PARAMETER DEFINITIONS
# ════════════════════════════════════════════════════════════════════
PARAM_NAMES = [
    "Ecorr", "icorr", "ba", "bc1", "iL",
    "i0_c2", "bc2",
    "Epp", "k_pass", "ipass",
    "Eb", "a_tp", "b_tp",
    "Esp", "k_sp", "ipass2",
    "Rs"
]
NP = 17
LOG_PARAMS = {1, 4, 5, 9, 11, 15}  # indices fitted in log-space

class CurveType:
    """Enumeration of supported polarization curve topologies."""
    ACTIVE      = "A"
    ACTIVE_D    = "AD"
    ACTIVE_H2   = "AH"
    PASSIVE     = "P"
    PASSIVE_D   = "PD"
    PASS_TP     = "PT"
    PASS_TP_SP  = "PTS"
    FULL        = "F"

    INFO = {
        "A":   ("Active Only",                  [0,1,2,3],                           4),
        "AD":  ("Active + Diffusion",           [0,1,2,3,4],                         5),
        "AH":  ("Active + Dual Cathodic",       [0,1,2,3,4,5,6],                     7),
        "P":   ("Active–Passive",               [0,1,2,3,7,8,9],                     7),
        "PD":  ("Active–Passive + Diff.",       [0,1,2,3,4,5,6,7,8,9],              10),
        "PT":  ("Passive–Transpassive",         [0,1,2,3,4,5,6,7,8,9,10,11,12],     13),
        "PTS": ("Full + Secondary Passivity",   list(range(16)),                     16),
        "F":   ("Full Multi-Region",            list(range(16)),                     16),
    }

    ALL = ["A", "AD", "AH", "P", "PD", "PT", "PTS", "F"]
    SIMPLE = ["A", "AD", "AH"]
    PASSIVE_TYPES = ["P", "PD", "PT", "PTS", "F"]

    @staticmethod
    def free_idx(ct):
        return CurveType.INFO.get(ct, ("", list(range(16)), 16))[1]

    @staticmethod
    def nfree(ct):
        return CurveType.INFO.get(ct, ("", [], 0))[2]

    @staticmethod
    def name(ct):
        return CurveType.INFO.get(ct, ("Unknown", [], 0))[0]

CT = CurveType


# ════════════════════════════════════════════════════════════════════
#  PHYSICS MODEL (17-parameter global)
# ════════════════════════════════════════════════════════════════════
def polarization_model(E, p, i_cap=None):
    """
    Full 17-parameter polarization model.
    Supports: active dissolution, O₂ diffusion-limited cathodic,
    H₂ evolution, active-passive transition, transpassive dissolution,
    secondary passivity, and ohmic drop (Rs).
    """
    E = np.asarray(E, dtype=float)
    Ecorr, icorr, ba, bc1, iL = p[0], p[1], p[2], p[3], p[4]
    i0_c2, bc2 = p[5], p[6]
    Epp, k_pass, ipass = p[7], p[8], p[9]
    Eb, a_tp, b_tp = p[10], p[11], p[12]
    Esp, k_sp, ipass2 = p[13], p[14], p[15]
    Rs = max(p[16], 0.0)
    ic = np.zeros_like(E) if i_cap is None else np.asarray(i_cap, dtype=float)

    E_eff = E.copy()
    for _ in range(6 if Rs > 0 else 1):
        eta = E_eff - Ecorr
        # Cathodic: O₂ reduction with diffusion limit
        ic1k = icorr * np.exp(np.clip(-2.303 * eta / max(bc1, 1e-12), -50, 50))
        i_c1 = ic1k / (1.0 + ic1k / max(iL, 1e-20))
        # Cathodic: H₂ evolution
        i_c2 = i0_c2 * np.exp(np.clip(-2.303 * eta / max(bc2, 1e-12), -50, 50))
        # Anodic: active dissolution
        i_act = icorr * np.exp(np.clip(2.303 * eta / max(ba, 1e-12), -50, 50))
        # Active-passive transition
        t1 = sig(E_eff - Epp, k_pass)
        i_p1 = i_act * (1 - t1) + ipass * t1
        # Transpassive
        i_tp = a_tp * np.exp(np.clip(b_tp * (E_eff - Eb), -50, 50)) * sig(E_eff - Eb, 40)
        # Secondary passivity
        t2 = sig(E_eff - Esp, k_sp)
        i_an = (i_p1 + i_tp) * (1 - t2) + ipass2 * t2
        # Net current
        i_net = i_an - (i_c1 + i_c2) + ic
        if Rs > 0:
            E_eff = E - i_net * Rs

    return i_net


def model_components(E, p):
    """Return individual current components for diagnostics."""
    E = np.asarray(E, dtype=float)
    Ecorr, icorr, ba, bc1, iL = p[0:5]
    i0_c2, bc2 = p[5:7]
    Epp, k_pass, ipass = p[7:10]
    Eb, a_tp, b_tp = p[10:13]
    Esp, k_sp, ipass2 = p[13:16]
    Rs = max(p[16], 0.0)

    E_eff = E.copy()
    for _ in range(6 if Rs > 0 else 1):
        eta = E_eff - Ecorr
        ic1k = icorr * np.exp(np.clip(-2.303 * eta / max(bc1, 1e-12), -50, 50))
        i_c1 = ic1k / (1 + ic1k / max(iL, 1e-20))
        i_c2 = i0_c2 * np.exp(np.clip(-2.303 * eta / max(bc2, 1e-12), -50, 50))
        i_act = icorr * np.exp(np.clip(2.303 * eta / max(ba, 1e-12), -50, 50))
        t1 = sig(E_eff - Epp, k_pass)
        i_p1 = i_act * (1 - t1) + ipass * t1
        i_tp = a_tp * np.exp(np.clip(b_tp * (E_eff - Eb), -50, 50)) * sig(E_eff - Eb, 40)
        t2 = sig(E_eff - Esp, k_sp)
        i_an = (i_p1 + i_tp) * (1 - t2) + ipass2 * t2
        i_net = i_an - (i_c1 + i_c2)
        if Rs > 0:
            E_eff = E - i_net * Rs

    return dict(ic1=i_c1, ic2=i_c2, iact=i_act, ip1=i_p1, itp=i_tp,
                ian=i_an, t1=t1, t2=t2, itot=i_net)


# ════════════════════════════════════════════════════════════════════
#  STAGE 1: SYNTHETIC DATA GENERATOR + ML CLASSIFIER
# ════════════════════════════════════════════════════════════════════

def _random_params(ct, rng):
    """Generate physically plausible random parameters for a given curve type."""
    passive_types = [CT.PASSIVE, CT.PASSIVE_D, CT.PASS_TP, CT.PASS_TP_SP, CT.FULL]
    tp_types = [CT.PASS_TP, CT.PASS_TP_SP, CT.FULL]
    dual_cat_types = [CT.ACTIVE_H2, CT.PASSIVE_D, CT.PASS_TP, CT.PASS_TP_SP, CT.FULL]
    diff_types = [CT.ACTIVE_D, CT.ACTIVE_H2, CT.PASSIVE_D, CT.PASS_TP, CT.PASS_TP_SP, CT.FULL]
    sp_types = [CT.PASS_TP_SP, CT.FULL]

    Ecorr = rng.uniform(-1.2, 0.2)
    icorr = 10**rng.uniform(-8, -3)
    ba = rng.uniform(0.030, 0.300)
    bc1 = rng.uniform(0.040, 0.400)
    iL = 10**rng.uniform(-6, -2) if ct in diff_types else 1e10
    i0_c2 = 10**rng.uniform(-10, -5) if ct in dual_cat_types else 1e-30
    bc2 = rng.uniform(0.080, 0.250) if ct in dual_cat_types else 0.150
    Epp = Ecorr + rng.uniform(0.10, 0.50) if ct in passive_types else Ecorr + 50
    k_pass = rng.uniform(20, 120) if ct in passive_types else 50
    ipass = icorr * 10**rng.uniform(-3, -0.5) if ct in passive_types else icorr
    Eb = Epp + rng.uniform(0.15, 0.60) if ct in tp_types else Epp + 50
    a_tp = ipass * rng.uniform(0.001, 0.1) if ct in tp_types else 1e-30
    b_tp = rng.uniform(3, 25) if ct in tp_types else 8
    Esp = Eb + rng.uniform(0.10, 0.40) if ct in sp_types else Eb + 50
    k_sp = rng.uniform(20, 120) if ct in sp_types else 50
    ipass2 = ipass * rng.uniform(0.3, 3.0) if ct in sp_types else icorr
    Rs = 0.0

    return np.array([Ecorr, icorr, ba, bc1, iL, i0_c2, bc2,
                     Epp, k_pass, ipass, Eb, a_tp, b_tp,
                     Esp, k_sp, ipass2, Rs])


def _extract_features(E, i, n_feat=60):
    """
    Extract hand-crafted spectral features from a polarization curve.
    Returns a fixed-length feature vector regardless of input size.
    """
    si = np.argsort(E)
    E_s, i_s = E[si], i[si]
    ai = np.abs(i_s)
    lg = slog(i_s)
    n = len(E_s)

    feats = []

    # 1. Normalized log|i| resampled to fixed grid (20 points)
    E_norm = np.linspace(0, 1, 20)
    E_01 = (E_s - E_s[0]) / max(E_s[-1] - E_s[0], 1e-6)
    lg_interp = np.interp(E_norm, E_01, lg)
    lg_interp -= lg_interp.mean()  # zero-mean
    feats.extend(lg_interp.tolist())  # 20 features

    # 2. Gradient features (10 points)
    dlg = np.gradient(lg_interp, E_norm)
    feats.extend(dlg[::2].tolist())  # 10 features

    # 3. Second derivative features (10 points)
    d2lg = np.gradient(dlg, E_norm)
    feats.extend(d2lg[::2].tolist())  # 10 features

    # 4. Statistical features
    feats.append(float(np.std(lg)))
    feats.append(float(np.max(lg) - np.min(lg)))
    feats.append(float(np.median(lg)))

    # 5. Zero-crossing info
    signs = np.sign(i_s)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    feats.append(float(crossings))

    # 6. Cathodic region slope
    Ecorr_est = E_s[np.argmin(ai)]
    cat = E_s < Ecorr_est
    if np.sum(cat) > 5:
        try:
            s, _, r, *_ = linregress(E_s[cat][-min(20, np.sum(cat)):],
                                     lg[cat][-min(20, np.sum(cat)):])
            feats.append(float(s))
            feats.append(float(r**2))
        except:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0, 0.0])

    # 7. Anodic region: plateau detection
    an = E_s > Ecorr_est
    if np.sum(an) > 5:
        lg_an = lg[an]
        dlg_an = np.abs(np.gradient(lg_an))
        feats.append(float(np.median(dlg_an)))
        feats.append(float(np.percentile(dlg_an, 90)))
        # Number of "flat" segments (potential passive regions)
        flat_frac = np.mean(dlg_an < 0.5)
        feats.append(float(flat_frac))
    else:
        feats.extend([0.0, 0.0, 0.0])

    # 8. Peaks in anodic region
    if np.sum(an) > 15:
        lg_sm = sm(lg[an], min(11, (np.sum(an)//2)*2-1 or 5))
        try:
            pks = argrelextrema(lg_sm, np.greater, order=max(3, np.sum(an)//15))[0]
            feats.append(float(len(pks)))
        except:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # 9. Asymmetry: difference between cathodic and anodic slopes near Ecorr
    near_cat = (E_s > Ecorr_est - 0.15) & (E_s < Ecorr_est - 0.02)
    near_an = (E_s > Ecorr_est + 0.02) & (E_s < Ecorr_est + 0.15)
    if np.sum(near_cat) > 3 and np.sum(near_an) > 3:
        try:
            sc, *_ = linregress(E_s[near_cat], lg[near_cat])
            sa, *_ = linregress(E_s[near_an], lg[near_an])
            feats.append(float(sa - sc))
            feats.append(float(abs(sa) / max(abs(sc), 0.1)))
        except:
            feats.extend([0.0, 1.0])
    else:
        feats.extend([0.0, 1.0])

    # Pad/truncate to exactly n_feat
    feats = feats[:n_feat]
    while len(feats) < n_feat:
        feats.append(0.0)

    return np.array(feats, dtype=float)


def build_classifier(n_samples=500, seed=42):
    """
    Build a self-trained Random Forest curve-type classifier.
    Generates synthetic polarization curves from the physics engine
    and trains on extracted features.
    """
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []

    types_to_train = [CT.ACTIVE, CT.ACTIVE_D, CT.ACTIVE_H2,
                      CT.PASSIVE, CT.PASSIVE_D, CT.PASS_TP, CT.FULL]

    for ct in types_to_train:
        generated = 0
        attempts = 0
        while generated < n_samples and attempts < n_samples * 5:
            attempts += 1
            try:
                p = _random_params(ct, rng)
                Ecorr = p[0]
                E = np.linspace(Ecorr - 0.8, Ecorr + 0.8, 200)
                i = polarization_model(E, p)
                if not np.all(np.isfinite(i)) or np.all(np.abs(i) < 1e-30):
                    continue
                noise_level = 10**rng.uniform(-2.5, -1.0)
                i_noisy = i * (1 + noise_level * rng.randn(len(i)))
                if not np.all(np.isfinite(i_noisy)):
                    continue
                feats = _extract_features(E, i_noisy)
                if np.all(np.isfinite(feats)):
                    X_all.append(feats)
                    y_all.append(ct)
                    generated += 1
            except:
                continue

    if len(X_all) < 10:
        raise RuntimeError(f"Only {len(X_all)} valid training samples generated — check _random_params")

    X = np.array(X_all)
    y = np.array(y_all)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=seed, n_jobs=-1, class_weight='balanced'
    )
    clf.fit(X_scaled, y)

    return clf, scaler


def build_surrogate(n_samples=400, seed=42):
    """
    Build a neural network surrogate that estimates initial parameters
    from curve features. Self-trained on synthetic data.
    """
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []

    p_lo = np.array([-1.5, -10, 0.01, 0.01, -8, -12, 0.04,
                     -1.5, 5, -10, -1.0, -30, 0.5, -0.5, 5, -10, 0])
    p_hi = np.array([0.5, -2, 0.5, 0.5, -1, -4, 0.35,
                     1.0, 200, -3, 1.5, -10, 40, 2.0, 200, -3, 0])

    for ct in [CT.ACTIVE_D, CT.ACTIVE_H2, CT.PASSIVE_D, CT.PASS_TP, CT.FULL]:
        generated = 0
        attempts = 0
        while generated < n_samples and attempts < n_samples * 5:
            attempts += 1
            try:
                p = _random_params(ct, rng)
                Ecorr = p[0]
                E = np.linspace(Ecorr - 0.8, Ecorr + 0.8, 200)
                i = polarization_model(E, p)
                if not np.all(np.isfinite(i)) or np.all(np.abs(i) < 1e-30):
                    continue
                noise_level = 10**rng.uniform(-2.5, -1.0)
                i_noisy = i * (1 + noise_level * rng.randn(len(i)))
                if not np.all(np.isfinite(i_noisy)):
                    continue
                feats = _extract_features(E, i_noisy)

                p_norm = np.zeros(NP)
                for j in range(NP):
                    val = np.log10(max(p[j], 1e-30)) if j in LOG_PARAMS else p[j]
                    p_norm[j] = (val - p_lo[j]) / max(p_hi[j] - p_lo[j], 1e-6)
                    p_norm[j] = np.clip(p_norm[j], 0, 1)

                if np.all(np.isfinite(feats)) and np.all(np.isfinite(p_norm)):
                    X_all.append(feats)
                    y_all.append(p_norm)
                    generated += 1
            except:
                continue

    if len(X_all) < 10:
        raise RuntimeError(f"Only {len(X_all)} valid surrogate samples generated")

    X = np.array(X_all)
    Y = np.array(y_all)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    net = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu', solver='adam',
        max_iter=500, random_state=seed,
        early_stopping=True, validation_fraction=0.15,
        learning_rate='adaptive', alpha=0.001
    )
    net.fit(X_scaled, Y)

    return net, scaler_x, p_lo, p_hi


# ════════════════════════════════════════════════════════════════════
#  STAGE 2: ROBUST Ecorr DETECTION
# ════════════════════════════════════════════════════════════════════

def detect_ecorr(E, i):
    """
    Robust Ecorr detection using zero-crossing analysis.
    Returns (Ecorr_value, index_in_sorted_array).
    """
    n = len(E)
    if n == 0:
        return 0.0, 0

    si = np.argsort(E)
    Es, Is = E[si], i[si]

    # Light smoothing
    max_w = max(5, (n // 5) | 1)
    w = min(max_w, 11)
    if w % 2 == 0:
        w -= 1
    w = max(5, w)

    try:
        ism = savgol_filter(Is, w, min(3, w-1)) if n >= w >= 5 else Is.copy()
    except:
        ism = Is.copy()

    def find_crossings(ia):
        cands = []
        for k in range(len(ia) - 1):
            if np.sign(ia[k]) * np.sign(ia[k+1]) < 0:
                denom = ia[k+1] - ia[k]
                if abs(denom) < 1e-30:
                    continue
                Ec = Es[k] - ia[k] * (Es[k+1] - Es[k]) / denom
                goes_anodic = (ia[k] < 0 and ia[k+1] > 0)
                cands.append(dict(Ec=Ec, k_orig=int(si[k]), goes_anodic=goes_anodic))
        return cands

    cands_raw = find_crossings(Is)
    cands_sm = find_crossings(ism)

    # Merge, dedup
    raw_Es = set(round(c["Ec"], 4) for c in cands_raw)
    all_cands = list(cands_raw)
    for c in cands_sm:
        if round(c["Ec"], 4) not in raw_Es:
            all_cands.append(c)

    if not all_cands:
        k_best = int(np.argmin(np.abs(Is)))
        return float(Es[k_best]), int(si[k_best])

    # Prefer most cathodic anodic-going crossing (true Ecorr)
    anodic = [c for c in all_cands if c["goes_anodic"]]
    if anodic:
        best = min(anodic, key=lambda c: c["Ec"])
        return best["Ec"], best["k_orig"]

    best = min(all_cands, key=lambda c: c["Ec"])
    return best["Ec"], best["k_orig"]


# ════════════════════════════════════════════════════════════════════
#  STAGE 2b: SEPARATE ANODIC / CATHODIC PRE-FITTING
# ════════════════════════════════════════════════════════════════════

def fit_cathodic_tafel(E, i, Ecorr, max_points=50):
    """
    Fit the cathodic branch separately using linear regression
    in log|i| vs E space. Returns (bc, icorr_cat, iL_est, has_H2).

    Strategy (following Gallant/NRC approach):
      - Find the linear Tafel region in the cathodic branch
      - Detect diffusion-limited plateau
      - Detect H₂ evolution at very negative potentials
    """
    cat = (E < Ecorr - 0.02)
    if np.sum(cat) < 5:
        return 0.120, 1e-6, None, False

    Ec = E[cat]
    ic = np.abs(i[cat])
    lg = slog(ic)
    si = np.argsort(Ec)
    Ec, ic, lg = Ec[si], ic[si], lg[si]

    # Find best Tafel linear region by sliding window
    best_r2, best_slope, best_inter = 0, -8.0, -6.0
    best_range = (0, len(Ec)-1)
    min_pts = max(5, len(Ec) // 5)

    for start in range(0, max(1, len(Ec) - min_pts)):
        for end in range(start + min_pts, min(start + max_points, len(Ec))):
            try:
                s, b, r, *_ = linregress(Ec[start:end], lg[start:end])
                if s < -1.0 and r**2 > best_r2 and abs(1/s) > 0.02 and abs(1/s) < 0.6:
                    best_r2 = r**2
                    best_slope = s
                    best_inter = b
                    best_range = (start, end)
            except:
                continue

    bc = abs(1 / best_slope) if abs(best_slope) > 0.1 else 0.120
    icorr_cat = 10**(best_inter + best_slope * Ecorr)

    # Detect diffusion-limited plateau
    iL_est = None
    if len(Ec) > 15:
        lg_sm = sm(lg, min(11, (len(Ec)//2)*2-1 or 5))
        dlg = np.abs(np.gradient(lg_sm, Ec))
        flat = dlg < max(np.percentile(dlg, 30), 0.7)
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            longest = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in longest[1]]
            rng = abs(Ec[idxs[-1]] - Ec[idxs[0]])
            if len(idxs) >= 3 and rng > 0.02:
                iL_est = float(np.median(ic[idxs]))

    # Detect H₂ evolution
    has_H2 = False
    if iL_est is not None:
        very_neg = Ec < Ec[best_range[0]] - 0.05
        if np.sum(very_neg) > 5:
            grad = np.gradient(lg[very_neg], Ec[very_neg])
            if np.mean(grad[:min(10, len(grad))]) < -1.5:
                has_H2 = True

    return bc, max(icorr_cat, 1e-15), iL_est, has_H2


def fit_anodic_tafel(E, i, Ecorr, max_points=50):
    """
    Fit the anodic branch separately. Returns (ba, passivity_info).

    passivity_info is a dict with keys:
      Epp, ipass, Eb, has_passive, has_transpassive
    """
    an = (E > Ecorr + 0.02)
    if np.sum(an) < 5:
        return 0.060, {}

    Ea = E[an]
    ia = np.abs(i[an])
    lg = slog(ia)
    si = np.argsort(Ea)
    Ea, ia, lg = Ea[si], ia[si], lg[si]

    # Find Tafel region
    best_r2, best_slope, best_inter = 0, 8.0, -6.0
    min_pts = max(5, len(Ea) // 5)

    for start in range(0, max(1, len(Ea) - min_pts)):
        for end in range(start + min_pts, min(start + max_points, len(Ea))):
            try:
                s, b, r, *_ = linregress(Ea[start:end], lg[start:end])
                if s > 1.0 and r**2 > best_r2 and abs(1/s) > 0.02 and abs(1/s) < 0.6:
                    best_r2 = r**2
                    best_slope = s
                    best_inter = b
            except:
                continue

    ba = abs(1 / best_slope) if abs(best_slope) > 0.1 else 0.060

    # Detect passive region
    info = {"has_passive": False, "has_transpassive": False}

    if len(Ea) > 15:
        lg_sm = sm(lg, min(15, (len(Ea)//2)*2-1 or 5))
        dlg = np.gradient(lg_sm, Ea)
        adl = np.abs(dlg)

        # Look for current peak followed by flat region
        pks = []
        if len(Ea) > 20:
            try:
                order = max(3, len(Ea)//15)
                pk_idx = argrelextrema(lg_sm, np.greater, order=order)[0]
                for pk in pk_idx:
                    rest = lg_sm[pk:]
                    if len(rest) > 5:
                        prom = lg_sm[pk] - np.min(rest[3:])
                        if prom > 0.3:
                            pks.append((pk, prom))
            except:
                pass

        # Look for flat (passive) region
        thr = max(np.percentile(adl, 25), 0.8)
        flat = adl < thr
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        passive_runs = []
        for _, ri in runs:
            idxs = [s[0] for s in ri]
            rng = abs(Ea[idxs[-1]] - Ea[idxs[0]])
            if len(idxs) >= 5 and rng > 0.08:
                im = float(np.median(ia[idxs]))
                pre = Ea < Ea[idxs[0]]
                # Passive current should be clearly lower than active peak
                if np.sum(pre) > 2 and im < np.max(ia[pre]) * 0.4:
                    passive_runs.append(dict(
                        Epp=float(Ea[idxs[0]]), Epe=float(Ea[idxs[-1]]),
                        ipass=im, rng=rng
                    ))

        if passive_runs:
            pr = passive_runs[0]
            info["has_passive"] = True
            info["Epp"] = pr["Epp"]
            info["ipass"] = pr["ipass"]

            # Look for transpassive rise after passive region
            post = Ea > pr["Epe"]
            if np.sum(post) > 5:
                dlg_post = np.gradient(lg[np.where(post)[0]], Ea[post])
                if np.max(dlg_post) > 3.0:
                    info["has_transpassive"] = True
                    jump_idx = np.argmax(dlg_post > 3.0)
                    info["Eb"] = float(Ea[post][jump_idx])

    return ba, info


# ════════════════════════════════════════════════════════════════════
#  STAGE 3: OPTIMIZER
# ════════════════════════════════════════════════════════════════════

class PhysicsOptimizer:
    """
    Physics-constrained optimizer with AICc model selection.
    Runs the fitting cascade and selects the best model.
    """

    def __init__(self, E, i, fit_rs=False, rs_max=200.0):
        self.E = E
        self.i = i
        self.ld = slog(i)
        self.n = len(E)
        self.fit_rs = fit_rs
        self.rs_max = rs_max
        self.log = []

    def _build_bounds(self, p0, ct, reg):
        """Build parameter bounds based on curve type and initial guess."""
        Ec = p0[0]
        ic = max(p0[1], 1e-14)
        Emax = float(np.max(self.E))
        Emin = float(np.min(self.E))
        Erange = Emax - Emin

        lo = np.array([
            Emin,                                                          # Ecorr
            max(ic * 1e-5, 1e-15),                                        # icorr
            0.010,                                                         # ba
            0.010,                                                         # bc1
            max(ic * 0.1, 1e-10),                                         # iL
            max(ic * 1e-8, 1e-18),                                        # i0_c2
            0.04,                                                          # bc2
            Ec - 0.05 if reg.get("has_passive") else Emax + 5,            # Epp
            5.0,                                                           # k_pass
            max(ic * 1e-5, 1e-15) if reg.get("has_passive") else 1e-15,   # ipass
            Ec if reg.get("has_transpassive") else Emax + 5,              # Eb
            1e-30,                                                         # a_tp
            0.5,                                                           # b_tp
            Ec if reg.get("has_sp") else Emax + 10,                       # Esp
            5.0,                                                           # k_sp
            max(ic * 1e-5, 1e-15),                                        # ipass2
            0.0                                                            # Rs
        ])

        hi = np.array([
            Emax,                                                          # Ecorr
            min(ic * 1e5, 1e1),                                           # icorr
            0.500,                                                         # ba
            0.500,                                                         # bc1
            max(ic * 1e6, 1e0),                                           # iL
            max(ic * 100, 1e-4),                                          # i0_c2
            0.400,                                                         # bc2
            Emax if reg.get("has_passive") else Emax + 15,                # Epp
            200.0,                                                         # k_pass
            max(ic * 1e3, 1e-2) if reg.get("has_passive") else 1e-2,     # ipass
            Emax + 0.2 if reg.get("has_transpassive") else Emax + 15,    # Eb
            max(ic * 1e4, 1e-6),                                          # a_tp
            40.0,                                                          # b_tp
            Emax + 0.5 if reg.get("has_sp") else Emax + 25,              # Esp
            200.0,                                                         # k_sp
            max(ic * 1e3, 1e-2),                                          # ipass2
            self.rs_max                                                    # Rs
        ])

        # Ensure lo < hi everywhere
        for j in range(NP):
            if lo[j] >= hi[j]:
                m = p0[j]
                lo[j] = m - abs(m) * 0.5 - 1e-6
                hi[j] = m + abs(m) * 0.5 + 1e-6

        return lo, hi

    def _pack(self, p, fidx):
        return np.array([np.log10(max(p[j], 1e-30)) if j in LOG_PARAMS else p[j]
                         for j in fidx])

    def _unpack(self, x, p_base, fidx):
        p = p_base.copy()
        for k, j in enumerate(fidx):
            p[j] = 10**x[k] if j in LOG_PARAMS else x[k]
        return p

    def _pbounds(self, lo, hi, fidx):
        return [(np.log10(max(lo[j], 1e-30)), np.log10(max(hi[j], 1e-30)))
                if j in LOG_PARAMS else (lo[j], hi[j]) for j in fidx]

    def _objective(self, x, p_base, fidx):
        p = self._unpack(x, p_base, fidx)
        # Reject physically unreasonable icorr
        if p[1] < 1e-14 or p[1] > 0.1:
            return 1e30
        try:
            pred = polarization_model(self.E, p)
            return float(np.sum((self.ld - slog(pred))**2))
        except:
            return 1e30

    def fit_single_model(self, ct, p0, reg, label=""):
        """Fit a single curve type model. Returns (best_params, r2, aicc_val)."""
        fidx = CT.free_idx(ct)
        if self.fit_rs:
            fidx = fidx + [16]
        nf = len(fidx)

        lo, hi = self._build_bounds(p0, ct, reg)
        bnds = self._pbounds(lo, hi, fidx)

        best_p = p0.copy()
        best_sse = self._objective(self._pack(p0, fidx), p0, fidx)

        def obj(x):
            return self._objective(x, p0, fidx)

        def update(x, tag):
            nonlocal best_p, best_sse
            s = obj(x)
            if s < best_sse:
                best_p = self._unpack(x, p0, fidx)
                best_sse = s
                return True
            return False

        # DE global search
        try:
            ps = max(12, min(20, nf * 2))
            mi = max(250, min(800, nf * 40))
            res = differential_evolution(obj, bnds, seed=42, maxiter=mi, popsize=ps,
                tol=1e-14, mutation=(0.5, 1.9), recombination=0.9, polish=False,
                workers=1, strategy='best1bin', atol=1e-14)
            update(res.x, "DE")
        except:
            pass

        # L-BFGS-B
        try:
            r2 = minimize(obj, self._pack(best_p, fidx), method="L-BFGS-B",
                bounds=bnds, options={"maxiter": 25000, "ftol": 1e-16})
            update(r2.x, "LBFGS")
        except:
            pass

        # Nelder-Mead
        try:
            r3 = minimize(obj, self._pack(best_p, fidx), method="Nelder-Mead",
                options={"maxiter": 15000, "xatol": 1e-14, "fatol": 1e-16, "adaptive": True})
            update(r3.x, "NM")
        except:
            pass

        # Powell
        try:
            r4 = minimize(obj, self._pack(best_p, fidx), method="Powell",
                options={"maxiter": 10000, "xtol": 1e-14, "ftol": 1e-16})
            update(r4.x, "Powell")
        except:
            pass

        # Compute metrics
        pred = polarization_model(self.E, best_p)
        r2_val = r2sc(self.ld, slog(pred))
        sse = np.sum((self.ld - slog(pred))**2)
        aicc_val = aicc(self.n, nf, sse)

        return best_p, r2_val, aicc_val

    def run(self, p0_dict, reg, ct_candidates):
        """
        Run multi-model fitting and select the best via AICc.

        Args:
            p0_dict: dict mapping CT -> initial parameter vector
            reg: detection results dict
            ct_candidates: list of curve types to try

        Returns:
            (best_params, best_r2, best_ct, results_table)
        """
        t0 = time.time()
        results = []

        for ct in ct_candidates:
            ct_name = CT.name(ct)
            nf = CT.nfree(ct)
            self.log.append(f"  [{ct}] {ct_name} ({nf} free)...")

            p0 = p0_dict.get(ct, p0_dict.get(CT.ACTIVE_D, np.zeros(NP)))
            bp, r2, aic = self.fit_single_model(ct, p0, reg, ct_name)

            results.append(dict(ct=ct, name=ct_name, nfree=nf,
                                r2=r2, aicc=aic, params=bp))
            self.log.append(f"       R²={r2:.6f}, AICc={aic:.1f}")

        # Select best model by AICc (penalizes complexity)
        results.sort(key=lambda x: x["aicc"])
        best = results[0]

        # Override: if a simpler model has nearly the same R² (within 0.002),
        # prefer it over a complex one even if AICc slightly favors complexity
        for r in results:
            if r["nfree"] < best["nfree"] and best["r2"] - r["r2"] < 0.002:
                best = r
                self.log.append(f"  → Parsimony override: {r['name']} (simpler, similar R²)")
                break

        elapsed = time.time() - t0
        self.log.append(f"  ═══ Best: {best['name']} | R²={best['r2']:.6f} | "
                        f"AICc={best['aicc']:.1f} | {elapsed:.1f}s")

        return best["params"], best["r2"], best["ct"], results


# ════════════════════════════════════════════════════════════════════
#  INITIAL GUESS BUILDER
# ════════════════════════════════════════════════════════════════════

def build_initial_guess(E, i, Ecorr, bc, ba, icorr_cat, iL_est, has_H2,
                        anodic_info, ct):
    """Build initial parameter vector from pre-fit results."""
    Emax = float(np.max(E))
    Emin = float(np.min(E))
    ic = max(icorr_cat, 1e-12)
    ai = np.abs(i)

    # For passive/transpassive models, we need to estimate passive params
    # even if the separate pre-fit didn't detect them
    Epp_est = anodic_info.get("Epp", Ecorr + 0.25)
    ipass_est = anodic_info.get("ipass", ic * 0.1)
    Eb_est = anodic_info.get("Eb", Epp_est + 0.30)

    # If no passive region was found by pre-fit, estimate from data
    if not anodic_info.get("has_passive") and ct in CT.PASSIVE_TYPES:
        an_mask = E > Ecorr + 0.05
        if np.sum(an_mask) > 10:
            # Look for minimum in anodic region as potential passive current
            lg_an = slog(ai[an_mask])
            E_an = E[an_mask]
            lg_sm = sm(lg_an, min(11, (np.sum(an_mask)//2)*2-1 or 5))
            min_idx = np.argmin(lg_sm)
            if min_idx > 2 and min_idx < len(lg_sm) - 2:
                Epp_est = float(E_an[max(0, min_idx - 3)])
                ipass_est = float(10**lg_sm[min_idx])
            else:
                Epp_est = Ecorr + 0.25
                ipass_est = ic * 0.01

    iL_val = iL_est if iL_est is not None else max(np.percentile(ai, 95), ic * 100)

    p0 = np.array([
        Ecorr,
        ic,
        max(ba, 0.020),
        max(bc, 0.020),
        iL_val,
        ic * 1e-3 if has_H2 else 1e-30,
        0.150 if has_H2 else 0.150,
        Epp_est if ct in CT.PASSIVE_TYPES else Emax + 10,
        40.0 if ct in CT.PASSIVE_TYPES else 50.0,
        ipass_est if ct in CT.PASSIVE_TYPES else ic,
        Eb_est if ct in [CT.PASS_TP, CT.PASS_TP_SP, CT.FULL] else Emax + 10,
        ipass_est * 0.01 if ct in [CT.PASS_TP, CT.PASS_TP_SP, CT.FULL] else 1e-30,
        8.0,
        Emax + 20,
        50.0,
        ic,
        0.0
    ])
    return p0


# ════════════════════════════════════════════════════════════════════
#  FILE I/O
# ════════════════════════════════════════════════════════════════════

COL_SIGNATURES = [
    (r"we.*potential", r"we.*current", "A"),
    (r"ewe", r"i/ma", "mA"),
    (r"ewe", r"<i>/ma", "mA"),
    (r"^vf$", r"^im$", "A"),
    (r"potential/v", r"current/a", "A"),
    (r"e/v", r"i/a", "A"),
    (r"potential|volt|^e$|e \(v\)|e_v", r"current|amps|^i$|i \(a\)|i_a", "A"),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]

UNIT_HINTS = {
    r"\(a\)|_a$|/a$": 1.0,
    r"\(ma\)|_ma$|/ma$": 1e-3,
    r"\(ua\)|_ua$|/ua$": 1e-6,
    r"a/cm": 1.0,
    r"ma/cm": 1e-3,
}


def auto_detect_columns(df):
    """Auto-detect potential and current columns."""
    cl = {c: c.lower().strip() for c in df.columns}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for Ep, Ip, u in COL_SIGNATURES:
        em = [c for c, v in cl.items() if re.search(Ep, v) and c in num]
        im = [c for c, v in cl.items() if re.search(Ip, v) and c in num and c not in em]
        if em and im:
            ec = sorted(em, key=lambda c: 0 if "we" in c.lower() else 1)[0]
            ic = im[0]
            f = 1e-3 if u == "mA" else 1.0
            for p, fv in UNIT_HINTS.items():
                if re.search(p, cl[ic]):
                    f = fv
                    break
            return ec, ic, f

    if len(num) >= 2:
        return num[0], num[1], 1.0
    raise ValueError("Cannot detect potential/current columns.")


def load_file(path):
    """Load data file (xlsx, csv, txt, tsv)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    lines = text.splitlines()
    skip = 0
    for idx, line in enumerate(lines):
        pts = re.split(r"[,;\t ]+", line.strip())
        if sum(1 for p in pts if re.match(r"^-?[\d.eE+\-]+$", p)) >= 2:
            if idx > 0:
                prev = re.split(r"[,;\t ]+", lines[idx-1].strip())
                if not all(re.match(r"^-?[\d.eE+\-]+$", p) for p in prev if p):
                    skip = idx - 1
                else:
                    skip = idx
            else:
                skip = idx
            break

    import io
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5:
                return df.dropna(axis=1, how="all")
        except:
            pass

    raise ValueError(f"Cannot parse {path}")


# ════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════

def save_plots(E, i, bp, Ecorr, ct, r2_val, out_prefix):
    """Generate publication-quality polarization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("    [!] matplotlib not installed — skipping plots.")
        return

    lg = slog(i)
    Em = np.linspace(E[0], E[-1], 1000)

    # ── Combined figure: data + fit + components ──
    fig = plt.figure(figsize=(14, 6), facecolor="#0f0f1a")
    gs = GridSpec(1, 2, figure=fig, wspace=0.32)

    # Colors
    COL_BG = "#1a1a2e"
    COL_DATA = "#64b5f6"
    COL_FIT = "#81c784"
    COL_ECORR = "#ef5350"
    COL_TEXT = "#e0e0e0"
    COL_GRID = "#2a2a3e"
    COL_SPINE = "#3a3a4e"

    # ── Panel 1: Measured vs Fit ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COL_BG)

    ax1.plot(E, lg, 'o', color=COL_DATA, markersize=2.5, alpha=0.6, label="Measured", zorder=2)
    try:
        im = polarization_model(Em, bp)
        ax1.plot(Em, slog(im), color=COL_FIT, linewidth=2.5,
                 label=f"Fit (R²={r2_val:.4f})", zorder=3)
    except:
        pass

    ax1.axvline(Ecorr, color=COL_ECORR, linestyle=":", linewidth=1.2, alpha=0.8,
                label=f"E_corr = {Ecorr:.4f} V")
    ax1.plot(bp[0], np.log10(max(bp[1], 1e-30)), 'x', color=COL_ECORR,
             markersize=14, markeredgewidth=3, label=f"i_corr = {bp[1]:.2e} A/cm²", zorder=4)

    ax1.set_xlabel("Potential (V vs Ref)", color=COL_TEXT, fontsize=11)
    ax1.set_ylabel("log₁₀|i| (A cm⁻²)", color=COL_TEXT, fontsize=11)
    ax1.set_title("Potentiodynamic Polarization — Global Fit", color=COL_TEXT, fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, facecolor=COL_BG, edgecolor=COL_SPINE, labelcolor=COL_TEXT, loc='best')
    ax1.tick_params(colors="#9e9e9e")
    ax1.grid(True, color=COL_GRID, alpha=0.5, linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color(COL_SPINE)

    # ── Panel 2: Component breakdown ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COL_BG)

    c = model_components(Em, bp)
    ax2.plot(Em, slog(c["ic1"]), ":", color="#ce93d8", linewidth=1.5, label="O₂ cathodic")
    ax2.plot(Em, slog(c["ic2"]), ":", color="#f48fb1", linewidth=1.5, label="H₂ cathodic")
    ax2.plot(Em, slog(c["iact"]), "--", color="#fff176", linewidth=1, alpha=0.8, label="Active Tafel")

    if ct not in CT.SIMPLE:
        ax2.plot(Em, slog(c["ip1"]), ":", color="#a5d6a7", linewidth=1.5, label="Passivated")
        ax2.plot(Em, slog(c["itp"]), ":", color="#ffab91", linewidth=1.5, label="Transpassive")

    ax2.plot(Em, slog(c["itot"]), "-", color=COL_FIT, linewidth=2.5, label="Net current")
    ax2.plot(E, lg, 'o', color=COL_DATA, markersize=1.5, alpha=0.3, zorder=1)

    ax2.set_xlabel("Potential (V vs Ref)", color=COL_TEXT, fontsize=11)
    ax2.set_ylabel("log₁₀|i| (A cm⁻²)", color=COL_TEXT, fontsize=11)
    ax2.set_title("Component Decomposition", color=COL_TEXT, fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, facecolor=COL_BG, edgecolor=COL_SPINE, labelcolor=COL_TEXT, loc='best')
    ax2.tick_params(colors="#9e9e9e")
    ax2.grid(True, color=COL_GRID, alpha=0.5, linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color(COL_SPINE)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_tafel_plot.png", dpi=180, facecolor="#0f0f1a",
                bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  MATERIALS DATABASE
# ════════════════════════════════════════════════════════════════════

MATERIALS = {
    "Carbon Steel / Iron":  (27.92, 7.87),
    "304 SS":               (25.10, 7.90),
    "316 SS":               (25.56, 8.00),
    "Copper":               (31.77, 8.96),
    "Aluminum":             (8.99,  2.70),
    "Nickel":               (29.36, 8.91),
    "Titanium":             (11.99, 4.51),
    "Zinc":                 (32.69, 7.14),
}


# ════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════

def process_file(fpath, area, ew, rho, fit_rs, rs_max, clf, clf_scaler,
                 surrogate, surr_scaler, surr_lo, surr_hi):
    """
    Full 3-stage processing pipeline for one file.
    """
    print(f"\n{'═'*70}")
    print(f"  FILE: {fpath}")
    print(f"{'═'*70}")
    t0 = time.time()

    # ── Load data ──
    try:
        df = load_file(fpath)
    except Exception as ex:
        print(f"  [ERROR] Load: {ex}")
        return None

    try:
        ec, ic, ifac = auto_detect_columns(df)
    except Exception as ex:
        print(f"  [ERROR] Columns: {ex}")
        return None

    E_raw = df[ec].values.astype(float)
    i_raw = df[ic].values.astype(float) * ifac
    ok = np.isfinite(E_raw) & np.isfinite(i_raw)
    E_raw, i_raw = E_raw[ok], i_raw[ok]
    i_density = i_raw / area

    # Sort by potential
    si = np.argsort(E_raw)
    E = E_raw[si]
    i = i_density[si]

    print(f"  Columns: {ec} / {ic} | Points: {len(i)} | "
          f"E: [{E.min():.3f}, {E.max():.3f}] V")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: ML Curve Classification
    # ════════════════════════════════════════════════════════════
    print(f"\n  ┌─ STAGE 1: ML Curve Classification")

    # Detect Ecorr first (needed for feature extraction)
    Ecorr, ec_idx = detect_ecorr(E, i)
    print(f"  │  Ecorr = {Ecorr:.4f} V")

    # Extract features and classify
    feats = _extract_features(E, i)
    feats_scaled = clf_scaler.transform(feats.reshape(1, -1))
    ct_pred = clf.predict(feats_scaled)[0]
    ct_proba = clf.predict_proba(feats_scaled)[0]
    ct_classes = clf.classes_

    # Get top-3 predictions
    top3_idx = np.argsort(ct_proba)[::-1][:3]
    top3 = [(ct_classes[j], ct_proba[j]) for j in top3_idx]

    print(f"  │  ML prediction: {CT.name(ct_pred)} ({ct_proba.max():.1%} confidence)")
    print(f"  │  Top-3: {' | '.join([f'{CT.name(c)}:{p:.1%}' for c, p in top3])}")

    # Build candidate list from ML predictions
    ct_candidates = []
    for c, p in top3:
        if p > 0.05 and c not in ct_candidates:
            ct_candidates.append(c)

    # Always include Active+Diffusion as baseline
    if CT.ACTIVE_D not in ct_candidates:
        ct_candidates.append(CT.ACTIVE_D)

    print(f"  └─ Candidates: {[CT.name(c) for c in ct_candidates]}")

    # ════════════════════════════════════════════════════════════
    # STAGE 2: Separate Pre-Fitting + Neural Surrogate
    # ════════════════════════════════════════════════════════════
    print(f"\n  ┌─ STAGE 2: Pre-Fitting & Neural Surrogate")

    # 2a. Separate cathodic/anodic fitting (Gallant/NRC approach)
    bc, icorr_cat, iL_est, has_H2 = fit_cathodic_tafel(E, i, Ecorr)
    ba, anodic_info = fit_anodic_tafel(E, i, Ecorr)

    print(f"  │  Cathodic: bc={bc:.4f} V/dec, icorr={icorr_cat:.2e}, "
          f"iL={'%.2e'%iL_est if iL_est else 'N/A'}, H₂={has_H2}")
    print(f"  │  Anodic:   ba={ba:.4f} V/dec, passive={anodic_info.get('has_passive', False)}, "
          f"transpassive={anodic_info.get('has_transpassive', False)}")

    # 2b. Neural surrogate initial guess
    surr_feats = surr_scaler.transform(feats.reshape(1, -1))
    p_norm = surrogate.predict(surr_feats)[0]
    p_surr = np.zeros(NP)
    for j in range(NP):
        val = surr_lo[j] + p_norm[j] * (surr_hi[j] - surr_lo[j])
        p_surr[j] = 10**val if j in LOG_PARAMS else val

    print(f"  │  Surrogate icorr={p_surr[1]:.2e}, ba={p_surr[2]:.4f}, bc={p_surr[3]:.4f}")

    # 2c. Build detection result dict
    reg = {
        "Ecorr": Ecorr,
        "iL": iL_est,
        "has_H2": has_H2,
        "has_passive": anodic_info.get("has_passive", False),
        "has_transpassive": anodic_info.get("has_transpassive", False),
        "has_sp": False,
    }
    if anodic_info.get("has_passive"):
        reg["Epp"] = anodic_info.get("Epp")
        reg["ipass"] = anodic_info.get("ipass")
    if anodic_info.get("has_transpassive"):
        reg["Eb"] = anodic_info.get("Eb")

    # 2d. Build initial guesses for each candidate model
    # Strategy: blend physics pre-fit with neural surrogate
    p0_dict = {}
    for ct in ct_candidates:
        # Start from physics-based initial guess
        p0_phys = build_initial_guess(E, i, Ecorr, bc, ba, icorr_cat, iL_est,
                                      has_H2, anodic_info, ct)
        # Blend with surrogate (trust physics more for Ecorr, icorr, ba, bc)
        p0 = p0_phys.copy()
        # Use surrogate for passive/transpassive params if physics didn't detect them
        if not anodic_info.get("has_passive") and ct in CT.PASSIVE_TYPES:
            for j in [7, 8, 9]:
                p0[j] = p_surr[j]
        if not anodic_info.get("has_transpassive") and ct in [CT.PT, CT.PTS, CT.FULL]:
            for j in [10, 11, 12]:
                p0[j] = p_surr[j]

        p0_dict[ct] = p0

    print(f"  └─ Initial guesses built for {len(ct_candidates)} models")

    # ════════════════════════════════════════════════════════════
    # STAGE 3: Physics-Constrained Optimization + AICc Selection
    # ════════════════════════════════════════════════════════════
    print(f"\n  ┌─ STAGE 3: Optimization & Model Selection")

    optimizer = PhysicsOptimizer(E, i, fit_rs, rs_max)
    bp, r2, best_ct, all_results = optimizer.run(p0_dict, reg, ct_candidates)

    for msg in optimizer.log:
        print(f"  │  {msg}")

    # ── Quality assessment ──
    stars = "★★★★★" if r2 >= 0.995 else "★★★★☆" if r2 >= 0.99 else \
            "★★★☆☆" if r2 >= 0.98 else "★★☆☆☆" if r2 >= 0.95 else "★☆☆☆☆"

    print(f"  │")
    print(f"  │  ═══════════════════════════════════════")
    print(f"  │  RESULT: {CT.name(best_ct)}")
    print(f"  │  R² = {r2:.6f}  {stars}")
    print(f"  │  Ecorr = {bp[0]:.4f} V")
    print(f"  │  icorr = {bp[1]:.3e} A/cm²")
    print(f"  │  ba = {bp[2]:.4f} V/dec | bc = {bp[3]:.4f} V/dec")
    if best_ct not in [CT.ACTIVE]:
        print(f"  │  iL = {bp[4]:.3e} A/cm²")
    if best_ct in CT.PASSIVE_TYPES:
        print(f"  │  Epp = {bp[7]:.4f} V | ipass = {bp[9]:.3e} A/cm²")
    print(f"  └─ ═══════════════════════════════════════")

    # ── Derived quantities ──
    B = (bp[2] * bp[3]) / (2.303 * (bp[2] + bp[3])) if bp[2] > 0 and bp[3] > 0 else 0
    CR = bp[1] * 3.27 * ew / rho  # mm/yr

    # ── Save results ──
    base = os.path.splitext(fpath)[0]

    # Results CSV
    results = {
        "File": os.path.basename(fpath),
        "Ecorr_V": bp[0], "icorr_A_cm2": bp[1],
        "ba_V_dec": bp[2], "bc_V_dec": bp[3],
        "B_Stern_Geary_V": B, "CR_mm_per_year": CR,
        "iL_A_cm2": bp[4] if best_ct not in [CT.ACTIVE] else None,
        "Epp_V": bp[7] if best_ct in CT.PASSIVE_TYPES else None,
        "ipass_A_cm2": bp[9] if best_ct in CT.PASSIVE_TYPES else None,
        "Eb_V": bp[10] if best_ct in [CT.PT, CT.PTS, CT.FULL] else None,
        "Rs_Ohm_cm2": bp[16] if fit_rs else 0.0,
        "R2_log": r2,
        "AICc": all_results[0]["aicc"] if all_results else None,
        "Curve_Type": CT.name(best_ct),
        "ML_Prediction": CT.name(ct_pred),
        "ML_Confidence": float(ct_proba.max()),
    }

    df_res = pd.DataFrame([results])
    csv_path = f"{base}_tafel_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n  ✓ Results  → {csv_path}")

    # Fit data CSV
    Z_fit = polarization_model(E, bp)
    df_data = pd.DataFrame({
        "E_V": E, "i_measured_A_cm2": i,
        "log_abs_i_measured": slog(i),
        "i_fit_A_cm2": Z_fit, "log_abs_i_fit": slog(Z_fit),
        "residual_log": slog(i) - slog(Z_fit)
    })
    data_path = f"{base}_tafel_fitdata.csv"
    df_data.to_csv(data_path, index=False)
    print(f"  ✓ Fit data → {data_path}")

    # Model comparison table
    comp_rows = []
    for r in all_results:
        comp_rows.append({
            "Model": r["name"], "N_free": r["nfree"],
            "R2": r["r2"], "AICc": r["aicc"],
            "Selected": "←" if r["ct"] == best_ct else ""
        })
    df_comp = pd.DataFrame(comp_rows)
    comp_path = f"{base}_tafel_model_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"  ✓ Models   → {comp_path}")

    # Plots
    save_plots(E, i, bp, Ecorr, best_ct, r2, base)
    print(f"  ✓ Plots    → {base}_tafel_plot.png")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    return dict(file=fpath, r2=r2, ct=best_ct, icorr=bp[1], Ecorr=bp[0],
                ba=bp[2], bc=bp[3], CR=CR, B=B, elapsed=elapsed)


# ════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TAFEL-PRO: ML-Augmented Global Polarization Curve Fitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python tafel_pro.py .
          python tafel_pro.py C:\\data --area 0.5 --material "304 SS"
          python tafel_pro.py ./experiments --fit-rs --pattern "**/*.csv"
        """))
    parser.add_argument("root", help="Root folder to search")
    parser.add_argument("--area", type=float, default=1.0, help="Electrode area (cm²)")
    parser.add_argument("--material", default="Carbon Steel / Iron",
                        choices=list(MATERIALS.keys()), help="Material for CR calc")
    parser.add_argument("--fit-rs", action="store_true", help="Enable Rs fitting")
    parser.add_argument("--rs-max", type=float, default=200.0, help="Rs upper bound (Ω·cm²)")
    parser.add_argument("--pattern", default="**/lsv.xlsx",
                        help="Glob pattern (default: **/lsv.xlsx)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ML models")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    ew, rho = MATERIALS[args.material]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  TAFEL-PRO v2.0 — ML-Augmented Polarization Curve Fitter   ║
╠══════════════════════════════════════════════════════════════╣
║  Root:     {root:<48s}║
║  Pattern:  {args.pattern:<48s}║
║  Area:     {args.area:<48.2f}║
║  Material: {args.material:<48s}║
║  Fit Rs:   {str(args.fit_rs):<48s}║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Build ML models ──
    print("  Building ML classifier (self-training on synthetic data)...")
    t_ml = time.time()
    clf, clf_scaler = build_classifier(n_samples=500, seed=args.seed)
    print(f"  ✓ Classifier ready ({time.time()-t_ml:.1f}s)")

    print("  Building neural surrogate (self-training)...")
    t_surr = time.time()
    surrogate, surr_scaler, surr_lo, surr_hi = build_surrogate(
        n_samples=400, seed=args.seed)
    print(f"  ✓ Surrogate ready ({time.time()-t_surr:.1f}s)")

    # ── Find files ──
    files = sorted(glob.glob(os.path.join(root, args.pattern), recursive=True))
    if not files:
        print(f"\n  No files matching '{args.pattern}' in {root}")
        sys.exit(1)
    print(f"\n  Found {len(files)} file(s)")

    # ── Process ──
    all_results = []
    ok, fail = 0, 0

    for fpath in files:
        try:
            result = process_file(fpath, args.area, ew, rho, args.fit_rs,
                                  args.rs_max, clf, clf_scaler,
                                  surrogate, surr_scaler, surr_lo, surr_hi)
            if result:
                all_results.append(result)
                ok += 1
            else:
                fail += 1
        except Exception as ex:
            print(f"  [FATAL] {fpath}: {ex}")
            fail += 1

    # ── Summary ──
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  BATCH COMPLETE                                             ║
╠══════════════════════════════════════════════════════════════╣
║  Files: {len(files):<52d}║
║  ✓ OK:  {ok:<52d}║
║  ✗ Fail:{fail:<52d}║
╚══════════════════════════════════════════════════════════════╝
""")

    if all_results:
        r2s = np.array([r["r2"] for r in all_results])
        print(f"  R² Distribution:")
        print(f"  ────────────────────────────────────────")
        print(f"  ★★★★★ Exceptional (≥0.995): {np.sum(r2s >= 0.995):>3d} ({100*np.mean(r2s >= 0.995):.0f}%)")
        print(f"  ★★★★☆ Excellent   (≥0.99):  {np.sum(r2s >= 0.99):>3d} ({100*np.mean(r2s >= 0.99):.0f}%)")
        print(f"  ★★★☆☆ Very Good   (≥0.98):  {np.sum(r2s >= 0.98):>3d} ({100*np.mean(r2s >= 0.98):.0f}%)")
        print(f"  ★★☆☆☆ Good        (≥0.95):  {np.sum(r2s >= 0.95):>3d} ({100*np.mean(r2s >= 0.95):.0f}%)")
        print(f"  ★☆☆☆☆ Below       (<0.95):  {np.sum(r2s < 0.95):>3d} ({100*np.mean(r2s < 0.95):.0f}%)")
        print(f"  ────────────────────────────────────────")
        print(f"  Mean:   {np.mean(r2s):.6f}")
        print(f"  Median: {np.median(r2s):.6f}")
        print(f"  Min:    {np.min(r2s):.6f}")
        print(f"  Max:    {np.max(r2s):.6f}")

        # Curve type distribution
        ct_counts = {}
        for r in all_results:
            ct_counts[r["ct"]] = ct_counts.get(r["ct"], 0) + 1
        print(f"\n  Curve Type Distribution:")
        for ct, count in sorted(ct_counts.items(), key=lambda x: -x[1]):
            print(f"    {CT.name(ct):<30s} {count:>3d} ({100*count/len(all_results):.0f}%)")

        # Export batch summary
        summary_path = os.path.join(root, "tafel_batch_summary.csv")
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  ✓ Batch summary → {summary_path}")

        # Flag problematic files
        poor = [r for r in all_results if r["r2"] < 0.95]
        if poor:
            print(f"\n  ⚠ Files needing review (R² < 0.95):")
            for r in poor[:15]:
                print(f"    • {os.path.basename(r['file'])} — R²={r['r2']:.4f}, "
                      f"CT={CT.name(r['ct'])}")

    print()


if __name__ == "__main__":
    main()
