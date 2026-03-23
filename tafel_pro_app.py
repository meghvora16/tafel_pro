"""
TAFEL-PRO v2.1 — Separate Anodic/Cathodic Fitting + Global Polish
Streamlit App — Run: streamlit run tafel_pro_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize, curve_fit
from scipy.signal import savgol_filter
from scipy.stats import linregress
from itertools import groupby
import warnings, re, io, time

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  MATH
# ═══════════════════════════════════════════════════════════════
def slog(x):   return np.log10(np.maximum(np.abs(x), 1e-30))
def sig(x, k=1.0): return 1.0 / (1.0 + np.exp(-np.clip(k * x, -50, 50)))

def sm(y, w=11, p=3):
    n = len(y); w = min(w, n if n % 2 == 1 else n - 1)
    return savgol_filter(y, w, min(p, w - 1)) if w >= 5 else y.copy()

def r2sc(yt, yp):
    sr = np.sum((yt - yp)**2); st = np.sum((yt - yt.mean())**2)
    return float(max(0, 1 - sr / st)) if st > 0 else 0.0

def aicc(n, k, sse):
    if n <= k + 1 or sse <= 0: return 1e30
    return n * np.log(sse / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)


# ═══════════════════════════════════════════════════════════════
#  CURVE TYPES
# ═══════════════════════════════════════════════════════════════
PARAM_NAMES = ["Ecorr","icorr","ba","bc1","iL","i0_c2","bc2",
               "Epp","k_pass","ipass","Eb","a_tp","b_tp",
               "Esp","k_sp","ipass2","Rs"]
NP = 17
LOG_P = {1, 4, 5, 9, 11, 15}

class CT:
    A="A"; AD="AD"; AH="AH"; P="P"; PD="PD"; PT="PT"; F="F"
    INFO = {
        "A":  ("Active",              [0,1,2,3],                        4),
        "AD": ("Active+Diffusion",    [0,1,2,3,4],                     5),
        "AH": ("Active+DualCathodic", [0,1,2,3,4,5,6],                 7),
        "P":  ("Active–Passive",      [0,1,2,3,7,8,9],                 7),
        "PD": ("Passive+Diffusion",   [0,1,2,3,4,5,6,7,8,9],          10),
        "PT": ("Transpassive",        [0,1,2,3,4,5,6,7,8,9,10,11,12], 13),
        "F":  ("Full",                list(range(16)),                  16),
    }
    ALL = ["A","AD","AH","P","PD","PT","F"]
    SIMPLE = ["A","AD","AH"]
    PASS = ["P","PD","PT","F"]
    @staticmethod
    def idx(ct):   return CT.INFO.get(ct,("",list(range(16)),16))[1]
    @staticmethod
    def nfree(ct): return CT.INFO.get(ct,("",[], 0))[2]
    @staticmethod
    def name(ct):  return CT.INFO.get(ct,("?",[], 0))[0]


# ═══════════════════════════════════════════════════════════════
#  PHYSICS MODEL
# ═══════════════════════════════════════════════════════════════
def pol_model(E, p):
    """Full 17-parameter polarization model."""
    E = np.asarray(E, float)
    Ec,ic,ba,bc1,iL = p[0],p[1],p[2],p[3],p[4]
    i0c2,bc2 = p[5],p[6]
    Epp,kp,ip = p[7],p[8],p[9]
    Eb,atp,btp = p[10],p[11],p[12]
    Esp,ksp,ip2 = p[13],p[14],p[15]
    Rs = max(p[16], 0.0)

    Ee = E.copy()
    for _ in range(6 if Rs > 0 else 1):
        eta = Ee - Ec
        # Cathodic: O₂ with diffusion limit
        ik = ic * np.exp(np.clip(-2.303*eta/max(bc1,1e-12), -50, 50))
        ic1 = ik / (1 + ik/max(iL,1e-20))
        # Cathodic: H₂ evolution
        ic2 = i0c2 * np.exp(np.clip(-2.303*eta/max(bc2,1e-12), -50, 50))
        # Anodic: active
        ia = ic * np.exp(np.clip(2.303*eta/max(ba,1e-12), -50, 50))
        # Passive transition
        t1 = sig(Ee - Epp, kp)
        ip1 = ia*(1-t1) + ip*t1
        # Transpassive
        itp = atp * np.exp(np.clip(btp*(Ee-Eb), -50, 50)) * sig(Ee-Eb, 40)
        # Secondary passivity
        t2 = sig(Ee - Esp, ksp)
        ian = (ip1 + itp)*(1-t2) + ip2*t2
        inet = ian - (ic1 + ic2)
        if Rs > 0: Ee = E - inet*Rs
    return inet


def pol_components(E, p):
    """Return dict of individual current components."""
    E = np.asarray(E, float)
    Ec,ic,ba,bc1,iL = p[0:5]; i0c2,bc2 = p[5:7]
    Epp,kp,ip = p[7:10]; Eb,atp,btp = p[10:13]
    Esp,ksp,ip2 = p[13:16]; Rs = max(p[16],0)
    Ee = E.copy()
    for _ in range(6 if Rs > 0 else 1):
        eta = Ee - Ec
        ik = ic*np.exp(np.clip(-2.303*eta/max(bc1,1e-12),-50,50))
        ic1 = ik/(1+ik/max(iL,1e-20))
        ic2 = i0c2*np.exp(np.clip(-2.303*eta/max(bc2,1e-12),-50,50))
        ia = ic*np.exp(np.clip(2.303*eta/max(ba,1e-12),-50,50))
        t1 = sig(Ee-Epp,kp); ip1 = ia*(1-t1)+ip*t1
        itp = atp*np.exp(np.clip(btp*(Ee-Eb),-50,50))*sig(Ee-Eb,40)
        t2 = sig(Ee-Esp,ksp); ian = (ip1+itp)*(1-t2)+ip2*t2
        inet = ian-(ic1+ic2)
        if Rs > 0: Ee = E - inet*Rs
    return dict(ic1=ic1, ic2=ic2, iact=ia, ip1=ip1, itp=itp, ian=ian, itot=inet)


# ═══════════════════════════════════════════════════════════════
#  STAGE 1: Ecorr DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_ecorr(E, i):
    n = len(E)
    if n == 0: return 0.0, 0
    si = np.argsort(E); Es, Is = E[si], i[si]
    w = min(max(5, (n//5)|1), 11)
    if w % 2 == 0: w -= 1
    w = max(5, w)
    try: ism = savgol_filter(Is, w, min(3,w-1)) if n >= w else Is.copy()
    except: ism = Is.copy()

    def cx(ia):
        c = []
        for k in range(len(ia)-1):
            if np.sign(ia[k])*np.sign(ia[k+1]) < 0:
                d = ia[k+1]-ia[k]
                if abs(d) < 1e-30: continue
                Ec = Es[k] - ia[k]*(Es[k+1]-Es[k])/d
                c.append((Ec, int(si[k]), ia[k]<0 and ia[k+1]>0))
        return c

    raw = cx(Is); smo = cx(ism)
    seen = set(round(c[0],4) for c in raw)
    all_c = raw + [c for c in smo if round(c[0],4) not in seen]
    if not all_c:
        k = int(np.argmin(np.abs(Is))); return float(Es[k]), int(si[k])
    anodic = [c for c in all_c if c[2]]
    best = min(anodic, key=lambda c: c[0]) if anodic else min(all_c, key=lambda c: c[0])
    return best[0], best[1]


# ═══════════════════════════════════════════════════════════════
#  STAGE 2: SEPARATE CATHODIC FITTING
# ═══════════════════════════════════════════════════════════════
def fit_cathodic_branch(E, i, Ecorr):
    """
    Fit cathodic branch separately.
    Returns dict: bc1, icorr, iL, has_diffusion, has_H2, bc2, i0_c2, r2
    """
    cat = (E < Ecorr - 0.015)
    if np.sum(cat) < 5:
        return dict(bc1=0.120, icorr=1e-6, iL=None, has_diff=False,
                    has_H2=False, bc2=0.15, i0_c2=1e-30, r2=0)

    Ec = E[cat]; ic = np.abs(i[cat]); lg = slog(ic)
    si = np.argsort(Ec); Ec,ic,lg = Ec[si],ic[si],lg[si]

    # ── Find Tafel region by sliding window ──
    best_r2, best_s, best_b = 0.0, -8.0, -6.0
    best_win = (0, len(Ec))
    mp = max(5, len(Ec)//5)
    for s0 in range(max(1, len(Ec)-mp)):
        for s1 in range(s0+mp, min(s0+60, len(Ec))):
            try:
                sl,b,r,*_ = linregress(Ec[s0:s1], lg[s0:s1])
                if sl < -1.0 and r**2 > best_r2 and 0.020 < abs(1/sl) < 0.600:
                    best_r2, best_s, best_b = r**2, sl, b
                    best_win = (s0, s1)
            except: continue

    bc1 = abs(1/best_s) if abs(best_s) > 0.1 else 0.120
    icorr = 10**(best_b + best_s * Ecorr)
    icorr = max(icorr, 1e-15)

    # ── Detect diffusion plateau ──
    iL = None; has_diff = False
    if len(Ec) > 12:
        lg_sm = sm(lg, min(11, (len(Ec)//2)*2-1 or 5))
        dlg = np.abs(np.gradient(lg_sm, Ec))
        flat = dlg < max(np.percentile(dlg, 30), 0.7)
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        if runs:
            br = max(runs, key=lambda x: len(x[1]))
            idxs = [s[0] for s in br[1]]
            rng = abs(Ec[idxs[-1]] - Ec[idxs[0]])
            if len(idxs) >= 3 and rng > 0.02:
                iL = float(np.median(ic[idxs]))
                has_diff = True

    # ── Detect H₂ evolution ──
    has_H2 = False; bc2 = 0.150; i0_c2 = 1e-30
    if has_diff and len(Ec) > 20:
        # Look for slope steepening beyond the plateau
        far = Ec < Ec[best_win[0]] - 0.03
        if np.sum(far) > 5:
            grad = np.gradient(lg[far], Ec[far])
            if np.mean(grad[:min(8, len(grad))]) < -1.5:
                has_H2 = True
                try:
                    n_pts = min(15, np.sum(far))
                    sl2,b2,r2,*_ = linregress(Ec[far][:n_pts], lg[far][:n_pts])
                    if sl2 < 0 and abs(1/sl2) > 0.04:
                        bc2 = abs(1/sl2)
                        i0_c2 = 10**(b2 + sl2*Ecorr)
                except: pass

    # ── Compute cathodic fit R² ──
    def cat_model(E_cat, bc_, ic_, iL_):
        eta = E_cat - Ecorr
        ik = ic_ * np.exp(np.clip(-2.303*eta/max(bc_,1e-12), -50, 50))
        return ik / (1 + ik/max(iL_, 1e-20))

    iL_fit = iL if iL else icorr * 1e4
    try:
        pred = cat_model(Ec, bc1, icorr, iL_fit)
        cat_r2 = r2sc(lg, slog(pred))
    except:
        cat_r2 = 0.0

    return dict(bc1=bc1, icorr=icorr, iL=iL, iL_fit=iL_fit,
                has_diff=has_diff, has_H2=has_H2, bc2=bc2, i0_c2=i0_c2,
                r2=cat_r2, E_cat=Ec, lg_cat=lg)


# ═══════════════════════════════════════════════════════════════
#  STAGE 3: SEPARATE ANODIC FITTING
# ═══════════════════════════════════════════════════════════════
def fit_anodic_branch(E, i, Ecorr, icorr_cat):
    """
    Fit anodic branch separately.
    Returns dict: ba, has_passive, Epp, ipass, has_tp, Eb, r2
    """
    an = (E > Ecorr + 0.015)
    if np.sum(an) < 5:
        return dict(ba=0.060, has_passive=False, Epp=None, ipass=None,
                    has_tp=False, Eb=None, r2=0)

    Ea = E[an]; ia = np.abs(i[an]); lg = slog(ia)
    si = np.argsort(Ea); Ea,ia,lg = Ea[si],ia[si],lg[si]

    # ── Find anodic Tafel region ──
    best_r2, best_s = 0.0, 8.0
    mp = max(5, len(Ea)//5)
    for s0 in range(max(1, len(Ea)-mp)):
        for s1 in range(s0+mp, min(s0+60, len(Ea))):
            try:
                sl,b,r,*_ = linregress(Ea[s0:s1], lg[s0:s1])
                if sl > 1.0 and r**2 > best_r2 and 0.020 < abs(1/sl) < 0.600:
                    best_r2, best_s = r**2, sl
            except: continue

    ba = abs(1/best_s) if abs(best_s) > 0.1 else 0.060

    # ── Detect passive region ──
    has_passive = False; Epp = None; ipass = None
    has_tp = False; Eb = None

    if len(Ea) > 12:
        lg_sm = sm(lg, min(15, (len(Ea)//2)*2-1 or 5))
        adl = np.abs(np.gradient(lg_sm, Ea))

        # Look for flat region (passive plateau)
        thr = max(np.percentile(adl, 25), 0.8)
        flat = adl < thr
        runs = [(k, list(g)) for k, g in groupby(enumerate(flat), key=lambda x: x[1]) if k]
        for _, ri in runs:
            idxs = [s[0] for s in ri]
            rng = abs(Ea[idxs[-1]] - Ea[idxs[0]])
            if len(idxs) >= 5 and rng > 0.08:
                im = float(np.median(ia[idxs]))
                pre = Ea < Ea[idxs[0]]
                if np.sum(pre) > 2 and im < np.max(ia[pre]) * 0.4:
                    has_passive = True
                    Epp = float(Ea[idxs[0]])
                    ipass = im
                    # Transpassive rise after passive
                    post_idx = Ea > Ea[idxs[-1]]
                    if np.sum(post_idx) > 5:
                        dlg_post = np.gradient(lg[np.where(post_idx)[0]], Ea[post_idx])
                        if np.max(dlg_post) > 3.0:
                            has_tp = True
                            Eb = float(Ea[post_idx][np.argmax(dlg_post > 3.0)])
                    break

    # ── Anodic Tafel R² ──
    # Fit just the active dissolution part
    near = (Ea < (Epp if Epp else Ea[-1]+1)) & (Ea > Ecorr + 0.02)
    if np.sum(near) > 3:
        try:
            sl,b,r,*_ = linregress(Ea[near], lg[near])
            an_r2 = r**2
        except: an_r2 = 0.0
    else:
        an_r2 = 0.0

    return dict(ba=ba, has_passive=has_passive, Epp=Epp, ipass=ipass,
                has_tp=has_tp, Eb=Eb, r2=an_r2, E_an=Ea, lg_an=lg)


# ═══════════════════════════════════════════════════════════════
#  STAGE 4: ASSEMBLE + GLOBAL POLISH
# ═══════════════════════════════════════════════════════════════
def classify_curve(cat_res, an_res):
    """Determine curve type from separate fits."""
    hd = cat_res["has_diff"]
    hh = cat_res["has_H2"]
    hp = an_res["has_passive"]
    ht = an_res["has_tp"]

    if hp and ht:  return CT.PT if hd else CT.PT
    if hp:         return CT.PD if hd else CT.P
    if hd and hh:  return CT.AH
    if hd:         return CT.AD
    return CT.A


def assemble_p0(Ecorr, cat_res, an_res, ct, Emax):
    """Build initial parameter vector from separate fits."""
    ic = cat_res["icorr"]
    bc1 = cat_res["bc1"]
    iL = cat_res.get("iL_fit", ic*1e4)
    ba = an_res["ba"]

    return np.array([
        Ecorr, ic, ba, bc1, iL,
        cat_res["i0_c2"], cat_res["bc2"],
        an_res.get("Epp") or Emax+10,
        40.0 if an_res["has_passive"] else 50.0,
        an_res.get("ipass") or ic,
        an_res.get("Eb") or Emax+10,
        (an_res.get("ipass") or ic)*0.01 if an_res["has_tp"] else 1e-30,
        8.0,
        Emax+20, 50.0, ic, 0.0
    ])


def global_polish(E, i, p0, ct, fit_rs=False, rs_max=200.0):
    """
    Light global optimization using the separate-fit initial guess.
    Much faster than full DE — just refines the combined model.
    """
    ld = slog(i)
    n = len(E)
    fidx = CT.idx(ct) + ([16] if fit_rs else [])
    nf = len(fidx)

    # Bounds
    Emax, Emin = float(np.max(E)), float(np.min(E))
    ic = max(p0[1], 1e-14)
    lo = np.array([
        Emin, max(ic*1e-4,1e-15), 0.01, 0.01, max(ic*0.1,1e-10),
        max(ic*1e-8,1e-18), 0.04,
        Emin if ct in CT.PASS else Emax+5, 5.0,
        max(ic*1e-5,1e-15) if ct in CT.PASS else 1e-15,
        Emin if ct in ["PT","F"] else Emax+5, 1e-30, 0.5,
        Emin if ct == "F" else Emax+10, 5.0, max(ic*1e-5,1e-15), 0.0
    ])
    hi = np.array([
        Emax, min(ic*1e5,1e1), 0.5, 0.5, max(ic*1e6,1e0),
        max(ic*100,1e-4), 0.4,
        Emax if ct in CT.PASS else Emax+15, 200.0,
        max(ic*1e3,1e-2) if ct in CT.PASS else 1e-2,
        Emax+0.2 if ct in ["PT","F"] else Emax+15, max(ic*1e4,1e-6), 40.0,
        Emax+0.5 if ct=="F" else Emax+25, 200.0, max(ic*1e3,1e-2), rs_max
    ])
    for j in range(NP):
        if lo[j] >= hi[j]:
            m = p0[j]; lo[j]=m-abs(m)*0.5-1e-6; hi[j]=m+abs(m)*0.5+1e-6

    def pack(pf):
        return np.array([np.log10(max(pf[j],1e-30)) if j in LOG_P else pf[j] for j in fidx])
    def unpack(x):
        p = p0.copy()
        for k,j in enumerate(fidx): p[j] = 10**x[k] if j in LOG_P else x[k]
        return p
    def pbounds():
        return [(np.log10(max(lo[j],1e-30)), np.log10(max(hi[j],1e-30)))
                if j in LOG_P else (lo[j],hi[j]) for j in fidx]
    def obj(x):
        p = unpack(x)
        if p[1] < 1e-14 or p[1] > 0.1: return 1e30
        try: return float(np.sum((ld - slog(pol_model(E,p)))**2))
        except: return 1e30

    best_p, best_s = p0.copy(), obj(pack(p0))
    bnds = pbounds()
    def up(x):
        nonlocal best_p, best_s
        s = obj(x)
        if s < best_s: best_p, best_s = unpack(x), s

    # DE (moderate — not exhaustive)
    try:
        ps = max(10, min(15, nf*2)); mi = max(150, min(400, nf*30))
        res = differential_evolution(obj, bnds, seed=42, maxiter=mi, popsize=ps,
            tol=1e-12, mutation=(0.5,1.7), recombination=0.85, polish=False, workers=1)
        up(res.x)
    except: pass

    # L-BFGS-B
    try:
        r = minimize(obj, pack(best_p), method="L-BFGS-B", bounds=bnds,
            options={"maxiter": 15000, "ftol": 1e-15})
        up(r.x)
    except: pass

    # Nelder-Mead
    try:
        r = minimize(obj, pack(best_p), method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-13, "fatol": 1e-15, "adaptive": True})
        up(r.x)
    except: pass

    pred = pol_model(E, best_p)
    r2 = r2sc(ld, slog(pred))
    sse = np.sum((ld - slog(pred))**2)
    aic = aicc(n, nf, sse)
    return best_p, r2, aic


# ═══════════════════════════════════════════════════════════════
#  FILE I/O
# ═══════════════════════════════════════════════════════════════
COL_SIG = [
    (r"we.*potential", r"we.*current", "A"), (r"ewe", r"i/ma", "mA"),
    (r"ewe", r"<i>/ma", "mA"), (r"^vf$", r"^im$", "A"),
    (r"potential/v", r"current/a", "A"), (r"e/v", r"i/a", "A"),
    (r"potential|volt|^e$|e \(v\)|e_v", r"current|amps|^i$|i \(a\)|i_a", "A"),
    (r"potential|volt|^e$", r"current.*ma|ima", "mA"),
]
UHINT = {r"\(a\)|_a$|/a$": 1.0, r"\(ma\)|_ma$|/ma$": 1e-3,
         r"\(ua\)|_ua$|/ua$": 1e-6, r"a/cm": 1.0, r"ma/cm": 1e-3}

def auto_cols(df):
    cl = {c: c.lower().strip() for c in df.columns}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for Ep, Ip, u in COL_SIG:
        em = [c for c,v in cl.items() if re.search(Ep,v) and c in num]
        im = [c for c,v in cl.items() if re.search(Ip,v) and c in num and c not in em]
        if em and im:
            ec = sorted(em, key=lambda c: 0 if "we" in c.lower() else 1)[0]
            ic = im[0]; f = 1e-3 if u == "mA" else 1.0
            for p, fv in UHINT.items():
                if re.search(p, cl[ic]): f = fv; break
            return ec, ic, f
    if len(num) >= 2: return num[0], num[1], 1.0
    raise ValueError("Cannot detect columns.")

def load_data(uploaded):
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(uploaded)
    content = uploaded.getvalue().decode("utf-8", errors="replace")
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 5:
                return df.dropna(axis=1, how="all")
        except: continue
    raise ValueError("Cannot parse file.")


MATERIALS = {
    "Carbon Steel / Iron": (27.92, 7.87), "304 Stainless Steel": (25.10, 7.90),
    "316 Stainless Steel": (25.56, 8.00), "Copper": (31.77, 8.96),
    "Aluminum": (8.99, 2.70), "Nickel": (29.36, 8.91),
    "Titanium": (11.99, 4.51), "Zinc": (32.69, 7.14),
}


# ═══════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════
C = dict(bg="#0d1117", pan="#161b22", dat="#58a6ff", fit="#3fb950",
         ec="#f85149", grid="#21262d", txt="#c9d1d9", sp="#30363d",
         ic1="#bc8cff", ic2="#f778ba", act="#d29922", pas="#7ee787",
         tp="#ffa657", acc="#a371f7")

def plot_main(E, i, bp, Ecorr, r2v, ct):
    Em = np.linspace(E[0], E[-1], 800)
    lg = slog(i)
    fig = make_subplots(1, 2, subplot_titles=(
        "Measured vs Fit", "Component Decomposition"), horizontal_spacing=0.12)

    fig.add_trace(go.Scattergl(x=E, y=lg, mode='markers', name='Data',
        marker=dict(color=C["dat"], size=3, opacity=0.5)), 1, 1)
    try:
        im = pol_model(Em, bp)
        fig.add_trace(go.Scattergl(x=Em, y=slog(im), mode='lines',
            name=f'Fit R²={r2v:.4f}', line=dict(color=C["fit"], width=2.5)), 1, 1)
    except: pass
    fig.add_vline(x=Ecorr, line=dict(color=C["ec"], dash="dot", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=[bp[0]], y=[slog(np.array([bp[1]]))], mode='markers',
        name=f'icorr={bp[1]:.2e}', marker=dict(color=C["ec"],size=12,symbol='x',
        line=dict(width=3))), 1, 1)

    c = pol_components(Em, bp)
    fig.add_trace(go.Scatter(x=E, y=lg, mode='markers', marker=dict(
        color=C["dat"],size=2,opacity=0.3), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=Em, y=slog(c["ic1"]), mode='lines', name='O₂ cath.',
        line=dict(color=C["ic1"],width=1.5,dash='dot')), 1, 2)
    fig.add_trace(go.Scatter(x=Em, y=slog(c["ic2"]), mode='lines', name='H₂ cath.',
        line=dict(color=C["ic2"],width=1.5,dash='dot')), 1, 2)
    fig.add_trace(go.Scatter(x=Em, y=slog(c["iact"]), mode='lines', name='Active',
        line=dict(color=C["act"],width=1,dash='dash')), 1, 2)
    if ct not in CT.SIMPLE:
        fig.add_trace(go.Scatter(x=Em, y=slog(c["ip1"]), mode='lines', name='Passive',
            line=dict(color=C["pas"],width=1.5,dash='dot')), 1, 2)
        fig.add_trace(go.Scatter(x=Em, y=slog(c["itp"]), mode='lines', name='Transp.',
            line=dict(color=C["tp"],width=1.5,dash='dot')), 1, 2)
    fig.add_trace(go.Scatter(x=Em, y=slog(c["itot"]), mode='lines', name='Net',
        line=dict(color=C["fit"],width=2.5)), 1, 2)

    fig.update_layout(height=480, template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["pan"],
        font=dict(color=C["txt"]), legend=dict(font=dict(size=9),
        bgcolor="rgba(0,0,0,0.3)"), margin=dict(t=45,b=45,l=55,r=25))
    fig.update_xaxes(title_text="E (V vs Ref)", gridcolor=C["grid"])
    fig.update_yaxes(title_text="log₁₀|i| (A/cm²)", gridcolor=C["grid"])
    return fig


def plot_separate(E, i, Ecorr, cat_res, an_res):
    """Plot showing the separate cathodic and anodic fits."""
    fig = make_subplots(1, 2, subplot_titles=(
        "Cathodic Branch Fit", "Anodic Branch Fit"), horizontal_spacing=0.12)

    lg = slog(i)
    # Full data (faded)
    fig.add_trace(go.Scattergl(x=E, y=lg, mode='markers', name='Full data',
        marker=dict(color=C["dat"], size=2, opacity=0.15), showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=E, y=lg, mode='markers',
        marker=dict(color=C["dat"], size=2, opacity=0.15), showlegend=False), 1, 2)

    # Cathodic
    if "E_cat" in cat_res:
        Ec, lgc = cat_res["E_cat"], cat_res["lg_cat"]
        fig.add_trace(go.Scattergl(x=Ec, y=lgc, mode='markers', name='Cathodic data',
            marker=dict(color=C["ic1"], size=4, opacity=0.7)), 1, 1)
        # Tafel line
        E_line = np.linspace(Ecorr-0.5, Ecorr-0.01, 100)
        lg_line = slog(np.array([cat_res["icorr"]])) + (E_line - Ecorr) * (-2.303/cat_res["bc1"])
        fig.add_trace(go.Scatter(x=E_line, y=lg_line, mode='lines',
            name=f'bc={cat_res["bc1"]:.3f} V/dec',
            line=dict(color=C["ic1"], width=2, dash='dash')), 1, 1)
        if cat_res["has_diff"] and cat_res["iL"]:
            fig.add_hline(y=float(slog(np.array([cat_res["iL"]]))[0]),
                line=dict(color=C["ic2"], dash="dot", width=1),
                annotation_text=f"iL={cat_res['iL']:.2e}", row=1, col=1)

    # Anodic
    if "E_an" in an_res:
        Ea, lga = an_res["E_an"], an_res["lg_an"]
        fig.add_trace(go.Scattergl(x=Ea, y=lga, mode='markers', name='Anodic data',
            marker=dict(color=C["act"], size=4, opacity=0.7)), 1, 2)
        E_line = np.linspace(Ecorr+0.01, Ecorr+0.5, 100)
        lg_line = slog(np.array([cat_res["icorr"]])) + (E_line - Ecorr) * (2.303/an_res["ba"])
        fig.add_trace(go.Scatter(x=E_line, y=lg_line, mode='lines',
            name=f'ba={an_res["ba"]:.3f} V/dec',
            line=dict(color=C["act"], width=2, dash='dash')), 1, 2)
        if an_res["has_passive"] and an_res["Epp"]:
            fig.add_vline(x=an_res["Epp"], line=dict(color=C["pas"], dash="dot", width=1),
                annotation_text=f"Epp={an_res['Epp']:.3f}V", row=1, col=2)

    fig.add_vline(x=Ecorr, line=dict(color=C["ec"], dash="dot", width=1), row=1, col=1)
    fig.add_vline(x=Ecorr, line=dict(color=C["ec"], dash="dot", width=1), row=1, col=2)

    fig.update_layout(height=400, template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["pan"],
        font=dict(color=C["txt"]), legend=dict(font=dict(size=9),
        bgcolor="rgba(0,0,0,0.3)"), margin=dict(t=45,b=45,l=55,r=25))
    fig.update_xaxes(title_text="E (V vs Ref)", gridcolor=C["grid"])
    fig.update_yaxes(title_text="log₁₀|i| (A/cm²)", gridcolor=C["grid"])
    return fig


def plot_residuals(E, i, bp):
    res = slog(i) - slog(pol_model(E, bp))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=res, mode='markers',
        marker=dict(color=C["dat"], size=3, opacity=0.6)))
    fig.add_hline(y=0, line=dict(color=C["ec"], dash="dash", width=1))
    fig.update_layout(height=250, template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["pan"],
        font=dict(color=C["txt"]), xaxis_title="E (V)", yaxis_title="Residual",
        margin=dict(t=15,b=45,l=55,r=25))
    fig.update_xaxes(gridcolor=C["grid"]); fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="TAFEL-PRO v2.1", page_icon="⚡", layout="wide")

    st.markdown("""<style>
    .stApp{background:#0d1117}
    h1{color:#a371f7 !important} h2,h3{color:#58a6ff !important}
    .stMetric label{color:#8b949e !important}
    .stMetric [data-testid="stMetricValue"]{color:#c9d1d9 !important}
    div[data-testid="stSidebar"]{background:#161b22}
    </style>""", unsafe_allow_html=True)

    st.title("⚡ TAFEL-PRO v2.1")
    st.caption("Separate Anodic/Cathodic Fitting → Global Polish → AICc Selection")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded = st.file_uploader("Upload LSV data",
            type=["xlsx","xls","csv","txt","tsv"])
        area = st.number_input("Electrode area (cm²)", 0.001, 1000.0, 1.0, 0.01, format="%.4f")
        material = st.selectbox("Material", list(MATERIALS.keys()))
        fit_rs = st.checkbox("Fit Rs", False)
        rs_max = st.number_input("Rs max (Ω·cm²)", 1.0, 10000.0, 200.0, disabled=not fit_rs)
        st.markdown("---")
        st.caption("v2.1 — Separate branch fitting\n"
                   "Cathodic → Anodic → Classify → Assemble → Global Polish")

    if not uploaded:
        st.info("👈 Upload a polarization data file to begin.")
        with st.expander("ℹ️ Pipeline", expanded=True):
            st.markdown("""
            **Stage 1** — Detect E_corr (zero-crossing)
            **Stage 2** — Fit cathodic branch: bc₁, i_corr, i_L, detect H₂
            **Stage 3** — Fit anodic branch: ba, detect passive/transpassive
            **Stage 4** — Classify curve type from branch results
            **Stage 5** — Assemble initial guess → Global polish (DE + L-BFGS-B + NM)
            **Stage 6** — AICc model selection if multiple candidates tried
            """)
        return

    # ── Load ──
    try: df = load_data(uploaded)
    except Exception as ex: st.error(f"Load error: {ex}"); return

    try: ec_auto, ic_auto, fac_auto = auto_cols(df); auto_ok = True
    except: ec_auto,ic_auto,fac_auto = None,None,1.0; auto_ok = False

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    with st.expander("📊 Column Selection", expanded=not auto_ok):
        c1,c2,c3 = st.columns(3)
        with c1: ec = st.selectbox("Potential", num_cols,
            index=num_cols.index(ec_auto) if auto_ok and ec_auto in num_cols else 0)
        with c2: ic = st.selectbox("Current", num_cols,
            index=num_cols.index(ic_auto) if auto_ok and ic_auto in num_cols else min(1,len(num_cols)-1))
        with c3: fac = st.selectbox("Unit", [("A",1.0),("mA",1e-3),("µA",1e-6)],
            format_func=lambda x: x[0],
            index=0 if fac_auto==1.0 else 1 if fac_auto==1e-3 else 2)[1]

    E_raw = df[ec].values.astype(float)
    i_raw = df[ic].values.astype(float) * fac
    ok = np.isfinite(E_raw) & np.isfinite(i_raw)
    E_raw, i_raw = E_raw[ok], i_raw[ok]
    i_d = i_raw / area
    si = np.argsort(E_raw); E, i = E_raw[si], i_d[si]

    st.markdown(f"**{len(E)} pts** | E: [{E.min():.3f}, {E.max():.3f}] V | "
                f"|i|: [{np.min(np.abs(i)):.2e}, {np.max(np.abs(i)):.2e}] A/cm²")

    # ═════════════════════════════════════════════════════════
    # RUN PIPELINE
    # ═════════════════════════════════════════════════════════
    run = st.button("🚀 Run Fitting", type="primary", use_container_width=True)
    if not run: return

    t0 = time.time()

    # Stage 1: Ecorr
    with st.status("⚡ Stage 1: Detecting E_corr...", expanded=True) as s1:
        Ecorr, _ = detect_ecorr(E, i)
        st.write(f"**E_corr = {Ecorr:.4f} V**")
        s1.update(label=f"✅ E_corr = {Ecorr:.4f} V", state="complete")

    # Stage 2: Cathodic
    with st.status("🔵 Stage 2: Fitting cathodic branch...", expanded=True) as s2:
        cat = fit_cathodic_branch(E, i, Ecorr)
        st.write(f"**bc₁ = {cat['bc1']:.4f} V/dec** | **i_corr = {cat['icorr']:.2e} A/cm²**")
        st.write(f"Diffusion limit: {'**iL = %.2e**' % cat['iL'] if cat['has_diff'] else 'Not detected'} "
                 f"| H₂ evolution: {'**Yes** (bc₂=%.3f)' % cat['bc2'] if cat['has_H2'] else 'No'}")
        st.write(f"Cathodic Tafel R² = {cat['r2']:.4f}")
        s2.update(label=f"✅ Cathodic: bc={cat['bc1']:.3f}, icorr={cat['icorr']:.2e}", state="complete")

    # Stage 3: Anodic
    with st.status("🟠 Stage 3: Fitting anodic branch...", expanded=True) as s3:
        an = fit_anodic_branch(E, i, Ecorr, cat["icorr"])
        st.write(f"**ba = {an['ba']:.4f} V/dec**")
        st.write(f"Passive: {'**Yes** (Epp=%.3fV, ipass=%.2e)' % (an['Epp'], an['ipass']) if an['has_passive'] else 'No'}")
        st.write(f"Transpassive: {'**Yes** (Eb=%.3fV)' % an['Eb'] if an['has_tp'] else 'No'}")
        s3.update(label=f"✅ Anodic: ba={an['ba']:.3f}, passive={an['has_passive']}", state="complete")

    # Separate fits plot
    st.plotly_chart(plot_separate(E, i, Ecorr, cat, an), use_container_width=True)

    # Stage 4: Classify
    ct_detected = classify_curve(cat, an)
    st.info(f"🔍 **Detected curve type: {CT.name(ct_detected)}** ({CT.nfree(ct_detected)} free params)")

    # Stage 5: Global polish
    with st.status("⚡ Stage 5: Global optimization...", expanded=True) as s5:
        # Try detected type + AD baseline + passive if wide anodic range
        candidates = [ct_detected]
        if CT.AD not in candidates: candidates.append(CT.AD)
        # If anodic scan extends >0.3V past Ecorr, always try passive models
        anodic_range = Emax - Ecorr
        if anodic_range > 0.25 and CT.PD not in candidates:
            candidates.append(CT.PD)
        if anodic_range > 0.40 and CT.PT not in candidates:
            candidates.append(CT.PT)

        results = []
        Emax = float(np.max(E))
        for ct_try in candidates:
            p0 = assemble_p0(Ecorr, cat, an, ct_try, Emax)
            bp_try, r2_try, aic_try = global_polish(E, i, p0, ct_try, fit_rs, rs_max)
            results.append(dict(ct=ct_try, r2=r2_try, aicc=aic_try, bp=bp_try))
            st.write(f"  {CT.name(ct_try)}: R²={r2_try:.6f}, AICc={aic_try:.0f}")

        # Select by AICc
        results.sort(key=lambda x: x["aicc"])
        best = results[0]
        # Parsimony: prefer simpler if R² similar
        for r in results:
            if CT.nfree(r["ct"]) < CT.nfree(best["ct"]) and best["r2"] - r["r2"] < 0.002:
                best = r; break

        bp = best["bp"]; r2 = best["r2"]; best_ct = best["ct"]
        s5.update(label=f"✅ Best: {CT.name(best_ct)}, R²={r2:.6f}", state="complete")

    elapsed = time.time() - t0

    # ═════════════════════════════════════════════════════════
    # RESULTS
    # ═════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📊 Results")

    stars = "★★★★★" if r2>=0.995 else "★★★★☆" if r2>=0.99 else \
            "★★★☆☆" if r2>=0.98 else "★★☆☆☆" if r2>=0.95 else "★☆☆☆☆"
    ql = "EXCEPTIONAL" if r2>=0.995 else "EXCELLENT" if r2>=0.99 else \
         "VERY GOOD" if r2>=0.98 else "GOOD" if r2>=0.95 else "ACCEPTABLE"
    st.markdown(f"### {stars} {ql} — {CT.name(best_ct)} — {elapsed:.1f}s")

    ew, rho = MATERIALS[material]
    B = (bp[2]*bp[3])/(2.303*(bp[2]+bp[3])) if bp[2]>0 and bp[3]>0 else 0
    CR = bp[1] * 3.27 * ew / rho

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("E_corr", f"{bp[0]:.4f} V")
    c2.metric("i_corr", f"{bp[1]:.2e} A/cm²")
    c3.metric("R² (log)", f"{r2:.6f}")
    c4.metric("CR", f"{CR:.4f} mm/yr")
    c5.metric("B", f"{B:.4f} V")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ba", f"{bp[2]:.4f} V/dec")
    c2.metric("bc₁", f"{bp[3]:.4f} V/dec")
    if best_ct not in [CT.A]: c3.metric("i_L", f"{bp[4]:.2e} A/cm²")
    if best_ct in CT.PASS: c4.metric("i_pass", f"{bp[9]:.2e} A/cm²")

    # Main plot
    st.plotly_chart(plot_main(E, i, bp, Ecorr, r2, best_ct), use_container_width=True)

    # Residuals
    with st.expander("📈 Residuals"):
        st.plotly_chart(plot_residuals(E, i, bp), use_container_width=True)

    # Model comparison
    if len(results) > 1:
        with st.expander("🏆 Model Comparison"):
            rows = [{"Model": CT.name(r["ct"]), "N_free": CT.nfree(r["ct"]),
                     "R²": f"{r['r2']:.6f}", "AICc": f"{r['aicc']:.0f}",
                     "✓": "✅" if r["ct"]==best_ct else ""}
                    for r in results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Parameters
    with st.expander("🔧 All Parameters"):
        descs = ["Corrosion potential","Corrosion current","Anodic Tafel slope",
                 "Cathodic Tafel slope (O₂)","O₂ diffusion limit","H₂ exchange current",
                 "Cathodic slope (H₂)","Passivation onset","Pass. sharpness",
                 "Passive current","Breakdown potential","Transp. pre-exp",
                 "Transp. exp factor","Secondary pass. onset","Sec. pass. sharpness",
                 "Secondary passive current","Solution resistance"]
        rows = [{"Param": PARAM_NAMES[j], "Value": f"{bp[j]:.4e}" if j in LOG_P else f"{bp[j]:.4f}",
                 "Active": "✅" if j in CT.idx(best_ct) else "—", "Description": descs[j]}
                for j in range(NP)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        res_row = {"Ecorr_V": bp[0], "icorr_A_cm2": bp[1], "ba_V_dec": bp[2],
                   "bc_V_dec": bp[3], "B_V": B, "CR_mm_yr": CR,
                   "iL": bp[4] if best_ct not in [CT.A] else None,
                   "Epp_V": bp[7] if best_ct in CT.PASS else None,
                   "ipass": bp[9] if best_ct in CT.PASS else None,
                   "R2": r2, "Model": CT.name(best_ct)}
        st.download_button("📄 Results CSV",
            pd.DataFrame([res_row]).to_csv(index=False),
            "tafel_results.csv", "text/csv")
    with c2:
        fit_df = pd.DataFrame({"E_V": E, "i_meas": i, "log_i_meas": slog(i),
            "i_fit": pol_model(E,bp), "log_i_fit": slog(pol_model(E,bp)),
            "residual": slog(i)-slog(pol_model(E,bp))})
        st.download_button("📄 Fit Data CSV",
            fit_df.to_csv(index=False), "tafel_fitdata.csv", "text/csv")


if __name__ == "__main__":
    main()
