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
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
# CONSTANTS & ROBUST LOADING
# ─────────────────────────────────────────────────────────────────────────────
TINY = 1e-30
MATERIALS = {
    "Carbon Steel / Iron": (27.92, 7.87),
    "304 Stainless Steel": (25.10, 7.90),
    "316 Stainless Steel": (25.56, 8.00),
    "Copper": (31.77, 8.96),
    "Aluminum": (8.99, 2.70)
}

def robust_load_csv(uploaded_file):
    """Try various encodings and delimiters to prevent UnicodeDecodeErrors."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            # engine='python' with sep=None allows pandas to guess the delimiter (comma, tab, semicolon)
            df = pd.read_csv(uploaded_file, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
            if df.shape[1] > 1: return df
        except:
            continue
    raise ValueError("Could not decode file. Please check if the CSV is valid.")

def slog(x): return np.log10(np.maximum(np.abs(x), TINY))

def _theil_sen(x, y):
    m = len(x)
    if m < 2: return 0.0, float(np.median(y))
    i_idx, j_idx = np.triu_indices(m, k=1)
    dx = x[j_idx] - x[i_idx]
    valid = np.abs(dx) > 1e-15
    slopes = (y[j_idx][valid] - y[i_idx][valid]) / dx[valid]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    return slope, float(np.median(y - slope * x))

# ─────────────────────────────────────────────────────────────────────────────
# FITTING LOGIC (ROBUST LOCAL DETECTION)
# ─────────────────────────────────────────────────────────────────────────────
def _sliding_regress_full(x, y, min_len):
    n = len(x)
    Sx, Sy, Sxx, Sxy, Syy = np.cumsum(x), np.cumsum(y), np.cumsum(x*x), np.cumsum(x*y), np.cumsum(y*y)
    starts, ends, slopes, inters, r2s = [], [], [], [], []
    for w in range(min_len, min(25, n)+1):
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

def fit_branch(E, i, Ecorr, cfg, side='cathodic'):
    mask = (i < 0) if side == 'cathodic' else (i > 0)
    Ex, Yx = E[mask], slog(i[mask])
    si = np.argsort(Ex); Ex, Yx = Ex[si], Yx[si]
    
    # Apply guard band to avoid the curved vertex
    guard = cfg["cath_guard"] if side == 'cathodic' else cfg["anod_guard"]
    valid = (Ex < Ecorr - guard) if side == 'cathodic' else (Ex > Ecorr + guard)
    Ex_fit, Yx_fit = Ex[valid], Yx[valid]
    
    if len(Ex_fit) < 5: return None
    
    s_idx, e_idx, slps, itps, r2s = _sliding_regress_full(Ex_fit, Yx_fit, 6)
    # Proximity score: prefer windows closer to Ecorr
    dist = np.abs(Ex_fit[e_idx-1] - Ecorr) if side == 'cathodic' else np.abs(Ex_fit[s_idx] - Ecorr)
    score = r2s * np.exp(-dist / 0.15)
    score[slps >= 0 if side == 'cathodic' else slps <= 0] = 0
    
    best = np.argmax(score)
    sl, itp = _theil_sen(Ex_fit[s_idx[best]:e_idx[best]], Yx_fit[s_idx[best]:e_idx[best]])
    return {"slope": sl, "intercept": itp, "win": (Ex_fit[s_idx[best]], Ex_fit[e_idx[best]-1]), "beta": abs(1.0/sl)}

# ─────────────────────────────────────────────────────────────────────────────
# UI & TAB STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mat_key = st.selectbox("Material", list(MATERIALS.keys()))
    ew, rho = MATERIALS[mat_key]
    area = st.number_input("Area (cm²)", value=1.0)
    st.divider()
    with st.expander("Detection Guards"):
        adv_cfg = {
            "anod_guard": st.number_input("Anodic Guard (V)", 0.0, 0.1, 0.015, 0.005),
            "cath_guard": st.number_input("Cathodic Guard (V)", 0.0, 0.2, 0.035, 0.005)
        }
    if st.button("🗑 Clear All"):
        st.session_state.results, st.session_state.figures = [], []
        st.rerun()

st.markdown('<div class="main-header">⚡ Polarization Curve Fitter</div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["📂 Upload & Fit", "📊 Results", "📋 Compare"])

with t1:
    files = st.file_uploader("Upload CSV", accept_multiple_files=True)
    sample_label = st.text_input("Sample Label", "Sample 1")
    
    if files:
        for idx, f in enumerate(files):
            try:
                df = robust_load_csv(f)
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
                c1, c2 = st.columns(2)
                e_col = c1.selectbox(f"E [V] ({f.name})", cols, key=f"e_{idx}")
                i_col = c2.selectbox(f"i [A] ({f.name})", cols, key=f"i_{idx}")
                
                E_vals = df[e_col].values
                I_vals = df[i_col].values / area
                
                if st.button(f"🚀 Fit {f.name}", key=f"fit_{idx}", type="primary"):
                    Ecorr = E_vals[np.argmin(np.abs(I_vals))]
                    cat = fit_branch(E_vals, I_vals, Ecorr, adv_cfg, 'cathodic')
                    ano = fit_branch(E_vals, I_vals, Ecorr, adv_cfg, 'anodic')
                    
                    if cat and ano:
                        # intersection icorr
                        E_int = (cat['intercept'] - ano['intercept']) / (ano['slope'] - cat['slope'])
                        I_int = 10**(cat['slope'] * E_int + cat['intercept'])
                        
                        fig = plt.figure(figsize=(10, 6))
                        plt.scatter(E_vals, slog(I_vals), s=5, alpha=0.3, color='gray')
                        # Draw Tafel lines
                        for res, c in [(cat, 'red'), (ano, 'green')]:
                            ex = np.linspace(res['win'][0], E_int, 50)
                            plt.plot(ex, res['slope']*ex + res['intercept'], '--', color=c, lw=2)
                        
                        plt.plot(E_int, np.log10(I_int), 'kx', ms=10, mew=2, label=f"icorr: {I_int:.2e}")
                        plt.title(f"Evans Plot: {sample_label}")
                        plt.xlabel("E (V)"); plt.ylabel("log|i| (A/cm²)")
                        st.pyplot(fig)
                        
                        CR = I_int * 3.27 * ew / rho
                        st.session_state.results.append({"Sample": sample_label, "Ecorr": E_int, "icorr": I_int, "ba": ano['beta']*1000, "bc": cat['beta']*1000, "CR": CR})
            except Exception as e:
                st.error(f"Error: {e}")

with t2:
    if st.session_state.results:
        st.table(pd.DataFrame(st.session_state.results))
    else:
        st.info("No data.")

with t3:
    if len(st.session_state.results) > 1:
        rdf = pd.DataFrame(st.session_state.results)
        st.bar_chart(rdf, x="Sample", y="icorr")
