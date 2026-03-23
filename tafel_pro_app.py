"""
Polarization Curve Fitting App
Full-featured Streamlit app for corrosion kinetics analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import io, os, warnings
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import savgol_filter
from copy import deepcopy
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polarization Curve Fitter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 3px solid #2e86de; padding-bottom: 8px; margin-bottom: 1rem;
    }
    .param-box {
        background: #f0f4ff; border-left: 4px solid #2e86de;
        padding: 12px 16px; border-radius: 4px; margin: 4px 0;
        font-family: monospace; font-size: 0.9rem;
    }
    .region-badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:0.78rem; font-weight:600; margin:2px;
    }
    .status-ok { background:#d4edda; color:#155724; }
    .status-warn { background:#fff3cd; color:#856404; }
    .metric-card {
        background: white; border: 1px solid #dee2e6; border-radius:8px;
        padding:12px; text-align:center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Physics model ────────────────────────────────────────────────────────────
F = 96485.0
R_gas = 8.314
T_K = 298.15

def butler_volmer(E, Ecorr, icorr, ba, bc):
    """Pure Butler-Volmer (Tafel regions)."""
    eta = E - Ecorr
    return icorr * (np.exp(eta / ba) - np.exp(-eta / bc))

def full_polarization_model(E, params, model_type='full'):
    """
    Full electrochemical polarization model including:
    - Cathodic Tafel branch
    - Anodic Tafel branch  
    - Passive plateau
    - Transpassive / pitting region
    """
    Ecorr = params['Ecorr']
    icorr = params['icorr']
    ba    = params['ba']
    bc    = params['bc']

    eta = E - Ecorr
    i_cathodic = -icorr * np.exp(-eta / bc)
    i_anodic   =  icorr * np.exp( eta / ba)

    if model_type == 'butler_volmer':
        return i_anodic + i_cathodic

    # Passive region
    ip     = params.get('ip', icorr * 0.01)
    Epass  = params.get('Epass', Ecorr + 0.2)
    k_pass = params.get('k_pass', 0.05)

    passive_weight = 1.0 / (1.0 + np.exp((E - Epass) / k_pass))
    active_weight  = 1.0 - passive_weight
    i_anodic_mod   = active_weight * i_anodic + passive_weight * ip

    if model_type == 'passive':
        return i_anodic_mod + i_cathodic

    # Transpassive / pitting
    Etrans  = params.get('Etrans', Ecorr + 0.6)
    itrans  = params.get('itrans', icorr * 100)
    k_trans = params.get('k_trans', 0.03)

    trans_weight = 1.0 / (1.0 + np.exp(-(E - Etrans) / k_trans))
    i_total = i_anodic_mod * (1 - trans_weight) + itrans * trans_weight + i_cathodic
    return i_total

def log_current_density(i):
    """Safe log10 of absolute current density."""
    return np.log10(np.abs(i) + 1e-15)

# ─── Region auto-detection ────────────────────────────────────────────────────
def detect_regions(E, log_i, smoothed_log_i):
    """
    Auto-detect electrochemical regions from polarization curve.
    Returns dict with boundary potentials for each region.
    """
    regions = {}
    dlogdi  = np.gradient(smoothed_log_i, E)

    # Find Ecorr as minimum of log|i| (minimum current density)
    min_idx = np.argmin(smoothed_log_i)
    regions['Ecorr_idx'] = min_idx
    regions['Ecorr_est'] = E[min_idx]

    # Cathodic Tafel: linear region left of Ecorr with dlogI/dE ~ constant
    cat_region = np.where(E < regions['Ecorr_est'])[0]
    if len(cat_region) > 5:
        regions['cathodic_start'] = E[cat_region[0]]
        regions['cathodic_end']   = regions['Ecorr_est'] - 0.05

    # Anodic Tafel: right of Ecorr, before passivation
    ano_region = np.where(E > regions['Ecorr_est'])[0]
    if len(ano_region) > 5:
        # Look for slope change (passivation = slope decrease)
        slopes = dlogdi[ano_region]
        slope_thresh = np.percentile(slopes, 20)
        passive_candidates = np.where(slopes < slope_thresh)[0]

        if len(passive_candidates) > 0:
            passive_start_idx = ano_region[passive_candidates[0]]
            regions['anodic_end']     = E[passive_start_idx]
            regions['passive_start']  = E[passive_start_idx]

            # Look for transpassive: current rises again after passive
            trans_region_idx = ano_region[passive_candidates[0]:]
            if len(trans_region_idx) > 10:
                post_passive_slope = dlogdi[trans_region_idx]
                trans_candidates   = np.where(post_passive_slope > 0.5)[0]
                if len(trans_candidates) > 0:
                    regions['transpassive_start'] = E[trans_region_idx[trans_candidates[0]]]
                    regions['passive_end']        = E[trans_region_idx[trans_candidates[0]]]
                else:
                    regions['passive_end'] = E[ano_region[-1]]
        else:
            regions['anodic_end'] = E[ano_region[-1]]

    return regions

# ─── Fitting engine ───────────────────────────────────────────────────────────
def build_bounds(data_E, model_type, user_bounds=None):
    """Build parameter bounds for optimizer."""
    E_min, E_max = data_E.min(), data_E.max()
    E_range = E_max - E_min

    bounds_base = {
        'Ecorr': (E_min + 0.1*E_range, E_max - 0.1*E_range),
        'icorr': (1e-10, 1e-1),
        'ba':    (0.005, 0.500),
        'bc':    (0.005, 0.500),
    }
    if model_type in ('passive', 'full'):
        bounds_base.update({
            'ip':     (1e-12, 1e-2),
            'Epass':  (E_min + 0.15*E_range, E_max - 0.05*E_range),
            'k_pass': (0.005, 0.10),
        })
    if model_type == 'full':
        bounds_base.update({
            'Etrans':  (E_min + 0.25*E_range, E_max),
            'itrans':  (1e-8, 1e0),
            'k_trans': (0.005, 0.08),
        })

    if user_bounds:
        for k, v in user_bounds.items():
            if k in bounds_base:
                bounds_base[k] = v
    return bounds_base

def pack_params(param_dict):
    return list(param_dict.values()), list(param_dict.keys())

def unpack_params(x, keys):
    return dict(zip(keys, x))

def residuals(x, keys, E, log_i_obs, model_type, weights):
    params = unpack_params(x, keys)
    i_pred = full_polarization_model(E, params, model_type)
    log_i_pred = log_current_density(i_pred)
    return weights * (log_i_pred - log_i_obs)

def fit_curve(E, i_obs, model_type='full', progress_cb=None):
    """
    Two-stage global fitting:
    1) Differential Evolution (global search)
    2) Levenberg-Marquardt refinement (local polish)
    """
    log_i_obs = log_current_density(i_obs)
    smoothed  = savgol_filter(log_i_obs, min(11, len(log_i_obs)//4*2+1), 3)

    # Weights: upweight Tafel region, down-weight noise near Ecorr
    min_idx = np.argmin(log_i_obs)
    dist_from_ecorr = np.abs(np.arange(len(E)) - min_idx)
    weights = 1.0 + 0.5 * (dist_from_ecorr / (len(E)/2))

    bounds_dict = build_bounds(E, model_type)
    bounds_list, keys = pack_params(bounds_dict)
    lower = [b[0] for b in bounds_list]
    upper = [b[1] for b in bounds_list]

    # Stage 1: Differential Evolution
    de_result = differential_evolution(
        lambda x: np.sum(residuals(x, keys, E, log_i_obs, model_type, weights)**2),
        bounds=list(zip(lower, upper)),
        maxiter=800,
        popsize=18,
        tol=1e-8,
        seed=42,
        polish=False,
        workers=1,
        callback=lambda xk, convergence: (progress_cb(0.5) if progress_cb else None) or False
    )

    if progress_cb: progress_cb(0.7)

    # Stage 2: Levenberg-Marquardt polish
    lm_result = least_squares(
        residuals,
        de_result.x,
        args=(keys, E, log_i_obs, model_type, weights),
        method='lm',
        max_nfev=5000
    )

    if progress_cb: progress_cb(0.95)

    params_final = unpack_params(lm_result.x, keys)

    # Goodness of fit
    i_fit     = full_polarization_model(E, params_final, model_type)
    log_i_fit = log_current_density(i_fit)
    ss_res    = np.sum((log_i_obs - log_i_fit)**2)
    ss_tot    = np.sum((log_i_obs - np.mean(log_i_obs))**2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse      = np.sqrt(np.mean((log_i_obs - log_i_fit)**2))

    # Parameter uncertainty via Jacobian
    J = lm_result.jac
    try:
        cov  = np.linalg.inv(J.T @ J) * (ss_res / max(len(E) - len(keys), 1))
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        perr = np.zeros(len(keys))

    uncertainties = dict(zip(keys, perr))

    return {
        'params':        params_final,
        'uncertainties': uncertainties,
        'r2':            r2,
        'rmse':          rmse,
        'i_fit':         i_fit,
        'log_i_fit':     log_i_fit,
        'success':       lm_result.success or de_result.success,
        'model_type':    model_type,
    }

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data(uploaded_file, e_col, i_col, skip_rows, delimiter):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            content = uploaded_file.read().decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            sep = delimiter if delimiter != 'auto' else None
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, sep=sep, engine='python')

        # Try to find E and i columns
        cols = list(df.columns)
        if e_col and e_col in cols:
            E = df[e_col].values.astype(float)
        else:
            E = df.iloc[:, 0].values.astype(float)

        if i_col and i_col in cols:
            i = df[i_col].values.astype(float)
        else:
            i = df.iloc[:, 1].values.astype(float)

        # Remove NaN / zero current rows
        mask = np.isfinite(E) & np.isfinite(i) & (i != 0)
        return E[mask], i[mask], df, None
    except Exception as ex:
        return None, None, None, str(ex)

# ─── Export functions ─────────────────────────────────────────────────────────
def export_excel(results_list):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fitting Results"

    header_fill = PatternFill("solid", fgColor="1A3A5C")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    alt_fill    = PatternFill("solid", fgColor="EEF2FF")
    border      = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    headers = ["Sample", "Model", "E_corr (V)", "±", "i_corr (A/cm²)", "±",
               "βa (V/dec)", "±", "βc (V/dec)", "±",
               "ip (A/cm²)", "E_pass (V)", "E_trans (V)",
               "R²", "RMSE (log)", "Status"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill   = header_fill
        cell.font   = header_font
        cell.border = border
        cell.alignment = Alignment(horizontal='center', wrap_text=True)

    for row_idx, res in enumerate(results_list, 2):
        p  = res['params']
        u  = res['uncertainties']
        fill = alt_fill if row_idx % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        vals = [
            res.get('name', f'Sample {row_idx-1}'),
            res['model_type'],
            round(p.get('Ecorr', np.nan), 4),
            round(u.get('Ecorr', np.nan), 5),
            f"{p.get('icorr', np.nan):.4e}",
            f"{u.get('icorr', np.nan):.2e}",
            round(p.get('ba', np.nan)*1000, 1),
            round(u.get('ba', np.nan)*1000, 2),
            round(p.get('bc', np.nan)*1000, 1),
            round(u.get('bc', np.nan)*1000, 2),
            f"{p.get('ip', np.nan):.3e}" if 'ip' in p else "—",
            round(p.get('Epass', np.nan), 4) if 'Epass' in p else "—",
            round(p.get('Etrans', np.nan), 4) if 'Etrans' in p else "—",
            round(res['r2'], 5),
            round(res['rmse'], 5),
            "✓ Converged" if res['success'] else "⚠ Check"
        ]
        for col, val in enumerate(vals, 1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.fill   = fill
            cell.border = border
            cell.alignment = Alignment(horizontal='center')

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 22)

    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

def export_pdf_report(results_list, fig_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                  fontSize=20, textColor=colors.HexColor('#1A3A5C'),
                                  spaceAfter=6)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, textColor=colors.HexColor('#2e86de'),
                               spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=9, leading=14)
    mono_style = ParagraphStyle('Mono', parent=styles['Normal'],
                                 fontName='Courier', fontSize=8,
                                 backColor=colors.HexColor('#F0F4FF'),
                                 leftIndent=10, leading=14)

    story = []
    story.append(Paragraph("Polarization Curve Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2e86de')))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Generated by Polarization Curve Fitter  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 0.5*cm))

    for idx, (res, fig_bytes) in enumerate(zip(results_list, fig_bytes_list)):
        story.append(Paragraph(f"Sample {idx+1}: {res.get('name', f'Sample {idx+1}')}", h2_style))

        p = res['params']
        u = res['uncertainties']

        param_lines = [
            f"Model type:          {res['model_type'].replace('_', ' ').title()}",
            f"E_corr:              {p.get('Ecorr', 0):.4f} ± {u.get('Ecorr', 0):.5f} V",
            f"i_corr:              {p.get('icorr', 0):.4e} ± {u.get('icorr', 0):.2e} A/cm²",
            f"beta_a (Tafel):      {p.get('ba', 0)*1000:.1f} ± {u.get('ba', 0)*1000:.2f} mV/dec",
            f"beta_c (Tafel):      {p.get('bc', 0)*1000:.1f} ± {u.get('bc', 0)*1000:.2f} mV/dec",
        ]
        if 'ip' in p:
            param_lines.append(f"i_passive:           {p.get('ip', 0):.3e} A/cm²")
            param_lines.append(f"E_passive:           {p.get('Epass', 0):.4f} V")
        if 'Etrans' in p:
            param_lines.append(f"E_transpassive:      {p.get('Etrans', 0):.4f} V")
        param_lines += [
            f"R²:                  {res['r2']:.5f}",
            f"RMSE (log-domain):   {res['rmse']:.5f}",
            f"Fit status:          {'Converged' if res['success'] else 'Check manually'}",
        ]
        story.append(Paragraph("<br/>".join(param_lines), mono_style))
        story.append(Spacer(1, 0.3*cm))

        if fig_bytes:
            img = RLImage(io.BytesIO(fig_bytes), width=16*cm, height=10*cm)
            story.append(img)
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

    doc.build(story)
    buf.seek(0)
    return buf

# ─── Plotting ─────────────────────────────────────────────────────────────────
PALETTE = ['#2e86de', '#e84393', '#27ae60', '#e67e22', '#8e44ad', '#16a085']

def make_figure(E, i_obs, fit_result, sample_name, regions=None, show_regions=True):
    fig = plt.figure(figsize=(12, 8), dpi=120)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax_main  = fig.add_subplot(gs[0, :])   # Top: main Tafel plot
    ax_lin   = fig.add_subplot(gs[1, 0])   # Bottom-left: linear plot
    ax_resid = fig.add_subplot(gs[1, 1])   # Bottom-right: residuals

    log_i_obs = log_current_density(i_obs)
    E_dense   = np.linspace(E.min(), E.max(), 2000)
    i_fit_dense = full_polarization_model(E_dense, fit_result['params'], fit_result['model_type'])

    # ── Main Tafel plot ──────────────────────────────────────────
    ax_main.plot(E, log_i_obs, 'o', color='#aab4c4', ms=3, alpha=0.6, label='Data', zorder=2)
    ax_main.plot(E_dense, log_current_density(i_fit_dense),
                 '-', color='#2e86de', lw=2.2, label='Global fit', zorder=3)

    # Region shading
    if show_regions and regions:
        p = fit_result['params']
        if 'Epass' in p and 'Etrans' in p:
            ax_main.axvspan(p['Epass'], p['Etrans'], alpha=0.08, color='#27ae60', label='Passive region')
        if 'Etrans' in p:
            ax_main.axvspan(p['Etrans'], E.max(), alpha=0.08, color='#e67e22', label='Transpassive')

    ax_main.axvline(fit_result['params']['Ecorr'], color='#e84393', ls='--', lw=1.4, label=f"E_corr = {fit_result['params']['Ecorr']:.4f} V", zorder=4)

    # Mark icorr
    icorr_log = np.log10(abs(fit_result['params']['icorr']))
    ax_main.axhline(icorr_log, color='#e84393', ls=':', lw=1.2, alpha=0.7)
    ax_main.text(E.min() + 0.02*(E.max()-E.min()), icorr_log + 0.05,
                 f"i_corr = {fit_result['params']['icorr']:.2e} A/cm²",
                 fontsize=8, color='#e84393')

    # Tafel slope tangents
    Ecorr = fit_result['params']['Ecorr']
    ba    = fit_result['params']['ba']
    bc    = fit_result['params']['bc']
    E_tan_a = np.linspace(Ecorr, Ecorr + 0.25, 100)
    E_tan_c = np.linspace(Ecorr - 0.25, Ecorr, 100)
    ax_main.plot(E_tan_a, icorr_log + (E_tan_a - Ecorr) / ba,
                 '--', color='#e67e22', lw=1.5, alpha=0.8, label=f"βa = {ba*1000:.0f} mV/dec")
    ax_main.plot(E_tan_c, icorr_log - (Ecorr - E_tan_c) / bc,
                 '--', color='#8e44ad', lw=1.5, alpha=0.8, label=f"βc = {bc*1000:.0f} mV/dec")

    ax_main.set_xlabel("E vs. Ref (V)", fontsize=10)
    ax_main.set_ylabel("log |i| (A/cm²)", fontsize=10)
    ax_main.set_title(f"Tafel Plot — {sample_name}", fontsize=12, fontweight='bold')
    ax_main.legend(fontsize=7.5, ncol=3, loc='upper right')
    ax_main.grid(True, ls='--', alpha=0.35)
    ax_main.set_facecolor('#fafbff')

    # ── Linear i vs E ───────────────────────────────────────────
    ax_lin.plot(E, i_obs * 1e3, 'o', color='#aab4c4', ms=2.5, alpha=0.6)
    ax_lin.plot(E_dense, i_fit_dense * 1e3, '-', color='#2e86de', lw=2)
    ax_lin.set_xlabel("E (V)", fontsize=9)
    ax_lin.set_ylabel("i (mA/cm²)", fontsize=9)
    ax_lin.set_title("Linear Scale", fontsize=10, fontweight='bold')
    ax_lin.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax_lin.axvline(fit_result['params']['Ecorr'], color='#e84393', ls='--', lw=1)
    ax_lin.grid(True, ls='--', alpha=0.3)
    ax_lin.set_facecolor('#fafbff')

    # ── Residuals ───────────────────────────────────────────────
    i_fit_at_data = full_polarization_model(E, fit_result['params'], fit_result['model_type'])
    residual_vals = log_current_density(i_obs) - log_current_density(i_fit_at_data)
    ax_resid.scatter(E, residual_vals, s=8, color='#2e86de', alpha=0.6)
    ax_resid.axhline(0, color='k', lw=0.8)
    ax_resid.axhline( 0.1, color='#e84393', ls=':', lw=1, alpha=0.6)
    ax_resid.axhline(-0.1, color='#e84393', ls=':', lw=1, alpha=0.6)
    ax_resid.set_xlabel("E (V)", fontsize=9)
    ax_resid.set_ylabel("Δlog|i|", fontsize=9)
    ax_resid.set_title(f"Residuals  (R²={fit_result['r2']:.4f})", fontsize=10, fontweight='bold')
    ax_resid.grid(True, ls='--', alpha=0.3)
    ax_resid.set_facecolor('#fafbff')

    fig.patch.set_facecolor('white')
    return fig

# ─── Session state ────────────────────────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = []
if 'figures' not in st.session_state:
    st.session_state.figures = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("**Model Selection**")
    model_type = st.selectbox(
        "Electrochemical model",
        options=['full', 'passive', 'butler_volmer'],
        format_func=lambda x: {
            'full': '🔵 Full (BV + Passive + Transpassive)',
            'passive': '🟢 BV + Passive plateau',
            'butler_volmer': '🟡 Butler-Volmer only'
        }[x],
        index=0
    )

    st.markdown("**Data Import**")
    skip_rows  = st.number_input("Skip header rows", 0, 20, 0)
    delimiter  = st.selectbox("CSV delimiter", ['auto', ',', ';', '\t', ' '], index=0)
    e_col_name = st.text_input("E column name (leave blank = col 1)", "")
    i_col_name = st.text_input("i column name (leave blank = col 2)", "")

    st.markdown("**Optimizer**")
    show_regions = st.toggle("Shade electrochemical regions", True)
    smooth_data  = st.toggle("Pre-smooth data (Savitzky-Golay)", False)

    st.markdown("---")
    st.markdown("**Export**")
    pub_dpi = st.slider("Publication figure DPI", 150, 600, 300, 50)

    if st.button("🗑 Clear all results"):
        st.session_state.results = []
        st.session_state.figures = []
        st.rerun()

# ─── Main layout ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Polarization Curve Fitter</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload & Fit", "📊 Results & Figures", "📋 Compare", "ℹ️ Help"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Upload & Fit
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_up, col_cfg = st.columns([1, 1])

    with col_up:
        st.markdown("### 📁 Upload Data File")
        uploaded_files = st.file_uploader(
            "Upload CSV / TXT / XLSX files",
            type=['csv', 'txt', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Autolab/NOVA exports, plain CSV, or Excel files with E and i columns"
        )

    with col_cfg:
        st.markdown("### 🔬 Fit Configuration")
        sample_name = st.text_input("Sample name / label", "Sample 1")
        i_unit = st.selectbox("Current density unit in file", ['A/cm²', 'mA/cm²', 'µA/cm²', 'A/m²'])
        scan   = st.selectbox("Scan direction", ['Both (full curve)', 'Anodic only', 'Cathodic only'])

        unit_factors = {'A/cm²': 1.0, 'mA/cm²': 1e-3, 'µA/cm²': 1e-6, 'A/m²': 1e-4}
        i_factor = unit_factors[i_unit]

    if uploaded_files:
        for uf in uploaded_files:
            with st.expander(f"📄 {uf.name}", expanded=True):
                E, i_raw, df_raw, err = load_data(
                    uf,
                    e_col_name or None,
                    i_col_name or None,
                    skip_rows,
                    delimiter
                )
                if err:
                    st.error(f"❌ Load error: {err}")
                    continue

                i_raw = i_raw * i_factor

                # Preview
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**Points loaded:** {len(E)}")
                    st.markdown(f"**E range:** [{E.min():.4f}, {E.max():.4f}] V")
                    st.markdown(f"**|i| range:** [{abs(i_raw).min():.2e}, {abs(i_raw).max():.2e}] A/cm²")
                    if df_raw is not None:
                        st.dataframe(df_raw.head(5), use_container_width=True, height=160)

                with c2:
                    fig_prev, ax_prev = plt.subplots(figsize=(6, 3.5))
                    ax_prev.plot(E, np.log10(np.abs(i_raw) + 1e-15), 'o-',
                                 ms=2, lw=0.8, color='#2e86de', alpha=0.7)
                    ax_prev.set_xlabel("E (V)"); ax_prev.set_ylabel("log |i|")
                    ax_prev.set_title("Raw Data Preview", fontsize=10)
                    ax_prev.grid(True, ls='--', alpha=0.35)
                    fig_prev.tight_layout()
                    st.pyplot(fig_prev, use_container_width=True)
                    plt.close(fig_prev)

                btn_col, _ = st.columns([1, 2])
                with btn_col:
                    run_fit = st.button(f"🚀 Fit {uf.name}", key=f"fit_{uf.name}")

                if run_fit:
                    prog = st.progress(0, text="Initializing global optimizer...")

                    def update_progress(val):
                        prog.progress(val, text={
                            0.5: "⚙️ Differential evolution running...",
                            0.7: "🔬 Levenberg-Marquardt refinement...",
                            0.95: "✅ Calculating uncertainties..."
                        }.get(val, "Running..."))

                    if smooth_data:
                        win = min(11, len(i_raw)//4*2+1)
                        i_fit_input = np.sign(i_raw) * np.abs(savgol_filter(i_raw, win, 3))
                    else:
                        i_fit_input = i_raw

                    try:
                        result = fit_curve(E, i_fit_input, model_type, update_progress)
                        result['name'] = sample_name or uf.name
                        prog.progress(1.0, text="✅ Fitting complete!")

                        # Detect regions for plotting
                        log_i = log_current_density(i_raw)
                        sm = savgol_filter(log_i, min(11, len(log_i)//4*2+1), 3)
                        regions = detect_regions(E, log_i, sm)

                        fig = make_figure(E, i_raw, result, result['name'], regions, show_regions)
                        fig.tight_layout()

                        # Save high-res bytes for export
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, dpi=pub_dpi, bbox_inches='tight', facecolor='white')
                        buf_png.seek(0)
                        png_bytes = buf_png.read()

                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg', bbox_inches='tight', facecolor='white')
                        buf_svg.seek(0)

                        st.session_state.results.append(result)
                        st.session_state.figures.append({'png': png_bytes, 'svg': buf_svg.read(), 'name': result['name']})

                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        # Parameter cards
                        p = result['params']
                        u = result['uncertainties']
                        st.markdown("#### 📐 Fitted Parameters")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("E_corr (V)", f"{p['Ecorr']:.4f}", f"±{u.get('Ecorr',0):.5f}")
                        m2.metric("i_corr (A/cm²)", f"{p['icorr']:.3e}", f"±{u.get('icorr',0):.1e}")
                        m3.metric("βa (mV/dec)", f"{p['ba']*1000:.1f}", f"±{u.get('ba',0)*1000:.2f}")
                        m4.metric("βc (mV/dec)", f"{p['bc']*1000:.1f}", f"±{u.get('bc',0)*1000:.2f}")

                        if 'ip' in p:
                            m5, m6, m7, m8 = st.columns(4)
                            m5.metric("i_passive (A/cm²)", f"{p['ip']:.3e}")
                            m6.metric("E_passive (V)", f"{p.get('Epass',0):.4f}")
                            if 'Etrans' in p:
                                m7.metric("E_transpassive (V)", f"{p.get('Etrans',0):.4f}")
                            m8.metric("R²", f"{result['r2']:.5f}")

                        # Download figure
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button("⬇ Download PNG", data=png_bytes,
                                               file_name=f"{result['name']}_fit.png",
                                               mime="image/png")
                        with dl2:
                            st.download_button("⬇ Download SVG", data=buf_svg.getvalue(),
                                               file_name=f"{result['name']}_fit.svg",
                                               mime="image/svg+xml")

                    except Exception as ex:
                        st.error(f"❌ Fitting failed: {ex}")
                        import traceback; st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Results & Export
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.results:
        st.info("No fitting results yet. Upload and fit data in the **Upload & Fit** tab.")
    else:
        st.markdown(f"### 📊 {len(st.session_state.results)} fitted sample(s)")

        # Summary table
        rows = []
        for r in st.session_state.results:
            p = r['params']
            rows.append({
                'Sample':        r.get('name', '?'),
                'Model':         r['model_type'],
                'E_corr (V)':    round(p.get('Ecorr', 0), 4),
                'i_corr (A/cm²)': f"{p.get('icorr', 0):.3e}",
                'βa (mV/dec)':   round(p.get('ba', 0)*1000, 1),
                'βc (mV/dec)':   round(p.get('bc', 0)*1000, 1),
                'i_pass (A/cm²)': f"{p.get('ip', 0):.2e}" if 'ip' in p else '—',
                'E_pass (V)':    round(p.get('Epass', 0), 4) if 'Epass' in p else '—',
                'E_trans (V)':   round(p.get('Etrans', 0), 4) if 'Etrans' in p else '—',
                'R²':            round(r['r2'], 5),
                'RMSE':          round(r['rmse'], 5),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")
        ecol1, ecol2, ecol3 = st.columns(3)

        with ecol1:
            xlsx_buf = export_excel(st.session_state.results)
            st.download_button(
                "📥 Export Excel (.xlsx)",
                data=xlsx_buf,
                file_name="polarization_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with ecol2:
            pdf_buf = export_pdf_report(
                st.session_state.results,
                [f['png'] for f in st.session_state.figures]
            )
            st.download_button(
                "📥 Export PDF Report",
                data=pdf_buf,
                file_name="polarization_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with ecol3:
            # Bulk figures ZIP
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for fig_data in st.session_state.figures:
                    zf.writestr(f"{fig_data['name']}_fit.png", fig_data['png'])
                    zf.writestr(f"{fig_data['name']}_fit.svg", fig_data['svg'])
            zip_buf.seek(0)
            st.download_button(
                "📥 Export All Figures (.zip)",
                data=zip_buf,
                file_name="polarization_figures.zip",
                mime="application/zip",
                use_container_width=True
            )

        # Show individual figures
        st.markdown("### 🖼 Fitted Plots")
        for fig_data in st.session_state.figures:
            st.markdown(f"**{fig_data['name']}**")
            st.image(fig_data['png'], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(st.session_state.results) < 2:
        st.info("Fit at least 2 samples to enable comparison.")
    else:
        st.markdown("### 📋 Multi-Sample Comparison")

        fig_cmp, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

        for idx, (res, fig_data) in enumerate(zip(st.session_state.results, st.session_state.figures)):
            color  = PALETTE[idx % len(PALETTE)]
            p      = res['params']
            E_plot = np.linspace(-1.5, 1.5, 3000)
            i_plot = full_polarization_model(E_plot, p, res['model_type'])
            label  = res.get('name', f"S{idx+1}")

            axes[0].plot(E_plot, log_current_density(i_plot), color=color, lw=2, label=label)
            axes[0].axvline(p['Ecorr'], color=color, ls=':', lw=1, alpha=0.6)

            axes[1].bar(idx, p['icorr'], color=color, alpha=0.8, label=label)

        axes[0].set_xlabel("E (V)"); axes[0].set_ylabel("log |i| (A/cm²)")
        axes[0].set_title("Overlay — Tafel Plots", fontweight='bold')
        axes[0].legend(fontsize=8); axes[0].grid(True, ls='--', alpha=0.3)
        axes[0].set_facecolor('#fafbff')

        axes[1].set_xticks(range(len(st.session_state.results)))
        axes[1].set_xticklabels([r.get('name','?') for r in st.session_state.results], rotation=20, ha='right', fontsize=8)
        axes[1].set_ylabel("i_corr (A/cm²)")
        axes[1].set_title("i_corr Comparison", fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, axis='y', ls='--', alpha=0.3)
        axes[1].set_facecolor('#fafbff')

        fig_cmp.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)

        # Tafel slopes comparison bar chart
        fig_beta, ax_beta = plt.subplots(figsize=(10, 4), dpi=100)
        x     = np.arange(len(st.session_state.results))
        width = 0.35
        ba_vals = [r['params']['ba']*1000 for r in st.session_state.results]
        bc_vals = [r['params']['bc']*1000 for r in st.session_state.results]
        names   = [r.get('name','?') for r in st.session_state.results]

        ax_beta.bar(x - width/2, ba_vals, width, label='βa (mV/dec)', color='#e67e22', alpha=0.85)
        ax_beta.bar(x + width/2, bc_vals, width, label='βc (mV/dec)', color='#8e44ad', alpha=0.85)
        ax_beta.set_xticks(x); ax_beta.set_xticklabels(names, rotation=15, ha='right')
        ax_beta.set_ylabel("Tafel slope (mV/dec)"); ax_beta.set_title("Tafel Slopes Comparison", fontweight='bold')
        ax_beta.legend(); ax_beta.grid(True, axis='y', ls='--', alpha=0.3)
        ax_beta.set_facecolor('#fafbff')
        fig_beta.tight_layout()
        st.pyplot(fig_beta, use_container_width=True)

        plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Help
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
### 📖 User Guide

#### Model Physics
| Model | Regions covered |
|---|---|
| **Butler-Volmer** | Cathodic Tafel + Anodic Tafel only |
| **BV + Passive** | + Passive plateau (ip, Epass) |
| **Full** | + Transpassive / pitting (Etrans, itrans) |

The full model uses a sigmoidal transition function between regions, enabling
**smooth global fitting** across the entire polarization curve in a single optimization pass.

#### Fitting Strategy
1. **Differential Evolution** (global, population-based) explores the full parameter space
2. **Levenberg-Marquardt** (local, gradient-based) polishes the result
3. All fitting is done in **log-domain** to handle the 5–8 decade span of current density
4. **Uncertainty estimates** are derived from the LM Jacobian covariance matrix

#### Extracted Parameters
| Symbol | Meaning |
|---|---|
| E_corr | Corrosion potential |
| i_corr | Corrosion current density |
| βa | Anodic Tafel slope |
| βc | Cathodic Tafel slope |
| ip | Passive current density |
| Epass | Passivation onset potential |
| Etrans | Transpassive / pitting potential |

#### Data Format
- First column: **Potential (V vs. reference)**
- Second column: **Current density** (select unit in sidebar)
- Supports Autolab/NOVA `.txt` / `.csv` exports, plain CSV, Excel

#### Tips
- Use **Full model** for stainless steels, passive alloys, black oxide coatings
- Use **BV only** for active metals or short scans near Ecorr
- Enable **Savitzky-Golay pre-smoothing** for noisy data
- **R² > 0.99** indicates excellent fit quality
    """)
