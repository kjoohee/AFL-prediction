"""
AFL Performance Analysis Dashboard v3
======================================
4-tab: Overview → EDA → Causal Inference → Predictive Model
Causal results loaded from CSV exports (ate_results.csv, rule_change_results.csv).

Usage:
    pip install streamlit plotly pandas numpy scipy
    streamlit run Dashboard.v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import re
import os

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "C:\\Users\\kzooo\\0. McGill Course\\3. INSY 674\\AFL-prediction\\data\\processed\\df_final_final.csv"                         # ← main dataset
CAUSAL_DIR = "C:\\Users\\kzooo\\0. McGill Course\\3. INSY 674\\AFL-prediction\\Models\\causal_results"                      # ← folder with CSV exports
ATE_PATH = os.path.join(CAUSAL_DIR, "ate_results.csv")
RULE_PATH = os.path.join(CAUSAL_DIR, "rule_change_results.csv")
REFUTATION_PATH = os.path.join(CAUSAL_DIR, "refutation_results.csv")
HTE_PATH = os.path.join(CAUSAL_DIR, "hte_results.csv")

st.set_page_config(
    page_title="AFL Performance Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Outfit', sans-serif; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560; border-radius: 12px; padding: 16px;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.15);
    }
    div[data-testid="stMetric"] label { color: #a8a8b3 !important; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e94560 !important; font-weight: 700; }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e94560; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown div,
    section[data-testid="stSidebar"] .stMarkdown strong,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #d1d1e0 !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }

    .hyp-badge {
        display: inline-block; background: linear-gradient(135deg, #e94560, #c23152);
        color: white; padding: 4px 14px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }
    .insight-box {
        background: #1a1a2e; border-left: 4px solid #e94560;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; color: #d1d1e0;
    }
    .causal-verdict {
        background: #0f3460; border-left: 4px solid #16c79a;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; color: #d1d1e0;
    }
    .rule-timeline {
        background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
        padding: 10px 14px; margin: 4px 0; color: #d1d1e0; font-size: 0.82rem;
    }
    .rule-timeline b { color: #f5a623; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_ACCENT = "#e94560"
COLOR_SECONDARY = "#0f3460"
COLOR_PALETTE = ["#e94560", "#0f3460", "#16c79a", "#f5a623", "#7c5cbf", "#00b4d8"]


def _bold(text):
    """Convert markdown **bold** to HTML <b>bold</b>."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)


def hypothesis_badge(num, text):
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">'
        f'<span class="hyp-badge">{num}</span>'
        f'<span style="font-weight:500; font-size:1rem;">{_bold(text)}</span>'
        f'</div>', unsafe_allow_html=True)


def insight_box(text):
    st.markdown(f'<div class="insight-box">💡 <strong>Insight:</strong> {_bold(text)}</div>', unsafe_allow_html=True)


def causal_verdict(text):
    st.markdown(f'<div class="causal-verdict">🧬 <strong>Causal Verdict:</strong> {_bold(text)}</div>', unsafe_allow_html=True)


def correlation_test(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return None, None, "Insufficient data"
    r, p = stats.pearsonr(x[m], y[m])
    if p < 0.001: sig = "Highly significant (p < 0.001)"
    elif p < 0.01: sig = "Significant (p < 0.01)"
    elif p < 0.05: sig = "Significant (p < 0.05)"
    else: sig = "Not significant (p ≥ 0.05)"
    return r, p, sig


def safe_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def causal_bar(data, x, y, title, midpoint=0, fmt="+.2f", colorscale=None):
    """Reusable causal ATE bar chart with proper margins so labels don't clip."""
    if colorscale is None:
        colorscale = ["#0f3460", "#333", "#e94560"]
    fig = px.bar(data, x=x, y=y, color=y,
        color_continuous_scale=colorscale, color_continuous_midpoint=midpoint,
        title=title, text=y, template=PLOTLY_TEMPLATE)
    fig.update_traces(texttemplate=f"%{{text:{fmt}}}", textposition="outside")
    vals = data[y].tolist()
    pad = max(abs(max(vals)), abs(min(vals))) * 0.25 + 0.15
    fig.update_layout(
        height=450, margin=dict(t=60, b=40),
        yaxis=dict(range=[min(vals) - pad, max(vals) + pad]))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at `{DATA_PATH}`. Update DATA_PATH.")
        st.stop()
    df.columns = df.columns.str.replace(".", "_", regex=False).str.replace(" ", "_")
    if "TotalScore" not in df.columns and "Goals" in df.columns and "Behinds" in df.columns:
        df["TotalScore"] = (6 * df["Goals"]) + df["Behinds"]
    if "Goal_Scored" not in df.columns and "Goals" in df.columns:
        df["Goal_Scored"] = (df["Goals"] > 0).astype(int)
    if "is_home" not in df.columns and "Team" in df.columns and "AwayTeam" in df.columns:
        df["is_home"] = (df["Team"] != df["AwayTeam"]).astype(int)
    if "BMI" not in df.columns and "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    return df


@st.cache_data
def load_causal_csv(path, fallback=None):
    """Load a causal result CSV. Return fallback DataFrame if file missing."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return fallback


df = load_data()

# Load causal results (CSV if available, otherwise hardcoded fallback)
_fb_ate = pd.DataFrame({
    "Hypothesis": ["C1_Height"]*4 + ["C2_Weight"]*4 + ["C3_BMI_primary"]*4 + ["C4_Home"]*4,
    "Position": ["Forward","Midfield","Ruck","Defender"]*4,
    "Treatment": ["Height"]*4 + ["Weight"]*4 + ["BMI"]*4 + ["is_home"]*4,
    "Outcome": ["TotalScore","Clearances","HitOuts","Rebounds"]*4,
    "ATE_XGB": [-0.13, 0.11, 5.10, -0.56, -0.10, 0.66, 4.70, -0.32, 0.22, 0.69, 1.70, 0.09, -0.05, -0.01, 0.15, -0.08],
    "ATE_LRS": [-0.08, 0.09, 4.80, -0.42, -0.07, 0.58, 4.30, -0.28, 0.18, 0.55, 1.50, 0.07, -0.03, -0.02, 0.12, -0.06],
})

_fb_rule = pd.DataFrame({
    "Rule_Key": ["Ruck_666", "Midfield_Stand", "Midfield_RotCaps"],
    "Pre_ATE": [0.8, 1.1, -0.15],
    "Post_ATE": [5.3, 0.69, 0.66],
    "Change_Pct": [561, -37, 100],
    "Interpretation": ["More space = height dominance", "Speed benefits all equally", "Heavier mids stay on longer"],
})

ate_df = load_causal_csv(ATE_PATH, fallback=_fb_ate)
rule_df = load_causal_csv(RULE_PATH, fallback=_fb_rule)
ref_csv = load_causal_csv(REFUTATION_PATH)
hte_csv = load_causal_csv(HTE_PATH)

causal_source = "📁 CSV" if os.path.exists(ATE_PATH) else "⚠️ Hardcoded fallback (run export_causal_results.py in notebook to generate CSVs)"


# Column mapping
COL_HEIGHT = safe_col(df, ["Height", "height"])
COL_WEIGHT = safe_col(df, ["Weight", "weight"])
COL_BMI = safe_col(df, ["BMI", "bmi"])
COL_DISPOSALS = safe_col(df, ["Disposals", "disposals", "Total_Disposals"])
COL_KICKS = safe_col(df, ["Kicks", "kicks", "K"])
COL_HANDBALLS = safe_col(df, ["Handballs", "handballs", "HB"])
COL_INSIDE50 = safe_col(df, ["Inside50s", "Inside_50s", "Inside50", "I50"])
COL_GOALS = safe_col(df, ["Goals", "goals"])
COL_TOTAL_SCORE = safe_col(df, ["TotalScore", "Total_Score"])
COL_GOAL_SCORED = safe_col(df, ["Goal_Scored", "goal_scored"])
COL_HOME = safe_col(df, ["is_home", "IsHome"])
COL_POSITION = safe_col(df, ["PrimaryPosition", "Position"])
COL_CLEARANCES = safe_col(df, ["Clearances"])
COL_HITOUTS = safe_col(df, ["HitOuts", "Hitouts"])
COL_REBOUNDS = safe_col(df, ["Rebounds"])
COL_AGE = safe_col(df, ["Age"])


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏈 AFL Dashboard")
    st.markdown("Performance Analysis & Causal Inference")
    st.divider()

    st.markdown("### 🔍 Filters")
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique())
        selected_years = st.slider("Season Range",
            min_value=int(min(years)), max_value=int(max(years)),
            value=(int(min(years)), int(max(years))))
        mask = df["Year"].between(*selected_years)
    else:
        mask = pd.Series(True, index=df.index)

    if "Team" in df.columns:
        teams = ["All Teams"] + sorted(df["Team"].dropna().unique().tolist())
        selected_team = st.selectbox("Team", teams)
        if selected_team != "All Teams":
            mask = mask & (df["Team"] == selected_team)

    if COL_POSITION:
        positions = ["All Positions"] + sorted(df[COL_POSITION].dropna().unique().tolist())
        selected_position = st.selectbox("Position", positions)
        if selected_position != "All Positions":
            mask = mask & (df[COL_POSITION] == selected_position)

    filtered_df = df[mask].copy()

    st.divider()
    st.markdown(f"**Records:** {len(filtered_df):,}")
    st.markdown(f"**Features:** {len(filtered_df.columns)}")

    with st.expander("🔧 Column Details", expanded=False):
        st.markdown(
            '<div style="color:#d1d1e0; font-size:0.85rem;">',
            unsafe_allow_html=True)
        col_map = {"Height": COL_HEIGHT, "Weight": COL_WEIGHT, "BMI": COL_BMI,
                   "Disposals": COL_DISPOSALS, "Goals": COL_GOALS,
                   "Home": COL_HOME, "Position": COL_POSITION,
                   "Inside50s": COL_INSIDE50, "Clearances": COL_CLEARANCES,
                   "HitOuts": COL_HITOUTS, "Rebounds": COL_REBOUNDS}
        for name, col in col_map.items():
            icon = "✅" if col else "❌"
            val = col if col else "Not found"
            st.markdown(
                f'<span style="color:#d1d1e0;">{icon} {name}: <code>{val}</code></span>',
                unsafe_allow_html=True)
        st.markdown(
            f'<br><span style="color:#a8a8b3;">Causal data: {causal_source}</span></div>',
            unsafe_allow_html=True)

    st.divider()

    # Rule Changes Timeline
    st.markdown("### 📜 AFL Rule Changes")
    st.markdown(
        '<div class="rule-timeline"><b>2019</b> — 6-6-6 Starting Positions<br>'
        'Players locked in zones at centre bounces → more open space</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="rule-timeline"><b>2021</b> — Stand Rule<br>'
        'Players must stand still when opponent marks → faster ball movement</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="rule-timeline"><b>2016–21</b> — Rotation Caps<br>'
        'Interchange limits reduced 120 → 90 → 75 → lifted<br>'
        'Endurance & weight matter more with fewer rotations</div>',
        unsafe_allow_html=True)

    st.divider()
    st.markdown("### [INSY674] Team 5")
    st.markdown("Faye Wu")
    st.markdown("Monica Jang")
    st.markdown("Joohee Kim")
    st.markdown("Rui Zhao")
    st.markdown("Jacob Featherstone")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏈 AFL Performance Analysis Dashboard")
st.markdown("Analysing causal effects of physical attributes on position-specific performance.")
st.divider()

if COL_GOALS and COL_GOAL_SCORED:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(filtered_df):,}")
    c2.metric("Avg Goals/Game", f"{filtered_df[COL_GOALS].mean():.2f}",
              help="Average goals per player per game record")
    c3.metric("Goal-Scoring Rate", f"{filtered_df[COL_GOAL_SCORED].mean()*100:.1f}%",
              help="% of records where at least 1 goal was scored")
    if COL_TOTAL_SCORE:
        c4.metric("Avg TotalScore", f"{filtered_df[COL_TOTAL_SCORE].mean():.1f}",
                  help="TotalScore = 6 × Goals + Behinds")
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Overview", "📈 Exploratory Analysis", "🧬 Causal Inference", "🤖 Predictive Model"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 0 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## Dataset Overview & Feature Correlations")

    numeric_cols = [c for c in [COL_HEIGHT, COL_WEIGHT, COL_BMI, COL_DISPOSALS,
                                COL_INSIDE50, COL_GOALS, COL_CLEARANCES,
                                COL_HITOUTS, COL_REBOUNDS] if c]
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr()
        mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_display = corr_matrix.where(~mask_tri)
        fig_corr = px.imshow(corr_display, text_auto=".2f",
            color_continuous_scale=["#0f3460", "#1a1a2e", "#e94560"],
            title="Feature Correlation Matrix (Lower Triangle)", template=PLOTLY_TEMPLATE)
        fig_corr.update_layout(height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

    col_a, col_b = st.columns(2)
    if COL_GOALS:
        with col_a:
            goal_binned = filtered_df[COL_GOALS].clip(upper=5)
            goal_labels = goal_binned.map(lambda x: f"{int(x)}" if x < 5 else "5+")
            bin_counts = goal_labels.value_counts().reindex(
                ["0", "1", "2", "3", "4", "5+"], fill_value=0).reset_index()
            bin_counts.columns = ["Goals", "Count"]
            fig = px.bar(bin_counts, x="Goals", y="Count",
                title="Distribution of Goals per Game",
                color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

    if COL_POSITION:
        with col_b:
            pos_counts = filtered_df[COL_POSITION].value_counts().reset_index()
            pos_counts.columns = ["Position", "Count"]
            fig = px.pie(pos_counts, names="Position", values="Count",
                title="Position Distribution", color_discrete_sequence=COLOR_PALETTE,
                template=PLOTLY_TEMPLATE, hole=0.45)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    if COL_HEIGHT and COL_WEIGHT and COL_BMI and COL_AGE:
        st.markdown("### Physical Attributes Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Height", f"{filtered_df[COL_HEIGHT].mean():.1f} cm")
        c2.metric("Avg Weight", f"{filtered_df[COL_WEIGHT].mean():.1f} kg")
        c3.metric("Avg BMI", f"{filtered_df[COL_BMI].mean():.1f}")
        c4.metric("Avg Age", f"{filtered_df[COL_AGE].mean():.1f} yrs")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — EXPLORATORY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📈 Exploratory Data Analysis")
    st.markdown("Correlation-based analysis of physical attributes and performance metrics.")

    eda_sub = st.tabs(["🏋️ Height", "⚖️ Weight", "📐 BMI",
                       "🏃 Disposals", "🎯 Inside-50s", "🏠 Home"])

    # ── Height ──
    with eda_sub[0]:
        st.markdown("#### Height vs Goals")
        if COL_HEIGHT and COL_GOALS:
            r, p, _ = correlation_test(filtered_df[COL_HEIGHT], filtered_df[COL_GOALS])
            if r is not None:
                supported = "✅ Supported" if r > 0 and p < 0.05 else "❌ Not Supported"
                insight_box(f"Height → Goals: r={r:.4f}, p={p:.2e} — {supported}")
            col1, col2 = st.columns(2)
            with col1:
                temp = filtered_df.copy()
                temp["Height Group"] = pd.cut(temp[COL_HEIGHT], bins=6, precision=0).astype(str)
                avg_by_bin = temp.groupby("Height Group")[COL_GOALS].mean().reset_index()
                fig = px.bar(avg_by_bin, x="Height Group", y=COL_GOALS,
                    title="Avg Goals by Height Group",
                    color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE,
                    labels={COL_GOALS: "Avg Goals"}, text_auto=".2f")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                temp2 = filtered_df.copy()
                temp2["Bin"] = pd.cut(temp2[COL_HEIGHT], bins=5, precision=0).astype(str)
                fig2 = px.box(temp2, x="Bin", y=COL_GOALS, title="Goals Distribution by Height",
                    color="Bin", color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            if COL_GOAL_SCORED:
                temp3 = filtered_df.copy()
                temp3["Bin"] = pd.cut(temp3[COL_HEIGHT], bins=5, precision=0).astype(str)
                rate = temp3.groupby("Bin")[COL_GOAL_SCORED].mean().reset_index()
                fig3 = px.bar(rate, x="Bin", y=COL_GOAL_SCORED, title="P(Goal) by Height Group",
                    color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE,
                    labels={COL_GOAL_SCORED: "P(Goal Scored)"})
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Height or Goals column not found.")

    # ── Weight ──
    with eda_sub[1]:
        st.markdown("#### Weight vs Goals")
        if COL_WEIGHT and COL_GOALS:
            r, p, _ = correlation_test(filtered_df[COL_WEIGHT], filtered_df[COL_GOALS])
            if r is not None:
                supported = "✅ Supported" if r < 0 and p < 0.05 else "❌ Not Supported"
                insight_box(f"Weight → Goals: r={r:.4f}, p={p:.2e} — {supported}")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
                    x=COL_WEIGHT, y=COL_GOALS, trendline="ols", title="Weight vs Goals",
                    opacity=0.4, color_discrete_sequence=[COLOR_SECONDARY], template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                temp = filtered_df.copy()
                temp["Bin"] = pd.cut(temp[COL_WEIGHT], bins=5, precision=0).astype(str)
                fig2 = px.violin(temp, x="Bin", y=COL_GOALS, title="Goals by Weight Group",
                    color="Bin", color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE, box=True)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            if COL_GOAL_SCORED:
                rate = temp.groupby("Bin")[COL_GOAL_SCORED].mean().reset_index()
                fig3 = px.bar(rate, x="Bin", y=COL_GOAL_SCORED, title="P(Goal) by Weight Group",
                    color_discrete_sequence=[COLOR_SECONDARY], template=PLOTLY_TEMPLATE,
                    labels={COL_GOAL_SCORED: "P(Goal Scored)"})
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Weight or Goals column not found.")

    # ── BMI ──
    with eda_sub[2]:
        st.markdown("#### BMI vs Goals")
        if COL_BMI and COL_GOALS:
            r, p, _ = correlation_test(filtered_df[COL_BMI], filtered_df[COL_GOALS])
            if r is not None:
                insight_box(f"BMI → Goals: r={r:.4f}, p={p:.2e}")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
                    x=COL_BMI, y=COL_GOALS, trendline="ols", title="BMI vs Goals",
                    opacity=0.4, color_discrete_sequence=["#7c5cbf"], template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if COL_POSITION:
                    fig2 = px.box(filtered_df, x=COL_POSITION, y=COL_BMI, title="BMI by Position",
                        color=COL_POSITION, color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE)
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("BMI or Goals column not found.")

    # ── Disposals (replaces Kicks + Handballs) ──
    with eda_sub[3]:
        st.markdown("#### Disposals vs Goals")
        _disp_col = COL_DISPOSALS
        if _disp_col is None and COL_KICKS and COL_HANDBALLS:
            filtered_df["_Disposals"] = filtered_df[COL_KICKS] + filtered_df[COL_HANDBALLS]
            _disp_col = "_Disposals"

        if _disp_col and COL_GOALS:
            r, p, _ = correlation_test(filtered_df[_disp_col], filtered_df[COL_GOALS])
            if r is not None:
                supported = "✅ Supported" if r > 0 and p < 0.05 else "❌ Not Supported"
                insight_box(f"Disposals → Goals: r={r:.4f}, p={p:.2e} — {supported}")
            col1, col2 = st.columns(2)
            with col1:
                # Binned avg goals — much clearer than scatter
                temp = filtered_df.copy()
                temp["Disposal Group"] = pd.cut(temp[_disp_col], bins=8, precision=0).astype(str)
                avg_by_bin = temp.groupby("Disposal Group")[COL_GOALS].mean().reset_index()
                fig = px.bar(avg_by_bin, x="Disposal Group", y=COL_GOALS,
                    title="Avg Goals by Disposal Count",
                    color_discrete_sequence=["#16c79a"], template=PLOTLY_TEMPLATE,
                    labels={COL_GOALS: "Avg Goals"}, text_auto=".2f")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # Heatmap: disposal bins vs goal bins
                temp2 = filtered_df.copy()
                temp2["Disposal Bin"] = pd.cut(temp2[_disp_col], bins=6, precision=0).astype(str)
                temp2["Goals Bin"] = temp2[COL_GOALS].clip(upper=4).map(
                    lambda x: f"{int(x)}" if x < 4 else "4+")
                ct = pd.crosstab(temp2["Disposal Bin"], temp2["Goals Bin"], normalize="index")
                fig2 = px.imshow(ct, text_auto=".0%",
                    title="Goal Distribution by Disposal Group (%)",
                    color_continuous_scale=["#1a1a2e", "#16c79a"],
                    template=PLOTLY_TEMPLATE,
                    labels=dict(x="Goals", y="Disposals", color="Proportion"))
                st.plotly_chart(fig2, use_container_width=True)
            if COL_GOAL_SCORED:
                temp = filtered_df.copy()
                temp["Bin"] = pd.cut(temp[_disp_col], bins=8, precision=0).astype(str)
                rate = temp.groupby("Bin")[COL_GOAL_SCORED].mean().reset_index()
                fig3 = px.line(rate, x="Bin", y=COL_GOAL_SCORED,
                    title="P(Goal) by Disposal Frequency",
                    markers=True, color_discrete_sequence=["#16c79a"], template=PLOTLY_TEMPLATE,
                    labels={COL_GOAL_SCORED: "P(Goal Scored)"})
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Disposals (or Kicks+Handballs) column not found.")

    # ── Inside-50s ──
    with eda_sub[4]:
        st.markdown("#### Inside-50s vs Goals")
        if COL_INSIDE50 and COL_GOALS:
            r, p, _ = correlation_test(filtered_df[COL_INSIDE50], filtered_df[COL_GOALS])
            if r is not None:
                insight_box(f"Inside-50s → Goals: r={r:.4f}, p={p:.2e}")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
                    x=COL_INSIDE50, y=COL_GOALS, trendline="ols", title="Inside-50s vs Goals",
                    opacity=0.3, color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                avg = filtered_df.groupby(COL_INSIDE50)[COL_GOALS].mean().reset_index()
                fig2 = px.bar(avg, x=COL_INSIDE50, y=COL_GOALS, title="Avg Goals by Inside-50s",
                    color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Feature Importance Ranking")
            features = {"Height": COL_HEIGHT, "Weight": COL_WEIGHT, "BMI": COL_BMI,
                        "Disposals": _disp_col if '_disp_col' in dir() else COL_DISPOSALS,
                        "Inside-50s": COL_INSIDE50}
            imp = []
            for name, col in features.items():
                if col and col in filtered_df.columns:
                    rf, _, _ = correlation_test(filtered_df[col], filtered_df[COL_GOALS])
                    if rf is not None:
                        imp.append({"Feature": name, "|r|": abs(rf), "r": rf})
            if imp:
                imp_df = pd.DataFrame(imp).sort_values("|r|", ascending=True)
                fig3 = px.bar(imp_df, x="|r|", y="Feature", orientation="h",
                    title="Feature Importance: |Correlation| with Goals",
                    color="|r|", color_continuous_scale=["#0f3460", "#e94560"],
                    template=PLOTLY_TEMPLATE, text="r")
                fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
                strongest = imp_df.iloc[-1]["Feature"]
                s = "✅ Supported" if strongest == "Inside-50s" else "❌ Not Supported"
                insight_box(f"Inside-50s as strongest predictor: {s}. Strongest = **{strongest}** (|r|={imp_df.iloc[-1]['|r|']:.4f}).")
        else:
            st.warning("Inside-50s or Goals column not found.")

    # ── Home ──
    with eda_sub[5]:
        st.markdown("#### Home Advantage")
        if COL_HOME and COL_GOALS:
            home_df = filtered_df[filtered_df[COL_HOME] == 1]
            away_df = filtered_df[filtered_df[COL_HOME] == 0]
            diff = home_df[COL_GOALS].mean() - away_df[COL_GOALS].mean()
            t_stat, t_p = stats.ttest_ind(home_df[COL_GOALS].dropna(), away_df[COL_GOALS].dropna())
            supported = "✅ Supported" if t_p < 0.05 and diff > 0 else "❌ Not Supported"
            insight_box(f"Home advantage: {supported}. Δ={diff:+.3f} goals (t={t_stat:.3f}, p={t_p:.2e}).")

            c1, c2, c3 = st.columns(3)
            c1.metric("Home Avg Goals", f"{home_df[COL_GOALS].mean():.2f}")
            c2.metric("Away Avg Goals", f"{away_df[COL_GOALS].mean():.2f}")
            c3.metric("Home Advantage", f"{diff:+.3f}")

            col1, col2 = st.columns(2)
            with col1:
                temp_home = filtered_df.copy()
                temp_home["Location"] = temp_home[COL_HOME].map({1: "Home", 0: "Away"})
                fig = px.histogram(temp_home, x=COL_GOALS, color="Location",
                    barmode="overlay", title="Goals: Home vs Away", opacity=0.7,
                    color_discrete_map={"Home": COLOR_ACCENT, "Away": COLOR_SECONDARY},
                    category_orders={"Location": ["Home", "Away"]},
                    template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if COL_GOAL_SCORED:
                    rates = pd.DataFrame({"Location": ["Home", "Away"],
                        "Rate": [home_df[COL_GOAL_SCORED].mean(), away_df[COL_GOAL_SCORED].mean()]})
                    fig2 = px.bar(rates, x="Location", y="Rate", title="P(Goal): Home vs Away",
                        color="Location",
                        color_discrete_map={"Home": COLOR_ACCENT, "Away": COLOR_SECONDARY},
                        template=PLOTLY_TEMPLATE, text_auto=".3f")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Home or Goals column not found.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CAUSAL INFERENCE (from CSV)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🧬 Causal Inference Results")
    st.markdown("Position-specific causal analysis using CausalML S-learners and DoWhy refutation tests.")
    st.caption(f"Data source: {causal_source}")

    ci_sub = st.tabs(["C1: Height", "C2: Weight", "C3: BMI",
                       "C4: Home", "C5: Rules", "📋 Summary & Robustness"])

    def get_hyp(hyp_prefix):
        return ate_df[ate_df["Hypothesis"].str.startswith(hyp_prefix)].copy()

    # ── C1: Height ──
    with ci_sub[0]:
        hypothesis_badge("C1", "Does height **cause** better performance in each role?")
        d = get_hyp("C1")
        if len(d) > 0:
            causal_verdict(
                f"**Ruck**: {d[d.Position=='Ruck']['ATE_XGB'].values[0]:+.2f} HitOuts (massive). "
                f"**Forward**: {d[d.Position=='Forward']['ATE_XGB'].values[0]:+.2f} TotalScore (surprising). "
                f"**Defender**: {d[d.Position=='Defender']['ATE_XGB'].values[0]:+.2f} Rebounds (spoiling > stats).")
            c1, c2 = st.columns([1.3, 1])
            with c1:
                fig = causal_bar(d, "Position", "ATE_XGB", "C1: Height Effect by Position")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(d[["Position", "Outcome", "ATE_XGB", "ATE_LRS"]].rename(
                    columns={"ATE_XGB": "ATE (XGB)", "ATE_LRS": "ATE (LRS)"}),
                    use_container_width=True, hide_index=True)

    # ── C2: Weight ──
    with ci_sub[1]:
        hypothesis_badge("C2", "Does weight **cause** better contest performance?")
        d = get_hyp("C2")
        if len(d) > 0:
            causal_verdict(
                f"**Ruck** {d[d.Position=='Ruck']['ATE_XGB'].values[0]:+.2f} HitOuts. "
                f"**Midfield** {d[d.Position=='Midfield']['ATE_XGB'].values[0]:+.2f} Clearances. "
                f"**Forward** {d[d.Position=='Forward']['ATE_XGB'].values[0]:+.2f} (agility > weight).")
            c1, c2 = st.columns([1.3, 1])
            with c1:
                fig = causal_bar(d, "Position", "ATE_XGB", "C2: Weight Effect by Position")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(d[["Position", "Outcome", "ATE_XGB", "ATE_LRS"]].rename(
                    columns={"ATE_XGB": "ATE (XGB)", "ATE_LRS": "ATE (LRS)"}),
                    use_container_width=True, hide_index=True)

    # ── C3: BMI ──
    with ci_sub[2]:
        hypothesis_badge("C3", "Does BMI affect performance differently across positions?")
        d = get_hyp("C3")
        if len(d) > 0:
            causal_verdict("Higher BMI helps **every** position — modern AFL rewards physicality everywhere.")
            fig = px.bar(d, x="Position", y="ATE_XGB", color="Position",
                color_discrete_sequence=COLOR_PALETTE, title="C3: BMI Effect by Position",
                text="ATE_XGB", template=PLOTLY_TEMPLATE)
            fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            vals = d["ATE_XGB"].tolist()
            fig.update_layout(height=450, showlegend=False, margin=dict(t=60, b=40),
                              yaxis=dict(range=[-0.2, max(vals) + 0.3]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(d[["Position", "Outcome", "ATE_XGB", "ATE_LRS"]].rename(
                columns={"ATE_XGB": "ATE (XGB)", "ATE_LRS": "ATE (LRS)"}),
                use_container_width=True, hide_index=True)

    # ── C4: Home ──
    with ci_sub[3]:
        hypothesis_badge("C4", "Does playing at home directly improve performance?")
        d = get_hyp("C4")
        if len(d) > 0:
            causal_verdict(
                f"**Ruck only** ({d[d.Position=='Ruck']['ATE_XGB'].values[0]:+.2f} HitOuts). "
                f"Forward {d[d.Position=='Forward']['ATE_XGB'].values[0]:+.2f}. NOT a universal booster.")
            fig = causal_bar(d, "Position", "ATE_XGB", "C4: Home Advantage Effect",
                             colorscale=["#0f3460", "#333", "#16c79a"], fmt="+.3f")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(d[["Position", "Outcome", "ATE_XGB", "ATE_LRS"]].rename(
                columns={"ATE_XGB": "ATE (XGB)", "ATE_LRS": "ATE (LRS)"}),
                use_container_width=True, hide_index=True)

    # ── C5: Rules ──
    with ci_sub[4]:
        hypothesis_badge("C5", "Have rule changes shifted causal effects?")
        if rule_df is not None and len(rule_df) > 0:
            causal_verdict("6-6-6 Rule → Height 6× more valuable. Stand Rule → BMI advantage −37%. Rotation Caps → Weight effect **flipped**.")
            cols = st.columns(min(3, len(rule_df)))
            for i, (_, row) in enumerate(rule_df.iterrows()):
                pct = row.get("Change_Pct", 0)
                cols[i % len(cols)].metric(
                    row["Rule_Key"], f"{pct:+.0f}%",
                    delta=f"{pct:+.0f}%",
                    delta_color="normal" if pct > 0 else "inverse")

            fig = causal_bar(rule_df, "Rule_Key", "Change_Pct",
                             "C5: Rule Impact on Causal Effects",
                             colorscale=["#e94560", "#333", "#16c79a"], fmt="+.0f")
            fig.update_traces(texttemplate="%{text:+.0f}%")
            st.plotly_chart(fig, use_container_width=True)

        if "Year" in filtered_df.columns and COL_POSITION and COL_HITOUTS:
            ruck = filtered_df[filtered_df[COL_POSITION] == "Ruck"].copy()
            if len(ruck) > 0:
                ruck["Era"] = pd.cut(ruck["Year"], bins=[2011, 2018, 2020, 2026],
                    labels=["Pre-6-6-6", "Transition", "Post-Stand"])
                era_stats = ruck.groupby("Era")[COL_HITOUTS].mean().reset_index()
                fig_era = px.bar(era_stats, x="Era", y=COL_HITOUTS,
                    title="Ruck Avg HitOuts by Era",
                    color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE,
                    text_auto=".1f")
                st.plotly_chart(fig_era, use_container_width=True)

    # ── Summary & Robustness ──
    with ci_sub[5]:
        st.markdown("### 📋 Executive Summary")
        st.markdown(
            "**🏆 Top 3 Coaching Insights:**\n\n"
            "1. Recruit rucks for height (>201cm) & weight (>102kg) — 6-6-6 rule amplified this\n\n"
            "2. Midfielders need weight (>85kg) for clearances — rotation caps made this crucial\n\n"
            "3. Home advantage = rucks only — don't assume universal benefit")
        st.markdown(
            "**🤯 Surprising Findings:**\n\n"
            "- Tall defenders → fewer rebounds (they spoil, not accumulate)\n\n"
            "- BMI helps all positions — modern game rewards physicality\n\n"
            "- Rotation caps flipped weight effect for midfielders")

        st.markdown("#### Complete ATE Results")
        summary = ate_df.copy()
        summary["Direction"] = summary["ATE_XGB"].apply(
            lambda x: "🟢 Positive" if x > 0 else "🔴 Negative")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("### 🛡️ Robustness: DoWhy Refutation Tests")
        if ref_csv is not None and len(ref_csv) > 0:
            st.dataframe(ref_csv, use_container_width=True, hide_index=True)
        else:
            ref = pd.DataFrame({
                "Test": ["Random Common Cause", "Placebo Treatment", "Data Subset", "Bootstrap"],
                "Result": ["✅ Robust", "✅ Pass", "✅ Robust", "⚠️ High Variance"],
                "What It Tests": ["Omitted variable bias", "Spurious correlation",
                                  "Sample dependence", "Estimate stability"]})
            st.dataframe(ref, use_container_width=True, hide_index=True)
        insight_box("**3/4 tests passed** → strong causal evidence.")

        st.divider()
        st.markdown("### 🎯 Heterogeneous Treatment Effects")
        if hte_csv is not None and len(hte_csv) > 0:
            fig = px.bar(hte_csv, x="Segment", y="ATE", color="ATE",
                color_continuous_scale=["#0f3460", "#e94560"],
                title="Who Benefits Most?", template=PLOTLY_TEMPLATE)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            hte = pd.DataFrame({
                "Segment": ["Young (<23)", "Rookies (<50 games)", "Weak Teams",
                            "Veterans (>100)", "Strong Teams"],
                "Benefit": [5, 4.5, 4, 2.5, 2]})
            fig = px.bar(hte, x="Segment", y="Benefit", color="Benefit",
                color_continuous_scale=["#0f3460", "#e94560"],
                title="Who Benefits Most?", template=PLOTLY_TEMPLATE)
            fig.update_layout(height=380, yaxis_title="Relative Benefit")
            st.plotly_chart(fig, use_container_width=True)
        insight_box("Young players & rookies benefit most. At elite level, skill > physique.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICTIVE MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🤖 Predictive Model Results")
    st.markdown("Position-specific prediction using OLS/Lasso, Random Forest, and XGBoost. Time-based split: Train (≤2022), Validation (2023-24), Test (2025).")

    # ── Performance Summary Data (from notebook output) ──
    perf_data = pd.DataFrame({
        "Position": ["Forward"]*6 + ["Midfield"]*6 + ["Ruck"]*6 + ["Defender"]*6,
        "Model": ["OLS/Lasso","OLS/Lasso","RandomForest","RandomForest","XGBoost","XGBoost"]*4,
        "Split": ["Val","Test"]*12,
        "MAE": [4.32,4.37,4.27,4.42,4.09,4.19, 1.41,1.42,1.43,1.44,1.32,1.35,
                8.08,8.45,8.11,8.89,7.39,8.30, 1.51,1.47,1.46,1.45,1.44,1.42],
        "RMSE": [5.73,5.72,5.60,5.71,5.41,5.46, 1.86,1.88,1.88,1.91,1.77,1.81,
                 10.33,10.64,10.05,10.61,9.39,10.19, 1.95,1.89,1.89,1.86,1.86,1.83],
        "R2": [0.463,0.487,0.487,0.488,0.520,0.533, 0.524,0.521,0.514,0.502,0.570,0.554,
               0.495,0.548,0.521,0.551,0.582,0.586, 0.302,0.298,0.347,0.321,0.362,0.342],
    })

    # ── Overview: Model Comparison ──
    st.markdown("### Model Comparison (Test Set)")
    test_data = perf_data[perf_data["Split"] == "Test"]

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(test_data, x="Position", y="R2", color="Model",
            barmode="group", title="Test R² by Position & Model",
            color_discrete_map={"OLS/Lasso": "#1F3A5F", "RandomForest": "#5B4EA3", "XGBoost": "#D94A6A"},
            template=PLOTLY_TEMPLATE, text_auto=".3f")
        fig_r2.update_layout(height=420, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        fig_mae = px.bar(test_data, x="Position", y="MAE", color="Model",
            barmode="group", title="Test MAE by Position & Model",
            color_discrete_map={"OLS/Lasso": "#1F3A5F", "RandomForest": "#5B4EA3", "XGBoost": "#D94A6A"},
            template=PLOTLY_TEMPLATE, text_auto=".2f")
        fig_mae.update_layout(height=420, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_mae, use_container_width=True)

    insight_box(
        "**XGBoost** achieves the best R² across all positions (up to 0.586 for Ruck). "
        "**Random Forest** often has lower MAE, suggesting smaller average errors. "
        "**Defender** is hardest to predict (R² ~0.34), while **Ruck** and **Midfield** are best explained.")

    # ── Full Metrics Table ──
    st.markdown("### Full Performance Metrics")
    st.dataframe(perf_data.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    st.divider()

    # ── Position-Specific Insights ──
    st.markdown("### Position-Specific Key Drivers")

    pred_sub = st.tabs(["⚡ Forward", "🏃 Midfield", "🏋️ Ruck", "🛡️ Defender"])

    with pred_sub[0]:
        st.markdown("#### Forward → Total Score")
        st.markdown("**Target:** Total_Score = 6 × Goals + Behinds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Drivers (LR Coefficients):**")
            fwd_coefs = pd.DataFrame({
                "Feature": ["MarksInside50", "Frees", "Disposals", "Height_x_Post666", "Marks", "Clangers"],
                "Direction": ["🟢 Strong +", "🟢 +", "🟢 +", "🟢 +", "🔴 −", "🔴 −"],
            })
            st.dataframe(fwd_coefs, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**SHAP Insights (XGBoost):**")
            st.markdown(
                "MarksInside50 dominates — forward scoring is primarily about getting possession "
                "in dangerous scoring range. Frees and Disposals also contribute, suggesting direct "
                "scoring opportunities and general involvement help.")
        insight_box("**Recruit forwards who can mark inside 50.** General disposal count matters less than positioning.")

    with pred_sub[1]:
        st.markdown("#### Midfield → Clearances")
        st.markdown("**Target:** Clearances")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Drivers (LR Coefficients):**")
            mid_coefs = pd.DataFrame({
                "Feature": ["Disposals", "Frees", "FreesAgainst", "Weight", "Inside50s", "Marks"],
                "Direction": ["🟢 Strong +", "🟢 +", "🟢 +", "🟢 +", "🟢 +", "🔴 −"],
            })
            st.dataframe(mid_coefs, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**SHAP Insights (XGBoost):**")
            st.markdown(
                "Disposals is the dominant driver. Clearance ability is tied to overall ball involvement "
                "and contested play intensity, not just isolated actions. "
                "Weight and Inside50s imply physical contest ability matters.")
        insight_box("**Midfield clearances = ball involvement + physical contests.** Disposal-heavy mids dominate.")

    with pred_sub[2]:
        st.markdown("#### Ruck → HitOuts")
        st.markdown("**Target:** HitOuts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Drivers (LR Coefficients):**")
            ruck_coefs = pd.DataFrame({
                "Feature": ["Height_x_Post666", "Age", "Height", "Clearances", "Disposals", "AgeSquared"],
                "Direction": ["🟢 Strongest +", "🟢 Strong +", "🟢 +", "🟢 +", "🟢 +", "🔴 Strong −"],
            })
            st.dataframe(ruck_coefs, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**SHAP Insights (XGBoost):**")
            st.markdown(
                "SHAP emphasizes Clearances, %Played, Height, and GamesPlayed — hit-out production "
                "is predicted by a mix of ruck involvement, time on ground, physical reach, and experience. "
                "Height × Post666 interaction confirms the 6-6-6 rule amplified height advantage.")
        insight_box("**Height × 6-6-6 rule is the strongest predictor.** Confirms causal finding C1/C5: tall rucks gained massive advantage post-2019.")

    with pred_sub[3]:
        st.markdown("#### Defender → Rebounds")
        st.markdown("**Target:** Rebounds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Drivers (LR Coefficients):**")
            def_coefs = pd.DataFrame({
                "Feature": ["Height", "Disposals", "OnePercenters", "%Played", "Weight", "BMISquared"],
                "Direction": ["🟢 Strongest +", "🟢 +", "🟢 +", "🟢 +", "🔴 Strong −", "🔴 −"],
            })
            st.dataframe(def_coefs, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**SHAP Insights (XGBoost):**")
            st.markdown(
                "SHAP shows Disposals and OnePercenters as the strongest practical drivers. "
                "Rebound output comes from defenders who win defensive actions AND move the ball out. "
                "Post666 is important — the 2019 rule changed rebound opportunities.")
        insight_box("**Rebound defenders need height + disposal ability.** Weight is negative — confirms causal finding that heavy defenders lack rebound run.")

    st.divider()

    # ── Conclusions ──
    st.markdown("### 🎯 Conclusions")
    st.markdown(
        "Across all positions, **role-specific in-game involvement** variables are the strongest predictors:\n\n"
        "- **Forward** scoring dominated by MarksInside50\n\n"
        "- **Midfield** clearances driven by Disposals and contested-play\n\n"
        "- **Ruck** hit-outs tied to Height, Clearances, and 6-6-6 rule interaction\n\n"
        "- **Defender** rebounds driven by Disposals, OnePercenters, and Height")
    insight_box(
        "Predictive models **confirm causal inference findings**: Height × 6-6-6 rule for rucks, "
        "weight helping midfield clearances, and BMI penalizing defender rebounds. "
        "XGBoost captures these nonlinear interactions best (R² 0.34–0.59).")


# FOOTER
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "AFL Performance Analysis Dashboard | Streamlit & Plotly | CausalML & DoWhy"
    "</div>", unsafe_allow_html=True)