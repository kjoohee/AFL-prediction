"""
AFL Performance Analysis Dashboard v3
======================================
4-tab: Overview → EDA → Causal Inference → Predictive Model
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
# CONFIG & PATHS
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "df_final_final.csv")
CAUSAL_DIR = os.path.join(PROJECT_ROOT, "Models", "causal_results")
PRED_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "reports", "tables", "predictive_model_summary.xlsx")

ATE_PATH = os.path.join(CAUSAL_DIR, "ate_results.csv")
RULE_PATH = os.path.join(CAUSAL_DIR, "rule_change_results.csv")
REFUTATION_PATH = os.path.join(CAUSAL_DIR, "refutation_results.csv")
HTE_PATH = os.path.join(CAUSAL_DIR, "hte_results.csv")

st.set_page_config(page_title="AFL Performance Dashboard", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Outfit', sans-serif; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #e94560; border-radius: 12px; padding: 16px; box-shadow: 0 4px 15px rgba(233, 69, 96, 0.15); }
    div[data-testid="stMetric"] label { color: #a8a8b3 !important; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e94560 !important; font-weight: 700; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    section[data-testid="stSidebar"] .stMarkdown h1 { color: #e94560; }
    section[data-testid="stSidebar"] * { color: #d1d1e0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }
    .hyp-badge { display: inline-block; background: linear-gradient(135deg, #e94560, #c23152); color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .insight-box { background: #1a1a2e; border-left: 4px solid #e94560; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; color: #d1d1e0; }
    .causal-verdict { background: #0f3460; border-left: 4px solid #16c79a; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; color: #d1d1e0; }
    .rule-timeline { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; margin: 4px 0; color: #d1d1e0; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_ACCENT = "#e94560"
COLOR_SECONDARY = "#0f3460"
COLOR_PALETTE = ["#e94560", "#0f3460", "#16c79a", "#f5a623", "#7c5cbf", "#00b4d8"]
POS_ORDER = ["Defender", "Forward", "Midfield", "Ruck"]
POS_COLOR_MAP = dict(zip(POS_ORDER, COLOR_PALETTE[:len(POS_ORDER)]))

def _bold(text): return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

def hypothesis_badge(num, text):
    st.markdown(f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;"><span class="hyp-badge">{num}</span><span style="font-weight:500; font-size:1rem;">{_bold(text)}</span></div>', unsafe_allow_html=True)

def hypothesis_result(status, reason):
    color = "#16c79a" if status == "Supported" else "#f5a623" if status == "Partially Supported" else "#e94560"
    icon = "✅" if status == "Supported" else "⚠️" if status == "Partially Supported" else "❌"
    st.markdown(f'<div style="margin-top:0px; margin-bottom:15px; padding:10px 15px; border-left:4px solid {color}; background-color:#1a1a2e; border-radius:0 5px 5px 0;"><span style="color:{color}; font-weight:bold; font-size:0.95rem;">{icon} {status}</span><br><span style="color:#d1d1e0; font-size:0.85rem;">{reason}</span></div>', unsafe_allow_html=True)

def insight_box(text): st.markdown(f'<div class="insight-box">💡 <strong>Insight:</strong> {_bold(text)}</div>', unsafe_allow_html=True)
def causal_verdict(text): st.markdown(f'<div class="causal-verdict">🧬 <strong>Causal Verdict:</strong> {_bold(text)}</div>', unsafe_allow_html=True)

def correlation_test(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 10: return None, None, "Insufficient data"
    r, p = stats.pearsonr(x[m], y[m])
    if p < 0.001: sig = "Highly significant (p < 0.001)"
    elif p < 0.01: sig = "Significant (p < 0.01)"
    elif p < 0.05: sig = "Significant (p < 0.05)"
    else: sig = "Not significant (p ≥ 0.05)"
    return r, p, sig

def safe_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at `{DATA_PATH}`.")
        st.stop()
    df.columns = df.columns.str.replace(".", "_", regex=False).str.replace(" ", "_")
    if "TotalScore" not in df.columns and "Goals" in df.columns and "Behinds" in df.columns: df["TotalScore"] = (6 * df["Goals"]) + df["Behinds"]
    if "Goal_Scored" not in df.columns and "Goals" in df.columns: df["Goal_Scored"] = (df["Goals"] > 0).astype(int)
    if "is_home" not in df.columns and "Team" in df.columns and "AwayTeam" in df.columns: df["is_home"] = (df["Team"] != df["AwayTeam"]).astype(int)
    if "BMI" not in df.columns and "Height" in df.columns and "Weight" in df.columns: df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    return df

@st.cache_data
def load_causal_csv(path, fallback=None):
    try:
        loaded = pd.read_csv(path)
        if loaded is None or loaded.empty: return fallback
        if len(loaded.columns) > 1 and str(loaded.iloc[0, 1]).strip().lower() in ["none", "nan"]: return fallback
        return loaded
    except Exception:
        return fallback

@st.cache_data
def load_predictive_excel(path, fallback_perf):
    try:
        df_ex = pd.read_excel(path)
        perf = df_ex.dropna(subset=["Split"]).copy()
        feat = df_ex.dropna(subset=["Feature_Set"]).copy()
        return perf, feat
    except Exception:
        return fallback_perf, None

df = load_data()

# Fallback Causal Data
_fb_ate = pd.DataFrame({"Hypothesis": ["C1_Height"]*4 + ["C2_Weight"]*4 + ["C3_BMI_primary"]*4 + ["C4_Home"]*4, "Position": ["Forward","Midfield","Ruck","Defender"]*4, "Outcome": ["TotalScore","Clearances","HitOuts","Rebounds"]*4, "ATE_XGB": [-0.307, 0.060, 4.842, -0.577,  -0.158, 0.675, 4.693, -0.303, 0.182, 0.687, 1.418, 0.102,  -0.051, -0.021, 0.322, -0.068], "ATE_LRS": [-0.261, 0.047, 4.770, -0.526,  -0.281, 0.591, 4.184, -0.303, 0.182, 0.687, 1.418, 0.102,  -0.051, -0.021, 0.322, -0.068]})
_fb_rule = pd.DataFrame({"Rule_Key": ["Ruck_666", "Midfield_Stand", "Midfield_RotCaps"], "Pre_ATE": [0.827, 0.775, 0.818], "Post_ATE": [8.236, 0.424, 0.101], "Change_Pct": [896.0, -45.3, -87.6]})
_fb_ref = pd.DataFrame({"Test": ["Random Common Cause", "Placebo Treatment", "Data Subset (50%)", "Bootstrap"], "Result": ["✅ Robust (Δ=0.000)", "✅ Pass (placebo≈0)", "✅ Stable (Δ<0.001)", "⚠️ High Variance"]})
_fb_hte = pd.DataFrame({"Segment": ["Young (<23)", "Rookies (<50)", "Weak Teams", "Prime (23-28)", "Veterans (>150)"], "Benefit": [7.291, 5.993, 5.442, 6.504, -2.521]})

ate_df = load_causal_csv(ATE_PATH, fallback=_fb_ate)
rule_df = load_causal_csv(RULE_PATH, fallback=_fb_rule)
ref_csv = load_causal_csv(REFUTATION_PATH, fallback=_fb_ref)
hte_csv = load_causal_csv(HTE_PATH, fallback=_fb_hte)

# Fallback Predictive Data
_fb_perf = pd.DataFrame({
    "Position": ["Forward"]*6 + ["Midfield"]*6 + ["Ruck"]*6 + ["Defender"]*6,
    "Model": ["OLS/Lasso","OLS/Lasso","RandomForest","RandomForest","XGBoost","XGBoost"]*4,
    "Split": ["Validation","Test"]*12,
    "MAE": [4.32,4.37,4.27,4.42,4.09,4.19, 1.41,1.42,1.43,1.44,1.32,1.35, 8.08,8.45,8.11,8.89,7.39,8.30, 1.51,1.47,1.46,1.45,1.44,1.42],
    "R2": [0.463,0.487,0.487,0.488,0.520,0.533, 0.524,0.521,0.514,0.502,0.570,0.554, 0.495,0.548,0.521,0.551,0.582,0.586, 0.302,0.298,0.347,0.321,0.362,0.342],
})

perf_data, feat_data = load_predictive_excel(PRED_SUMMARY_PATH, _fb_perf)

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
        selected_years = st.slider("Season Range", min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))
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
    st.markdown("### 📜 AFL Rule Changes")
    st.markdown('<div class="rule-timeline"><b>2019</b> — 6-6-6 Starting Positions<br>Players locked in zones at centre bounces → more open space</div>', unsafe_allow_html=True)
    st.markdown('<div class="rule-timeline"><b>2021</b> — Stand Rule<br>Players must stand still when opponent marks → faster ball movement</div>', unsafe_allow_html=True)
    st.markdown('<div class="rule-timeline"><b>2016–21</b> — Rotation Caps<br>Interchange limits reduced 120 → 90 → 75 → lifted<br>Endurance & weight matter more with fewer rotations</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### [INSY674] Team 5")
    st.markdown("Faye Wu\n\nMonica Jang\n\nJoohee Kim\n\nRui Zhao\n\nJacob Featherstone")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏈 AFL Performance Analysis Dashboard")
st.markdown("Analysing causal effects of physical attributes on position-specific performance.")
st.divider()

if COL_GOALS and COL_GOAL_SCORED:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(filtered_df):,}")
    c2.metric("Avg Goals/Game", f"{filtered_df[COL_GOALS].mean():.2f}", help="Average goals per player per game record")
    c3.metric("Goal-Scoring Rate", f"{filtered_df[COL_GOAL_SCORED].mean()*100:.1f}%", help="% of records where at least 1 goal was scored")
    if COL_TOTAL_SCORE:
        c4.metric("Avg TotalScore", f"{filtered_df[COL_TOTAL_SCORE].mean():.1f}", help="TotalScore = 6 × Goals + Behinds")
    st.divider()

tabs = st.tabs(["📊 Overview", "📈 Exploratory Analysis", "🧬 Causal Inference", "🤖 Predictive Model"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 0 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## Dataset Overview")

    numeric_cols = [c for c in [COL_HEIGHT, COL_WEIGHT, COL_BMI, COL_DISPOSALS,
                                COL_INSIDE50, COL_GOALS, COL_CLEARANCES,
                                COL_HITOUTS, COL_REBOUNDS] if c]
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr()
        mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_display = corr_matrix.where(~mask_tri)
        fig_corr = px.imshow(corr_display, text_auto=".2f",
            color_continuous_scale=["#0f3460", "#1a1a2e", "#e94560"],
            title="Correlation Matrix", template=PLOTLY_TEMPLATE)
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

    eda_sub = st.tabs(["🏋️ Height", "⚖️ Weight", "📐 BMI", "🏃 Disposals", "🎯 Inside-50s", "🏠 Home"])

    def eda_physical_charts(feature_col, feature_name, custom_insight="", n_bins=6, accent="#e94560"):
        if custom_insight: insight_box(custom_insight)

        temp = filtered_df.copy()
        col_min, col_max = temp[feature_col].min(), temp[feature_col].max()
        edges = np.round(np.linspace(np.floor(col_min), np.ceil(col_max), n_bins + 1), 1)
        temp["Bin"] = pd.cut(temp[feature_col], bins=edges, include_lowest=True)
        bin_cats = list(temp["Bin"].cat.categories)
        bin_labels = [f"({b.left:.1f}, {b.right:.1f}]" for b in bin_cats]
        temp["Bin"] = temp["Bin"].astype(str).map({str(b): lbl for b, lbl in zip(bin_cats, bin_labels)})

        col1, col2, col3 = st.columns(3)
        with col1:
            avg = temp.groupby("Bin")[COL_GOALS].mean().reindex(bin_labels).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=avg["Bin"], y=avg[COL_GOALS], name="Avg Goals", marker_color=accent, text=avg[COL_GOALS].round(2), textposition="outside"))
            if COL_GOAL_SCORED:
                rate = temp.groupby("Bin")[COL_GOAL_SCORED].mean().reindex(bin_labels).reset_index()
                fig.add_trace(go.Scatter(x=rate["Bin"], y=rate[COL_GOAL_SCORED], name="P(Goal)", yaxis="y2", mode="lines+markers+text", line=dict(color="#f5a623", width=2), text=[f"{v:.1%}" for v in rate[COL_GOAL_SCORED]], textposition="top center"))
            fig.update_layout(title=f"Avg Goals & P(Goal)", template=PLOTLY_TEMPLATE, height=420, yaxis2=dict(overlaying="y", side="right", range=[0, 1]), legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=60))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(temp, x="Bin", y=COL_GOALS, title=f"Goals Distribution", color="Bin", color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE, category_orders={"Bin": bin_labels})
            fig2.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            if COL_POSITION:
                fig3 = px.box(filtered_df, x=COL_POSITION, y=feature_col, title=f"by Position", color=COL_POSITION, color_discrete_map=POS_COLOR_MAP, category_orders={COL_POSITION: POS_ORDER}, template=PLOTLY_TEMPLATE)
                fig3.update_layout(showlegend=False, height=420)
                st.plotly_chart(fig3, use_container_width=True)

    with eda_sub[0]:
        if COL_HEIGHT and COL_GOALS: eda_physical_charts(COL_HEIGHT, "Height", n_bins=6, accent=COLOR_ACCENT, custom_insight="This reflects the two distinct ways to score in the AFL: Tall players (Key Forwards/Rucks) dominate aerial contested marks, while smaller players (Small Forwards) excel at ground-level agility.")

    with eda_sub[1]:
        if COL_WEIGHT and COL_GOALS: eda_physical_charts(COL_WEIGHT, "Weight", n_bins=6, accent=COLOR_SECONDARY, custom_insight="Raw physical mass matters in the forward line. Heavier players can use their body weight to win physical contests.")

    with eda_sub[2]:
        if COL_BMI and COL_GOALS: eda_physical_charts(COL_BMI, "BMI", n_bins=6, accent="#7c5cbf", custom_insight="It’s not just about being tall or heavy; having a dense, muscular build (higher BMI) maximizes a player's ability to score effectively under pressure.")

    with eda_sub[3]:
        _disp_col = COL_DISPOSALS
        if _disp_col is None and COL_KICKS and COL_HANDBALLS:
            filtered_df["_Disposals"] = filtered_df[COL_KICKS] + filtered_df[COL_HANDBALLS]
            _disp_col = "_Disposals"
        if _disp_col and COL_GOALS and COL_POSITION:
            insight_box("In the AFL, being a \"ball magnet\" does not equate to scoreboard impact. A Forward's single touch inside the 50m arc is worth infinitely more than a Defender's 10 passes in the backline.")
            col1, col2 = st.columns(2)
            with col1:
                pos_disp = filtered_df.groupby(COL_POSITION)[_disp_col].mean().reset_index()
                pos_disp.columns = ["Position", "Avg Disposals"]
                fig = px.bar(pos_disp, x="Position", y="Avg Disposals", title="Avg Disposals by Position", text_auto=".1f", color="Position", color_discrete_map=POS_COLOR_MAP, category_orders={"Position": POS_ORDER}, template=PLOTLY_TEMPLATE)
                fig.update_layout(showlegend=False, height=420)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                temp = filtered_df.copy()
                d_edges = np.round(np.linspace(0, np.ceil(temp[_disp_col].max()), 6), 0)
                temp["Disposal Group"] = pd.cut(temp[_disp_col], bins=d_edges, include_lowest=True)
                dg_cats = list(temp["Disposal Group"].cat.categories)
                dg_labels = [f"({int(b.left)}, {int(b.right)}]" for b in dg_cats]
                dg_labels[0] = f"[{int(dg_cats[0].left)}, {int(dg_cats[0].right)}]"
                temp["Disposal Group"] = temp["Disposal Group"].astype(str).map({str(b): lbl for b, lbl in zip(dg_cats, dg_labels)})
                avg_pos = temp.groupby(["Disposal Group", COL_POSITION])[COL_GOALS].mean().reset_index()
                fig2 = px.bar(avg_pos, x="Disposal Group", y=COL_GOALS, color=COL_POSITION, barmode="group", title="Avg Goals by Disposal Group & Position", color_discrete_map=POS_COLOR_MAP, category_orders={"Disposal Group": dg_labels, COL_POSITION: POS_ORDER}, template=PLOTLY_TEMPLATE)
                fig2.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig2, use_container_width=True)

    with eda_sub[4]:
        if COL_INSIDE50 and COL_GOALS and COL_POSITION:
            insight_box("This perfectly captures team synergy. Players with 13+ Inside-50s are elite Midfielders acting as playmakers. They aren't kicking goals themselves; they feed the Forwards.")
            col1, col2 = st.columns(2)
            with col1:
                temp = filtered_df.copy()
                i50_edges = np.round(np.linspace(0, np.ceil(temp[COL_INSIDE50].max()), 7), 0)
                temp["I50 Group"] = pd.cut(temp[COL_INSIDE50], bins=i50_edges, include_lowest=True)
                i50_cats = list(temp["I50 Group"].cat.categories)
                i50_labels = [f"({int(b.left)}, {int(b.right)}]" for b in i50_cats]
                i50_labels[0] = f"[{int(i50_cats[0].left)}, {int(i50_cats[0].right)}]"
                temp["I50 Group"] = temp["I50 Group"].astype(str).map({str(b): lbl for b, lbl in zip(i50_cats, i50_labels)})
                avg = temp.groupby("I50 Group")[COL_GOALS].mean().reindex(i50_labels).reset_index()
                fig = px.bar(avg, x="I50 Group", y=COL_GOALS, title="Avg Goals by Inside-50 Group", color_discrete_sequence=[COLOR_ACCENT], template=PLOTLY_TEMPLATE, text_auto=".2f")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                pos_i50 = filtered_df.groupby(COL_POSITION)[COL_INSIDE50].mean().reset_index()
                pos_i50.columns = ["Position", "Avg Inside-50s"]
                fig2 = px.bar(pos_i50, x="Position", y="Avg Inside-50s", title="Avg Inside-50s by Position", text_auto=".1f", color="Position", color_discrete_map=POS_COLOR_MAP, category_orders={"Position": POS_ORDER}, template=PLOTLY_TEMPLATE)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    with eda_sub[5]:
        if COL_HOME and COL_GOALS and COL_GOAL_SCORED:
            home_df, away_df = filtered_df[filtered_df[COL_HOME] == 1], filtered_df[filtered_df[COL_HOME] == 0]
            insight_box("Home advantage is statistically significant but acts as a slight tailwind rather than a primary driver.")
            col1, col2 = st.columns(2)
            with col1:
                temp_home = filtered_df.copy()
                temp_home["Location"] = temp_home[COL_HOME].map({1: "Home", 0: "Away"})
                fig = px.histogram(
                    temp_home, x=COL_GOALS, color="Location", barmode="overlay", 
                    title="Goals: Home vs Away", 
                    color_discrete_map={"Home": COLOR_ACCENT, "Away": COLOR_SECONDARY}, 
                    category_orders={"Location": ["Home", "Away"]}, 
                    opacity=0.8,
                    template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                rates = pd.DataFrame({"Location": ["Home", "Away"], "Rate": [home_df[COL_GOAL_SCORED].mean(), away_df[COL_GOAL_SCORED].mean()]})
                fig2 = px.bar(rates, x="Location", y="Rate", title="P(Goal): Home vs Away", color="Location", color_discrete_map={"Home": COLOR_ACCENT, "Away": COLOR_SECONDARY}, template=PLOTLY_TEMPLATE, text_auto=".3f")
                st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CAUSAL INFERENCE 
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🧬 Causal Inference")
    st.markdown("CausalML analysis using S-Learners (LRS + XGBoost). Binary treatment via median split.")

    POS_CAUSAL_COLORS = {"Defender": "#e94560", "Forward": "#0f3460", "Midfield": "#16c79a", "Ruck": "#f5a623"}
    ci_sub = st.tabs(["H1: Height", "H2: Weight", "H3: BMI", "H4: Home", "H5: Rules", "📋 Summary"])

    def get_hyp(hyp_prefix): return ate_df[ate_df["Hypothesis"].str.startswith(hyp_prefix)].copy()

    def causal_pos_bar(d, title):
        fig = px.bar(d, x="Position", y="ATE_XGB", color="Position", color_discrete_map=POS_CAUSAL_COLORS, title=title, template=PLOTLY_TEMPLATE)
        fig.update_traces(texttemplate="%{y:+.3f}", textposition="outside", cliponaxis=False, textfont=dict(size=14))
        y_max, y_min = d["ATE_XGB"].max(), d["ATE_XGB"].min()
        rng = [y_min * 1.5 if y_min < 0 else 0, y_max * 1.5 if y_max > 0 else 0]
        if y_min == 0 and y_max == 0: rng = [-1, 1]
        fig.update_layout(height=450, showlegend=False, margin=dict(t=80, b=40, l=40, r=40), yaxis=dict(range=rng))
        return fig

    with ci_sub[0]:
        hypothesis_badge("H1", "Does height cause better performance?")
        hypothesis_result("Partially Supported", "Height provides a massive advantage for Rucks (+4.84), but surprisingly negative effects for Forwards/Defenders.")
        d = get_hyp("C1")
        if len(d) > 0:
            st.plotly_chart(causal_pos_bar(d, "H1: Height Effect (ATE)"), use_container_width=True)
            causal_verdict("**Ruck (+4.842)**: Tall rucks (>201cm) dominate — MASSIVE effect. **Forward (−0.307)**: Agility matters more.")

    with ci_sub[1]:
        hypothesis_badge("H2", "Does weight cause better contest performance?")
        hypothesis_result("Partially Supported", "Weight helps Rucks and Midfielders, but hurts Forwards and Defenders.")
        d = get_hyp("C2")
        if len(d) > 0:
            st.plotly_chart(causal_pos_bar(d, "H2: Weight Effect (ATE)"), use_container_width=True)
            causal_verdict("**Midfield (+0.675)**: Heavier mids win contested ball. **Defender (−0.303)**: Lighter defenders rebound better.")

    with ci_sub[2]:
        hypothesis_badge("H3", "Does BMI affect performance?")
        hypothesis_result("Supported", "Higher BMI benefits ALL positions, disproving the 'lean' athlete myth.")
        d = get_hyp("C3_BMI_primary")
        if len(d) > 0:
            st.plotly_chart(causal_pos_bar(d, "H3: BMI Effect (ATE)"), use_container_width=True)
            causal_verdict("**Ruck (+1.418)**: Body positioning dominates. **Midfield (+0.687)**: Muscle mass aids clearances.")

    with ci_sub[3]:
        hypothesis_badge("H4", "Does playing at home improve performance?")
        hypothesis_result("Partially Supported", "Home advantage is real ONLY for Rucks.")
        d = get_hyp("C4")
        if len(d) > 0:
            st.plotly_chart(causal_pos_bar(d, "H4: Home Advantage Effect (ATE)"), use_container_width=True)
            causal_verdict("**Ruck (+0.322)**: Familiar bounce rhythms matter. **Others (~0)**: No significant boost.")

    with ci_sub[4]:
        hypothesis_badge("H5", "Have AFL rule changes shifted causal effects?")
        hypothesis_result("Supported", "Rules drastically alter physical advantages.")
        r666 = rule_df[rule_df["Rule_Key"] == "Ruck_666"]
        if len(r666) > 0:
            era_data = pd.DataFrame({"Era": ["Pre-2019", "Post-2019"], "ATE": [r666.iloc[0]["Pre_ATE"], r666.iloc[0]["Post_ATE"]]})
            fig = px.bar(era_data, x="Era", y="ATE", color_discrete_sequence=["#f5a623"], template=PLOTLY_TEMPLATE, title="Ruck: Height → HitOuts by Era")
            fig.update_traces(texttemplate="%{y:+.3f}", textposition="outside", cliponaxis=False, textfont=dict(size=14))
            fig.update_layout(height=400, margin=dict(t=80, b=30), yaxis=dict(range=[0, max(era_data["ATE"]) * 1.5]))
            st.plotly_chart(fig, use_container_width=True)

    with ci_sub[5]:
        st.markdown("### 🛡️ Heterogeneous Treatment Effects (HTE) Analysis")
        st.markdown("*\"Who benefits most? Player segments reveal the story behind the averages\"*")

        st.divider()

        # 1. KEY TAKEAWAYS (Streamlit Info/Success Boxes)
        st.markdown("#### 🧠 Coaching Takeaways")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("**👴 Age is the Dominant Factor**\n\nPhysical attributes help young/prime players but consistently **hurt veterans** across every position.")
        with c2:
            st.info("**🌱 The Rookie Advantage**\n\nPlayers early in their career (<50 games) consistently show stronger positive effects from height and weight.")
        with c3:
            st.warning("**📉 Averages Lie**\n\nA tall young ruck gains **+7.2 hitouts**; a tall veteran loses **-5.0**. Recruit based on career cycle, not just specs.")

        st.divider()

        # 2. SEGMENT SUMMARY TABLE (The "Gap" Analysis)
        st.markdown("#### 📊 Segment Summary: Best vs Worst")
        gap_data = pd.DataFrame({
            "Position": ["Ruck", "Ruck", "Forward", "Forward", "Defender", "Midfield", "Midfield"],
            "Treatment": ["Height", "Weight", "Weight", "Height", "Weight", "Height", "Weight"],
            "Best Segment": ["Young (<23): +7.24", "Rookies (<50 games): +5.19", "Young (<23): +0.15", "Prime (23-28): +0.91", "Rookies (<50 games): -0.11", "Young (<23): +0.04", "Rookies (<50 games): +0.27"],
            "Worst Segment": ["Veterans (>150 games): -5.03", "Veterans (>150 games): -0.53", "Veterans (>150 games): -1.56", "Veterans (>28): -0.51", "Veterans (>28): -0.77", "Veterans (>28): -0.54", "Established (50-150 games): -0.11"],
            "Gap": [12.27, 5.72, 1.71, 1.42, 0.66, 0.58, 0.38]
        })
        st.dataframe(gap_data.style.background_gradient(subset=['Gap'], cmap='Reds'), use_container_width=True, hide_index=True)

        st.divider()

        # 3. DEEP DIVE METRICS BY POSITION (Using Expanders and Metrics for clean UI)
        st.markdown("#### 🔍 Detailed Deep Dive by Position")
        
        with st.expander("🏋️ Ruck (Massive Height & Weight Effects)", expanded=True):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("**Height → HitOuts (ATE: +1.05)**")
                r1, r2 = st.columns(2)
                r1.metric("Best: Young (<23)", "+7.24", "Massive Advantage")
                r2.metric("Worst: Veterans (>150 games)", "-5.03", "Liability", delta_color="inverse")
                st.caption("Height is a massive advantage for young rucks but a liability for veterans.")
            with r_col2:
                st.markdown("**Weight → HitOuts (ATE: +4.03)**")
                r3, r4 = st.columns(2)
                r3.metric("Best: Rookies (<50 games)", "+5.19", "Strong Positive")
                r4.metric("Worst: Veterans (>150 games)", "-0.53", "Negative", delta_color="inverse")
                st.caption("Weight helps rucks of all ages EXCEPT veterans.")

        with st.expander("🎯 Forward (Height vs Weight)"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown("**Height → TotalScore (ATE: +0.31)**")
                f1, f2 = st.columns(2)
                f1.metric("Best: Prime (23-28)", "+0.91", "Helps Prime")
                f2.metric("Worst: Veterans (>28)", "-0.51", "Hurts Veterans", delta_color="inverse")
                st.caption("Height helps prime-age forwards but hurts veterans.")
            with f_col2:
                st.markdown("**Weight → TotalScore (ATE: -0.29)**")
                f3, f4 = st.columns(2)
                f3.metric("Best: Young (<23)", "+0.15", "Only positive segment")
                f4.metric("Worst: Veterans (>150 games)", "-1.56", "Destroys performance", delta_color="inverse")
                st.caption("'Power forwards' thrive early but decline faster.")

        with st.expander("🏃 Midfield & 🛡️ Defender"):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("**Midfield: Weight → Clearances (ATE: +0.05)**")
                m1, m2 = st.columns(2)
                m1.metric("Best: Rookies (<50 games)", "+0.27", "Peak physical years")
                m2.metric("Worst: Established (50-150 games)", "-0.11", "Advantage fades", delta_color="inverse")
                st.caption("Weight helps young/rookie midfielders, but advantage disappears with age.")
            with m_col2:
                st.markdown("**Defender: Weight → Rebounds (ATE: -0.32)**")
                d1, d2 = st.columns(2)
                d1.metric("Best: Rookies (<50 games)", "-0.11", "Least negative")
                d2.metric("Worst: Veterans (>28)", "-0.77", "Hurts badly", delta_color="inverse")
                st.caption("Weight hurts all defenders. Veterans with bulk are most disadvantaged.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICTIVE MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🤖 Predictive Model Results")
    st.markdown("Position-specific prediction pipeline comparing interpretable linear models (OLS/Lasso) against non-linear tree models (Random Forest, XGBoost). Split is chronological: **Train (≤2022)**, **Validation (2023-24)**, **Test (2025)**.")
    
    st.markdown("### 🏆 Model Comparison (Test Set 2025)")
    test_data = perf_data[perf_data["Split"] == "Test"]
    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(test_data, x="Position", y="R2", color="Model", barmode="group", title="Test R² by Position & Model", color_discrete_map={"OLS/Lasso": "#1F3A5F", "RandomForest": "#5B4EA3", "XGBoost": "#D94A6A"}, template=PLOTLY_TEMPLATE)
        fig_r2.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
        fig_r2.update_layout(height=420, margin=dict(t=80, b=40), legend=dict(orientation="h", y=-0.15), yaxis=dict(range=[0, max(test_data["R2"])*1.5]))
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        fig_mae = px.bar(test_data, x="Position", y="MAE", color="Model", barmode="group", title="Test MAE by Position & Model", color_discrete_map={"OLS/Lasso": "#1F3A5F", "RandomForest": "#5B4EA3", "XGBoost": "#D94A6A"}, template=PLOTLY_TEMPLATE)
        fig_mae.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
        fig_mae.update_layout(height=420, margin=dict(t=80, b=40), legend=dict(orientation="h", y=-0.15), yaxis=dict(range=[0, max(test_data["MAE"])*1.5]))
        st.plotly_chart(fig_mae, use_container_width=True)

    insight_box("**XGBoost is the best overall model** for capturing variance (highest R²). Random Forest is competitive on MAE. Defender stats are the hardest to predict (R² ~0.34), while Ruck hit-outs are highly predictable (R² ~0.59).")
    st.divider()
    
    st.markdown("### 📋 Full Performance Metrics")
    
    # 🚀 FIX: Drop columns that are entirely 'None' or NaN so they don't clutter the UI
    perf_display = perf_data.replace("None", np.nan).dropna(axis=1, how='all')
    
    # Apply formatting only to columns that actually exist after dropping 'None' ones
    format_dict = {}
    for col in ["MAE", "RMSE", "R2"]:
        if col in perf_display.columns:
            format_dict[col] = "{:.3f}"
            
    st.dataframe(perf_display.style.format(format_dict), use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("### 🔍 Feature Importance & SHAP Insights")
    pred_sub = st.tabs(["🎯 Forward (Total Score)", "🏃 Midfield (Clearances)", "🏋️ Ruck (HitOuts)", "🛡️ Defender (Rebounds)"])
    
    for i, pos in enumerate(["Forward", "Midfield", "Ruck", "Defender"]):
        with pred_sub[i]:
            col_a, col_b = st.columns([1, 1.5])
            with col_a:
                st.markdown("#### 📂 Features Used")
                if feat_data is not None:
                    pos_feat = feat_data[feat_data["Position"] == pos].copy()
                    # 🚀 FIX: Drop 'None' columns here too (e.g., Target(y) if empty)
                    pos_feat_display = pos_feat.replace("None", np.nan).dropna(axis=1, how='all')
                    st.dataframe(pos_feat_display, use_container_width=True, hide_index=True)
                
                if pos == "Forward":
                    st.markdown("**Top Drivers (Linear):**\n1. 🟢 MarksInside50\n2. 🟢 Frees\n3. 🟢 Disposals\n4. 🔴 Clangers")
                elif pos == "Midfield":
                    st.markdown("**Top Drivers (Linear):**\n1. 🟢 Disposals\n2. 🟢 Frees / FreesAgainst\n3. 🟢 Weight")
                elif pos == "Ruck":
                    st.markdown("**Top Drivers (Linear):**\n1. 🟢 Height_x_Post666\n2. 🟢 Age\n3. 🟢 Height")
                elif pos == "Defender":
                    st.markdown("**Top Drivers (Linear):**\n1. 🟢 Disposals\n2. 🟢 OnePercenters\n3. 🟢 Height")

            with col_b:
                st.markdown("#### 🧠 XGBoost SHAP Insights")
                if pos == "Forward":
                    st.markdown("Forward scoring is completely dominated by **MarksInside50**. It's not just about touching the ball anywhere; it's about getting possession in the dangerous scoring zone.")
                    insight_box("Recruit forwards based on contested marking ability inside 50, not general disposal counts.")
                elif pos == "Midfield":
                    st.markdown("Midfield clearances are primarily driven by **Disposals** and **contested-play indicators**. Weight also appears as a strong positive driver for securing the ball in heavy traffic.")
                    insight_box("Clearance ability = High ball involvement + Physical robustness in contests.")
                elif pos == "Ruck":
                    st.markdown("Hit-outs are strongly tied to stoppage involvement, playing time, and physical profile. The interaction term (**Height × 6-6-6 rule**) is a massive predictor, confirming the causal inference findings.")
                    insight_box("Height became the ultimate weapon for Rucks after the 2019 rule change.")
                elif pos == "Defender":
                    st.markdown("Defender rebounds are best explained by **transition-defense activity** (Disposals, OnePercenters) rather than body profile alone. Heavy weight acts as a negative predictor for rebound generation.")
                    insight_box("Elite rebounding requires athletic agility to move the ball out, which heavy defenders lack.")
# FOOTER
st.divider()
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem;'>AFL Performance Analysis Dashboard | Team 5</div>", unsafe_allow_html=True)