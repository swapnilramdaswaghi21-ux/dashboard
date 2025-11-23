
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =======================================
# Page + Theme
# =======================================
st.set_page_config(
    page_title="BRIIM Decision Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Global */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: 0.2px;}
.small-muted {color:#6b7280; font-size:0.9rem;}
.section-title {font-size:1.2rem; font-weight:700; margin-bottom:0.5rem;}
.card {
  border:1px solid rgba(0,0,0,0.08);
  border-radius:14px;
  padding:14px 16px;
  background: rgba(255,255,255,0.7);
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-title {font-size:0.85rem; color:#6b7280; margin-bottom:4px;}
.metric-value {font-size:1.8rem; font-weight:750;}
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:0.9rem;
  border:1px solid rgba(0,0,0,0.12);
}
.badge-high {background:#ecfdf3;}
.badge-med  {background:#fff7ed;}
.badge-low  {background:#fef2f2;}
.hr {height:1px; background:rgba(0,0,0,0.08); margin: 12px 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =======================================
# 1) Model coefficients (EDIT if you retrain)
# =======================================
COEFS = {
    "Intercept": -1.2,
    "FBI": -2.5,
    "SearchSpike": 2.1,
    "ContentDepth": 1.8,
    "RFV": 0.9,
    "CommitteeRate": 1.5,
    "Velocity": 1.2,
    "PastPurchaseValue": 0.6,
    "BudgetProximity": 1.0,
    "RenewalCliff": 0.8,
}

TIMING = {
    "Intercept": 6.8,
    "FBI": 10.0,
    "Velocity": -4.0,
    "IntentScore": -5.5,
    "RenewalCliff": -1.5,
}

FEATURE_BOUNDS = {
    "FBI": (0.0, 0.10),
    "SearchSpike": (0.0, 1.0),
    "ContentDepth": (0.0, 1.0),
    "RFV": (0.0, 1.0),
    "CommitteeRate": (0.0, 1.0),
    "Velocity": (0.0, 1.0),
    "BudgetProximity": (0.0, 1.0),
    "PastPurchaseValue": (0.0, 1.0),
}

def logistic(z):
    return 1/(1+np.exp(-z))

def compute_intent(x):
    z = COEFS["Intercept"]
    for k, v in COEFS.items():
        if k != "Intercept":
            z += v * x[k]
    return float(logistic(z))

def compute_timing(x, intent):
    t = (TIMING["Intercept"]
         + TIMING["FBI"] * x["FBI"]
         + TIMING["Velocity"] * x["Velocity"]
         + TIMING["IntentScore"] * intent
         + TIMING["RenewalCliff"] * x["RenewalCliff"])
    return float(max(t, 0.5))

def classify_stage(intent, content_depth, committee_rate):
    if intent < 0.35 or content_depth < 0.30:
        return "Awareness"
    if intent < 0.70 or committee_rate < 0.40:
        return "Consideration"
    return "Decision"

def decision_label(intent, months_to_close, stage, fbi, intent_thr=0.70, close_thr=2.0):
    if intent >= intent_thr and months_to_close <= close_thr and stage == "Decision":
        return "PRIORITIZE NOW", "badge-high", "High purchase likelihood and near‑term close window."
    if intent >= 0.50 and months_to_close <= 4.0:
        if fbi >= 0.03:
            return "NURTURE + RISK MITIGATION", "badge-med", "Intent is building, but financial posture suggests higher governance friction."
        return "NURTURE (MID‑FUNNEL)", "badge-med", "Likely to buy, but committee isn’t fully decision‑ready."
    return "WATCHLIST / DEPRIORITIZE", "badge-low", "Low near‑term intent or early‑stage behavior."

def top_drivers(x):
    contribs = {k: abs(COEFS.get(k,0) * x.get(k,0)) for k in x.keys() if k in COEFS}
    return sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:5]

# =======================================
# 2) Sidebar
# =======================================
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Input mode", ["Manual inputs", "Upload Excel (Behavioral_Data)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Decision thresholds")
intent_thr = st.sidebar.slider("High‑intent threshold", 0.50, 0.90, 0.70, 0.01)
close_thr = st.sidebar.slider("Near‑term close window (months)", 1.0, 6.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: keep FBI small (0–0.05) unless strong anomalies are observed.")

# =======================================
# 3) Header
# =======================================
st.markdown("## BRIIM Decision Dashboard")
st.markdown('<div class="small-muted">Benford‑based Financial Behavior (FBI) + Digital & CRM Intent → Real‑time Purchase Decision</div>', unsafe_allow_html=True)
st.write("")

tab_decision, tab_diagnostics = st.tabs(["🧭 Decision View", "🔎 Diagnostics View"])

# =======================================
# 4) Manual Mode (Decision View)
# =======================================
def render_manual():
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Financial posture (slow‑moving trait)**")
            FBI = st.slider("FBI — Financial Behavior Index", *FEATURE_BOUNDS["FBI"], 0.02, 0.001,
                            help="Composite Benford deviation across IS/BS/CFS. Higher = more anomaly / governance friction.")
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            st.markdown("**Digital intent (fast‑moving signals)**")
            SearchSpike = st.slider("Search Spike", *FEATURE_BOUNDS["SearchSpike"], 0.40, 0.01,
                                    help="Relative burst in category/solution searches.")
            ContentDepth = st.slider("Content Depth Ratio", *FEATURE_BOUNDS["ContentDepth"], 0.50, 0.01,
                                     help="Share of mid‑/low‑funnel assets consumed.")
            RFV = st.slider("RFV Score", *FEATURE_BOUNDS["RFV"], 0.55, 0.01,
                            help="Recency‑Frequency‑Value summary of engagements.")
            CommitteeRate = st.slider("Committee Expansion Rate", *FEATURE_BOUNDS["CommitteeRate"], 0.30, 0.01,
                                      help="New stakeholders joining the evaluation per month.")
            Velocity = st.slider("Engagement Velocity", *FEATURE_BOUNDS["Velocity"], 0.50, 0.01,
                                 help="Acceleration in engagement intensity.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Context / advanced inputs", expanded=False):
            BudgetProximity = st.slider("Budget Cycle Proximity", *FEATURE_BOUNDS["BudgetProximity"], 0.60, 0.01,
                                        help="How close the account is to budget allocation windows.")
            PastPurchaseValue = st.slider("Past Purchase Value (normalized)", *FEATURE_BOUNDS["PastPurchaseValue"], 0.35, 0.01,
                                          help="Strength of supplier history with this account.")
            RenewalCliff = st.selectbox("Renewal Cliff?", [0,1],
                                        help="1 if renewal/contract cliff occurs in next 90 days.")

        x = dict(
            FBI=FBI, SearchSpike=SearchSpike, ContentDepth=ContentDepth, RFV=RFV,
            CommitteeRate=CommitteeRate, Velocity=Velocity,
            BudgetProximity=BudgetProximity, PastPurchaseValue=PastPurchaseValue,
            RenewalCliff=float(RenewalCliff)
        )
        intent = compute_intent(x)
        months_to_close = compute_timing(x, intent)
        stage = classify_stage(intent, ContentDepth, CommitteeRate)
        decision, badge_cls, reason = decision_label(intent, months_to_close, stage, FBI, intent_thr, close_thr)
        drivers = top_drivers(x)

    with right:
        st.markdown('<div class="section-title">Decision Summary</div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="card"><div class="metric-title">Intent Score</div>'
                        f'<div class="metric-value">{intent:.3f}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="card"><div class="metric-title">Months to Close</div>'
                        f'<div class="metric-value">{months_to_close:.2f}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="card"><div class="metric-title">Buying Stage</div>'
                        f'<div class="metric-value">{stage}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown(f'<div class="card"><div style="font-size:0.9rem;color:#6b7280;">Decision</div>'
                    f'<div class="badge {badge_cls}" style="margin-top:6px;">{decision}</div>'
                    f'<div style="margin-top:8px;font-size:0.95rem;">{reason}</div></div>', unsafe_allow_html=True)

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=intent,
            number={'valueformat':'.3f'},
            gauge={
                'axis': {'range': [0,1]},
                'steps': [{'range':[0,0.35]}, {'range':[0.35,0.70]}, {'range':[0.70,1.0]}],
                'threshold': {'line': {'width': 4}, 'value': intent_thr}
            },
            title={'text':'Purchase Intent Gauge'}
        ))
        gauge.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("**Top intent drivers (local)**")
        st.write(" → ".join([d[0] for d in drivers]))

    return x, intent, months_to_close, stage, drivers


# =======================================
# 5) Excel Mode (Decision View)
# =======================================
def render_excel():
    st.markdown('<div class="section-title">Upload BRIIM case workbook</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload .xlsx containing sheet: Behavioral_Data", type=["xlsx"])

    if not file:
        st.info("Upload your Excel to compute live outputs.")
        return None

    df = pd.read_excel(file, sheet_name="Behavioral_Data")

    required = ["Month","SearchSpike","ContentDepth","RFV","CommitteeRate",
                "Velocity","BudgetProximity","PastPurchaseValue","RenewalCliff","FBI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in Behavioral_Data: {missing}")
        return None

    def row_intent(r):
        x = {k: float(r[k]) for k in required if k != "Month"}
        return compute_intent(x)

    df["IntentScore"] = df.apply(row_intent, axis=1)
    df["MonthsToClose"] = df.apply(lambda r: compute_timing(
        {k: float(r[k]) for k in required if k != "Month"}, r["IntentScore"]), axis=1)
    df["PredictedStage"] = df.apply(lambda r: classify_stage(
        r["IntentScore"], r["ContentDepth"], r["CommitteeRate"]), axis=1)

    df["Decision"] = df.apply(lambda r: decision_label(
        r["IntentScore"], r["MonthsToClose"], r["PredictedStage"], r["FBI"], intent_thr, close_thr)[0], axis=1)

    top = df.loc[df["IntentScore"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Intent", f"{df['IntentScore'].mean():.3f}")
    k2.metric("Peak Intent", f"{top['IntentScore']:.3f}")
    k3.metric("Peak Month", str(top["Month"]))
    k4.metric("Most Frequent Decision", df["Decision"].value_counts().idxmax())

    c1, c2 = st.columns([1.3,1], gap="large")
    with c1:
        fig_int = px.line(df, x="Month", y="IntentScore", markers=True, title="Intent Trajectory (Monthly)")
        fig_int.update_layout(height=320, margin=dict(l=10,r=10,t=45,b=10), yaxis_range=[0,1])
        st.plotly_chart(fig_int, use_container_width=True)

        fig_close = px.line(df, x="Month", y="MonthsToClose", markers=True, title="Expected Months‑to‑Close")
        fig_close.update_layout(height=300, margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_close, use_container_width=True)

    with c2:
        stage_counts = df["PredictedStage"].value_counts().reset_index()
        stage_counts.columns = ["Stage","Count"]
        fig_stage = px.bar(stage_counts, x="Stage", y="Count", title="Stage Mix")
        fig_stage.update_layout(height=320, margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_stage, use_container_width=True)

        fig_fbi = px.line(df, x="Month", y="FBI", markers=True, title="Financial Behavior (FBI) Trend")
        fig_fbi.update_layout(height=300, margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_fbi, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Monthly Decisions Table</div>', unsafe_allow_html=True)
    st.dataframe(df[["Month","IntentScore","MonthsToClose","PredictedStage","Decision"]], use_container_width=True)

    return df


# =======================================
# Render Tabs
# =======================================
with tab_decision:
    if mode == "Manual inputs":
        x, intent, months_to_close, stage, drivers = render_manual()
    else:
        df = render_excel()

with tab_diagnostics:
    st.markdown('<div class="section-title">Signal Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">These views explain *why* the dashboard made its decision.</div>',
                unsafe_allow_html=True)

    if mode == "Manual inputs":
        if "x" not in locals():
            st.info("Go to Decision View and set inputs first.")
        else:
            d1, d2 = st.columns([1,1], gap="large")
            with d1:
                feats = ["SearchSpike","ContentDepth","RFV","CommitteeRate","Velocity","BudgetProximity"]
                radar_df = pd.DataFrame({"feature": feats, "value": [x[f] for f in feats]})
                fig_radar = px.line_polar(radar_df, r="value", theta="feature", line_close=True,
                                         title="Behavioral Signal Profile")
                fig_radar.update_traces(fill='toself')
                fig_radar.update_layout(height=360, margin=dict(l=20,r=20,t=60,b=10))
                st.plotly_chart(fig_radar, use_container_width=True)

            with d2:
                contrib_df = pd.DataFrame(drivers, columns=["Feature","Contribution"])
                fig_bar = px.bar(contrib_df, x="Contribution", y="Feature", orientation="h",
                                 title="Top Feature Contributions to Intent")
                fig_bar.update_layout(height=360, margin=dict(l=20,r=20,t=60,b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            st.write("")
            st.markdown("**Interpretation tips**")
            st.write(
                "- Large spikes in SearchSpike/ContentDepth/Velocity typically indicate mid‑late funnel intent.\n"
                "- A higher FBI suggests more rigorous governance; pair with risk‑mitigation content.\n"
                "- CommitteeRate rising is a strong indicator of formal evaluation."
            )
    else:
        if "df" not in locals() or df is None:
            st.info("Upload Excel in Decision View first.")
        else:
            d1, d2 = st.columns([1.3,1], gap="large")

            with d1:
                cols = ["SearchSpike","ContentDepth","RFV","CommitteeRate","Velocity","BudgetProximity"]
                fig_heat = px.imshow(df[cols].T, aspect="auto", title="Monthly Signal Heatmap")
                fig_heat.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=10))
                st.plotly_chart(fig_heat, use_container_width=True)

            with d2:
                avg_feats = df[cols].mean().reset_index()
                avg_feats.columns = ["Feature","AvgValue"]
                fig_avg = px.bar(avg_feats, x="AvgValue", y="Feature", orientation="h",
                                 title="Average Signal Strength (Year)")
                fig_avg.update_layout(height=420, margin=dict(l=20,r=20,t=60,b=10))
                st.plotly_chart(fig_avg, use_container_width=True)
