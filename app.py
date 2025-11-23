import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BRIIM Decision Dashboard", layout="wide")

# ===========================
# 1) Model coefficients (EDIT if you retrain)
# ===========================
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
    "RenewalCliff": (0.0, 1.0),
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

def decision_label(intent, months_to_close, stage, fbi):
    """
    Single simplified objective -> a decision for marketing/sales.
    """
    if intent >= 0.70 and months_to_close <= 2.0 and stage == "Decision":
        return "PRIORITIZE NOW", "High purchase likelihood, near-term close window."
    if intent >= 0.50 and months_to_close <= 4.0:
        if fbi >= 0.03:
            return "NURTURE WITH RISK-MITIGATION", "Intent is building but financial posture suggests higher governance friction."
        return "NURTURE (MID-FUNNEL)", "Likely to buy, but not fully decision-ready yet."
    return "DEPRIORITIZE / WATCHLIST", "Low near-term intent or early-stage behavior."

def normalize(value, min_v, max_v):
    if max_v == min_v:
        return 0.0
    return float((value - min_v) / (max_v - min_v))

def top_drivers(x):
    # simple local contribution proxy
    contribs = {k: abs(COEFS.get(k,0) * x.get(k,0)) for k in x.keys() if k in COEFS}
    return sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:4]

# ===========================
# 2) Header
# ===========================
st.title("BRIIM Decision Dashboard")
st.caption("Benford-based Financial Behavior (FBI) + Digital & CRM Intent → Real-time Purchase Decision")

# ===========================
# 3) Sidebar controls
# ===========================
st.sidebar.header("Input mode")
mode = st.sidebar.radio("Choose input mode", ["Manual inputs", "Upload Excel (Behavioral_Data)"])

st.sidebar.header("Decision thresholds")
intent_thr = st.sidebar.slider("High-intent threshold", 0.50, 0.90, 0.70, 0.01)
close_thr = st.sidebar.slider("Near-term close window (months)", 1.0, 6.0, 2.0, 0.5)

# ===========================
# 4) Manual mode
# ===========================
if mode == "Manual inputs":
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("Model Inputs (real-time)")
        FBI = st.slider("FBI (Financial Behavior Index)", *FEATURE_BOUNDS["FBI"], 0.02, 0.001)
        SearchSpike = st.slider("Search Spike", *FEATURE_BOUNDS["SearchSpike"], 0.40, 0.01)
        ContentDepth = st.slider("Content Depth Ratio", *FEATURE_BOUNDS["ContentDepth"], 0.50, 0.01)
        RFV = st.slider("RFV Score", *FEATURE_BOUNDS["RFV"], 0.55, 0.01)
        CommitteeRate = st.slider("Committee Expansion Rate", *FEATURE_BOUNDS["CommitteeRate"], 0.30, 0.01)
        Velocity = st.slider("Engagement Velocity", *FEATURE_BOUNDS["Velocity"], 0.50, 0.01)
        BudgetProximity = st.slider("Budget Cycle Proximity", *FEATURE_BOUNDS["BudgetProximity"], 0.60, 0.01)
        PastPurchaseValue = st.slider("Past Purchase Value (normalized)", *FEATURE_BOUNDS["PastPurchaseValue"], 0.35, 0.01)
        RenewalCliff = st.selectbox("Renewal Cliff?", [0,1])

        x = dict(
            FBI=FBI, SearchSpike=SearchSpike, ContentDepth=ContentDepth, RFV=RFV,
            CommitteeRate=CommitteeRate, Velocity=Velocity,
            BudgetProximity=BudgetProximity, PastPurchaseValue=PastPurchaseValue,
            RenewalCliff=float(RenewalCliff)
        )

        intent = compute_intent(x)
        close_months = compute_timing(x, intent)
        stage = classify_stage(intent, ContentDepth, CommitteeRate)
        decision, reason = decision_label(intent, close_months, stage, FBI)

    with right:
        st.subheader("Decision Summary")
        k1, k2, k3 = st.columns(3)
        k1.metric("Intent Score", f"{intent:.3f}")
        k2.metric("Months to Close", f"{close_months:.2f}")
        k3.metric("Buying Stage", stage)

        # Decision badge
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:12px;border:1px solid #e6e6e6;">
              <div style="font-size:20px;font-weight:700;">Decision: {decision}</div>
              <div style="font-size:14px;color:#555;margin-top:4px;">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Intent gauge (Plotly)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=intent,
            number={'valueformat':'.3f'},
            gauge={
                'axis': {'range': [0,1]},
                'steps': [
                    {'range':[0,0.35]},
                    {'range':[0.35,0.70]},
                    {'range':[0.70,1.0]}
                ],
                'threshold': {'line': {'width': 4}, 'value': intent_thr}
            },
            title={'text':'Purchase Intent Gauge'}
        ))
        gauge.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(gauge, use_container_width=True)

        # Driver contributions
        st.markdown("**Top intent drivers (local contributions)**")
        drivers = top_drivers(x)
        st.write(", ".join([f"{d[0]}" for d in drivers]))

    # ---------------------------
    # Diagnostics panel
    # ---------------------------
    st.divider()
    st.subheader("Signal Diagnostics")

    d1, d2 = st.columns([1,1], gap="large")

    with d1:
        # Radar of current feature levels
        feats = ["SearchSpike","ContentDepth","RFV","CommitteeRate","Velocity","BudgetProximity"]
        vals = [x[f] for f in feats]
        radar_df = pd.DataFrame({"feature": feats, "value": vals})
        fig_radar = px.line_polar(radar_df, r="value", theta="feature", line_close=True,
                                 title="Behavioral Signal Profile")
        fig_radar.update_traces(fill='toself')
        fig_radar.update_layout(height=360, margin=dict(l=30,r=30,t=60,b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with d2:
        # Waterfall-style contribution plot
        contrib_df = pd.DataFrame(drivers, columns=["Feature","Contribution"])
        fig_bar = px.bar(contrib_df, x="Contribution", y="Feature", orientation="h",
                         title="Top Feature Contributions to Intent")
        fig_bar.update_layout(height=360, margin=dict(l=30,r=30,t=60,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

# ===========================
# 5) Excel mode
# ===========================
else:
    st.subheader("Upload your BRIIM Excel")
    file = st.file_uploader("Upload .xlsx with sheet: Behavioral_Data", type=["xlsx"])

    if file:
        df = pd.read_excel(file, sheet_name="Behavioral_Data")

        required = ["Month","SearchSpike","ContentDepth","RFV","CommitteeRate",
                    "Velocity","BudgetProximity","PastPurchaseValue","RenewalCliff","FBI"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns in Behavioral_Data: {missing}")
        else:
            def row_intent(r):
                x = {k: float(r[k]) for k in required if k != "Month"}
                return compute_intent(x)

            df["IntentScore"] = df.apply(row_intent, axis=1)
            df["MonthsToClose"] = df.apply(lambda r: compute_timing(
                {k: float(r[k]) for k in required if k != "Month"}, r["IntentScore"]), axis=1)
            df["PredictedStage"] = df.apply(lambda r: classify_stage(
                r["IntentScore"], r["ContentDepth"], r["CommitteeRate"]), axis=1)

            # Final decision per month
            decs = df.apply(lambda r: decision_label(r["IntentScore"], r["MonthsToClose"], r["PredictedStage"], r["FBI"])[0], axis=1)
            df["Decision"] = decs

            # Summary KPIs
            top = df.loc[df["IntentScore"].idxmax()]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Intent", f"{df['IntentScore'].mean():.3f}")
            k2.metric("Peak Intent", f"{top['IntentScore']:.3f}")
            k3.metric("Peak Month", str(top["Month"]))
            k4.metric("Most Frequent Decision", df["Decision"].value_counts().idxmax())

            # Trend charts
            c1, c2 = st.columns([1.3,1], gap="large")

            with c1:
                fig_int = px.line(df, x="Month", y="IntentScore", markers=True,
                                  title="Intent Trajectory (Monthly)")
                fig_int.update_layout(height=340, margin=dict(l=20,r=20,t=50,b=20), yaxis_range=[0,1])
                st.plotly_chart(fig_int, use_container_width=True)

                fig_close = px.line(df, x="Month", y="MonthsToClose", markers=True,
                                    title="Expected Months-to-Close")
                fig_close.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_close, use_container_width=True)

            with c2:
                stage_counts = df["PredictedStage"].value_counts().reset_index()
                stage_counts.columns = ["Stage","Count"]
                fig_stage = px.bar(stage_counts, x="Stage", y="Count",
                                   title="Stage Mix")
                fig_stage.update_layout(height=340, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_stage, use_container_width=True)

                # FBI snapshot
                fig_fbi = px.line(df, x="Month", y="FBI", markers=True,
                                  title="Financial Behavior (FBI) Trend")
                fig_fbi.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_fbi, use_container_width=True)

            st.divider()
            st.subheader("Monthly Decisions Table")
            st.dataframe(df[["Month","IntentScore","MonthsToClose","PredictedStage","Decision"]], use_container_width=True)

    else:
        st.info("Upload your Excel to compute live outputs.")
