import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="BRIIM Real-Time Dashboard", layout="wide")

# ---------------------------
# 1) Coefficients (EDIT if you retrain)
# ---------------------------
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

# ---------------------------
# 2) Sidebar: data source choice
# ---------------------------
st.sidebar.title("Data Source")
mode = st.sidebar.radio("Choose input mode", ["Manual inputs", "Load from Excel (Behavioral_Data sheet)"])

st.title("BRIIM Real-Time Dashboard")
st.caption("Benford (FBI) + Behavioral Signals → Regression Intent & Timing → Stage → Role Actions")

# ---------------------------
# 3) Manual input UI
# ---------------------------
if mode == "Manual inputs":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Features")
        FBI = st.slider("FBI (Financial Behavior Index)", 0.0, 0.10, 0.02, 0.001)
        SearchSpike = st.slider("Search Spike", 0.0, 1.0, 0.40, 0.01)
        ContentDepth = st.slider("Content Depth Ratio", 0.0, 1.0, 0.50, 0.01)
        RFV = st.slider("RFV Score", 0.0, 1.0, 0.55, 0.01)
        CommitteeRate = st.slider("Committee Expansion Rate", 0.0, 1.0, 0.30, 0.01)
        Velocity = st.slider("Engagement Velocity", 0.0, 1.0, 0.50, 0.01)
        BudgetProximity = st.slider("Budget Cycle Proximity", 0.0, 1.0, 0.60, 0.01)
        PastPurchaseValue = st.slider("Past Purchase Value (normalized)", 0.0, 1.0, 0.35, 0.01)
        RenewalCliff = st.selectbox("Renewal Cliff?", [0, 1])

        x = dict(
            FBI=FBI, SearchSpike=SearchSpike, ContentDepth=ContentDepth, RFV=RFV,
            CommitteeRate=CommitteeRate, Velocity=Velocity,
            BudgetProximity=BudgetProximity, PastPurchaseValue=PastPurchaseValue,
            RenewalCliff=RenewalCliff
        )

        intent = compute_intent(x)
        close_months = compute_timing(x, intent)
        stage = classify_stage(intent, ContentDepth, CommitteeRate)

    with col2:
        st.subheader("Live Outputs")
        st.metric("Predicted Intent Score", f"{intent:.3f}")
        st.metric("Expected Months to Close", f"{close_months:.2f}")
        st.metric("Predicted Buying Stage", stage)

        st.write("Intent Gauge")
        fig, ax = plt.subplots()
        ax.bar(["Intent"], [intent])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# ---------------------------
# 4) Excel-driven mode
# ---------------------------
else:
    st.subheader("Upload your case Excel")
    file = st.file_uploader("Upload .xlsx with a 'Behavioral_Data' sheet", type=["xlsx"])

    if file:
        @st.cache_data(ttl=5)  # refresh every 5 seconds if re-run
        def load_data(f):
            return pd.read_excel(f, sheet_name="Behavioral_Data")

        df = load_data(file)

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
            df["Expected_Months_to_Close"] = df.apply(lambda r: compute_timing(
                {k: float(r[k]) for k in required if k != "Month"}, r["IntentScore"]), axis=1)
            df["PredictedStage"] = df.apply(lambda r: classify_stage(
                r["IntentScore"], r["ContentDepth"], r["CommitteeRate"]), axis=1)

            c1, c2 = st.columns([1,1])
            with c1:
                st.metric("Avg Intent Score", f"{df['IntentScore'].mean():.3f}")
                st.metric("Peak Intent", f"{df['IntentScore'].max():.3f}")
                st.metric("Peak Intent Month", df.loc[df["IntentScore"].idxmax(),"Month"])

            with c2:
                stage_counts = df["PredictedStage"].value_counts()
                st.write("Stage Mix")
                fig2, ax2 = plt.subplots()
                ax2.bar(stage_counts.index, stage_counts.values)
                st.pyplot(fig2)

            st.write("Monthly Outputs")
            st.dataframe(df[["Month","IntentScore","Expected_Months_to_Close","PredictedStage"]])

            st.write("Intent Trajectory")
            fig3, ax3 = plt.subplots()
            ax3.plot(df["Month"], df["IntentScore"], marker="o")
            ax3.set_xticklabels(df["Month"], rotation=45, ha="right")
            ax3.set_ylim(0,1)
            ax3.set_ylabel("Intent Score")
            st.pyplot(fig3)

            st.write("Benford / FBI Snapshot (from input)")
            fig4, ax4 = plt.subplots()
            ax4.plot(df["Month"], df["FBI"], marker="s")
            ax4.set_xticklabels(df["Month"], rotation=45, ha="right")
            ax4.set_ylabel("FBI (Trait)")
            st.pyplot(fig4)
    else:
        st.info("Upload your Excel to see live outputs.")
