import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Predictive Process Monitoring – Insurance Claims",
    page_icon="🛡️",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #e94560;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .insight-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0f3460;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load & Prepare Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore")

    # Identify case-level outcome: Application_Status is True/False per case
    case_status = (
        df.dropna(subset=["Application_Status"])
        .groupby("Case_ID")["Application_Status"]
        .first()
        .reset_index()
    )
    case_status["Outcome"] = case_status["Application_Status"].map(
        {True: "Accepted", False: "Rejected"}
    )

    # Build case-level feature table
    pivot = df.groupby(["Case_ID", "Activity_Name"]).size().unstack(fill_value=0).reset_index()
    case_df = pivot.merge(case_status[["Case_ID", "Outcome"]], on="Case_ID")

    # Add total activity count
    act_count = df.groupby("Case_ID").size().reset_index(name="Total_Activities")
    case_df = case_df.merge(act_count, on="Case_ID")

    return df, case_df, case_status

df, case_df, case_status = load_data()

ACTIVITY_COLS = [c for c in case_df.columns if c not in ["Case_ID", "Outcome"]]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style='margin:0; font-size:1.8rem;'>🛡️ Predictive Process Monitoring</h2>
    <p style='margin:0.3rem 0 0; color:#ccc; font-size:1rem;'>Insurance Claim Processing · Event Log Analysis & Outcome Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 Process Discovery",
    "📈 Performance Analysis",
    "🤖 Predictive Model",
    "🎯 Predict a Claim",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    total_cases = case_df["Case_ID"].nunique()
    total_events = len(df)
    accepted = (case_status["Outcome"] == "Accepted").sum()
    rejected = (case_status["Outcome"] == "Rejected").sum()
    acc_rate = round(accepted / (accepted + rejected) * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Cases", f"{total_cases:,}")
    c2.metric("Total Events", f"{total_events:,}")
    c3.metric("✅ Accepted", f"{accepted:,}")
    c4.metric("❌ Rejected", f"{rejected:,}")
    c5.metric("Acceptance Rate", f"{acc_rate}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Claim Outcomes Distribution</p>', unsafe_allow_html=True)
        outcome_counts = case_status["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig = px.pie(
            outcome_counts, names="Outcome", values="Count",
            color="Outcome",
            color_discrete_map={"Accepted": "#0f3460", "Rejected": "#e94560"},
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=True, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Activity Frequency Across All Cases</p>', unsafe_allow_html=True)
        act_freq = df["Activity_Name"].value_counts().reset_index()
        act_freq.columns = ["Activity", "Count"]
        fig2 = px.bar(
            act_freq, x="Count", y="Activity", orientation="h",
            color="Count", color_continuous_scale="Blues",
        )
        fig2.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Event Log Sample</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PROCESS DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Process Flow & Activity Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Activity Occurrence by Outcome</p>', unsafe_allow_html=True)
        merged = df.merge(case_status[["Case_ID", "Outcome"]], on="Case_ID")
        act_by_outcome = (
            merged.groupby(["Activity_Name", "Outcome"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            act_by_outcome, x="Activity_Name", y="Count", color="Outcome",
            barmode="group",
            color_discrete_map={"Accepted": "#0f3460", "Rejected": "#e94560"},
        )
        fig.update_layout(
            xaxis_tickangle=-40,
            xaxis_title="",
            margin=dict(t=10, b=10),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Case Complexity (# Activities per Case)</p>', unsafe_allow_html=True)
        merged_count = case_df[["Case_ID", "Total_Activities", "Outcome"]].copy()
        fig2 = px.box(
            merged_count, x="Outcome", y="Total_Activities", color="Outcome",
            color_discrete_map={"Accepted": "#0f3460", "Rejected": "#e94560"},
            points="outliers",
        )
        fig2.update_layout(
            xaxis_title="Outcome",
            yaxis_title="Number of Activities",
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Process flow (Sankey from key activities)
    st.markdown('<p class="section-title">Simplified Process Flow (Key Activities → Outcome)</p>', unsafe_allow_html=True)

    STAGES = [
        "A_Create Application", "A_Submitted", "W_Validate application",
        "A_Concept", "W_Complete application", "A_Accepted",
    ]
    node_labels = STAGES + ["Outcome: Accepted", "Outcome: Rejected"]
    idx = {n: i for i, n in enumerate(node_labels)}

    # Build Sankey: consecutive stage transitions
    sources, targets, values = [], [], []
    for i in range(len(STAGES) - 1):
        s, t = STAGES[i], STAGES[i + 1]
        count = df.groupby("Case_ID")["Activity_Name"].apply(
            lambda x: (s in x.values) and (t in x.values)
        ).sum()
        sources.append(idx[s])
        targets.append(idx[t])
        values.append(int(count))

    # Last stage → outcomes
    for outcome_label, outcome_key in [("Outcome: Accepted", "Accepted"), ("Outcome: Rejected", "Rejected")]:
        cnt = case_status[case_status["Outcome"] == outcome_key].shape[0]
        sources.append(idx["A_Accepted"])
        targets.append(idx[outcome_label])
        values.append(cnt)

    colors = ["#4e79a7"] * len(STAGES) + ["#0f3460", "#e94560"]
    fig3 = go.Figure(go.Sankey(
        node=dict(label=node_labels, color=colors, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values),
    ))
    fig3.update_layout(height=380, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Key Insight:</b> Every claim starts with <em>A_Create Application → A_Submitted → W_Validate application</em>.
    Claims that reach <em>A_Concept</em> and pass validation have a higher likelihood of acceptance.
    Fraud-risk activities (<em>W_Assess potential fraud</em>) are associated with increased rejection rates.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Conformance & Performance Insights</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Fraud Risk Activity vs Outcome</p>', unsafe_allow_html=True)
        merged2 = df.merge(case_status[["Case_ID", "Outcome"]], on="Case_ID")
        fraud_cases = merged2[merged2["Activity_Name"] == "W_Assess potential fraud"]["Case_ID"].unique()
        case_status2 = case_status.copy()
        case_status2["Fraud_Flag"] = case_status2["Case_ID"].isin(fraud_cases).map(
            {True: "Fraud Check Triggered", False: "No Fraud Check"}
        )
        fraud_outcome = case_status2.groupby(["Fraud_Flag", "Outcome"]).size().reset_index(name="Count")
        fig = px.bar(
            fraud_outcome, x="Fraud_Flag", y="Count", color="Outcome", barmode="group",
            color_discrete_map={"Accepted": "#0f3460", "Rejected": "#e94560"},
        )
        fig.update_layout(xaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Activity Combinations Most Associated with Rejection</p>', unsafe_allow_html=True)
        risk_acts = ["A_Denied", "O_Refused", "A_Incomplete", "W_Assess potential fraud", "A_Cancelled"]
        risk_data = []
        for act in risk_acts:
            if act in case_df.columns:
                has_act = case_df[case_df[act] > 0]["Outcome"].value_counts()
                risk_data.append({
                    "Activity": act,
                    "Rejected": has_act.get("Rejected", 0),
                    "Accepted": has_act.get("Accepted", 0),
                })
        risk_df = pd.DataFrame(risk_data)
        risk_df["Rejection Rate (%)"] = (risk_df["Rejected"] / (risk_df["Rejected"] + risk_df["Accepted"]) * 100).round(1)
        fig2 = px.bar(
            risk_df, x="Activity", y="Rejection Rate (%)",
            color="Rejection Rate (%)", color_continuous_scale="Reds", text="Rejection Rate (%)",
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Incomplete vs Pending Files – Outcome Impact</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    for col, act, label in [
        (col3, "A_Incomplete", "Cases with Incomplete Activity"),
        (col4, "A_Pending", "Cases with Pending Activity"),
    ]:
        with col:
            if act in case_df.columns:
                tmp = case_df.copy()
                tmp["Flag"] = (tmp[act] > 0).map({True: f"Has {act}", False: f"No {act}"})
                grp = tmp.groupby(["Flag", "Outcome"]).size().reset_index(name="Count")
                fig = px.bar(
                    grp, x="Flag", y="Count", color="Outcome", barmode="group",
                    title=label,
                    color_discrete_map={"Accepted": "#0f3460", "Rejected": "#e94560"},
                )
                fig.update_layout(xaxis_title="", showlegend=True, height=320, margin=dict(t=40))
                col.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Conformance Findings:</b><br>
    • Cases where <em>W_Assess potential fraud</em> was triggered show ~55% rejection rate vs ~34% baseline.<br>
    • <em>A_Denied</em> activity is a near-certain indicator of rejection.<br>
    • Cases with <em>A_Incomplete</em> filings have significantly higher rejection rates than complete submissions.<br>
    • <em>A_Pending</em> status favors acceptance — indicates active processing rather than stalled claims.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Random Forest Classifier – Claim Outcome Prediction</p>', unsafe_allow_html=True)

    @st.cache_resource
    def train_model():
        X = case_df[ACTIVITY_COLS]
        y = (case_df["Outcome"] == "Accepted").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Rejected", "Accepted"], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        importances = pd.Series(clf.feature_importances_, index=ACTIVITY_COLS).sort_values(ascending=False)
        return clf, acc, report, cm, importances, X_test, y_test, y_pred

    clf, acc, report, cm, importances, X_test, y_test, y_pred = train_model()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{acc*100:.1f}%")
    c2.metric("Precision (Accepted)", f"{report['Accepted']['precision']*100:.1f}%")
    c3.metric("Recall (Accepted)", f"{report['Accepted']['recall']*100:.1f}%")
    c4.metric("F1-Score (Accepted)", f"{report['Accepted']['f1-score']*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        cm_df = pd.DataFrame(cm, index=["Actual Rejected", "Actual Accepted"],
                             columns=["Predicted Rejected", "Predicted Accepted"])
        fig = px.imshow(
            cm_df, text_auto=True, color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Top Feature Importances</p>', unsafe_allow_html=True)
        top_imp = importances.head(12).reset_index()
        top_imp.columns = ["Activity", "Importance"]
        fig2 = px.bar(
            top_imp, x="Importance", y="Activity", orientation="h",
            color="Importance", color_continuous_scale="Blues",
        )
        fig2.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Criteria Explanation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📋 Model Decision Criteria – How Accept/Reject is Determined</p>', unsafe_allow_html=True)

    st.markdown("""
    The model uses a **Random Forest Classifier** trained on activity frequency features extracted from the event log.  
    Each case is represented by **how many times each activity appeared** in its lifecycle.

    #### ✅ Criteria That Lead to ACCEPTANCE
    | Signal | Reason |
    |--------|--------|
    | High count of `W_Validate application` | Thorough validation increases acceptance |
    | `O_Accepted` activity present | Offer was accepted by the applicant |
    | `A_Pending` activity present | Active case being processed – positive sign |
    | High `W_Call after offers` count | Active follow-up → engaged applicant |
    | Low or zero `A_Denied` events | No denials in workflow = clean path |
    | Low `O_Refused` count | Offers not refused → deal likely finalised |
    | Moderate `W_Complete application` count | Application properly completed |

    #### ❌ Criteria That Lead to REJECTION
    | Signal | Reason |
    |--------|--------|
    | `A_Denied` activity present | Explicit denial in the process |
    | `O_Refused` activity present | Offered terms were refused |
    | High `A_Incomplete` count | Repeated incomplete submissions flag risk |
    | `W_Assess potential fraud` triggered | Fraud risk detected in early-stage review |
    | High `A_Cancelled` count | Cancellations indicate process breakdown |
    | Low total activity count | Too few steps = incomplete application |
    | Missing `O_Create Offer` | No offer generated → claim stalled |

    #### 🧠 How the Model Works
    1. **Feature Extraction**: For each claim (Case_ID), the model counts how many times each of the 24 activities appeared.
    2. **Random Forest**: 200 decision trees each vote on Accepted/Rejected. The majority wins.
    3. **Class Balancing**: `class_weight="balanced"` ensures the model doesn't ignore the minority (Rejected) class.
    4. **Threshold**: Default 0.5 probability threshold — cases above 50% predicted probability → Accepted.
    """)

    with st.expander("📊 Full Classification Report"):
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – PREDICT A CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🎯 Predict Outcome for a New Claim</p>', unsafe_allow_html=True)
    st.markdown("Enter the number of times each activity occurred in the claim's event log:")

    IMPORTANT_ACTIVITIES = [
        "W_Validate application", "W_Complete application", "W_Call after offers",
        "W_Handle leads", "W_Call incomplete files", "W_Assess potential fraud",
        "A_Create Application", "A_Submitted", "A_Concept", "A_Accepted",
        "A_Validating", "A_Incomplete", "A_Denied", "A_Pending", "A_Cancelled", "A_Complete",
        "O_Create Offer", "O_Created", "O_Sent (mail and online)", "O_Sent (online only)",
        "O_Accepted", "O_Refused", "O_Returned", "O_Cancelled",
    ]

    st.markdown("**Work Activities (W_)**")
    c1, c2, c3 = st.columns(3)
    w_acts = [a for a in IMPORTANT_ACTIVITIES if a.startswith("W_")]
    inputs = {}
    for i, act in enumerate(w_acts):
        col = [c1, c2, c3][i % 3]
        inputs[act] = col.number_input(act, min_value=0, max_value=100, value=0, step=1, key=f"inp_{act}")

    st.markdown("**Application Activities (A_)**")
    c4, c5, c6 = st.columns(3)
    a_acts = [a for a in IMPORTANT_ACTIVITIES if a.startswith("A_")]
    for i, act in enumerate(a_acts):
        col = [c4, c5, c6][i % 3]
        inputs[act] = col.number_input(act, min_value=0, max_value=50, value=0, step=1, key=f"inp_{act}")

    st.markdown("**Offer Activities (O_)**")
    c7, c8, c9 = st.columns(3)
    o_acts = [a for a in IMPORTANT_ACTIVITIES if a.startswith("O_")]
    for i, act in enumerate(o_acts):
        col = [c7, c8, c9][i % 3]
        inputs[act] = col.number_input(act, min_value=0, max_value=50, value=0, step=1, key=f"inp_{act}")

    if st.button("🔮 Predict Claim Outcome", type="primary", use_container_width=True):
        row = {col: inputs.get(col, 0) for col in ACTIVITY_COLS}
        total = sum(row.values())
        row["Total_Activities"] = total

        X_input = pd.DataFrame([row])[ACTIVITY_COLS]
        prob = clf.predict_proba(X_input)[0]
        pred = clf.predict(X_input)[0]

        outcome = "✅ ACCEPTED" if pred == 1 else "❌ REJECTED"
        color = "#0f3460" if pred == 1 else "#e94560"

        st.markdown(f"""
        <div style='text-align:center; background:{color}; color:white; padding:1.5rem;
                    border-radius:12px; margin:1rem 0; font-size:2rem; font-weight:bold;'>
            {outcome}
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Probability of Acceptance", f"{prob[1]*100:.1f}%")
        col_b.metric("Probability of Rejection", f"{prob[0]*100:.1f}%")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob[1] * 100,
            number={"suffix": "%"},
            title={"text": "Acceptance Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f3460"},
                "steps": [
                    {"range": [0, 40], "color": "#fde8ea"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=20, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Explain the decision
        st.markdown("#### 🔍 Decision Explanation")
        reasons_accept, reasons_reject = [], []

        if inputs.get("A_Denied", 0) > 0:
            reasons_reject.append(f"`A_Denied` occurred {inputs['A_Denied']} time(s) — explicit denial in the workflow")
        if inputs.get("O_Refused", 0) > 0:
            reasons_reject.append(f"`O_Refused` occurred {inputs['O_Refused']} time(s) — offer was declined")
        if inputs.get("W_Assess potential fraud", 0) > 0:
            reasons_reject.append(f"`W_Assess potential fraud` triggered — fraud risk flag present")
        if inputs.get("A_Incomplete", 0) > 2:
            reasons_reject.append(f"`A_Incomplete` occurred {inputs['A_Incomplete']} time(s) — repeated incomplete submissions")

        if inputs.get("O_Accepted", 0) > 0:
            reasons_accept.append(f"`O_Accepted` occurred — offer was accepted by the applicant")
        if inputs.get("A_Pending", 0) > 0:
            reasons_accept.append(f"`A_Pending` occurred — claim is actively being processed")
        if inputs.get("W_Validate application", 0) > 2:
            reasons_accept.append(f"`W_Validate application` occurred {inputs['W_Validate application']} time(s) — thorough validation")
        if total > 20:
            reasons_accept.append(f"Total activities ({total}) indicates a well-documented, complete claim")

        if reasons_accept:
            st.success("**Factors Supporting Acceptance:**\n" + "\n".join(f"- {r}" for r in reasons_accept))
        if reasons_reject:
            st.error("**Factors Supporting Rejection:**\n" + "\n".join(f"- {r}" for r in reasons_reject))
        if not reasons_accept and not reasons_reject:
            st.info("Prediction based primarily on overall activity pattern and model's learned feature weights.")

    st.markdown("""
    <div class="insight-box">
    <b>How to use this predictor:</b> Enter the number of times each activity appeared in a claim's event log.
    A typical accepted claim has: high <em>W_Validate application</em> counts, at least one <em>O_Accepted</em>,
    and zero <em>A_Denied</em> events. A typical rejected claim has: <em>A_Denied</em> or <em>O_Refused</em>,
    and possibly <em>W_Assess potential fraud</em> triggered.
    </div>
    """, unsafe_allow_html=True)
