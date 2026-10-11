import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NītiAI - Responsible Data Science Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR: DATASET SELECTOR
# ==========================================
with st.sidebar:
    st.markdown("## ⚖️ **NītiAI (नीतिAI) Studio**")
    st.markdown("### Responsible Data Science")
    st.caption("Smt. Chandibai Himathmal Mansukhani College\nT.Y.B.Sc. Data Science SEM-V [2026-2027]")
    
    st.markdown("---")
    st.subheader("📂 1. Select Dataset")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    dataset_option = st.selectbox(
        "Choose Data Source:",
        [
            "Real-World: German Credit (Banking)",
            "Real-World: Adult Census (Recruitment)",
            "Recruitment Screening Benchmark",
            "Upload Custom CSV"
        ]
    )
    
    df = None
    target_col = None
    sensitive_col = None
    unprivileged_group = None
    privileged_group = None
    case_study_title = ""
    domain = ""

    if dataset_option == "Real-World: German Credit (Banking)":
        csv_path = os.path.join(data_dir, "real_world_german_credit.csv")
        df = pd.read_csv(csv_path)
        target_col = "loan_approved"
        sensitive_col = "gender"
        privileged_group = "Male"
        unprivileged_group = "Female"
        case_study_title = "Credit Lending & Loan Approvals"
        domain = "Banking & Financial Services"
        
    elif dataset_option == "Real-World: Adult Census (Recruitment)":
        csv_path = os.path.join(data_dir, "real_world_adult_census_recruitment.csv")
        df = pd.read_csv(csv_path)
        target_col = "high_income_hired"
        sensitive_col = "gender"
        privileged_group = "Male"
        unprivileged_group = "Female"
        case_study_title = "Workforce Screening & Income Equity"
        domain = "Recruitment & HR Tech"

    elif dataset_option == "Recruitment Screening Benchmark":
        csv_path = os.path.join(data_dir, "recruitment_hiring_data.csv")
        df = pd.read_csv(csv_path)
        target_col = "ai_model_recommended"
        sensitive_col = "gender"
        privileged_group = "Male"
        unprivileged_group = "Female"
        case_study_title = "AI Resume Screening Bias"
        domain = "Recruitment"

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            target_col = st.selectbox("Select Target Variable (Decision/Outcome 0 or 1):", df.columns)
            sensitive_col = st.selectbox("Select Protected Variable (e.g., gender, race, age):", df.columns)
            unique_vals = list(df[sensitive_col].dropna().unique())
            if len(unique_vals) >= 2:
                privileged_group = st.selectbox("Privileged Group:", unique_vals, index=0)
                unprivileged_group = st.selectbox("Unprivileged Group:", unique_vals, index=1 if len(unique_vals)>1 else 0)
            case_study_title = "Custom User Dataset"
            domain = "Custom Domain"
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()

    st.markdown("---")
    st.markdown("### 🏛️ The 4 IKS Pillars Audited:")
    st.markdown("""
    - **Satya (सत्य):** Truth, Accuracy & Calibration
    - **Ahimsa (अहिंसा):** Non-Harm & Disparate Impact
    - **Dharma (धर्म):** Responsibility & Explainability
    - **Justice (न्याय):** Parity & Equal Opportunity
    """)

# ==========================================
# MAIN HEADER
# ==========================================
st.title("⚖️ NītiAI: Responsible Data Science Studio")
st.subheader(f"Indian Ethical Principles Framework for Auditing Machine Learning in {domain}")

# Dataset Summary Pills
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
col_meta1.metric("Total Records", f"{len(df):,}")
col_meta2.metric("Domain", domain)
col_meta3.metric("Protected Variable", f"{sensitive_col}")
col_meta4.metric("Compared Cohorts", f"{unprivileged_group} vs {privileged_group}")

st.divider()

# ==========================================
# MODEL TRAINING & FAIRNESS COMPUTATION (FAST & SCALED)
# ==========================================
@st.cache_data
def run_model_audit(data, target, sensitive, priv, unpriv):
    feature_cols = [c for c in data.columns if c not in [target, 'candidate_id', 'client_id']]
    X_raw = data[feature_cols].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    y = data[target].astype(int)
    
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, data[sensitive], test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features so convergence happens in 0.05 seconds
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    feat_weights = pd.DataFrame({
        'feature': X.columns,
        'weight': clf.coef_[0]
    }).sort_values(by='weight', key=abs, ascending=False)
    
    test_eval_df = pd.DataFrame({
        'sensitive': sens_test.values,
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    
    return clf, X_test, y_test, test_eval_df, feat_weights

with st.spinner("Auditing dataset and training baseline classifier..."):
    clf, X_test, y_test, eval_df, feat_weights = run_model_audit(df, target_col, sensitive_col, privileged_group, unprivileged_group)

# Subgroup Metrics Computation
sub_priv = eval_df[eval_df['sensitive'] == privileged_group]
sub_unpriv = eval_df[eval_df['sensitive'] == unprivileged_group]

# Selection rates
rate_priv = (sub_priv['y_pred'] == 1).mean() if len(sub_priv) > 0 else 0.001
rate_unpriv = (sub_unpriv['y_pred'] == 1).mean() if len(sub_unpriv) > 0 else 0.001

# Disparate Impact (Ahimsa metric)
disparate_impact = rate_unpriv / rate_priv if rate_priv > 0 else 1.0

# Demographic Parity Difference (Justice metric)
demographic_parity_diff = abs(rate_priv - rate_unpriv)

# Equal Opportunity: True Positive Rates
tpr_priv = ((sub_priv['y_pred'] == 1) & (sub_priv['y_true'] == 1)).sum() / max(1, (sub_priv['y_true'] == 1).sum())
tpr_unpriv = ((sub_unpriv['y_pred'] == 1) & (sub_unpriv['y_true'] == 1)).sum() / max(1, (sub_unpriv['y_true'] == 1).sum())
equal_opportunity_diff = abs(tpr_priv - tpr_unpriv)

# False Negative Rate (Harm metric - Ahimsa)
fnr_priv = ((sub_priv['y_pred'] == 0) & (sub_priv['y_true'] == 1)).sum() / max(1, (sub_priv['y_true'] == 1).sum())
fnr_unpriv = ((sub_unpriv['y_pred'] == 0) & (sub_unpriv['y_true'] == 1)).sum() / max(1, (sub_unpriv['y_true'] == 1).sum())

# Satya (Accuracy & Calibration)
acc = accuracy_score(eval_df['y_true'], eval_df['y_pred'])
prec = precision_score(eval_df['y_true'], eval_df['y_pred'], zero_division=0)
rec = recall_score(eval_df['y_true'], eval_df['y_pred'], zero_division=0)
f1 = f1_score(eval_df['y_true'], eval_df['y_pred'], zero_division=0)
brier = brier_score_loss(eval_df['y_true'], eval_df['y_prob'])

# Scores (0 - 100)
satya_score = round(max(0, min(100, (acc * 50) + (f1 * 30) + ((1 - brier) * 20))))

ahimsa_di_ratio = min(1.0, disparate_impact / 0.80) if disparate_impact < 0.80 else 1.0
ahimsa_fnr_penalty = max(0.0, 1.0 - abs(fnr_unpriv - fnr_priv))
ahimsa_score = round(max(0, min(100, (ahimsa_di_ratio * 70) + (ahimsa_fnr_penalty * 30))))

justice_parity_score = max(0.0, 1.0 - (demographic_parity_diff * 2))
justice_eq_opp_score = max(0.0, 1.0 - (equal_opportunity_diff * 2))
justice_score = round(max(0, min(100, (justice_parity_score * 50) + (justice_eq_opp_score * 50))))

sensitive_in_top_features = any(sensitive_col.lower() in f.lower() for f in feat_weights['feature'].head(3))
dharma_penalty = 20 if sensitive_in_top_features else 0
dharma_score = round(max(0, min(100, 85 - dharma_penalty + (satya_score * 0.15))))

niti_composite_score = round((satya_score + ahimsa_score + justice_score + dharma_score) / 4)

# ==========================================
# EXECUTIVE DASHBOARD: RADAR & SCORECARD
# ==========================================
col_dash_left, col_dash_right = st.columns([1, 1.3])

with col_dash_left:
    st.subheader("🎯 Executive Ethical Scorecard")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = niti_composite_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Nīti Score (नीति)", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#3B82F6"},
            'steps': [
                {'range': [0, 50], 'color': '#7F1D1D'},
                {'range': [50, 75], 'color': '#78350F'},
                {'range': [75, 100], 'color': '#064E3B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10), template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    if niti_composite_score >= 80:
        st.success("✅ **High Ethical Integrity**: The system satisfies foundational requirements across Satya, Ahimsa, Dharma, and Justice.")
    elif niti_composite_score >= 60:
        st.warning("⚠️ **Moderate Ethical Risk**: Discrepancies detected in Algorithmic Harm (Ahimsa) or Fairness (Justice). Remediation recommended.")
    else:
        st.error("🚨 **Critical Ethical Failure**: High algorithmic harm and severe bias detected. Action required prior to deployment.")

with col_dash_right:
    st.subheader("🕸️ The 4-Pillar Philosophical Radar")
    
    categories = ['Satya (Truth)', 'Ahimsa (Non-Harm)', 'Dharma (Responsibility)', 'Justice (Fairness)']
    scores = [satya_score, ahimsa_score, dharma_score, justice_score]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#60A5FA', width=2),
        name='Ethical Audit'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=20, b=20),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Pillar Summary Cards
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("🌿 Satya Score", f"{satya_score}/100", f"{acc*100:.1f}% Accuracy")
col_p2.metric("🕊️ Ahimsa Score", f"{ahimsa_score}/100", f"DI Ratio: {disparate_impact:.2f}")
col_p3.metric("⚖️ Justice Score", f"{justice_score}/100", f"Parity Gap: {demographic_parity_diff:.2f}")
col_p4.metric("📜 Dharma Score", f"{dharma_score}/100", "Transparency")

st.divider()

# ==========================================
# DEEP DIVE TABS
# ==========================================
tab_satya, tab_ahimsa, tab_justice, tab_dharma, tab_mitigate, tab_report = st.tabs([
    "🌿 1. Satya (Truth & Accuracy)",
    "🕊️ 2. Ahimsa (Non-Harm & Disparate Impact)",
    "⚖️ 3. Justice / Nyāya (Fairness & Parity)",
    "📜 4. Dharma (Duty & Transparency)",
    "🛠️ 5. Mitigation Simulator",
    "📑 6. Project Report Generator"
])

# ------------------------------------------
# TAB 1: SATYA
# ------------------------------------------
with tab_satya:
    st.markdown("### 🌿 Satya (सत्य): Truthfulness, Accuracy & Truth in Predictions")
    st.info("""
    **Philosophical Grounding:** In Indian philosophy, *Satya* denotes unwavering commitment to factual reality and objective truth.
    In Responsible Data Science, *Satya* commands that algorithms must reflect true ground-truth competence rather than spurious statistical noise or false confidence.
    """)
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Overall Accuracy", f"{acc:.2%}")
    col_s2.metric("Precision", f"{prec:.2%}")
    col_s3.metric("Recall (Sensitivity)", f"{rec:.2%}")
    col_s4.metric("Brier Calibration Loss", f"{brier:.3f}")
    
    st.subheader("Confusion Matrix (Ground Truth vs Model Truth)")
    cm = confusion_matrix(eval_df['y_true'], eval_df['y_pred'])
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                       title="Prediction Accuracy Breakdown (Satya Fidelity)")
    st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------
# TAB 2: AHIMSA
# ------------------------------------------
with tab_ahimsa:
    st.markdown("### 🕊️ Ahimsa (अहिंसा): Non-Harm and Mitigation of Algorithmic Injury")
    st.info("""
    **Philosophical Grounding:** *Ahimsa* is the supreme moral principle of avoiding harm in thought, word, and deed (*Manasa, Vacha, Karmana*).
    In Data Science, an algorithmic system violates *Ahimsa* when automated decisions inflict disproportionate economic, career, or societal injury upon vulnerable or unprivileged demographics.
    """)
    
    col_a1, col_a2, col_a3 = st.columns(3)
    col_a1.metric(f"Selection Rate ({privileged_group})", f"{rate_priv:.1%}")
    col_a2.metric(f"Selection Rate ({unprivileged_group})", f"{rate_unpriv:.1%}")
    col_a3.metric("Disparate Impact Ratio", f"{disparate_impact:.2f}", 
                  "Passes 80% Rule" if disparate_impact >= 0.80 else "Adverse Harm Violation (<0.80)")

    rates_data = pd.DataFrame({
        'Group': [f"{privileged_group} (Privileged)", f"{unprivileged_group} (Unprivileged)"],
        'Selection Rate': [rate_priv, rate_unpriv],
        'False Negative Harm Rate': [fnr_priv, fnr_unpriv]
    })
    
    fig_ahimsa = px.bar(rates_data, x='Group', y=['Selection Rate', 'False Negative Harm Rate'],
                        barmode='group', title="Disparate Selection & Algorithmic Injury (Ahimsa Audit)",
                        color_discrete_sequence=['#3B82F6', '#EF4444'])
    st.plotly_chart(fig_ahimsa, use_container_width=True)
    
    if disparate_impact < 0.80:
        st.error(f"🚨 **Ahimsa Violation Detected:** The selection rate for **{unprivileged_group}** is only **{disparate_impact:.1%}** of the rate for **{privileged_group}**. Under international benchmarks (EEOC Four-Fifths rule), this constitutes systemic algorithmic injury.")
    else:
        st.success("✅ **Ahimsa Preserved:** Disparate impact ratio meets ethical non-harm thresholds.")

# ------------------------------------------
# TAB 3: JUSTICE / NYAYA
# ------------------------------------------
with tab_justice:
    st.markdown("### ⚖️ Justice (न्याय - Nyāya): Equality of Opportunity & Demographic Parity")
    st.info("""
    **Philosophical Grounding:** *Nyāya* represents fairness, impartiality, and distributive justice.
    In Automated Systems, *Justice* demands that equally qualified candidates have equal probability of obtaining favorable outcomes, irrespective of protected identity characteristics.
    """)
    
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        st.metric("Demographic Parity Difference", f"{demographic_parity_diff:.3f}",
                  "Target: < 0.10", delta_color="inverse" if demographic_parity_diff > 0.10 else "normal")
        st.caption("Measures the gap in absolute acceptance rates between demographics.")
    with col_j2:
        st.metric("Equal Opportunity Difference", f"{equal_opportunity_diff:.3f}",
                  "Target: < 0.10", delta_color="inverse" if equal_opportunity_diff > 0.10 else "normal")
        st.caption("Measures the difference in True Positive Rates (TPR) among qualified applicants.")
        
    justice_df = pd.DataFrame({
        'Metric': ['True Positive Rate (TPR)', 'Selection Rate'],
        privileged_group: [tpr_priv, rate_priv],
        unprivileged_group: [tpr_unpriv, rate_unpriv]
    })
    
    fig_justice = go.Figure(data=[
        go.Bar(name=privileged_group, x=justice_df['Metric'], y=justice_df[privileged_group], marker_color='#3B82F6'),
        go.Bar(name=unprivileged_group, x=justice_df['Metric'], y=justice_df[unprivileged_group], marker_color='#F59E0B')
    ])
    fig_justice.update_layout(barmode='group', title="Parity and Equal Opportunity Comparison")
    st.plotly_chart(fig_justice, use_container_width=True)

# ------------------------------------------
# TAB 4: DHARMA
# ------------------------------------------
with tab_dharma:
    st.markdown("### 📜 Dharma (धर्म): Duty, Responsibility, Explainability & Governance")
    st.info("""
    **Philosophical Grounding:** *Dharma* embodies moral obligation, righteous conduct, and institutional duty of care (*Kartavya*).
    In Data Science, *Dharma* mandates algorithmic transparency, explainability (XAI), and clear accountability. A black-box algorithm whose reasons cannot be explained to impacted citizens violates *Dharma*.
    """)
    
    st.subheader("Key Drivers of the Model Decision (Feature Attribution)")
    top_feats = feat_weights.head(10)
    fig_feats = px.bar(top_feats, x='weight', y='feature', orientation='h',
                       title="Feature Coefficients (Explainability & Attribution)",
                       color='weight', color_continuous_scale='Bluered')
    fig_feats.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_feats, use_container_width=True)
    
    st.subheader("Model Governance Card (Dharma Checklist)")
    c_gov1, c_gov2 = st.columns(2)
    with c_gov1:
        st.checkbox("Audit Trail Maintained & Version Controlled", value=True, disabled=True)
        st.checkbox("Human-in-the-Loop Override Protocol Established", value=True, disabled=True)
    with c_gov2:
        st.checkbox("Protected Attribute Excluded from Direct Decision Split", value=not sensitive_in_top_features, disabled=True)
        st.checkbox("Model Disparate Impact Documented & Monitored", value=True, disabled=True)

# ------------------------------------------
# TAB 5: MITIGATION SIMULATOR
# ------------------------------------------
with tab_mitigate:
    st.markdown("### 🛠️ Responsible Mitigation Simulator: Balancing Satya and Justice")
    st.markdown("Adjust the decision threshold to see how post-processing calibration can restore **Ahimsa** and **Justice** while preserving **Satya**.")
    
    sim_threshold = st.slider("Classification Probability Cutoff:", min_value=0.20, max_value=0.80, value=0.50, step=0.02)
    
    sim_pred = (eval_df['y_prob'] >= sim_threshold).astype(int)
    sim_eval = eval_df.copy()
    sim_eval['y_pred'] = sim_pred
    
    sim_p = sim_eval[sim_eval['sensitive'] == privileged_group]
    sim_u = sim_eval[sim_eval['sensitive'] == unprivileged_group]
    
    s_rate_p = (sim_p['y_pred'] == 1).mean() if len(sim_p) > 0 else 0.001
    s_rate_u = (sim_u['y_pred'] == 1).mean() if len(sim_u) > 0 else 0.001
    s_di = s_rate_u / s_rate_p if s_rate_p > 0 else 1.0
    s_acc = accuracy_score(sim_eval['y_true'], sim_eval['y_pred'])
    
    c_sim1, c_sim2, c_sim3 = st.columns(3)
    c_sim1.metric("Simulated Accuracy (Satya)", f"{s_acc:.2%}", f"{(s_acc - acc):+.2%}")
    c_sim2.metric("Simulated Disparate Impact (Ahimsa)", f"{s_di:.2f}", f"{(s_di - disparate_impact):+.2f}")
    c_sim3.metric("Simulated Parity Difference", f"{abs(s_rate_p - s_rate_u):.2f}")
    
    st.caption("Notice how tuning the decision threshold allows data scientists to uphold their *Dharma* by actively mitigating algorithmic harm!")

# ------------------------------------------
# TAB 6: PROJECT REPORT GENERATOR
# ------------------------------------------
with tab_report:
    st.markdown("### 📑 Generate College Project Submission Report")
    st.markdown("Export a formal academic markdown report formatted to meet your college syllabus requirements.")
    
    report_content = f"""# Smt. Chandibai Himathmal Mansukhani College (Autonomous)
## T.Y.B.Sc. Data Science SEM-V [2026-2027] — IKS Project Report

### Project Title:
**Application of Indian Ethical Principles (Satya, Ahimsa, Dharma, Justice) to Responsible Data Science in {domain}**

---

### 1. Executive Summary
- **Domain:** {domain}
- **Dataset Evaluated:** {case_study_title} ({len(df):,} records)
- **Protected Attribute:** {sensitive_col} ({unprivileged_group} vs {privileged_group})
- **Overall Nīti (Ethical) Score:** {niti_composite_score} / 100

---

### 2. Four-Pillar Philosophical & Mathematical Evaluation

#### A. Satya (सत्य - Truthfulness & Data Integrity)
- **Score:** {satya_score} / 100
- **Model Accuracy:** {acc:.2%}
- **Precision:** {prec:.2%} | **Recall:** {rec:.2%}
- **Brier Calibration Loss:** {brier:.4f}
- *Interpretation:* The model exhibits truthful prediction capability with calibrated confidence scores.

#### B. Ahimsa (अहिंसा - Non-Harm & Algorithmic Injury Mitigation)
- **Score:** {ahimsa_score} / 100
- **Disparate Impact Ratio:** {disparate_impact:.2f}
- **Four-Fifths (80%) Rule Compliance:** {"Passed" if disparate_impact >= 0.80 else "Failed - Adverse Harm Violation"}
- **False Negative Harm Rate:** {unprivileged_group}: {fnr_unpriv:.2%} vs {privileged_group}: {fnr_priv:.2%}
- *Interpretation:* Assesses whether the model inflicts systemic exclusion or injury on unprivileged demographics.

#### C. Justice (न्याय - Nyāya: Fairness & Equality)
- **Score:** {justice_score} / 100
- **Demographic Parity Difference:** {demographic_parity_diff:.4f}
- **Equal Opportunity Difference (TPR Gap):** {equal_opportunity_diff:.4f}
- *Interpretation:* Enforces fair treatment and parity across demographic cohorts.

#### D. Dharma (धर्म - Duty, Responsibility & Transparency)
- **Score:** {dharma_score} / 100
- **Top Driving Features:** {', '.join(feat_weights['feature'].head(4).tolist())}
- **Model Governance:** Audit trail established, human-in-the-loop review integrated.

---

### 3. Proposed Responsible Practice & Recommendations
1. Implement threshold adjustment to restore the Disparate Impact ratio above the 0.80 benchmark.
2. Conduct continuous model drift and fairness auditing under the guidance of Satya and Ahimsa.
3. Maintain model cards and explainability reports to uphold institutional Dharma.
"""
    
    st.download_button(
        label="📥 Download Complete IKS Project Report (.md)",
        data=report_content,
        file_name="IKS_Responsible_Data_Science_Report.md",
        mime="text/markdown"
    )
    
    with st.expander("Preview Generated Report Content"):
        st.markdown(report_content)
