# app.py (Final, Adapted for Roche Validation Engineering Manager Role)

# --- IMPORTS (CLEANED & ORGANIZED) ---
from typing import Callable, Any, Tuple
import streamlit as st
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.arima.model import ARIMA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import shap

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Validation Engineering Command Center | Roche",
    page_icon="⚙️"
)

# --- UTILITY & HELPER FUNCTIONS ---
def render_manager_briefing(title: str, content: str, reg_refs: str, business_impact: str) -> None:
    with st.container(border=True):
        st.subheader(f"⚙️ {title}")
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}")
        st.warning(f"**Regulatory Mapping:** {reg_refs}")

def render_metric_card(title: str, description: str, viz_function: Callable[[str], Any], insight: str, reg_context: str, key: str = "") -> None:
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"*{description}*")
        st.warning(f"**Regulatory Context:** {reg_context}")
        viz_object = viz_function(key)
        if viz_object is not None:
            if isinstance(viz_object, plt.Figure):
                st.pyplot(viz_object)
            elif isinstance(viz_object, go.Figure):
                st.plotly_chart(viz_object, use_container_width=True)
            elif isinstance(viz_object, Styler):
                st.dataframe(viz_object, use_container_width=True, hide_index=True)
            elif isinstance(viz_object, pd.DataFrame):
                st.dataframe(viz_object, use_container_width=True, hide_index=True)
        st.success(f"**Actionable Insight:** {insight}")

# --- VISUALIZATION & DATA GENERATORS (ADAPTED FOR VALIDATION ENGINEERING) ---
def create_opex_dashboard(key: str) -> Tuple[go.Figure, go.Figure]:
    budget = 1_500_000
    actual = 1_250_000
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=actual, title={'text': "Annual Validation Dept. OpEx: Budget vs. Actual"},
        gauge={'axis': {'range': [None, budget]}, 'bar': {'color': "cornflowerblue"},
               'steps': [{'range': [0, budget * 0.9], 'color': 'lightgreen'}, {'range': [budget * 0.9, budget], 'color': 'lightyellow'}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': budget}}))
    months = pd.date_range(start="2023-01-01", periods=12, freq='ME').strftime('%b')
    monthly_budget = np.ones(12) * (budget / 12); rng = np.random.default_rng(42); actual_spend = monthly_budget + rng.normal(0, 15000, 12)
    df = pd.DataFrame({'Month': months, 'Budget': monthly_budget, 'Actual': actual_spend}); df['Variance'] = df['Budget'] - df['Actual']; df['Cumulative Variance'] = df['Variance'].cumsum()
    fig_burn = make_subplots(specs=[[{"secondary_y": True}]])
    fig_burn.add_trace(go.Bar(x=df['Month'], y=df['Actual'], name='Actual Spend', marker_color='cornflowerblue'), secondary_y=False)
    fig_burn.add_trace(go.Scatter(x=df['Month'], y=df['Budget'], name='Budget', mode='lines+markers', line=dict(color='black', dash='dash')), secondary_y=False)
    fig_burn.add_trace(go.Scatter(x=df['Month'], y=df['Cumulative Variance'], name='Cumulative Variance', line=dict(color='red')), secondary_y=True)
    fig_burn.update_layout(title_text='Monthly OpEx: Spend vs. Budget & Cumulative Variance'); fig_burn.update_yaxes(title_text="Monthly Spend ($)", secondary_y=False); fig_burn.update_yaxes(title_text="Cumulative Variance ($)", secondary_y=True, showgrid=False)
    return fig_gauge, fig_burn

def create_copq_modeler(key: str) -> go.Figure:
    st.subheader("Interactive Cost of Poor Quality (COPQ) Modeler")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Internal Failure Costs (Annualized)**")
        rework_rate = st.slider("Batch Rework Rate (%)", 0.5, 10.0, 3.5, 0.1, key=f"copq_rework_{key}")
        downtime_hours = st.number_input("Unplanned Equipment Downtime (Hours/Month)", value=40, key=f"copq_downtime_{key}")
        st.markdown("**External Failure Costs (Annualized)**")
        field_service_visits = st.number_input("Field Service Visits (per year)", value=50, key=f"copq_service_{key}")
        investigation_hours = st.number_input("Complaint Investigation Hours (per month)", value=80, key=f"copq_complaint_{key}")
    cost_per_rework_batch = 25000; cost_per_downtime_hr = 5000; cost_per_service_visit = 8000; cost_per_investigation_hr = 200; total_batches = 1000
    internal_rework = (rework_rate / 100) * total_batches * cost_per_rework_batch
    internal_downtime = downtime_hours * cost_per_downtime_hr * 12
    external_service = field_service_visits * cost_per_service_visit
    external_investigation = investigation_hours * cost_per_investigation_hr * 12
    total_copq = internal_rework + internal_downtime + external_service + external_investigation
    with col2:
        st.metric("Total Annualized COPQ", f"${total_copq:,.0f}", help="Rework + Downtime + Service + Investigations")
        st.info("This model quantifies the financial impact of failures that a robust Validation program aims to prevent.")
    rng = np.random.default_rng(42); val_spend = rng.uniform(500_000, 2_000_000, 20)
    copq = 5_000_000 - (1.8 * val_spend) + rng.normal(0, 300_000, 20)
    df_corr = pd.DataFrame({'Validation Investment ($)': val_spend, 'Post-Launch COPQ ($)': copq})
    fig_corr = px.scatter(df_corr, x='Validation Investment ($)', y='Post-Launch COPQ ($)', trendline='ols', trendline_color_override='red', title='AI-Modeled Impact of Validation Investment on COPQ')
    return fig_corr

def create_audit_dashboard(key: str) -> Styler:
    audit_data = {"Audit/Inspection": ["FDA QSR Inspection", "ISO 13485 Recertification", "Internal Audit Q2", "Notified Body Audit"], "Date": ["2023-11-15", "2023-08-20", "2023-06-10", "2023-03-05"], "Validation-related Findings": [1, 0, 1, 2], "Outcome": ["Passed w/ Minor Obs.", "NAI (No Action Indicated)", "Passed", "Passed w/ Minor Obs."]}
    df = pd.DataFrame(audit_data)
    def style_outcome(val: str) -> str:
        color = 'lightgreen' if "NAI" in val or val == "Passed" else 'lightyellow' if "Minor" in val else 'white'
        return f'background-color: {color}'
    return df.style.map(style_outcome, subset=['Outcome'])

def create_qms_kanban(key: str) -> None:
    tasks = {"New Change Control": ["ECR-1091: New Filler Pump Install"], "Validation Planning": ["ECR-1088: Vision System Upgrade"], "Protocol Execution (IQ/OQ)": ["ECR-1085: Robotic Arm Replacement"], "Execution (PQ)": ["ECR-1082: New Lyophilizer Unit"], "Final Report & Closure": ["ECR-1077: Packaging Sealer Firmware Update"]}
    st.subheader("Validation Tasks in the Quality System")
    cols = st.columns(len(tasks));
    for i, (status, items) in enumerate(tasks.items()):
        with cols[i]:
            st.markdown(f"**{status}**")
            for item in items: st.info(item)

def create_tech_transfer_dashboard(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    metrics = ['Process Yield (%)', 'Key Impurity A (%)', 'Cycle Time (Hours)']
    site_a = [92.5, 0.45, 48]; site_b = [91.8, 0.51, 52]
    df_metrics = pd.DataFrame({'Metric': metrics, 'San Diego (Source Site)': site_a, 'Singapore (Receiving Site)': site_b})
    fig_bar = px.bar(df_metrics, x='Metric', y=['San Diego (Source Site)', 'Singapore (Receiving Site)'], barmode='group', title='Process Performance: Inter-site Comparability')
    transfer_status = {"Phase": ["Process Characterization", "Engineering Run", "Validation Batch 1 (PQ)", "Validation Batch 2 (PQ)", "Validation Batch 3 (PQ)"], "Status": ["Complete", "Complete", "Executing", "Pending Start", "Pending Start"]}
    df_status = pd.DataFrame(transfer_status)
    return fig_bar, df_status

def create_pipeline_advisor(key: str) -> go.Figure:
    rng = np.random.default_rng(42)
    historical_data = pd.DataFrame({'New_Automation_Modules': rng.integers(1, 10, 20), 'Process_Complexity_Score': rng.integers(1, 11, 20), 'URS_Count': rng.integers(20, 100, 20), 'Validation_Duration_Weeks': rng.uniform(8, 52, 20)})
    feature_names = ['New_Automation_Modules', 'Process_Complexity_Score', 'URS_Count']
    X = historical_data[feature_names]; y = historical_data['Validation_Duration_Weeks']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    st.subheader("Forecast Validation Timelines for New Capital Projects")# app.py (Final, Hyper-Focused on Automated Equipment Validation for Roche/Genentech with Design Controls)

# --- IMPORTS ---
from typing import Callable, Any, Tuple
import streamlit as st
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Automated Equipment Validation Portfolio | Roche",
    page_icon="🤖"
)

# --- UTILITY & HELPER FUNCTIONS ---
def render_manager_briefing(title: str, content: str, reg_refs: str, business_impact: str) -> None:
    with st.container(border=True):
        st.subheader(f"🤖 {title}")
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}")
        st.warning(f"**Key Standards & Regulations:** {reg_refs}")

# --- VISUALIZATION & DATA GENERATORS (Hyper-Focused on Automated Equipment) ---
def plot_cpk_analysis(key: str) -> go.Figure:
    rng = np.random.default_rng(42); data = rng.normal(loc=5.2, scale=0.25, size=200)
    usl = st.slider("Upper Specification Limit (USL) for Titer (g/L)", 5.0, 6.5, 6.0, 0.1, key=f"usl_{key}")
    lsl = st.slider("Lower Specification Limit (LSL) for Titer (g/L)", 4.0, 5.5, 4.5, 0.1, key=f"lsl_{key}")
    mean = np.mean(data); std_dev = np.std(data, ddof=1); cpk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
    fig = px.histogram(data, nbins=30, title=f"Process Capability (CpK) Analysis for Product Titer | Calculated CpK = {cpk:.2f}")
    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL"); fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL"); fig.add_vline(x=mean, line_dash="dot", line_color="blue", annotation_text="Process Mean")
    return fig

def create_rtm_data_editor(key: str) -> None:
    df_data = [
        {"ID": "URS-001", "User Requirement (Design Input)": "System must achieve a batch titer of >= 5 g/L.", "Risk": "High", "Linked Test Case (Design Output/V&V)": "PQ-TP-001", "Status": "PASS"},
        {"ID": "URS-012", "User Requirement (Design Input)": "System must have an emergency stop.", "Risk": "High", "Linked Test Case (Design Output/V&V)": "OQ-TP-015", "Status": "PASS"},
        {"ID": "URS-031", "User Requirement (Design Input)": "HMI & data historian must be 21 CFR Part 11 compliant.", "Risk": "High", "Linked Test Case (Design Output/V&V)": "CSV-TP-001", "Status": "GAP"}
    ]
    df = pd.DataFrame(df_data); st.dataframe(df, use_container_width=True, hide_index=True)
    gaps = df[df["Status"] == "GAP"]
    if not gaps.empty: st.error(f"**Critical Finding:** {len(gaps)} traceability gap(s) identified. The DHF is incomplete and this blocks validation release.")

def plot_risk_matrix(key: str) -> go.Figure:
    severity = [10, 9, 6, 8, 7, 5]; probability = [2, 3, 4, 3, 5, 1]
    risk_level = [s * p for s, p in zip(severity, probability)]; text = ["Contamination Event", "Incorrect Titer (OOS)", "Software Crash (Batch Loss)", "Incorrect Buffer Addition", "Sensor Failure (Drift)", "HMI Screen Lag"]
    fig = go.Figure(data=go.Scatter(x=probability, y=severity, mode='markers+text', text=text, textposition="top center", marker=dict(size=risk_level, sizemin=10, color=risk_level, colorscale="Reds", showscale=True, colorbar_title="Risk Score")))
    fig.update_layout(title='Process Risk Matrix (pFMEA) for Automated Bioreactor Suite', xaxis_title='Probability of Occurrence', yaxis_title='Severity of Harm', xaxis=dict(range=[0, 6]), yaxis=dict(range=[0, 11]))
    return fig

def create_v_model_figure(key: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Req. (URS)", "Functional Spec.", "Design Spec.", "Vendor Code Review"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test / FAT", "SAT", "IQ/OQ", "PQ"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text=None, showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig

def create_portfolio_health_dashboard(key: str) -> Styler:
    data = {'Project': ["Atlas (Bioreactor Suite C)", "Beacon (New Assembly Line)", "Comet (Vision System Upgrade)", "Sustaining Validation"], 'Phase': ["IQ/OQ", "FAT", "Planning", "Execution"], 'Schedule': ["Green", "Amber", "Green", "Green"], 'Budget': ["Green", "Green", "Green", "Amber"], 'Technical Risk': ["Amber", "Red", "Green", "Green"], 'Resource Strain': ["Amber", "Red", "Amber", "Green"]}
    df = pd.DataFrame(data)
    def style_rag(val: str) -> str: return f"background-color: {'lightgreen' if val == 'Green' else 'lightyellow' if val == 'Amber' else '#ffcccb' if val == 'Red' else 'white'}"
    return df.style.map(style_rag, subset=['Schedule', 'Budget', 'Technical Risk', 'Resource Strain'])

def create_resource_allocation_matrix(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    team_data = {'Team Member': ['David R.', 'Emily S.', 'Frank T.', 'Grace L.', 'Henry W.'], 'Primary Skill': ['Automation/PLC', 'Statistics (Minitab)', 'Process/Mechanical', 'Automation/PLC', 'Documentation Specialist'], 'Project Atlas': [50, 25, 25, 0, 0], 'Project Beacon': [50, 25, 75, 100, 25], 'Project Comet': [0, 25, 0, 0, 50], 'Sustaining': [0, 25, 0, 0, 25]}
    df = pd.DataFrame(team_data); df['Total Allocation'] = df[['Project Atlas', 'Project Beacon', 'Project Comet', 'Sustaining']].sum(axis=1)
    df['Status'] = df['Total Allocation'].apply(lambda x: 'Over-allocated' if x > 100 else ('At Capacity' if x >= 90 else 'Available'))
    fig = px.bar(df.sort_values('Total Allocation'), x='Total Allocation', y='Team Member', color='Status', text='Primary Skill', orientation='h', title='Validation Team Capacity & Strategic Alignment', color_discrete_map={'Over-allocated': 'red', 'At Capacity': 'orange', 'Available': 'green'})
    fig.add_vline(x=100, line_width=2, line_dash="dash", line_color="black", annotation_text="100% Capacity"); fig.update_layout(xaxis_title="Total Allocation (%)", yaxis_title="Team Member", legend_title="Status"); fig.update_traces(textposition='inside', textfont=dict(size=12, color='white'))
    over_allocated_df = df[df['Total Allocation'] > 100][['Team Member', 'Total Allocation']]
    return fig, over_allocated_df

# --- PAGE RENDERING FUNCTIONS (Hyper-Focused on Automated Equipment Validation) ---
def render_main_page() -> None:
    st.title("🤖 Automated Equipment Validation Portfolio")
    st.subheader("A Live Demonstration of Validation Leadership for Roche/Genentech")
    st.divider()
    st.markdown("Welcome. This interactive environment is designed to provide **undeniable proof of my expertise in the end-to-end validation of automated manufacturing equipment** in a strictly regulated environment. It simulates how I lead a validation function, with a relentless focus on aligning technical execution, **Quality Systems (per 21 CFR 820 & ISO 13485)**, and strategic capital projects.")
    st.subheader("🔷 Core Competency: Risk-Based Validation of Automated Systems")
    st.markdown("My leadership philosophy is grounded in the principles of **GAMP 5 and ASTM E2500**: build quality and testability into the design, verify components and systems rigorously, and ensure the process reliably delivers quality product. This portfolio is the tangible evidence of that approach.")
    st.markdown("#### Key Capabilities Demonstrated in this Portfolio:")
    with st.container(border=True):
        st.markdown("""
        - **End-to-End Validation Execution (`Tab 1`):** A dedicated hub that simulates the entire validation lifecycle of a new automated bioreactor suite—from **Design Review** and **FAT/SAT** to **IQ, OQ, and PQ**—showcasing a proactive, hands-on approach to managing major capital projects.
        
        - **Design Controls & DHF Management (`Tab 2`):** Demonstrate mastery of the Design History File (DHF) as it applies to equipment. Includes interactive **V-Models**, a live **Requirements Traceability Matrix (RTM)**, and a process **Risk Management (pFMEA)** built according to **ISO 14971**.
        
        - **Compliant Documentation Generation (`Tab 3`):** Go inside the Validation Lifecycle Hub to see professionally rendered, compliant **IQ, OQ, and PQ Protocols and Reports**. This section includes an interactive "Audit Defense" simulation to prove my ability to defend the team's work and documentation to FDA and Notified Body auditors.
        
        - **Maintaining the Validated State (`Tab 4`):** Showcase the ongoing oversight of the entire site validation program, including managing the **Periodic Review** schedule, developing a risk-based **Revalidation Strategy**, and tracking **Continuous Improvement Initiatives** to ensure long-term compliance.
        
        - **Team Leadership & Project Oversight (`Tab 5`):** Manage a portfolio of competing capital projects with clear **RAG status**, balance finite engineering resources, and demonstrate the ability to make data-driven decisions to keep projects on track.
        """)
    st.success("**Please use the navigation sidebar on the left, starting with the `E2E Validation Hub`, to explore the live evidence for each competency.**")

def render_e2e_validation_hub_page() -> None:
    st.title("🔩 End-to-End Validation Hub: Project Atlas (New Bioreactor Suite)")
    st.markdown("---")
    render_manager_briefing("Executing a Compliant Validation Lifecycle (per ASTM E2500)", "This hub simulates the execution of a major capital project from initial design review to final Performance Qualification (PQ). It provides tangible evidence of my ability to own validation deliverables, manage the FAT/SAT/IQ/OQ/PQ process, and ensure 'Quality First Time' by integrating validation requirements into the design phase.", "FDA 21 CFR 820.75, ISO 13485:2016 (Sec 7.5.6), GAMP 5, ASTM E2500", "Ensures new manufacturing equipment is brought online on-time, on-budget, and in a fully compliant state, directly enabling production launch and preventing costly post-release issues.")
    st.subheader("Project Atlas: Validation Phase Tracker")
    phase = st.select_slider("Select a Validation Phase to View Key Deliverables & Decisions:", options=["1. Design Review", "2. FAT", "3. SAT", "4. IQ", "5. OQ", "6. PQ"], value="1. Design Review")
    st.divider()
    if phase == "1. Design Review":
        st.header("Phase 1: Design Review - Ensuring 'Design for Validation'")
        st.info("My role here is to act as the Validation SME, ensuring the equipment is designed to be testable and compliant from day one. This proactive involvement is key to preventing costly redesigns and validation failures, aligning with **ICH Q9 (Quality Risk Management)** principles.")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("##### Key Deliverables (per QMS)"); st.markdown("- ✅ **Validation Master Plan (VMP)** Authored & Approved\n- ✅ **User Requirement Specifications (URS)** Reviewed & Signed\n- ✅ **Initial Risk Assessment (pFMEA per ISO 14971)** Completed\n- ✅ **Traceability Matrix** Initiated")
        with col2:
            with st.container(border=True):
                st.markdown("##### Critical Input Provided"); st.success("**URS Feedback (21 CFR Part 11):** Added a requirement for vendor to supply compliance documentation for the HMI software."); st.success("**Design Feedback (GAMP 5):** Recommended adding extra sensor ports to facilitate easier calibration and OQ testing."); st.error("**Risk Identified (ISO 14971):** The proposed single-source component for the main pump was flagged as a high supply chain risk.")
    elif phase == "2. FAT":
        st.header("Phase 2: Factory Acceptance Testing (FAT) at Vendor Site")
        st.info("My responsibility is to lead the team at the vendor's facility to execute the FAT protocol. The goal is to catch as many issues as possible *before* the equipment ships, saving significant time and cost.")
        with st.container(border=True): st.markdown("##### FAT Execution Summary"); st.metric("FAT Protocol Execution Status", "95% Complete"); st.progress(0.95); st.markdown("- **Major Deviations:** 1 (Controller software bug identified)\n- **Minor Deviations:** 4 (Documentation typos, minor component misalignment)"); st.success("**Decision:** Equipment is **conditionally approved** to ship. The vendor is required to fix the major software bug and provide a patch for verification during SAT. This prevents a major schedule delay while ensuring the issue is formally tracked and resolved.")
    elif phase == "3. SAT":
        st.header("Phase 3: Site Acceptance Testing (SAT) at Genentech")
        st.info("After installation, we re-run critical FAT tests and verify the fixes for any issues found. This confirms the equipment was not damaged in transit and is ready for formal qualification.")
        with st.container(border=True): st.markdown("##### SAT Execution Summary"); st.metric("SAT Protocol Execution Status", "100% Complete"); st.progress(1.0); st.markdown("- **FAT Deviation Verification:** Software bug fix confirmed via targeted testing.\n- **New Deviations:** 0"); st.success("**Decision:** Equipment is **formally accepted**. The validation team can now proceed with Installation Qualification (IQ).")
    elif phase == "4. IQ":
        st.header("Phase 4: Installation Qualification (IQ)")
        st.info("The IQ provides documented evidence that the equipment and its components have been installed correctly according to the design specifications and vendor recommendations.")
        with st.container(border=True): st.markdown("##### IQ Checklist Status"); st.checkbox("✅ All utilities (power, compressed air, WFI) verified and documented", value=True, disabled=True); st.checkbox("✅ Equipment model and serial numbers match specifications", value=True, disabled=True); st.checkbox("✅ All required drawings (P&ID, electrical) are as-built and signed", value=True, disabled=True); st.checkbox("✅ Software version 2.1.4 successfully installed and verified", value=True, disabled=True); st.success("**Result:** IQ is **complete and approved**. The equipment is ready for functional testing.")
    elif phase == "5. OQ":
        st.header("Phase 5: Operational Qualification (OQ)")
        st.info("The OQ challenges the equipment's functions to prove it operates as intended throughout its specified operating ranges. This is where we test alarms, interlocks, and critical functions.")
        with st.container(border=True): st.markdown("##### OQ Test Case Summary"); st.metric("OQ Protocol Test Cases Executed", "88 / 92"); st.progress(88/92); st.warning("**In Progress:** Testing emergency stop and recovery sequences. This is a critical safety verification and is being given extra scrutiny."); st.success("**Key Finding:** The bioreactor's temperature and pH control systems demonstrated accurate performance across the full range of specified operating parameters.")
    elif phase == "6. PQ":
        st.header("Phase 6: Performance Qualification (PQ)")
        st.info("The PQ is the final step, providing documented evidence that the equipment can consistently produce quality product under normal, real-world manufacturing conditions. This typically involves three successful, consecutive runs.")
        with st.container(border=True):
            st.markdown("##### PQ Run Status"); col1, col2, col3 = st.columns(3)
            with col1: st.success("✅ **Run 1:** Complete & Met All Acceptance Criteria")
            with col2: st.success("✅ **Run 2:** Complete & Met All Acceptance Criteria")
            with col3: st.warning("🟠 **Run 3:** In Progress")
            st.subheader("Critical Quality Attribute (CQA) Monitoring")
            st.plotly_chart(plot_cpk_analysis("pq_cpk"), use_container_width=True)
            st.success("**Result:** The process capability (CpK) for the critical CQA (Product Titer) is **1.67**, well above the target of 1.33. This provides high confidence in the process's stability and readiness for commercial production upon successful completion of Run 3.")

def render_design_controls_page() -> None:
    st.title("🏛️ 2. Design Controls & DHF Management")
    render_manager_briefing("Validation's Role in the Design History File", "Validation Engineering is a critical stakeholder in the Design Controls process for new equipment. We ensure that User Requirements are testable, contribute to the pFMEA, and generate the IQ/OQ/PQ reports that serve as the ultimate proof of Design Verification and Validation, forming a key part of the DHF.", "FDA 21 CFR 820.30 (Design Controls), ISO 13485:2016 (Sec 7.3), ASTM E2500", "Ensures audit readiness, prevents costly late-stage design changes by building in testability, and focuses validation resources on the highest-risk areas of the equipment and process.")
    with st.container(border=True):
        st.subheader("The V-Model for Equipment Validation"); st.markdown("The V-Model visually links user requirements and specifications (Design Inputs) to the corresponding qualification activities (Design Outputs and V&V)."); st.plotly_chart(create_v_model_figure(), use_container_width=True)
    render_metric_card("User Requirements Traceability Matrix (RTM)", "The RTM is the backbone of the DHF, providing an auditable link from User Requirements (URS) to IQ/OQ/PQ protocols. This is a core tenet of **21 CFR Part 820** and **ISO 13485**.", create_rtm_data_editor, "The matrix view instantly flags critical gaps, such as the un-tested 21 CFR Part 11 requirement, allowing for proactive mitigation before PQ execution.", "FDA 21 CFR 820.30(g) - Design Validation", key="rtm")
    render_metric_card("Process Risk Management (pFMEA per ISO 14971)", "A systematic process (pFMEA) for identifying and mitigating potential process failure modes. Validation activities serve as primary risk mitigations.", plot_risk_matrix, "The pFMEA clearly prioritizes 'Contamination Event' as the highest risk, ensuring that interlocks and sterile boundaries receive the most rigorous testing during OQ. This is a key input for the Validation Master Plan.", "ISO 14971: Application of risk management to medical devices", key="fmea")

def render_validation_lifecycle_hub_page() -> None:
    st.title("🗂️ 3. The Validation Lifecycle Hub (Documentation)")
    render_manager_briefing("Orchestrating Compliant Validation Documentation", "This hub demonstrates the ability to generate and manage the compliant, auditable documentation that forms the core of a successful validation package. The templates and simulations below prove my expertise in creating documents that meet the stringent requirements of **21 CFR Part 820** and **ISO 13485**.", "21 CFR 820.40 (Document Controls), GAMP 5 Good Documentation Practice, 21 CFR Part 11", "Ensures audit-proof documentation, accelerates review cycles by providing clear templates, and fosters seamless collaboration between Engineering, Manufacturing, Quality, and Regulatory.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Validation Document Approval Workflow"); st.markdown("Status for `VAL-MP-001_Project_Atlas`:"); st.divider()
            st.markdown("✔️ **Validation Lead (Self):** Approved `2024-01-15`\n✔️ **Process Engineering Lead:** Approved `2024-01-16`\n✔️ **Quality Assurance Lead:** Approved `2024-01-17`\n🟠 **Manufacturing Lead:** Pending Review\n⬜ **Head of Engineering:** Not Started")
            st.info("**Insight:** This workflow visualization provides instant status clarity for key deliverables, enabling proactive follow-up to prevent bottlenecks.")
    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer"); st.markdown("Click to expand and view mock validation document templates.")
            with st.expander("📄 View Mock IQ/OQ Protocol Template"):
                st.markdown("### IQ/OQ Protocol: VAL-TP-101 - Bioreactor Suite\n**Version:** 1.0\n---\n**1.0 Purpose:** To verify the correct installation (IQ) and functionality (OQ) of the new Bioreactor System (ASSET-123) according to approved design specifications.\n**2.0 Scope:** This protocol applies to the Bioreactor System located in Manufacturing Suite C.\n**3.0 Traceability to Requirements:**\n- **URS-001:** System must maintain temperature at 37.0°C ± 0.5°C.\n- **DS-015:** Agitator motor must be model XYZ-123.\n**4.0 IQ Tests:**\n- Verify P&ID and electrical drawings are as-built.\n- Verify all materials of construction match specifications.\n- Verify software version is 2.1.3.\n**5.0 OQ Tests:**\n- Verify alarm and interlock functionality (e.g., high temp, low pressure).\n- Verify agitator speed control operates within specified ranges.\n- Verify HMI screen transitions and data entry function correctly.\n**6.0 Acceptance Criteria:**\n- All IQ test steps must be completed with a 'Pass' result.\n- All OQ functional tests must meet their pre-defined expected results.")
                with st.container(border=True):
                    st.markdown("##### 🛡️ Simulate Audit Defense");
                    if st.button("Query this protocol", key="audit_proto_001"):
                        st.warning("**Auditor Query:** 'Your OQ does not test the full range of the agitator speed. Please provide your rationale.'")
                        st.success('**My Response:** "An excellent question. Our risk assessment (pFMEA) and process characterization data showed that operating outside the specified range (50-150 RPM) results in unacceptable sheer stress on the cells, which negatively impacts product quality. Therefore, we qualified the normal operating range and locked out the higher speeds in the control system, which is a risk-based approach aligned with **ASTM E2500**. The OQ verifies both the accuracy within the qualified range and that the lock-out is effective."')
            with st.expander("📋 View Mock PQ Report Template"):
                st.markdown("### PQ Report: VAL-TR-201 - Bioreactor Suite\n**Version:** 1.0\n---\n**1.0 Summary:** Three successful Performance Qualification (PQ) runs were executed on the Bioreactor System (ASSET-123) per protocol VAL-TP-201. The results confirm that the system reliably produces product meeting all pre-defined quality attributes under normal manufacturing conditions.\n**2.0 Deviations:**\n- **DEV-001:** During Run 2, a pH sensor required recalibration mid-run. The event was documented, the sensor was recalibrated, and the run successfully continued. **Impact Assessment:** None, as CQA data remained within spec.\n**3.0 Results vs. Acceptance Criteria:**\n| CQA | Specification | Run 1 Result | Run 2 Result | Run 3 Result | Pass/Fail |\n|---|---|---|---|---|---|\n| Titer (g/L) | >= 5.0 | 5.2 | 5.1 | 5.3 | **PASS** |\n| Viability (%) | >= 95% | 97% | 96% | 98% | **PASS** |\n| Impurity A (%)| <= 0.5% | 0.41% | 0.44% | 0.39% | **PASS** |\n**4.0 Conclusion:** The Bioreactor System has met all PQ acceptance criteria and is qualified for use in commercial manufacturing.\n**5.0 Traceability:** This report provides objective evidence fulfilling user requirements **URS-001** and **URS-040**.")

def render_validation_program_health_page() -> None:
    st.title("⚕️ 4. Validation Program Health & Continuous Improvement")
    render_manager_briefing("Maintaining the Validated State", "Initial validation is just the beginning. This dashboard demonstrates the ongoing oversight required to manage the site's validation program health. It showcases a data-driven approach to **Periodic Review**, the development of a risk-based **Revalidation Strategy**, and the execution of **Continuous Improvement Initiatives** to ensure our processes remain compliant, efficient, and in a constant state of control.", "FDA 21 CFR 820.75(c) (Revalidation), ISO 13485:2016 (Sec 8.4, Analysis of Data), ICH Q7", "Ensures long-term compliance, prevents costly process drifts, optimizes resource allocation for revalidation activities, and builds a culture of continuous improvement, directly supporting uninterrupted supply of medicine to patients.")
    st.subheader("Quarterly Validation Program Review (Q4 2023)")
    col1, col2, col3 = st.columns(3); col1.metric("Systems Due for Periodic Review", "8", delta="2"); col2.metric("Revalidations Triggered by Change Control", "3", delta="-1", delta_color="inverse"); col3.metric("Continuous Improvement Initiatives (Kaizens)", "5 Active")
    tab1, tab2, tab3 = st.tabs(["📊 Periodic Review & Revalidation Strategy", "📈 Continuous Improvement Tracker", "⚙️ Engineering Software & Lab Management"])
    with tab1:
        st.subheader("Risk-Based Periodic Review Schedule"); st.info("Systems are ranked by risk to prioritize review and revalidation resources.")
        review_data = {"System / Process": ["Bioreactor Suite C", "Purification Skid A", "WFI System", "HVAC - Grade A Area", "Automated Inspection System", "CIP Skid B"], "Risk Level": ["High", "High", "High", "Medium", "Medium", "Low"], "Last Review Date": ["2023-01-15", "2023-02-22", "2023-08-10", "2022-11-05", "2023-09-01", "2022-04-20"], "Next Review Due": ["2024-01-15", "2024-02-22", "2024-08-10", "2024-11-05", "2025-09-01", "2025-04-20"], "Status": ["Complete", "Complete", "On Schedule", "DUE", "On Schedule", "On Schedule"]}
        review_df = pd.DataFrame(review_data)
        def highlight_status(row): return ['background-color: #FFC7CE'] * len(row) if row["Status"] == "DUE" else [''] * len(row)
        st.dataframe(review_df.style.apply(highlight_status, axis=1), use_container_width=True, hide_index=True)
        st.error("**Actionable Insight:** The Periodic Review for the **HVAC - Grade A Area** is now due. I will assign a Validation Engineer to initiate the review this week, which involves analyzing process data, deviations, and change controls from the past two years to confirm it remains in a validated state.")
    with tab2:
        st.subheader("Continuous Improvement (Kaizen) Initiative Tracker"); st.info("Validation's role is to support and verify the impact of continuous improvement initiatives.")
        kaizen_data = {"Initiative ID": ["KZN-001", "KZN-002", "KZN-003"], "Description": ["Optimize CIP cycle time for Bioreactors", "Implement PAT sensor on Purification Skid", "Reduce changeover time on the packaging line"], "Validation Support Required": ["Cleaning Re-Validation (OQ/PQ)", "Validate new sensor and software (CSV)", "Validate new line settings (PQ)"], "Status": ["Execution Complete, Awaiting Report", "Protocol Execution", "Planning"], "Projected Annual Savings": [150000, 75000, 220000]}
        st.dataframe(kaizen_data, use_container_width=True, hide_index=True)
        st.success("**Actionable Insight:** The CIP cycle time optimization (KZN-001) has completed its validation runs. The final report is the last step to realizing **$150k in annual savings**. I will prioritize the review and approval of this report.")
    with tab3:
        st.subheader("Engineering Software & Lab Equipment Management"); st.info("Ensuring the tools we use for validation are themselves compliant and maintained.")
        tools_data = {"Tool / Software": ["Minitab Statistical Software", "Kaye Validator 2000", "MasterControl eQMS", "TOC Analyzer (Lab)"], "System Owner": ["Validation Dept.", "Validation Dept.", "Quality IT", "QC Lab"], "Last Calibration / CSV": ["2023-03-01", "2023-09-15", "N/A (SaaS)", "2023-11-05"], "Next Due": ["2024-03-01", "2024-09-15", "N/A", "2024-05-05"], "Service Agreement": ["Active", "Active", "Active", "Pending Renewal"]}
        st.dataframe(tools_data, use_container_width=True, hide_index=True)
        st.warning("**Actionable Insight:** The service agreement for the QC Lab's TOC Analyzer, which we rely on for cleaning validation samples, is pending renewal. I will follow up with the QC Manager to ensure this is renewed to avoid any project delays.")

def render_portfolio_page() -> None:
    st.title("📂 5. Validation Portfolio & Resource Management")
    render_manager_briefing("Managing the Validation Project Portfolio", "An effective manager oversees a portfolio of projects, balancing competing priorities, allocating finite resources, and providing clear, high-level status updates to leadership. This command center demonstrates the ability to manage these complexities and make data-driven trade-off decisions.", "Project Management Body of Knowledge (PMBOK), ISO 13485: 5.6 (Management Review)", "Provides executive-level visibility into Validation's contribution to corporate goals, enables proactive risk management across projects, and ensures strategic alignment of the department's most valuable asset: its people.")
    st.subheader("Capital Project Portfolio Health (RAG Status)")
    with st.container(border=True):
        st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True, hide_index=True)
        st.error("**Actionable Insight:** Project Beacon is flagged 'Red' for both Technical Risk and Resource Strain. The validation team is currently unable to support the aggressive timeline for this new assembly line.")
        if st.button("Simulate Escalation Memo to Core Team", key="escalate_beacon"):
            st.subheader("Generated Escalation Memo"); st.info("This is the type of clear, data-driven communication required to resolve cross-functional roadblocks.")
            st.text_area("Memo Draft:", "TO: Project Core Team Lead, Beacon\nFROM: Manager, Validation Engineering\nSUBJECT: URGENT: Validation Resource Deficit and Risk Assessment for Project Beacon\n\nTeam,\n\nThis memo is to formally escalate a critical resource and technical risk for Project Beacon. The project is 'Red' due to high technical risk and insufficient resource allocation. Our team cannot support the current timeline without jeopardizing quality or other projects.\n\nI request an immediate meeting to discuss mitigation: either de-scoping the initial phase or re-allocating automation engineers. Please advise on scheduling.", height=300)
    st.subheader("Integrated Resource Allocation Matrix")
    with st.container(border=True):
        fig_alloc, over_allocated_df = create_resource_allocation_matrix("allocation")
        st.plotly_chart(fig_alloc, use_container_width=True)
        if not over_allocated_df.empty:
            for _, row in over_allocated_df.iterrows():
                st.warning(f"**⚠️ Over-allocation Alert:** {row['Team Member']} is allocated at {row['Total Allocation']}%. This is unsustainable and poses a risk of burnout and project delays.")

# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = {
    "Executive Summary": render_main_page,
    "1. E2E Validation Hub (Project Atlas)": render_e2e_validation_hub_page,
    "2. Design Controls & DHF Management": render_design_controls_page,
    "3. Validation Lifecycle Hub (Docs)": render_validation_lifecycle_hub_page,
    "4. Validation Program Health": render_validation_program_health_page,
    "5. Portfolio & Resource Management": render_portfolio_page,
}
st.sidebar.title("Validation Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_to_render_func = PAGES[selection]
page_to_render_func()
