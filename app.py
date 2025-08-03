# app.py (Final, SME World-Class Version for Roche/Genentech)

# --- IMPORTS ---
from typing import Callable, Any, Tuple
import streamlit as st
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Automated Equipment Validation Portfolio | Roche",
    page_icon="🤖"
)

# --- AESTHETIC & THEME CONSTANTS ---
ROCHE_BLUE = '#0066CC'
SUCCESS_GREEN = '#2E7D32'
WARNING_AMBER = '#FFC107'
ERROR_RED = '#D32F2F'
NEUTRAL_GREY = '#B0BEC5'

# --- UTILITY & HELPER FUNCTIONS ---
def render_manager_briefing(title: str, content: str, reg_refs: str, business_impact: str) -> None:
    with st.container(border=True):
        st.subheader(f"🤖 {title}"); st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}")
        st.warning(f"**Key Standards & Regulations:** {reg_refs}")

# --- ENHANCED VISUALIZATION & DATA GENERATORS (50+ Artifacts) ---
def plot_budget_variance(key: str) -> go.Figure:
    df = pd.DataFrame({'Category': ['CapEx Projects', 'OpEx (Team)', 'OpEx (Lab)', 'Contractors'], 'Budgeted': [2500, 850, 300, 400], 'Actual': [2650, 820, 310, 350]})
    df['Variance'] = df['Actual'] - df['Budgeted']
    df['Color'] = df['Variance'].apply(lambda x: ERROR_RED if x > 0 else SUCCESS_GREEN)
    fig = go.Figure(go.Bar(x=df['Variance'], y=df['Category'], orientation='h', marker_color=df['Color'], text=df['Variance'].apply(lambda x: f'{x:+,}k')))
    fig.update_layout(title_text='Annual Budget Variance by Category (in $ thousands)', xaxis_title="Variance (Actual - Budget)", yaxis_title="", bargap=0.5)
    return fig

def plot_headcount_forecast(key: str) -> go.Figure:
    df = pd.DataFrame({'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'], 'Current FTEs': [8, 8, 8, 8], 'Forecasted Need': [8, 9, 10, 10]})
    df['Gap'] = df['Forecasted Need'] - df['Current FTEs']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current Headcount', x=df['Quarter'], y=df['Current FTEs'], marker_color=ROCHE_BLUE))
    fig.add_trace(go.Bar(name='Resource Gap', x=df['Quarter'], y=df['Gap'], base=df['Current FTEs'], marker_color=WARNING_AMBER))
    fig.update_layout(barmode='stack', title='Resource Gap Analysis: Headcount Plan vs. Forecasted Need', yaxis_title="Full-Time Equivalents (FTEs)")
    return fig

def display_departmental_okrs(key: str) -> None:
    st.table(pd.DataFrame({
        "Objective": ["Improve Validation Efficiency", "Enhance Team Capabilities", "Strengthen Compliance Posture"],
        "Key Result": ["Reduce Avg. Doc Cycle Time by 15%", "Certify 2 Engineers in GAMP 5", "Achieve Zero Major Findings in Next Audit"],
        "Status": ["On Track (12% reduction)", "Complete", "On Track"]
    }))

def run_project_duration_forecaster(key: str) -> None:
    rng = np.random.default_rng(42)
    historical_data = pd.DataFrame({'New_Automation_Modules': rng.integers(1, 10, 20), 'Process_Complexity_Score': rng.integers(1, 11, 20), 'URS_Count': rng.integers(20, 100, 20), 'Validation_Duration_Weeks': rng.uniform(8, 52, 20)})
    feature_names = ['New_Automation_Modules', 'Process_Complexity_Score', 'URS_Count']; X = historical_data[feature_names]; y = historical_data['Validation_Duration_Weeks']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    col1, col2, col3 = st.columns(3)
    with col1: new_modules = st.slider("# of New Automation Modules", 1, 10, 4, key=f"pipe_modules_{key}")
    with col2: complexity = st.slider("Process Complexity (1-10)", 1, 10, 6, key=f"pipe_comp_{key}")
    with col3: urs_count = st.slider("# of URS", 20, 100, 50, key=f"pipe_urs_{key}")
    new_project_data = pd.DataFrame([[new_modules, complexity, urs_count]], columns=feature_names)
    predicted_duration = model.predict(new_project_data)[0]
    st.metric("AI-Predicted Validation Duration (Weeks)", f"{predicted_duration:.1f}", help="Includes IQ, OQ, and PQ phases.")

def plot_gantt_chart(key: str) -> go.Figure:
    df = pd.DataFrame([
        dict(Task="Project Atlas (Bioreactor)", Start='2023-01-01', Finish='2023-12-31', Phase='Execution', Completion=0.8),
        dict(Task="Project Beacon (Assembly)", Start='2023-06-01', Finish='2024-06-30', Phase='Execution', Completion=0.4),
        dict(Task="Project Comet (Vision)", Start='2023-09-01', Finish='2024-03-31', Phase='Planning', Completion=0.9),
    ])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Phase", title="Major Capital Project Timelines", text="Task")
    fig.update_traces(textposition='inside'); return fig

def plot_risk_burndown(key: str) -> go.Figure:
    df = pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar', 'Apr'], 'Open Risks (RPN > 25)': [12, 10, 7, 4], 'Target Burndown': [12, 9, 6, 3]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Month'], y=df['Open Risks (RPN > 25)'], name='Actual Open Risks', fill='tozeroy', line=dict(color=ROCHE_BLUE)))
    fig.add_trace(go.Scatter(x=df['Month'], y=df['Target Burndown'], name='Target Burndown', line=dict(color=NEUTRAL_GREY, dash='dash')))
    fig.update_layout(title='Project Atlas: High-Risk Burndown', yaxis_title="Count of Open Risks"); return fig

def display_vendor_scorecard(key: str) -> None:
    st.table(pd.DataFrame({
        "Vendor": ["Vendor A (Automation)", "Vendor B (Components)"],
        "On-Time Delivery (%)": [95, 88],
        "FAT First-Pass Yield (%)": [92, 98],
        "Doc Package GDP Error Rate (%)": [2, 8],
        "Overall Score (weighted)": [91, 85]
    }))

def create_rtm_data_editor(key: str) -> None:
    df_data = [{"ID": "URS-001", "User Requirement": "System must achieve a batch titer of >= 5 g/L.", "Risk": "High", "Test Case": "PQ-TP-001", "Status": "PASS"}, {"ID": "URS-012", "User Requirement": "System must have an E-Stop.", "Risk": "High", "Test Case": "OQ-TP-015", "Status": "PASS"}, {"ID": "URS-031", "User Requirement": "HMI must be 21 CFR Part 11 compliant.", "Risk": "High", "Test Case": "CSV-TP-001", "Status": "GAP"}]
    df = pd.DataFrame(df_data); coverage = len(df[df["Test Case"] != ""]) / len(df) * 100
    st.metric("Traceability Coverage", f"{coverage:.1f}%", help="Percentage of requirements linked to a test case."); st.dataframe(df, use_container_width=True, hide_index=True)
    if any(df["Status"] == "GAP"): st.error("Critical traceability gap identified, blocking validation release.")

def create_v_model_figure(key: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["URS", "Func. Spec", "Design Spec", "Code/Config"], textposition="top right", line=dict(color=ROCHE_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["FAT", "SAT", "IQ/OQ", "PQ"], textposition="top left", line=dict(color=SUCCESS_GREEN, width=2)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color=NEUTRAL_GREY, width=1, dash="dot"))
    fig.update_layout(title_text="The Validation V-Model", showlegend=False, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False)); return fig

def plot_risk_matrix(key: str) -> None:
    severity = [10, 9, 6, 8, 7, 5]; probability = [2, 3, 4, 3, 5, 1]
    risk_level = [s * p for s, p in zip(severity, probability)]; text = ["Contamination", "Incorrect Titer", "Software Crash", "Incorrect Buffer Add", "Sensor Failure", "HMI Lag"]
    fig = go.Figure(data=go.Scatter(x=probability, y=severity, mode='markers+text', text=text, textposition="top center", marker=dict(size=risk_level, sizemin=10, color=risk_level, colorscale="Reds", showscale=True, colorbar_title="RPN")))
    fig.update_layout(title='Process Risk Matrix (pFMEA)', xaxis_title='Probability', yaxis_title='Severity', xaxis=dict(range=[0, 6]), yaxis=dict(range=[0, 11]))
    col1, col2 = st.columns([2,1])
    with col1: st.plotly_chart(fig, use_container_width=True)
    with col2: st.metric("RPN Mitigation Threshold", "25", help="Risks with an RPN > 25 require mandatory mitigation per SOP-QA-045."); st.info("This risk-based approach ensures validation efforts are focused on the highest-impact failure modes, a core principle of **ISO 14971**.")

def display_fat_sat_summary(key: str) -> None:
    col1, col2, col3 = st.columns(3); col1.metric("FAT Protocol Pass Rate", "95%"); col2.metric("SAT Protocol Pass Rate", "100%"); col3.metric("FAT-to-SAT Deviations", "0 New Major")
    st.info("For critical parameters, we use **Two One-Sided T-Tests (TOST)** to prove statistical equivalence between FAT and SAT results, ensuring no performance degradation during shipping.")

def plot_oq_challenge_results(key: str) -> go.Figure:
    setpoints = [30, 37, 45, 37, 30]; rng = np.random.default_rng(10); actuals = [rng.normal(sp, 0.1) for sp in setpoints]
    time = pd.to_datetime(['2023-10-01 08:00', '2023-10-01 09:00', '2023-10-01 10:00', '2023-10-01 11:00', '2023-10-01 12:00'])
    fig = go.Figure(); fig.add_trace(go.Scatter(x=time, y=setpoints, name='Setpoint (°C)', mode='lines+markers', line=dict(shape='hv', dash='dash', color=NEUTRAL_GREY))); fig.add_trace(go.Scatter(x=time, y=actuals, name='Actual (°C)', mode='lines+markers', line=dict(color=ROCHE_BLUE)))
    fig.update_layout(title='OQ Challenge: Bioreactor Temperature Control', xaxis_title='Time', yaxis_title='Temperature (°C)'); return fig

def plot_process_stability_chart(key: str) -> go.Figure:
    rng = np.random.default_rng(22); data = rng.normal(5.2, 0.25, 25); df = pd.DataFrame({'Titer': data}); df['MR'] = df['Titer'].diff().abs()
    I_CL = df['Titer'].mean(); MR_CL = df['MR'].mean(); I_UCL = I_CL + 2.66 * MR_CL; I_LCL = I_CL - 2.66 * MR_CL; MR_UCL = 3.267 * MR_CL
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Individuals (I) Chart", "Moving Range (MR) Chart"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Titer'], name='Titer (g/L)', mode='lines+markers'), row=1, col=1); fig.add_hline(y=I_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1); fig.add_hline(y=I_UCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1); fig.add_hline(y=I_LCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MR'], name='Moving Range', mode='lines+markers', line=dict(color=WARNING_AMBER)), row=2, col=1); fig.add_hline(y=MR_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1); fig.add_hline(y=MR_UCL, line_dash="dot", line_color=ERROR_RED, row=2, col=1)
    fig.update_layout(height=500, showlegend=False, title_text="Process Stability (I-MR Chart) for PQ Run 1 Titer"); return fig

def plot_csv_dashboard(key: str) -> None:
    st.metric("21 CFR Part 11 Compliance Status", "PASS", help="Electronic records and signatures meet requirements."); 
    st.table(pd.DataFrame({ "GAMP 5 Category": ["Cat 4: Configured", "Cat 5: Custom"], "System": ["HMI Software", "LIMS Interface"], "Status": ["Validation Complete", "IQ/OQ In Progress"] }))

def plot_cleaning_validation_results(key: str) -> go.Figure:
    df = pd.DataFrame({'Sample Location': ['Swab 1 (Reactor Wall)', 'Swab 2 (Agitator Blade)', 'Swab 3 (Fill Nozzle)', 'Final Rinse'], 'TOC Result (ppb)': [150, 180, 165, 25], 'Acceptance Limit (ppb)': [500, 500, 500, 50]})
    fig = px.bar(df, x='Sample Location', y='TOC Result (ppb)', title='Cleaning Validation Results (Total Organic Carbon)')
    fig.add_trace(go.Scatter(x=df['Sample Location'], y=df['Acceptance Limit (ppb)'], name='Acceptance Limit', mode='lines', line=dict(color=ERROR_RED, dash='dash'))); return fig

def plot_shipping_validation_temp(key: str) -> go.Figure:
    rng = np.random.default_rng(30); time = pd.to_datetime(pd.date_range("2023-01-01", periods=48, freq="H")); temp = rng.normal(4, 0.5, 48)
    fig = px.line(x=time, y=temp, title='Shipping Lane PQ: Temperature Profile'); fig.add_hrect(y0=2, y1=8, line_width=0, fillcolor=SUCCESS_GREEN, opacity=0.2, annotation_text="In Spec", annotation_position="top left");
    fig.update_layout(yaxis_title="Temperature (°C)"); return fig

def plot_doe_optimization(key: str) -> go.Figure:
    temp = np.linspace(30, 40, 10); ph = np.linspace(6.8, 7.6, 10); temp_grid, ph_grid = np.meshgrid(temp, ph)
    signal = 100 - (temp_grid - 37)**2 - 20*(ph_grid - 7.2)**2 + np.random.rand(10, 10)*2
    fig = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis')])
    fig.update_layout(title='DOE Response Surface for Process Optimization', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Product Yield (%)'))
    return fig

def plot_kaizen_roi_chart(key: str) -> go.Figure:
    fig = go.Figure(go.Waterfall(name="2023 Savings", orientation="v", measure=["relative", "relative", "relative", "total"], x=["Optimize CIP Cycle Time", "Implement PAT Sensor", "Reduce Line Changeover", "Total Annual Savings"], text=[f"+${v}k" for v in [150, 75, 220, 445]], y=[150, 75, 220, 445], connector={"line": {"color": "rgb(63, 63, 63)"}}, increasing={"marker":{"color":SUCCESS_GREEN}}, decreasing={"marker":{"color":ERROR_RED}}, totals={"marker":{"color":ROCHE_BLUE}}))
    fig.update_layout(title="Continuous Improvement (Kaizen) ROI Tracker", yaxis_title="Annual Savings ($k)"); return fig

def plot_deviation_trend_chart(key: str) -> go.Figure:
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=6, freq="ME")); data = {'Month': dates, 'Bioreactor Suite C': [5, 6, 8, 7, 9, 11], 'Purification Skid A': [4, 5, 4, 6, 5, 6], 'Packaging Line 2': [2, 1, 3, 2, 4, 3]}
    df = pd.DataFrame(data); fig = px.area(df, x='Month', y=['Bioreactor Suite C', 'Purification Skid A', 'Packaging Line 2'], title='Deviation Trend by Manufacturing System')
    fig.update_layout(yaxis_title="Number of Deviations"); return fig

def run_urs_risk_nlp_model(key: str) -> go.Figure:
    reqs = ["System shall have 99.5% uptime.", "Arm must move quickly.", "UI should be easy.", "System shall process 200 units/hour.", "System must be robust.", "Pump shall dispense 5.0 mL +/- 0.05 mL."]; labels = [0, 1, 1, 0, 1, 0]; criticality = [9, 6, 4, 10, 7, 10]
    df = pd.DataFrame({'Requirement': reqs, 'Risk_Label': labels, 'Criticality': criticality})
    tfidf = TfidfVectorizer(stop_words='english'); X = tfidf.fit_transform(df['Requirement']); y = df['Risk_Label']
    model = LogisticRegression(random_state=42).fit(X, y); df['Ambiguity Score'] = model.predict_proba(X)[:, 1]
    fig = px.scatter(df, x='Ambiguity Score', y='Criticality', text=df.index, title='URS Ambiguity Risk Model', labels={'Ambiguity Score': 'Predicted Ambiguity', 'Criticality': 'Process Impact'})
    fig.update_traces(textposition='top center'); fig.add_vline(x=0.5, line_dash="dash"); fig.add_hline(y=7.5, line_dash="dash"); return fig

# --- PAGE RENDERING FUNCTIONS (Hyper-Focused, 6-Pillar Structure) ---
def render_main_page() -> None:
    st.title("🤖 Automated Equipment Validation Portfolio"); st.subheader("A Live Demonstration of Validation Leadership for Roche/Genentech"); st.divider()
    st.markdown("Welcome. This interactive environment provides **undeniable proof of my expertise in the end-to-end validation of automated manufacturing equipment** in a strictly regulated environment. It simulates how I lead a validation function, with a relentless focus on aligning technical execution, **Quality Systems (per 21 CFR 820 & ISO 13485)**, and strategic capital projects.")
    st.subheader("🔷 Core Competency: Risk-Based Validation of Automated Systems"); st.markdown("My leadership philosophy is grounded in the principles of **GAMP 5 and ASTM E2500**: build quality and testability into the design, verify systems rigorously, and ensure the process reliably delivers quality product. This portfolio is the tangible evidence of that approach.")
    st.subheader("Key Program Health KPIs"); col1, col2, col3 = st.columns(3); col1.metric("Validation Program Compliance", "98%"); col2.metric("Quality First Time Rate", "91%"); col3.metric("Capital Project On-Time Delivery", "95%")
    with st.expander("Click here for a guided tour of this portfolio's key capabilities"):
        st.info("1. **Start with Strategy (`Tab 1`):** See how I manage budgets, forecast resources, and set departmental goals (OKRs).\n2. **Oversee the Portfolio (`Tab 2`):** View the Gantt chart, RAG status, and key project metrics (SPI/CPI) for all validation projects.\n3. **Deep Dive into Execution (`Tab 3`):** Walk through a live simulation of a major capital project from FAT to PQ.\n4. **Verify Specialized Expertise (`Tab 4`):** Explore my capabilities in Computer System, Cleaning, and Shipping Validation.\n5. **Confirm Lifecycle Management (`Tab 5`):** Understand my approach to maintaining the validated state through periodic review and continuous improvement.\n6. **Inspect the Documentation (`Tab 6`):** See how compliant, auditable protocols and reports are generated and defended.")

def render_strategic_management_page() -> None:
    st.title("📈 1. Strategic Management & Business Acumen")
    render_manager_briefing("Leading Validation as a Business Unit", "An effective manager must translate technical excellence into business value. This dashboard demonstrates my ability to manage budgets, plan for future headcount needs based on the project pipeline, and align departmental goals with the strategic objectives of the site.", "ISO 13485:2016 (Sec 5 & 6), 21 CFR 820.20", "Ensures the validation department is a strategic, financially responsible partner that enables the company's growth and compliance goals.")
    with st.container(border=True): st.subheader("Departmental OKRs (Objectives & Key Results)"); st.info("Purpose: OKRs align the team's daily work with high-level company goals."); display_departmental_okrs(key="okrs"); st.success("**Actionable Insight:** The team is on track to meet its efficiency and compliance goals for the year.")
    with st.container(border=True): st.subheader("Annual Budget Performance"); st.info("Purpose: This chart tracks actual spend against the department's annual budget, broken down by major cost centers."); st.plotly_chart(plot_budget_variance(key="budget"), use_container_width=True); st.success("**Actionable Insight:** The department is operating within its overall budget. The slight overage in CapEx was an approved expenditure for accelerated project timelines, offset by savings in contractor spend.")
    with st.container(border=True): st.subheader("Headcount & Resource Forecasting"); st.info("Purpose: This analysis compares the current team size against the forecasted resource needs demanded by the upcoming capital project pipeline."); st.plotly_chart(plot_headcount_forecast(key="headcount"), use_container_width=True); st.success("**Actionable Insight:** The forecast indicates a resource gap of 2 FTEs beginning in Q3. This data will be used to justify the hiring requisition for one Automation Engineer and one Validation Specialist.")
    with st.container(border=True): st.subheader("AI-Powered Capital Project Duration Forecaster"); run_project_duration_forecaster("duration_ai"); st.success("**Actionable Insight:** The AI model provides data-driven timeline estimates to the PMO, improving the accuracy of site-wide project planning and resource allocation.")

def render_project_portfolio_page() -> None:
    st.title("📂 2. Project & Portfolio Management")
    render_manager_briefing("Managing the Validation Project Portfolio", "This command center demonstrates the ability to manage a portfolio of competing capital projects, balancing priorities, allocating finite resources, and providing clear, high-level status updates to the PMO and site leadership.", "Project Management Body of Knowledge (PMBOK), ISO 13485: 5.6", "Provides executive-level visibility into Validation's contribution to corporate goals, enables proactive risk management, and ensures strategic alignment of the department's people.")
    with st.container(border=True): st.subheader("Capital Project Timelines (Gantt Chart)"); st.info("Purpose: This provides a high-level overview of major validation projects and their dependencies."); st.plotly_chart(plot_gantt_chart(key="gantt"), use_container_width=True)
    with st.container(border=True):
        st.subheader("Capital Project Portfolio Health"); col1, col2 = st.columns(2)
        with col1: st.markdown("##### RAG Status"); st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True)
        with col2: st.markdown("##### Key Project Metrics"); st.metric("Project Beacon: Schedule Performance Index (SPI)", "0.92", "-8%", help="SPI < 1.0 indicates the project is behind schedule."); st.metric("Project Beacon: Cost Performance Index (CPI)", "0.85", "-15%", help="CPI < 1.0 indicates the project is over budget."); st.plotly_chart(plot_risk_burndown("risk_burn"), use_container_width=True)
    with st.container(border=True): st.subheader("Validation Team Resource Allocation"); fig_alloc, over_allocated_df = create_resource_allocation_matrix("allocation"); st.plotly_chart(fig_alloc, use_container_width=True);
    if not over_allocated_df.empty:
        for _, row in over_allocated_df.iterrows(): st.warning(f"**⚠️ Over-allocation Alert:** {row['Team Member']} at {row['Total Allocation']}%.")
    with st.container(border=True): st.subheader("Key Vendor Performance Scorecard"); st.info("Purpose: This scorecard tracks the performance of key equipment vendors against critical, quantitative metrics."); display_vendor_scorecard("vendor"); st.success("**Actionable Insight:** Vendor A's strong performance and low documentation error rate on the 'Atlas' project make them a preferred partner for the upcoming 'Beacon' project. Vendor B's performance requires a follow-up meeting to establish a performance improvement plan.")

def render_e2e_validation_hub_page() -> None:
    st.title("🔩 3. End-to-End Validation Hub: Project Atlas")
    render_manager_briefing("Executing a Compliant Validation Lifecycle (per ASTM E2500)", "This hub simulates the execution of a major capital project from initial design review to final Performance Qualification (PQ). It provides tangible evidence of my ability to own validation deliverables, manage the FAT/SAT/IQ/OQ/PQ process, and ensure 'Quality First Time' by integrating validation requirements into the design phase.", "FDA 21 CFR 820.75, ISO 13485:2016 (Sec 7.5.6), GAMP 5, ASTM E2500", "Ensures new manufacturing equipment is brought online on-time, on-budget, and in a fully compliant state, directly enabling production launch.")
    phase = st.select_slider("Select a Validation Phase to View Key Deliverables:", options=["1. Design Review & Planning", "2. FAT & SAT", "3. IQ & OQ", "4. PQ"], value="1. Design Review & Planning")
    st.divider()
    if phase == "1. Design Review & Planning":
        st.header("Phase 1: Design Controls & Risk-Based Planning"); st.info("My role is to act as the Validation SME, ensuring the equipment is designed to be testable and compliant from day one. This proactive involvement is key to preventing costly redesigns and validation failures.")
        with st.container(border=True): st.subheader("Validation V-Model"); st.plotly_chart(create_v_model_figure("vmodel"), use_container_width=True)
        with st.container(border=True): st.subheader("User Requirements Traceability (RTM)"); create_rtm_data_editor("rtm")
        with st.container(border=True): st.subheader("Process Risk Management (pFMEA)"); plot_risk_matrix("fmea")
        with st.container(border=True): st.subheader("AI-Powered URS Risk Analysis"); st.plotly_chart(run_urs_risk_nlp_model("urs_risk"), use_container_width=True)
    elif phase == "2. FAT & SAT":
        st.header("Phase 2: Factory & Site Acceptance Testing"); st.info("Purpose: To execute acceptance testing at the vendor's facility (FAT) and our site (SAT). The goal is to catch as many issues as possible *before* formal qualification begins.")
        with st.container(border=True): st.subheader("FAT & SAT Summary Metrics"); display_fat_sat_summary("fat_sat")
    elif phase == "3. IQ & OQ":
        st.header("Phase 3: Installation & Operational Qualification"); st.info("Purpose: The IQ provides documented evidence of correct installation. The OQ challenges the equipment's functions to prove it operates as intended throughout its specified operating ranges."); st.plotly_chart(plot_oq_challenge_results("oq_plot"), use_container_width=True)
    elif phase == "4. PQ":
        st.header("Phase 4: Performance Qualification"); st.info("Purpose: The PQ is the final step, providing documented evidence that the equipment can consistently produce quality product under normal, real-world manufacturing conditions.")
        col1, col2 = st.columns(2);
        with col1: st.subheader("Process Capability (PQ CQA)"); st.plotly_chart(plot_cpk_analysis("pq_cpk"), use_container_width=True)
        with col2: st.subheader("Process Stability (PQ CPP)"); st.plotly_chart(plot_process_stability_chart("pq_spc"), use_container_width=True)

def render_specialized_validation_page() -> None:
    st.title("🧪 4. Specialized Validation Hubs")
    render_manager_briefing("Demonstrating Breadth of Expertise", "Beyond standard equipment qualification, a Validation Manager must be fluent in specialized validation disciplines critical to GMP manufacturing. This hub showcases expertise in Computer System Validation (CSV), Cleaning Validation, and Process Characterization.", "21 CFR Part 11, GAMP 5, PDA TR 29 (Cleaning Validation)", "Ensures all aspects of the manufacturing process, including supporting systems and processes, are fully compliant and controlled, preventing common sources of regulatory findings.")
    tab1, tab2, tab3, tab4 = st.tabs(["🖥️ Computer System Validation (CSV)", "🧼 Cleaning Validation", "🔬 Process Characterization (DOE)", "📦 Shipping Validation"])
    with tab1: st.subheader("GAMP 5 CSV for Automated Systems"); st.info("Purpose: This dashboard tracks the validation status of all GxP computerized systems associated with a project, following the GAMP 5 risk-based approach."); plot_csv_dashboard("csv")
    with tab2: st.subheader("Cleaning Validation for Multi-Product Facility"); st.info("Purpose: This plot shows the results from a cleaning validation study, confirming that residual product levels are below the pre-defined acceptance limits to prevent cross-contamination."); st.plotly_chart(plot_cleaning_validation_results("cleaning"), use_container_width=True)
    with tab3: st.subheader("Process Characterization using Design of Experiments (DOE)"); st.info("Purpose: DOE is a powerful statistical tool used during process development to identify the optimal settings (e.g., temperature, pH) that maximize product yield. The validation team uses this data to set and challenge the normal operating ranges during OQ and PQ."); st.plotly_chart(plot_doe_optimization("doe"), use_container_width=True)
    with tab4: st.subheader("Shipping Lane Performance Qualification"); st.info("Purpose: This PQ study confirms that the validated shipping container and process can maintain the required temperature range over a simulated transit duration."); st.plotly_chart(plot_shipping_validation_temp("shipping"), use_container_width=True)

def render_validation_program_health_page() -> None:
    st.title("⚕️ 5. Validation Program Health & Continuous Improvement")
    render_manager_briefing("Maintaining the Validated State", "This dashboard demonstrates the ongoing oversight required to manage the site's validation program health. It showcases a data-driven approach to **Periodic Review**, the development of a risk-based **Revalidation Strategy**, and the execution of **Continuous Improvement Initiatives**.", "FDA 21 CFR 820.75(c) (Revalidation), ISO 13485:2016 (Sec 8.4)", "Ensures long-term compliance, prevents costly process drifts, optimizes resource allocation for revalidation, and supports uninterrupted supply of medicine to patients.")
    st.subheader("Quarterly Validation Program Review")
    col1, col2, col3 = st.columns(3); col1.metric("Systems Due for Periodic Review", "8"); col2.metric("Revalidations from Change Control", "3"); col3.metric("CAPA Effectiveness Rate", "95%")
    tab1, tab2 = st.tabs(["📊 Periodic Review & Revalidation Strategy", "📈 Continuous Improvement Tracker"])
    with tab1:
        st.subheader("Risk-Based Periodic Review Schedule"); st.info("Purpose: This table tracks the periodic review schedule for all critical GxP systems, prioritized by risk level. This ensures that resources are focused on the systems with the greatest impact on product quality and patient safety.")
        review_data = {"System": ["Bioreactor C", "Purification A", "WFI System", "HVAC - Grade A", "Inspection System", "CIP Skid B"], "Risk Level": ["High", "High", "High", "Medium", "Medium", "Low"], "Last Review": ["2023-01-15", "2023-02-22", "2023-08-10", "2022-11-05", "2023-09-01", "2022-04-20"], "Next Due": ["2024-01-15", "2024-02-22", "2024-08-10", "2024-11-05", "2025-09-01", "2025-04-20"], "Status": ["Complete", "Complete", "On Schedule", "DUE", "On Schedule", "On Schedule"]}
        review_df = pd.DataFrame(review_data)
        def highlight_status(row):
            return ['background-color: #FFC7CE'] * len(row) if row["Status"] == "DUE" else [''] * len(row)
        st.dataframe(review_df.style.apply(highlight_status, axis=1), use_container_width=True, hide_index=True)
        st.error("**Actionable Insight:** The Periodic Review for the **HVAC - Grade A Area** is now due. I will assign a Validation Engineer to initiate the review this week.")
    with tab2:
        st.subheader("Continuous Improvement (Kaizen) Initiative Tracker"); st.info("Purpose: Validation's role is to support and verify the impact of continuous improvement initiatives. These charts link operational problems (deviations) to financial solutions (ROI).")
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(plot_kaizen_roi_chart("kaizen_roi"), use_container_width=True)
        with col2: st.plotly_chart(plot_deviation_trend_chart("deviation_trend"), use_container_width=True)
        st.success("**Actionable Insight:** The deviation trend chart validates the focus of our continuous improvement efforts. The ROI tracker provides a strong business case for future Kaizen events.")

def render_documentation_hub_page() -> None:
    st.title("🗂️ 6. Validation Documentation & Audit Defense Hub")
    render_manager_briefing("Orchestrating Compliant Validation Documentation", "This hub demonstrates the ability to generate and manage the compliant, auditable documentation that forms the core of a successful validation package. The templates and simulations below prove my expertise in creating documents that meet the stringent requirements of **21 CFR Part 820** and **ISO 13485**.", "21 CFR 820.40 (Document Controls), GAMP 5 Good Documentation Practice, 21 CFR Part 11", "Ensures audit-proof documentation, accelerates review cycles by providing clear templates, and fosters seamless collaboration between Engineering, Manufacturing, Quality, and Regulatory.")
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.subheader("Document Approval Workflow"); st.info("This simulates the eQMS workflow for key validation deliverables."); st.markdown("Status for `VAL-MP-001_Project_Atlas`:"); st.divider()
            st.markdown("✔️ **Validation Lead (Self):** Approved `2024-01-15`\n✔️ **Process Engineering Lead:** Approved `2024-01-16`\n✔️ **Quality Assurance Lead:** Approved `2024-01-17`\n🟠 **Manufacturing Lead:** Pending Review\n⬜ **Head of Engineering:** Not Started")
    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer"); st.info("The following are professionally rendered digital artifacts that simulate documents within a validated eQMS.")
            with st.expander("📄 **View Professional IQ/OQ Protocol Template**"):
                _render_professional_protocol_template()
            with st.expander("📋 **View Professional PQ Report Template**"):
                _render_professional_report_template()

def _render_professional_protocol_template() -> None:
    """
    Renders a world-class, professional IQ/OQ Protocol, mimicking an eQMS.
    This version fixes a critical IndentationError.
    """
    st.header("IQ/OQ Protocol: VAL-TP-101")
    st.subheader("Automated Bioreactor Suite (ASSET-123)")
    st.divider()

    meta_cols = st.columns(4)
    meta_cols[0].metric("Document ID", "VAL-TP-101")
    meta_cols[1].metric("Version", "1.0")
    meta_cols[2].metric("Status", "Approved")
    meta_cols[3].metric("Effective Date", "2024-01-15")
    st.divider()

    st.subheader("1.0 Purpose")
    st.write("To provide documented evidence that the Automated Bioreactor Suite (ASSET-123) and its ancillary components are installed correctly (Installation Qualification) and operate according to their functional specifications (Operational Qualification).")
    
    st.subheader("2.0 Scope")
    st.write("This protocol applies to the Bioreactor System located in Manufacturing Suite C, including the vessel, control skid, HMI, and associated critical utilities (WFI, CSG).")
    
    with st.container(border=True):
        st.subheader("3.0 Traceability & Risk Management")
        st.markdown("This protocol provides test evidence for URS items as defined in the **Validation Master Plan (VAL-MP-001)** and the **Traceability Matrix (QA-DOC-105)**. All tests are derived from the system's **pFMEA (RISK-034)** to ensure a risk-based approach per **ISO 14971**.")
    
    with st.container(border=True):
        st.subheader("4.0 Installation Qualification (IQ)")
        st.markdown("##### 4.1 Documentation Verification\n- Verify P&ID and electrical drawings are as-built.\n- Confirm receipt of vendor documentation package, including material certifications.")
        st.markdown("##### 4.2 Equipment Verification\n- Verify equipment model and serial numbers match the bill of materials (BOM).\n- Confirm software and firmware versions match design specifications.")
        st.markdown("##### 4.3 Statistical Sampling Plan\nFor repeated checks (e.g., gasket verification), a sampling plan based on **ANSI/ASQ Z1.4** will be used, with an AQL of 1.0.")
    
    with st.container(border=True):
        st.subheader("5.0 Operational Qualification (OQ)")
        st.markdown("##### 5.1 Critical Function & Interlock Challenges\n- **Alarm Tests:** Challenge high/low alarms for temperature, pressure, and pH.\n- **Interlock Tests:** Verify agitator does not run when vessel pressure is high; verify E-Stop functionality.")
        st.markdown("##### 5.2 Computer System Validation (CSV) Tests\n- Verify HMI screen transitions and data entry function correctly.\n- Confirm audit trail functionality per **21 CFR Part 11** requirements.")
    
    # --- START OF THE FIX ---
    # This container and its contents have been un-indented by one level to align
    # correctly with the main flow of the function.
    with st.container(border=True):
        st.markdown("##### 🛡️ Simulate Audit Defense")
        if st.button("Query this protocol's strategy", key="audit_proto_001"):
            st.warning("**Auditor Query:** 'Your OQ does not test the full operating range of the agitator speed. Please provide your rationale.'")
            st.success('**My Response:** "An excellent question. Our risk assessment (pFMEA) and process characterization data (from the DOE in the Specialized Hub) showed that operating outside the specified range of 50-150 RPM results in unacceptable sheer stress on the cells, which negatively impacts a Critical Quality Attribute. Therefore, we qualified the normal operating range and formally locked out the higher speeds in the control system. This is a risk-based approach aligned with **ASTM E2500**. The OQ verifies both the accuracy within the qualified range and that the software lock-out is effective."')
    # --- END OF THE FIX ---
def _render_professional_report_template() -> None:
    st.header("PQ Report: VAL-TR-201"); st.subheader("Automated Bioreactor Suite (ASSET-123)"); st.divider()
    meta_cols = st.columns(4); meta_cols[0].metric("Document ID", "VAL-TR-201"); meta_cols[1].metric("Version", "1.0"); meta_cols[2].metric("Status", "Final"); meta_cols[3].metric("Approval Date", "2024-03-01"); st.divider()
    st.subheader("1.0 Summary & Conclusion"); col1, col2 = st.columns([2, 1])
    with col1: st.write("Three successful, consecutive Performance Qualification (PQ) runs were executed on the Bioreactor System per protocol VAL-TP-201. The results confirm that the system reliably produces product meeting all pre-defined Critical Quality Attributes (CQAs) under normal manufacturing conditions."); st.success("**Conclusion:** The Automated Bioreactor System (ASSET-123) has met all PQ acceptance criteria and is **qualified for use in commercial GMP manufacturing.**")
    with col2: st.metric("Overall Result", "PASS"); st.metric("Final CpK (Product Titer)", "1.67", help="Exceeds target of >= 1.33")
    st.subheader("2.0 Deviations & Impact Assessment");
    with st.container(border=True): st.info("**DEV-001 (Run 2):** A pH sensor required recalibration mid-run. The event was documented in the batch record, the sensor was recalibrated per SOP, and the run successfully continued."); st.success("**Impact Assessment:** None. All CQA data for the batch remained within specification. The event and its resolution were reviewed and approved by QA.")
    st.subheader("3.0 Results vs. Acceptance Criteria")
    results_data = {'Critical Quality Attribute (CQA)': ['Titer (g/L)', 'Viability (%)', 'Impurity A (%)'], 'Specification': ['>= 5.0', '>= 95%', '<= 0.5%'], 'Run 1 Result': [5.2, 97, 0.41], 'Run 2 Result': [5.1, 96, 0.44], 'Run 3 Result': [5.3, 98, 0.39], 'Pass/Fail': ['PASS', 'PASS', 'PASS']}
    results_df = pd.DataFrame(results_data); def style_pass_fail(val): return f"background-color: {SUCCESS_GREEN}; color: white;" if val == 'PASS' else f"background-color: {ERROR_RED}; color: white;"
    styled_df = results_df.style.applymap(style_pass_fail, subset=['Pass/Fail'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.subheader("4.0 Traceability"); st.warning("This report provides the objective evidence that fulfills user requirements **URS-001** (Titer) and **URS-040** (Purity) as documented in the Requirements Traceability Matrix (QA-DOC-105).")
    st.subheader("5.0 Approvals"); st.markdown("---"); sig_cols = st.columns(3); sig_cols[0].success("✔️ **Validation Manager:** Approved `2024-03-01`"); sig_cols[1].success("✔️ **Manufacturing Head:** Approved `2024-03-01`"); sig_cols[2].success("✔️ **Quality Assurance Head:** Approved `2024-03-02`")

# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = {
    "Executive Summary": render_main_page,
    "1. Strategic Management": render_strategic_management_page,
    "2. Project & Portfolio Management": render_project_portfolio_page,
    "3. E2E Validation Hub (Project Atlas)": render_e2e_validation_hub_page,
    "4. Specialized Validation Hubs": render_specialized_validation_page,
    "5. Validation Program Health": render_validation_program_health_page,
    "6. Documentation & Audit Defense": render_documentation_hub_page,
}
st.sidebar.title("Validation Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_to_render_func = PAGES[selection]
page_to_render_func()
