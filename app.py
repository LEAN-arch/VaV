# app.py (Final, SME World-Class Version - Fully Corrected)

# --- IMPORTS ---
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
from scipy.stats import norm
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Automated Equipment Validation Portfolio",
    page_icon="🤖"
)

# --- AESTHETIC & THEME CONSTANTS ---
PRIMARY_COLOR = '#0460A9'
SUCCESS_GREEN = '#4CAF50'
WARNING_AMBER = '#FFC107'
ERROR_RED = '#D32F2F'
NEUTRAL_GREY = '#9E9E9E'
BACKGROUND_GREY = '#F5F5F5'

# --- UTILITY & HELPER FUNCTIONS ---
def render_manager_briefing(title: str, content: str, reg_refs: str, business_impact: str, quality_pillar: str, risk_mitigation: str) -> None:
    """Renders a standardized, professional briefing container."""
    with st.container(border=True):
        st.subheader(f"🤖 {title}", divider='blue')
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}", icon="🎯")
        st.warning(f"**Key Standards & Regulations:** {reg_refs}", icon="📜")
        st.success(f"**Quality Culture Pillar:** {quality_pillar}", icon="🌟")
        st.error(f"**Strategic Risk Mitigation:** {risk_mitigation}", icon="🛡️")

def style_dataframe(df: pd.DataFrame) -> Styler:
    """Applies a professional, consistent style to a DataFrame."""
    return df.style.set_properties(**{
        'background-color': '#FFFFFF', 'color': '#000000', 'border': f'1px solid {NEUTRAL_GREY}'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', PRIMARY_COLOR), ('color', 'white'), ('font-weight', 'bold')]}
    ]).hide(axis="index")

# --- DATA GENERATORS & VISUALIZATIONS ---

def create_portfolio_health_dashboard(key: str) -> Styler:
    """Creates a styled RAG status dashboard for the project portfolio."""
    health_data = {
        'Project': ["Project Atlas (Bioreactor)", "Project Beacon (Assembly)", "Project Comet (Vision)"],
        'Overall Status': ["Green", "Amber", "Green"], 'Schedule': ["On Track", "At Risk", "Ahead"],
        'Budget': ["On Track", "Over", "On Track"], 'Lead': ["J. Doe", "S. Smith", "J. Doe"]
    }
    df = pd.DataFrame(health_data)
    def style_status(val: str) -> str:
        color_map = {"Green": SUCCESS_GREEN, "Amber": WARNING_AMBER, "Red": ERROR_RED,
                     "On Track": SUCCESS_GREEN, "Ahead": SUCCESS_GREEN, 
                     "At Risk": WARNING_AMBER, "Over": ERROR_RED}
        bg_color = color_map.get(val, 'white')
        font_color = 'white' if val in color_map else 'black'
        return f"background-color: {bg_color}; color: {font_color};"
    return df.style.map(style_status, subset=['Overall Status', 'Schedule', 'Budget']).set_properties(**{'text-align': 'center'}).hide(axis="index")

def create_resource_allocation_matrix(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    """Creates a resource allocation heatmap and identifies over-allocated staff."""
    data = {'J. Doe (Lead)': [0.5, 0.4, 0.2], 'S. Smith (Eng.)': [0.1, 0.8, 0.0],
            'A. Wong (Spec.)': [0.4, 0.4, 0.3], 'B. Zeller (Eng.)': [0.0, 0.2, 0.7]}
    df = pd.DataFrame(data, index=["Project Atlas", "Project Beacon", "Project Comet"])
    df_transposed = df.T
    
    color_range_max = 1.1
    normalized_colorscale = [
        [0.0, 'white'], [0.5 / color_range_max, SUCCESS_GREEN],
        [1.0 / color_range_max, WARNING_AMBER], [1.0, ERROR_RED]
    ]
    
    fig = px.imshow(df_transposed, text_auto=".0%", aspect="auto",
                    color_continuous_scale=normalized_colorscale,
                    range_color=[0, color_range_max], 
                    labels=dict(x="Project", y="Team Member", color="Allocation"),
                    title="<b>Team Allocation by Project</b>")
    fig.update_traces(textfont_color='black')
    fig.update_layout(title_x=0.5, title_font_size=20, plot_bgcolor=BACKGROUND_GREY)
    
    allocations = df_transposed.sum(axis=1).reset_index()
    allocations.columns = ['Team Member', 'Total Allocation']
    over_allocated = allocations[allocations['Total Allocation'] > 1.0]
    return fig, over_allocated

def run_project_duration_forecaster(key: str) -> None:
    """Trains a model and explains its predictions using SHAP."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({
        'New Automation Modules': rng.integers(1, 10, 20), 
        'Process Complexity Score': rng.integers(1, 11, 20), 
        '# of URS': rng.integers(20, 100, 20)
    })
    y_train = pd.Series(rng.uniform(8, 52, 20), name="Validation Duration (Weeks)")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    st.markdown("##### Adjust Project Parameters to Forecast Timeline:")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_modules = st.slider("New Automation Modules", 1, 10, 4, key=f"pipe_modules_{key}")
    with col2:
        complexity = st.slider("Process Complexity (1-10)", 1, 10, 6, key=f"pipe_comp_{key}")
    with col3:
        urs_count = st.slider("# of URS", 20, 100, 50, key=f"pipe_urs_{key}")
    
    new_project_data = pd.DataFrame([[new_modules, complexity, urs_count]], columns=X_train.columns)
    predicted_duration = model.predict(new_project_data)[0]
    
    st.metric("AI-Predicted Validation Duration (Weeks)", f"{predicted_duration:.1f}", 
              help="Based on a Random Forest model trained on 20 historical projects. Includes IQ, OQ, and PQ phases.")

    st.subheader("AI Prediction Analysis (Why This Forecast?)")
    st.info("""
    **Purpose:** This SHAP (SHapley Additive exPlanations) force plot provides transparent, undeniable proof of how the AI model arrived at its prediction. 
    It's not a black box; we can see exactly which project features are driving the timeline estimate.
    - **Base Value:** The average predicted duration across all historical projects.
    - **Red Arrows:** Features pushing the prediction **higher** (increasing the duration).
    - **Blue Arrows:** Features pushing the prediction **lower** (decreasing the duration).
    """)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(new_project_data)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(explainer.expected_value, shap_values[0], new_project_data.iloc[0], matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.success("""
    **Actionable Insight:** The SHAP analysis reveals that the high '# of URS' is the primary factor increasing the project's predicted duration. To shorten this timeline, we should focus on consolidating or simplifying user requirements during the planning phase.
    """)

def plot_cpk_analysis(key: str) -> go.Figure:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.15, scale=0.05, size=100)
    LSL, USL = 5.0, 5.3
    mu, std = np.mean(data), np.std(data, ddof=1)
    cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=20, name='Observed Data', histnorm='probability density', marker_color=PRIMARY_COLOR, opacity=0.7))
    x_fit = np.linspace(min(data), max(data), 200)
    y_fit = norm.pdf(x_fit, mu, std)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Normal Distribution', line=dict(color=SUCCESS_GREEN, width=2)))
    fig.add_vline(x=LSL, line_dash="dash", line_color=ERROR_RED, annotation_text="LSL", annotation_position="top left")
    fig.add_vline(x=USL, line_dash="dash", line_color=ERROR_RED, annotation_text="USL", annotation_position="top right")
    fig.add_vline(x=mu, line_dash="dot", line_color=NEUTRAL_GREY, annotation_text=f"Mean={mu:.2f}", annotation_position="bottom right")
    fig.update_layout(title_text=f'Process Capability (Cpk) Analysis - Titer (g/L)<br><b>Cpk = {cpk:.2f}</b> (Target: ≥1.33)',
                      xaxis_title="Titer (g/L)", yaxis_title="Density", showlegend=False,
                      title_font_size=20, title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def _render_professional_protocol_template() -> None:
    st.header("IQ/OQ Protocol: VAL-TP-101"); st.subheader("Automated Bioreactor Suite (ASSET-123)"); st.divider()
    st.markdown("##### 1.0 Purpose")
    st.write("The purpose of this protocol is to provide documented evidence that the Automated Bioreactor Suite (ASSET-123) is installed correctly (Installation Qualification - IQ) and operates according to its functional specifications (Operational Qualification - OQ).")
    st.markdown("##### 4.0 Test Procedures - OQ Section (Example)")
    test_case_data = {'Test ID': ['OQ-TC-001', 'OQ-TC-002', 'OQ-TC-003'], 'Test Description': ['Verify Temperature Control Loop', 'Challenge Agitator Speed Control', 'Test Critical Alarms (High Temp)'], 'Acceptance Criteria': ['Maintain setpoint ± 0.5°C for 60 mins', 'Maintain setpoint ± 2 RPM across range', 'Alarm activates within 5s of exceeding setpoint'], 'Result (Pass/Fail)': ['', '', '']}
    st.dataframe(style_dataframe(pd.DataFrame(test_case_data)), use_container_width=True)
    st.warning("This is a simplified template for demonstration purposes.")

def plot_budget_variance(key: str) -> go.Figure:
    df = pd.DataFrame({'Category': ['CapEx Projects', 'OpEx (Team)', 'OpEx (Lab)', 'Contractors'], 'Budgeted': [2500, 850, 300, 400], 'Actual': [2650, 820, 310, 350]})
    df['Variance'] = df['Actual'] - df['Budgeted']
    df['Color'] = df['Variance'].apply(lambda x: ERROR_RED if x > 0 else SUCCESS_GREEN)
    df['Text'] = df['Variance'].apply(lambda x: f'${x:+,}k')
    fig = go.Figure(go.Bar(x=df['Variance'], y=df['Category'], orientation='h', marker_color=df['Color'], text=df['Text'], customdata=df[['Budgeted', 'Actual']],
                           hovertemplate='<b>%{y}</b><br>Variance: %{x:+,}k<br>Budgeted: $%{customdata[0]}k<br>Actual: $%{customdata[1]}k<extra></extra>'))
    fig.update_traces(textposition='inside', textfont=dict(color='white', size=14, family="Arial, sans-serif"))
    fig.update_layout(title_text='<b>Annual Budget Variance (Actual vs. Budgeted)</b>', xaxis_title="Variance (in $ thousands)", yaxis_title="",
                      bargap=0.4, plot_bgcolor=BACKGROUND_GREY, title_x=0.5, font=dict(family="Arial, sans-serif", size=12))
    return fig

def plot_headcount_forecast(key: str) -> go.Figure:
    df = pd.DataFrame({'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'], 'Current FTEs': [8, 8, 8, 8], 'Forecasted Need': [8, 9, 10, 10]})
    df['Gap'] = df['Forecasted Need'] - df['Current FTEs']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current Headcount', x=df['Quarter'], y=df['Current FTEs'], marker_color=PRIMARY_COLOR, text=df['Current FTEs']))
    fig.add_trace(go.Bar(name='Resource Gap', x=df['Quarter'], y=df['Gap'], base=df['Current FTEs'], marker_color=WARNING_AMBER, text=df['Gap'].apply(lambda g: f'+{g}' if g > 0 else '')))
    fig.update_traces(textposition='inside', textfont_size=14)
    fig.update_layout(barmode='stack', title='<b>Resource Gap Analysis: Headcount vs. Forecasted Need</b>',
                      yaxis_title="Full-Time Equivalents (FTEs)", xaxis_title="Fiscal Quarter",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def display_departmental_okrs(key: str) -> None:
    df = pd.DataFrame({"Objective": ["Improve Validation Efficiency", "Enhance Team Capabilities", "Strengthen Compliance Posture"], "Key Result": ["Reduce Avg. Doc Cycle Time by 15%", "Certify 2 Engineers in GAMP 5", "Achieve Zero Major Findings in Next Audit"], "Status": ["On Track", "Complete", "On Track"]})
    def style_status(val: str) -> str:
        color = SUCCESS_GREEN if val in ["On Track", "Complete"] else WARNING_AMBER
        return f"background-color: {color}; color: white; text-align: center; font-weight: bold;"
    styled_df = df.style.map(style_status, subset=['Status']).set_properties(**{'text-align': 'left'}).hide(axis="index")
    st.dataframe(styled_df, use_container_width=True)

def plot_gantt_chart(key: str) -> go.Figure:
    df = pd.DataFrame([dict(Task="Project Atlas (Bioreactor)", Start='2023-01-01', Finish='2023-12-31', Phase='Execution'), dict(Task="Project Beacon (Assembly)", Start='2023-06-01', Finish='2024-06-30', Phase='Execution'), dict(Task="Project Comet (Vision)", Start='2023-09-01', Finish='2024-03-31', Phase='Planning')])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Phase", title="<b>Major Capital Project Timelines</b>", color_discrete_map={'Execution': PRIMARY_COLOR, 'Planning': WARNING_AMBER})
    today_date = pd.to_datetime('today')
    fig.add_shape(type="line", x0=today_date, y0=0, x1=today_date, y1=1, yref='paper', line=dict(color="black", width=2, dash="dash"))
    fig.add_annotation(x=today_date, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="black"))
    fig.update_layout(title_x=0.5, yaxis_title=None, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_risk_burndown(key: str) -> go.Figure:
    df = pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar', 'Apr'], 'Open Risks (RPN > 25)': [12, 10, 7, 4], 'Target Burndown': [12, 9, 6, 3]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Month'], y=df['Open Risks (RPN > 25)'], name='Actual Open Risks', fill='tozeroy', line=dict(color=PRIMARY_COLOR, width=3), mode='lines+markers+text', text=df['Open Risks (RPN > 25)'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df['Month'], y=df['Target Burndown'], name='Target Burndown', line=dict(color=NEUTRAL_GREY, dash='dash')))
    fig.update_layout(title='<b>Project Atlas: High-Risk Burndown</b>', yaxis_title="Count of Open Risks", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def display_vendor_scorecard(key: str) -> None:
    df = pd.DataFrame({"Vendor": ["Vendor A (Automation)", "Vendor B (Components)"], "On-Time Delivery (%)": [95, 88], "FAT First-Pass Yield (%)": [92, 98], "Doc Package GDP Error Rate (%)": [2, 8], "Overall Score": [91, 85]})
    styled_df = df.style.background_gradient(cmap='RdYlGn', subset=['On-Time Delivery (%)', 'FAT First-Pass Yield (%)', 'Overall Score'], vmin=70, vmax=100).background_gradient(cmap='RdYlGn_r', subset=['Doc Package GDP Error Rate (%)'], vmin=0, vmax=10).format('{:.0f}', subset=df.columns.drop('Vendor')).hide(axis="index")
    st.dataframe(styled_df, use_container_width=True)

def create_rtm_data_editor(key: str) -> None:
    df_data = [{"ID": "URS-001", "User Requirement": "System must achieve a batch titer of >= 5 g/L.", "Risk": "High", "Test Case": "PQ-TP-001", "Status": "PASS"}, {"ID": "URS-012", "User Requirement": "System must have an E-Stop.", "Risk": "High", "Test Case": "OQ-TP-015", "Status": "PASS"}, {"ID": "URS-031", "User Requirement": "HMI must be 21 CFR Part 11 compliant.", "Risk": "High", "Test Case": "CSV-TP-001", "Status": "GAP"}]
    df = pd.DataFrame(df_data)
    coverage = len(df[df["Test Case"] != ""]) / len(df) * 100
    st.metric("Traceability Coverage", f"{coverage:.1f}%", help="Percentage of requirements linked to a test case.")
    st.info("This is an interactive editor. Change 'Status' from 'GAP' to 'PASS' to see the alert disappear.")
    edited_df = st.data_editor(df, use_container_width=True, hide_index=True, column_config={"Status": st.column_config.SelectboxColumn("Status", options=["PASS", "FAIL", "GAP", "In Progress"], required=True,)})
    if any(edited_df["Status"] == "GAP"):
        st.error("Critical traceability gap identified! This blocks validation release until a test case is linked and passed.", icon="🚨")

def create_v_model_figure(key: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["<b>URS</b>", "<b>Functional Spec</b>", "<b>Design Spec</b>", "<b>Code/Config</b>"], textposition="bottom center", line=dict(color=PRIMARY_COLOR, width=3), marker=dict(size=15)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["<b>Unit/FAT</b>", "<b>SAT</b>", "<b>IQ/OQ</b>", "<b>PQ</b>"], textposition="top center", line=dict(color=SUCCESS_GREEN, width=3), marker=dict(size=15)))
    for i in range(4):
        fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color=NEUTRAL_GREY, width=1, dash="dot"))
    fig.add_annotation(x=2.5, y=4.5, text="<b>Specification / Design</b>", showarrow=False, font=dict(color=PRIMARY_COLOR, size=14))
    fig.add_annotation(x=6.5, y=4.5, text="<b>Verification / Qualification</b>", showarrow=False, font=dict(color=SUCCESS_GREEN, size=14))
    fig.update_layout(title_text="<b>The Validation V-Model (per GAMP 5)</b>", title_x=0.5, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_risk_matrix(key: str) -> None:
    severity = [10, 9, 6, 8, 7, 5]; probability = [2, 3, 4, 3, 5, 1]; risk_level = [s * p for s, p in zip(severity, probability)]; text = ["Contamination", "Incorrect Titer", "Software Crash", "Incorrect Buffer Add", "Sensor Failure", "HMI Lag"]
    fig = go.Figure(data=go.Scatter(x=probability, y=severity, mode='markers+text', text=text, textposition="top center", marker=dict(size=[r*1.5 for r in risk_level], sizemin=10, color=risk_level, colorscale="YlOrRd", showscale=True, colorbar_title="RPN")))
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=5.5, fillcolor=SUCCESS_GREEN, opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=2.5, y0=5.5, x1=6, y1=11, fillcolor=ERROR_RED, opacity=0.2, layer="below", line_width=0)
    fig.add_annotation(x=1.25, y=2.75, text="Acceptable", showarrow=False, font_size=12, textangle=-45)
    fig.add_annotation(x=4.25, y=8.25, text="Unacceptable<br>(Mitigation Required)", showarrow=False, font_size=12, textangle=-45)
    fig.update_layout(title='<b>Process Risk Matrix (pFMEA)</b>', xaxis_title='Probability of Occurrence', yaxis_title='Severity of Effect', xaxis=dict(range=[0, 6], tickmode='linear', tick0=1, dtick=1), yaxis=dict(range=[0, 11], tickmode='linear', tick0=1, dtick=1), plot_bgcolor=BACKGROUND_GREY, title_x=0.5)
    col1, col2 = st.columns([2,1])
    with col1: st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("RPN Mitigation Threshold", "25", help="Risks with a Risk Priority Number (RPN = Sev x Prob) > 25 require mandatory mitigation per SOP-QA-045.")
        st.info("This risk-based approach ensures validation efforts are focused on the highest-impact failure modes, a core principle of **ISO 14971** and **GAMP 5**.")

def display_fat_sat_summary(key: str) -> None:
    col1, col2, col3 = st.columns(3); col1.metric("FAT Protocol Pass Rate", "95%"); col2.metric("SAT Protocol Pass Rate", "100%"); col3.metric("FAT-to-SAT Deviations", "0 New Major")
    st.info("For critical parameters, we use **Two One-Sided T-Tests (TOST)** to prove statistical equivalence between FAT and SAT results, ensuring no performance degradation during shipping and handling.")

def plot_oq_challenge_results(key: str) -> go.Figure:
    setpoints = [30, 37, 45, 37, 30]; rng = np.random.default_rng(10); actuals = [rng.normal(sp, 0.1) for sp in setpoints]
    time = pd.to_datetime(['2023-10-01 08:00', '2023-10-01 09:00', '2023-10-01 10:00', '2023-10-01 11:00', '2023-10-01 12:00'])
    fig = go.Figure(); fig.add_trace(go.Scatter(x=time, y=setpoints, name='Setpoint (°C)', mode='lines+markers', line=dict(shape='hv', dash='dash', color=NEUTRAL_GREY, width=3))); fig.add_trace(go.Scatter(x=time, y=actuals, name='Actual (°C)', mode='lines+markers', line=dict(color=PRIMARY_COLOR, width=3)))
    fig.add_hrect(y0=36.5, y1=37.5, line_width=0, fillcolor=SUCCESS_GREEN, opacity=0.1, annotation_text="Acceptance Band", annotation_position="bottom right")
    fig.update_layout(title='<b>OQ Challenge: Bioreactor Temperature Control</b>', xaxis_title='Time', yaxis_title='Temperature (°C)', title_x=0.5, plot_bgcolor=BACKGROUND_GREY); return fig

def plot_process_stability_chart(key: str) -> go.Figure:
    rng = np.random.default_rng(22); data = rng.normal(5.2, 0.25, 25); df = pd.DataFrame({'Titer': data}); df['MR'] = df['Titer'].diff().abs()
    I_CL = df['Titer'].mean(); MR_CL = df['MR'].mean(); I_UCL = I_CL + 2.66 * MR_CL; I_LCL = I_CL - 2.66 * MR_CL; MR_UCL = 3.267 * MR_CL
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart</b>", "<b>Moving Range (MR) Chart</b>"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Titer'], name='Titer (g/L)', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1)
    fig.add_hline(y=I_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="CL"); fig.add_hline(y=I_UCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL"); fig.add_hline(y=I_LCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL")
    fig.add_trace(go.Scatter(x=df.index, y=df['MR'], name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1)
    fig.add_hline(y=MR_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="CL"); fig.add_hline(y=MR_UCL, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL")
    fig.update_layout(height=500, showlegend=False, title_text="<b>Process Stability (I-MR Chart) for PQ Run 1 Titer</b>", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_csv_dashboard(key: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("21 CFR Part 11 Compliance Status", "PASS", "✔️", help="Electronic records and signatures meet all technical and procedural requirements.")
    with col2:
        st.metric("Data Integrity Risk Score", "Low", "-5% vs Last Quarter", help="Calculated based on ALCOA+ principles.")
    df = pd.DataFrame({ "GAMP 5 Category": ["Cat 4: Configured", "Cat 5: Custom"], "System": ["HMI Software", "LIMS Interface"], "Status": ["Validation Complete", "IQ/OQ In Progress"] })
    st.dataframe(style_dataframe(df), use_container_width=True)

def plot_cleaning_validation_results(key: str) -> go.Figure:
    df = pd.DataFrame({'Sample Location': ['Swab 1 (Reactor Wall)', 'Swab 2 (Agitator Blade)', 'Swab 3 (Fill Nozzle)', 'Final Rinse'], 'TOC Result (ppb)': [150, 180, 165, 25], 'Acceptance Limit (ppb)': [500, 500, 500, 50]})
    df['Status'] = df.apply(lambda row: SUCCESS_GREEN if row['TOC Result (ppb)'] < row['Acceptance Limit (ppb)'] else ERROR_RED, axis=1)
    fig = px.bar(df, x='Sample Location', y='TOC Result (ppb)', title='<b>Cleaning Validation Results (Total Organic Carbon)</b>', text='TOC Result (ppb)')
    fig.update_traces(marker_color=df['Status'])
    fig.add_trace(go.Scatter(x=df['Sample Location'], y=df['Acceptance Limit (ppb)'], name='Acceptance Limit', mode='lines+markers', line=dict(color=ERROR_RED, dash='dash', width=3)))
    fig.update_layout(title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_shipping_validation_temp(key: str) -> go.Figure:
    rng = np.random.default_rng(30); time = pd.to_datetime(pd.date_range("2023-01-01", periods=48, freq="h")); temp = rng.normal(4, 0.5, 48); temp[24] = 8.5 
    fig = px.line(x=time, y=temp, title='<b>Shipping Lane PQ: Temperature Profile</b>', markers=True)
    fig.add_hrect(y0=2, y1=8, line_width=0, fillcolor=SUCCESS_GREEN, opacity=0.2, annotation_text="In Spec (2-8°C)", annotation_position="top left")
    excursion_time = time[temp > 8]
    if not excursion_time.empty:
        fig.add_annotation(x=excursion_time[0], y=temp[24], text="Excursion!", showarrow=True, arrowhead=1, ax=0, ay=-40, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor=ERROR_RED, opacity=0.8, font=dict(color="white"))
    fig.update_layout(yaxis_title="Temperature (°C)", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_doe_optimization(key: str) -> go.Figure:
    temp = np.linspace(30, 40, 10); ph = np.linspace(6.8, 7.6, 10); temp_grid, ph_grid = np.meshgrid(temp, ph); signal = 100 - (temp_grid - 37)**2 - 20*(ph_grid - 7.2)**2 + np.random.rand(10, 10)*2
    fig = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis', colorbar_title='Yield')])
    fig.update_layout(title='<b>DOE Response Surface for Process Optimization</b>', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Product Yield (%)'), title_x=0.5, margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_kaizen_roi_chart(key: str) -> go.Figure:
    fig = go.Figure(go.Waterfall(name="2023 Savings", orientation="v", measure=["relative", "relative", "relative", "total"], x=["Optimize CIP Cycle Time", "Implement PAT Sensor", "Reduce Line Changeover", "<b>Total Annual Savings</b>"], text=[f"+${v}k" for v in [150, 75, 220, 445]], y=[150, 75, 220, 445], connector={"line": {"color": NEUTRAL_GREY}}, increasing={"marker":{"color":SUCCESS_GREEN}}, decreasing={"marker":{"color":ERROR_RED}}, totals={"marker":{"color":PRIMARY_COLOR, "line": dict(color='white', width=2)}}))
    fig.update_layout(title="<b>Continuous Improvement (Kaizen) ROI Tracker</b>", yaxis_title="Annual Savings ($k)", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_deviation_trend_chart(key: str) -> go.Figure:
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=6, freq="ME")); data = {'Month': dates, 'Bioreactor Suite C': [5, 6, 8, 7, 9, 11], 'Purification Skid A': [4, 5, 4, 6, 5, 6], 'Packaging Line 2': [2, 1, 3, 2, 4, 3]}
    df = pd.DataFrame(data); fig = px.area(df, x='Month', y=['Bioreactor Suite C', 'Purification Skid A', 'Packaging Line 2'], title='<b>Deviation Trend by Manufacturing System</b>', markers=True); fig.update_layout(yaxis_title="Number of Deviations", title_x=0.5, plot_bgcolor=BACKGROUND_GREY, legend_title_text='System')
    return fig

def run_urs_risk_nlp_model(key: str) -> go.Figure:
    reqs = ["System shall have 99.5% uptime.", "Arm must move quickly.", "UI should be easy.", "System shall process 200 units/hour.", "System must be robust.", "Pump shall dispense 5.0 mL +/- 0.05 mL."]; labels = [0, 1, 1, 0, 1, 0]; criticality = [9, 6, 4, 10, 7, 10]
    df = pd.DataFrame({'Requirement': reqs, 'Risk_Label': labels, 'Criticality': criticality}); tfidf = TfidfVectorizer(stop_words='english'); X = tfidf.fit_transform(df['Requirement']); y = df['Risk_Label']
    model = LogisticRegression(random_state=42).fit(X, y); df['Ambiguity Score'] = model.predict_proba(X)[:, 1]
    fig = px.scatter(df, x='Ambiguity Score', y='Criticality', text=df.index, title='<b>URS Ambiguity Risk Model</b>', labels={'Ambiguity Score': 'Predicted Ambiguity Score (0=Clear, 1=Ambiguous)', 'Criticality': 'Process Impact'}, hover_data=['Requirement'], size=[15]*len(df), color='Ambiguity Score', color_continuous_scale='YlOrRd')
    fig.update_traces(textposition='top center'); fig.add_vline(x=0.5, line_dash="dash", annotation_text="Action Threshold"); fig.add_hline(y=7.5, line_dash="dash", annotation_text="High Criticality")
    fig.add_annotation(x=0.75, y=5, text="High Ambiguity, <br>Low Impact:<br><b>Clarify</b>", showarrow=False, bgcolor="#FFC107", borderpad=4)
    fig.add_annotation(x=0.75, y=9, text="High Ambiguity, <br>High Impact:<br><b>REJECT/REWRITE!</b>", showarrow=False, bgcolor="#D32F2F", font=dict(color='white'), borderpad=4)
    fig.update_layout(title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

# --- PAGE RENDERING FUNCTIONS ---
def render_main_page() -> None:
    st.title("🤖 Automated Equipment Validation Portfolio"); st.subheader("A Live Demonstration of Modern Validation Leadership"); st.divider()
    st.markdown("Welcome. This interactive environment provides **undeniable proof of expertise in the end-to-end validation of automated manufacturing equipment** in a strictly regulated GMP environment. It simulates how an effective leader manages a validation function, with a relentless focus on aligning technical execution, **Quality Systems (per 21 CFR 820 & ISO 13485)**, and strategic capital projects.")
    
    st.subheader("Key Program Health KPIs", divider='blue');
    c1, c2, c3 = st.columns(3)
    c1.metric("Validation Program Compliance", "98%", delta="1%", help="Percentage of systems in a validated state and on their periodic review schedule.")
    c2.metric("Quality First Time Rate", "91%", delta="-2%", help="Percentage of validation protocols executed without deviation.")
    c3.metric("Capital Project On-Time Delivery", "95%", delta="5%", help="Validation deliverables completed on or before schedule for major capital projects.")
    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    c4.metric("CAPEX Validation Spend vs. Budget", "97%", delta="-3%", help="Total validation capital expenditure against the annual fiscal plan.", delta_color="inverse")
    c5.metric("Avg. Protocol Review Cycle Time", "8.2 Days", delta="1.5 Days", help="Average time from document submission to final QA approval.", delta_color="inverse")
    c6.metric("Open Validation-Related CAPAs", "3", delta="1", help="Number of open Corrective/Preventive Actions where Validation is the owner.", delta_color="inverse")
    c7, c8, c9 = st.columns(3)
    c7.metric("Systems Overdue for Periodic Review", "1", delta="1", help="Critical GxP systems that have passed their scheduled periodic review date.", delta_color="inverse")
    c8.metric("Vendor Documentation Quality", "96%", delta="4%", help="Percentage of vendor documents (e.g., FAT, drawings) accepted on first pass without GDP errors.")
    c9.metric("Team Utilization Rate", "88%", delta="-5%", help="Percentage of available engineering hours logged against active projects.", delta_color="inverse")

    with st.expander("Click here for a guided tour of this portfolio's key capabilities"):
        st.info("1.  **Start with Strategy (`Tab 1`):** See how a leader manages budgets, forecasts resources, and sets departmental goals (OKRs).\n2.  **Oversee the Portfolio (`Tab 2`):** View the Gantt chart, RAG status, and resource allocation for all validation projects.\n3.  **Deep Dive into Execution (`Tab 3`):** Walk through a live simulation of a major capital project from FAT to PQ.\n4.  **Verify Specialized Expertise (`Tab 4`):** Explore capabilities in CSV, Cleaning, and Shipping Validation.\n5.  **Confirm Lifecycle Management (`Tab 5`):** Understand the approach to maintaining the validated state.\n6.  **Inspect the Documentation (`Tab 6`):** See how compliant, auditable protocols and reports are generated.")

def render_strategic_management_page() -> None:
    st.title("📈 1. Strategic Management & Business Acumen")
    render_manager_briefing(title="Leading Validation as a Business Unit", content="An effective manager must translate technical excellence into business value. This dashboard demonstrates the ability to manage budgets, plan for future headcount needs based on the project pipeline, and align departmental goals with the strategic objectives of the site.", reg_refs="ISO 13485:2016 (Sec 5 & 6), 21 CFR 820.20", business_impact="Ensures the validation department is a strategic, financially responsible partner that enables the company's growth and compliance goals.", quality_pillar="Resource Management & Financial Acumen.", risk_mitigation="Proactively identifies and mitigates resource shortfalls and budget variances before they impact project timelines.")
    with st.container(border=True): st.subheader("Departmental OKRs (Objectives & Key Results)", help="Aligns team's daily work with high-level company goals."); display_departmental_okrs(key="okrs"); st.success("**Actionable Insight:** The team is on track to meet its efficiency and compliance goals for the year.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True): st.subheader("Annual Budget Performance", help="Tracks actual spend against the department's annual budget."); st.plotly_chart(plot_budget_variance(key="budget"), use_container_width=True); st.success("**Actionable Insight:** Operating within overall budget. The slight CapEx overage was an approved expenditure for accelerated project timelines, offset by contractor savings.")
    with col2:
        with st.container(border=True): st.subheader("Headcount & Resource Forecasting", help="Compares current team size against forecasted resource needs."); st.plotly_chart(plot_headcount_forecast(key="headcount"), use_container_width=True); st.success("**Actionable Insight:** The forecast indicates a resource gap of 2 FTEs by Q3. This data justifies the hiring requisition for one Automation Engineer and one Validation Specialist.")
    with st.container(border=True): st.subheader("AI-Powered Capital Project Duration Forecaster"); run_project_duration_forecaster("duration_ai")

def render_project_portfolio_page() -> None:
    st.title("📂 2. Project & Portfolio Management")
    render_manager_briefing(title="Managing the Validation Project Portfolio", content="This command center demonstrates the ability to manage a portfolio of competing capital projects, balancing priorities, allocating finite resources, and providing clear, high-level status updates to the PMO and site leadership.", reg_refs="Project Management Body of Knowledge (PMBOK), ISO 13485: 5.6", business_impact="Provides executive-level visibility into Validation's contribution to corporate goals, enables proactive risk management, and ensures strategic alignment of the department's people.", quality_pillar="Project Governance & Oversight.", risk_mitigation="Prevents budget overruns and schedule delays through proactive monitoring of CPI/SPI metrics and resource allocation.")
    with st.container(border=True): st.subheader("Capital Project Timelines (Gantt Chart)"); st.info("**Purpose:** The Gantt chart provides a high-level visual timeline for all major capital projects, highlighting dependencies and overall schedule health."); st.plotly_chart(plot_gantt_chart(key="gantt"), use_container_width=True)
    with st.container(border=True):
        st.subheader("Capital Project Portfolio Health")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Purpose:** The RAG (Red-Amber-Green) status provides an immediate, at-a-glance summary of portfolio health for executive review and PMO meetings.")
            st.markdown("##### RAG Status"); st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True)
        with col2:
            st.info("**Purpose:** Key Performance Indicators (KPIs) like SPI and CPI are industry-standard metrics (per PMBOK) to quantitatively track project performance against the established schedule and budget baselines.")
            st.markdown("##### Key Project Metrics (Project Beacon)")
            st.metric("Schedule Performance Index (SPI)", "0.92", "-8%", help="SPI < 1.0 indicates the project is behind schedule.")
            st.metric("Cost Performance Index (CPI)", "0.85", "-15%", help="CPI < 1.0 indicates the project is over budget.")
            st.success("**Actionable Insight:** Project Beacon's SPI and CPI are below 1.0, indicating it is both behind schedule and over budget. An urgent review meeting with the project lead is required to develop a recovery plan.")
    with st.container(border=True):
        st.info("**Purpose:** The Risk Burndown chart visually tracks the team's effectiveness at mitigating and closing high-impact project risks over time. The goal is for the 'Actual' line to be at or below the 'Target' line.")
        st.plotly_chart(plot_risk_burndown("risk_burn"), use_container_width=True)
        st.success("**Actionable Insight:** The team is effectively mitigating risks ahead of schedule, which reduces the probability of future project delays or quality issues.")

def render_e2e_validation_hub_page() -> None:
    st.title("🔩 Live E2E Validation Walkthrough: Project Atlas")
    render_manager_briefing(title="Executing a Compliant Validation Lifecycle (per ASTM E2500)", content="This hub presents the entire validation lifecycle in a single, comprehensive view, simulating the execution of a major capital project. It provides tangible evidence of owning deliverables from design and risk management through to final performance qualification.", reg_refs="FDA 21 CFR 820.75, ISO 13485:2016 (Sec 7.5.6), GAMP 5, ASTM E2500", business_impact="Ensures new manufacturing equipment is brought online on-time, on-budget, and in a fully compliant state, directly enabling production launch.", quality_pillar="Design Controls & Risk-Based Verification.", risk_mitigation="Prevents costly redesigns and validation failures by ensuring testability is built-in from the URS phase using tools like the V-Model and pFMEA.")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Phase 1: Design & Risk Management"); st.info("The 'left side of the V-Model' focuses on proactive planning, ensuring quality and testability are designed into the system from the start.")
        with st.container(border=True): st.subheader("Validation V-Model"); st.plotly_chart(create_v_model_figure("vmodel"), use_container_width=True)
        with st.container(border=True): st.subheader("AI-Powered URS Risk Analysis"); st.plotly_chart(run_urs_risk_nlp_model("urs_risk"), use_container_width=True); st.success("**Actionable Insight:** Requirements 2, 3, and 5 flagged for rewrite due to high ambiguity.")
        with st.container(border=True): st.subheader("User Requirements Traceability (RTM)"); create_rtm_data_editor("rtm")
        with st.container(border=True): st.subheader("Process Risk Management (pFMEA)"); plot_risk_matrix("fmea")
    with col2:
        st.header("Phases 2-4: Execution & Qualification"); st.info("The 'right side of the V-Model' focuses on generating objective evidence that the system meets all requirements and is fit for its intended use.")
        st.subheader("Phase 2: Factory & Site Acceptance Testing", divider='blue')
        with st.container(border=True): st.markdown("Purpose: To execute acceptance testing at the vendor's facility (FAT) and our site (SAT). The goal is to catch as many issues as possible *before* formal qualification begins."); display_fat_sat_summary("fat_sat")
        st.subheader("Phase 3: Installation & Operational Qualification", divider='blue')
        with st.container(border=True): st.markdown("Purpose: The IQ provides documented evidence of correct installation. The OQ challenges the equipment's functions to prove it operates as intended throughout its specified operating ranges."); st.plotly_chart(plot_oq_challenge_results("oq_plot"), use_container_width=True)
        st.subheader("Phase 4: Performance Qualification", divider='blue')
        with st.container(border=True):
            st.markdown("Purpose: The PQ is the final step, providing documented evidence that the equipment can consistently produce quality product under normal, real-world manufacturing conditions.")
            c1, c2 = st.columns(2)
            with c1: st.subheader("Process Capability"); st.plotly_chart(plot_cpk_analysis("pq_cpk"), use_container_width=True)
            with c2: st.subheader("Process Stability"); st.plotly_chart(plot_process_stability_chart("pq_spc"), use_container_width=True)

def render_specialized_validation_page() -> None:
    st.title("🧪 4. Specialized Validation Hubs")
    render_manager_briefing(title="Demonstrating Breadth of Expertise", content="Beyond standard equipment qualification, a Validation Manager must be fluent in specialized validation disciplines critical to GMP manufacturing. This hub showcases expertise in Computer System Validation (CSV), Cleaning Validation, and Process Characterization.", reg_refs="21 CFR Part 11, GAMP 5, PDA TR 29 (Cleaning Validation)", business_impact="Ensures all aspects of the manufacturing process, including supporting systems and processes, are fully compliant and controlled, preventing common sources of regulatory findings.", quality_pillar="Cross-functional Technical Leadership.", risk_mitigation="Ensures compliance in niche, high-risk areas like data integrity (CSV) and cross-contamination (Cleaning) that are frequent targets of audits.")
    tab1, tab2, tab3, tab4 = st.tabs(["🖥️ Computer System Validation (CSV)", "🧼 Cleaning Validation", "🔬 Process Characterization (DOE)", "📦 Shipping Validation"])
    with tab1: st.subheader("GAMP 5 CSV for Automated Systems"); st.info("**Purpose:** This dashboard tracks the validation status of all GxP computerized systems associated with a project, ensuring compliance with data integrity and 21 CFR Part 11 requirements for electronic records and signatures."); plot_csv_dashboard("csv"); st.success("**Actionable Insight:** The successful validation of the HMI confirms 21 CFR Part 11 compliance for electronic signatures, unblocking the system for GMP use. The LIMS interface validation is the next critical path item.")
    with tab2: st.subheader("Cleaning Validation for Multi-Product Facility"); st.info("**Purpose:** This plot shows the results from a cleaning validation study, confirming that residual product and cleaning agent levels are below the pre-defined, toxicologically-based acceptance limits to prevent cross-contamination."); st.plotly_chart(plot_cleaning_validation_results("cleaning"), use_container_width=True); st.success("**Actionable Insight:** All results are well below 50% of the acceptance limit, providing a high degree of assurance that the cleaning process effectively prevents cross-contamination. The cleaning procedure can be approved and finalized.")
    with tab3: st.subheader("Process Characterization using Design of Experiments (DOE)"); st.info("**Purpose:** DOE is a powerful statistical tool used during process development to identify the optimal settings (e.g., temperature, pH) that maximize product yield and robustness. This data is critical for defining and defending the Normal Operating Range (NOR) during validation."); st.plotly_chart(plot_doe_optimization("doe"), use_container_width=True); st.success("**Actionable Insight:** The response surface clearly defines the NOR for Temperature (36-38°C) and pH (7.1-7.3). These parameters will be specified in the batch record and challenged at their limits during OQ.")
    with tab4: st.subheader("Shipping Lane Performance Qualification"); st.info("**Purpose:** This PQ study uses calibrated temperature loggers to confirm that the validated shipping container and process can maintain the required temperature range (e.g., 2-8°C) over a simulated, worst-case transit duration."); st.plotly_chart(plot_shipping_validation_temp("shipping"), use_container_width=True); st.success("**Actionable Insight:** Despite the brief external temperature excursion to 30°C at hour 24, the qualified shipper maintained internal temperatures within the required 2-8°C range, validating the robustness of the packaging configuration for this shipping lane.")

def render_validation_program_health_page() -> None:
    st.title("⚕️ 5. Validation Program Health & Continuous Improvement")
    render_manager_briefing(title="Maintaining the Validated State", content="This dashboard demonstrates the ongoing oversight required to manage the site's validation program health. It showcases a data-driven approach to **Periodic Review**, the development of a risk-based **Revalidation Strategy**, and the execution of **Continuous Improvement Initiatives**.", reg_refs="FDA 21 CFR 820.75(c) (Revalidation), ISO 13485:2016 (Sec 8.4)", business_impact="Ensures long-term compliance, prevents costly process drifts, optimizes resource allocation for revalidation, and supports uninterrupted supply of medicine to patients.", quality_pillar="Lifecycle Management & Continuous Improvement.", risk_mitigation="Guards against compliance drift and ensures systems remain in a validated state throughout their operational life, preventing production holds or recalls.")
    tab1, tab2 = st.tabs(["📊 Periodic Review & Revalidation Strategy", "📈 Continuous Improvement Tracker"])
    with tab1:
        st.subheader("Risk-Based Periodic Review Schedule")
        review_data = {"System": ["Bioreactor C", "Purification A", "WFI System", "HVAC - Grade A", "Inspection System", "CIP Skid B"], "Risk Level": ["High", "High", "High", "Medium", "Medium", "Low"], "Last Review": ["2023-01-15", "2023-02-22", "2023-08-10", "2022-11-05", "2023-09-01", "2022-04-20"], "Next Due": ["2024-01-15", "2024-02-22", "2024-08-10", "2024-11-05", "2025-09-01", "2025-04-20"], "Status": ["Complete", "Complete", "On Schedule", "DUE", "On Schedule", "On Schedule"]}
        review_df = pd.DataFrame(review_data)
        def highlight_status(row):
            return ['background-color: #FFC7CE; color: black; font-weight: bold;'] * len(row) if row["Status"] == "DUE" else [''] * len(row)
        st.dataframe(review_df.style.apply(highlight_status, axis=1), use_container_width=True, hide_index=True)
        st.error("**Actionable Insight:** The Periodic Review for the **HVAC - Grade A Area** is now due. A Validation Engineer will be assigned to initiate the review this week.")
    with tab2:
        st.subheader("Continuous Improvement (Kaizen) Initiative Tracker")
        st.info("**Context:** An effective validation program uses data to drive improvement. The **Deviation Trend** chart identifies operational problems (the 'why'), while the **ROI Tracker** provides the business case for funding solutions (the 'what for').")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_kaizen_roi_chart("kaizen_roi"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_deviation_trend_chart("deviation_trend"), use_container_width=True)
        st.success("**Actionable Insight:** The rising deviation trend in the Bioreactor Suite C directly validates the focus of our Kaizen efforts (e.g., 'Implement PAT Sensor'). The ROI tracker provides a strong business case to leadership for continuing these improvement projects.")

def render_documentation_hub_page() -> None:
    st.title("🗂️ 6. Validation Documentation & Audit Defense Hub")
    render_manager_briefing(title="Orchestrating Compliant Validation Documentation", content="This hub demonstrates the ability to generate and manage the compliant, auditable documentation that forms the core of a successful validation package. The templates and simulations below prove expertise in creating documents that meet the stringent requirements of **21 CFR Part 820** and **ISO 13485**.", reg_refs="21 CFR 820.40 (Document Controls), GAMP 5 Good Documentation Practice, 21 CFR Part 11", business_impact="Ensures audit-proof documentation, accelerates review cycles by providing clear templates, and fosters seamless collaboration between Engineering, Manufacturing, Quality, and Regulatory.", quality_pillar="Good Documentation Practice (GDP) & Audit Readiness.", risk_mitigation="Minimizes review cycles and audit findings by ensuring documentation is attributable, legible, contemporaneous, original, and accurate (ALCOA+).")
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True): st.subheader("Document Approval Workflow"); st.info("Simulates the eQMS workflow."); st.markdown("Status for `VAL-MP-001_Project_Atlas`:"); st.divider(); st.markdown("✔️ **Validation Lead (Self):** Approved `2024-01-15`\n✔️ **Process Engineering Lead:** Approved `2024-01-16`\n✔️ **Quality Assurance Lead:** Approved `2024-01-17`\n🟠 **Manufacturing Lead:** Pending Review\n⬜ **Head of Engineering:** Not Started")
    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer"); st.info("The following are professionally rendered digital artifacts that simulate documents within a validated eQMS.")
            # --- FIX: Correct IndentationError by creating a proper 'with' block ---
            with st.expander("📄 **View Professional IQ/OQ Protocol Template**"):
                _render_professional_protocol_template()
            with st.expander("📋 **View Professional PQ Report Template**"):
                _render_professional_report_template()

def _render_professional_report_template() -> None:
    st.header("PQ Report: VAL-TR-201"); st.subheader("Automated Bioreactor Suite (ASSET-123)"); st.divider()
    meta_cols = st.columns(4); meta_cols[0].metric("Document ID", "VAL-TR-201"); meta_cols[1].metric("Version", "1.0"); meta_cols[2].metric("Status", "Final"); meta_cols[3].metric("Approval Date", "2024-03-01"); st.divider()
    st.markdown("##### 1.0 Summary & Conclusion")
    col1, col2 = st.columns([2, 1])
    with col1: st.write("Three successful, consecutive Performance Qualification (PQ) runs were executed on the Bioreactor System per protocol VAL-TP-201. The results confirm that the system reliably produces product meeting all pre-defined Critical Quality Attributes (CQAs) under normal manufacturing conditions."); st.success("**Conclusion:** The Automated Bioreactor System (ASSET-123) has met all PQ acceptance criteria and is **qualified for use in commercial GMP manufacturing.**")
    with col2: st.metric("Overall Result", "PASS"); st.metric("Final CpK (Product Titer)", "1.67", help="Exceeds target of >= 1.33")
    st.markdown("##### 3.0 Results vs. Acceptance Criteria")
    results_data = {'CQA': ['Titer (g/L)', 'Viability (%)', 'Impurity A (%)'], 'Specification': ['>= 5.0', '>= 95%', '<= 0.5%'], 'Run 1 Result': [5.2, 97, 0.41], 'Run 2 Result': [5.1, 96, 0.44], 'Run 3 Result': [5.3, 98, 0.39], 'Pass/Fail': ['PASS', 'PASS', 'PASS']}
    results_df = pd.DataFrame(results_data)
    def style_pass_fail(val: str) -> str:
        color = SUCCESS_GREEN if val == 'PASS' else ERROR_RED
        return f"background-color: {color}; color: white; text-align: center; font-weight: bold;"
    styled_df = results_df.style.map(style_pass_fail, subset=['Pass/Fail'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = { "Executive Summary": render_main_page, "1. Strategic Management": render_strategic_management_page, "2. Project & Portfolio Management": render_project_portfolio_page, "3. E2E Validation Walkthrough": render_e2e_validation_hub_page, "4. Specialized Validation Hubs": render_specialized_validation_page, "5. Validation Program Health": render_validation_program_health_page, "6. Documentation & Audit Defense": render_documentation_hub_page }
st.sidebar.title("🛠️ Validation Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()
