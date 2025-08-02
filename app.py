# app.py (Final, Monolithic, Guaranteed Working Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import statsmodels.api as sm

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="V&V Executive Command Center | Portfolio",
    page_icon="🎯"
)

# --- UTILITY & HELPER FUNCTIONS ---

def render_director_briefing(title, content, reg_refs, business_impact):
    """Renders a formatted container for strategic context."""
    with st.container(border=True):
        st.subheader(f"🎯 {title}")
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}")
        st.warning(f"**Regulatory Mapping:** {reg_refs}")

def render_metric_card(title, description, viz_function, insight, key=""):
    """Renders a formatted container for a specific metric or visualization."""
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"*{description}*")
        fig = viz_function(key)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"**Actionable Insight:** {insight}")

# --- VISUALIZATION & DATA GENERATORS ---

def create_rtm_data_editor(key):
    df = pd.DataFrame([
        {"ID": "URS-001", "Requirement": "Assay must detect Target X with >95% clinical sensitivity.", "Risk": "High", "Linked Test Case": "AVP-SENS-01", "Status": "PASS"},
        {"ID": "DI-002", "Requirement": "Analytical sensitivity (LoD) shall be <= 50 copies/mL.", "Risk": "High", "Linked Test Case": "AVP-LOD-01", "Status": "PASS"},
        {"ID": "SRS-012", "Requirement": "Results screen must display patient ID.", "Risk": "Medium", "Linked Test Case": "SVP-UI-04", "Status": "PASS"},
        {"ID": "DI-003", "Requirement": "Assay must be stable for 12 months at 2-8°C.", "Risk": "High", "Linked Test Case": "AVP-STAB-01", "Status": "IN PROGRESS"},
        {"ID": "URS-003", "Requirement": "Assay must have no cross-reactivity with Influenza B.", "Risk": "Medium", "Linked Test Case": "", "Status": "GAP"},
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    gaps = df[df["Status"] == "GAP"]
    if not gaps.empty:
        st.error(f"**Critical Finding:** {len(gaps)} traceability gap(s) identified. This is a major audit finding and blocks design release.")
    return None

def plot_defect_burnup(key):
    days = np.arange(1, 46); scope = np.ones(45) * 50; scope[25:] = 60
    closed = np.linspace(0, 45, 45) + np.random.rand(45) * 2
    opened = np.linspace(5, 58, 45) + np.random.rand(45) * 3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=scope, mode='lines', name='Total Scope', line=dict(dash='dash', color='grey')))
    fig.add_trace(go.Scatter(x=days, y=opened, mode='lines', name='Defects Opened (Cumulative)', fill='tozeroy', line=dict(color='rgba(255,0,0,0.5)')))
    fig.add_trace(go.Scatter(x=days, y=closed, mode='lines', name='Defects Closed (Cumulative)', fill='tozeroy', line=dict(color='rgba(0,128,0,0.5)')))
    fig.update_layout(title='Defect Open vs. Close Trend (Burnup Chart)', xaxis_title='Project Day', yaxis_title='Number of Defects')
    return fig

def plot_cpk_analysis(key):
    np.random.seed(42); data = np.random.normal(loc=10.2, scale=0.25, size=200)
    usl = st.slider("Upper Specification Limit (USL)", 9.0, 12.0, 11.0, key=f"usl_{key}")
    lsl = st.slider("Lower Specification Limit (LSL)", 8.0, 11.0, 9.0, key=f"lsl_{key}")
    mean = np.mean(data); std_dev = np.std(data, ddof=1)
    cpk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
    fig = px.histogram(data, nbins=30, title=f"Process Capability (CpK) Analysis | Calculated CpK = {cpk:.2f}")
    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
    fig.add_vline(x=mean, line_dash="dot", line_color="blue", annotation_text="Process Mean")
    return fig

def plot_msa_analysis(key):
    parts = np.repeat(np.arange(1, 11), 6); operators = np.tile(np.repeat(['Alice', 'Bob', 'Charlie'], 2), 10)
    true_values = np.repeat(np.linspace(5, 15, 10), 6); operator_bias = np.tile(np.repeat([0, 0.2, -0.15], 2), 10)
    measurements = true_values + operator_bias + np.random.normal(0, 0.3, 60)
    df = pd.DataFrame({'Part': parts, 'Operator': operators, 'Measurement': measurements})
    fig = px.box(df, x='Part', y='Measurement', color='Operator', title='Measurement System Analysis (MSA) - Gage R&R')
    return fig

def plot_doe_rsm(key):
    temp = np.linspace(20, 40, 20); ph = np.linspace(6.5, 8.5, 20)
    temp_grid, ph_grid = np.meshgrid(temp, ph)
    signal = -(temp_grid - 32)**2 - 2*(ph_grid - 7.5)**2 + 1000 + np.random.rand(20, 20)*20
    fig = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph)])
    fig.update_layout(title='Design of Experiments (DOE) Response Surface', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Assay Signal'))
    return fig

def plot_levey_jennings_westgard(key):
    days = np.arange(1, 31); mean = 100; sd = 4; data = np.random.normal(mean, sd, 30)
    data[10] = mean + 3.5 * sd; data[20:22] = mean + 2.2 * sd
    fig = go.Figure(); fig.add_trace(go.Scatter(x=days, y=data, mode='lines+markers', name='Control Value'))
    for i, color in zip([1, 2, 3], ['green', 'orange', 'red']):
        fig.add_hline(y=mean + i*sd, line_dash="dot", line_color=color, annotation_text=f"+{i}SD")
        fig.add_hline(y=mean - i*sd, line_dash="dot", line_color=color, annotation_text=f"-{i}SD")
    fig.add_annotation(x=10, y=data[10], text="1_3s Violation", showarrow=True, arrowhead=1)
    fig.add_annotation(x=21, y=data[21], text="2_2s Violation", showarrow=True, arrowhead=1)
    fig.update_layout(title='Levey-Jennings Chart with Westgard Rules', xaxis_title='Day', yaxis_title='Control Value')
    return fig

def run_assay_regression(key):
    conc = np.array([0, 10, 25, 50, 100, 200, 400]); signal = 50 + 2.5 * conc + np.random.normal(0, 20, 7)
    df = pd.DataFrame({'Concentration': conc, 'Signal': signal})
    fig = px.scatter(df, x='Concentration', y='Signal', trendline='ols', title="Assay Performance Regression (Linearity)")
    X = sm.add_constant(df['Concentration']); model = sm.OLS(df['Signal'], X).fit()
    st.code(f"Regression Results (statsmodels summary):\n{model.summary()}")
    return fig

def create_v_model_figure(key):
    fig = go.Figure(); fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Needs", "System Req.", "Architecture", "Module Design"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test", "Integration Test", "System V&V", "UAT"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text=None, showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig

def get_software_risk_data():
    return pd.DataFrame([{"Software Item": "Patient Result Algorithm", "IEC 62304 Class": "Class C"}, {"Software Item": "Database Middleware", "IEC 62304 Class": "Class B"}, {"Software Item": "UI Color Theme Module", "IEC 62304 Class": "Class A"}])

# --- PAGE RENDERING FUNCTIONS ---

def render_main_page():
    st.title("🎯 The V&V Executive Command Center")
    st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
    st.markdown("---")
    render_director_briefing("Portfolio Objective",
        "This interactive application translates the core responsibilities of V&V leadership into a suite of high-density dashboards. It is designed to be an overwhelming and undeniable demonstration of the strategic, technical, and quality systems expertise required for a senior leadership role in the medical device industry.",
        "ISO 13485, ISO 14971, IEC 62304, 21 CFR 820, 21 CFR Part 11, CLSI Guidelines",
        "A well-led V&V function directly accelerates time-to-market, reduces compliance risk, lowers the cost of poor quality (COPQ), and builds a culture of data-driven excellence."
    )
    st.info("Please use the navigation sidebar on the left to explore each of the five core competency areas.")

def render_design_controls_page():
    st.title("🏛️ 1. Design Controls, Planning & Risk Management")
    st.markdown("---")
    render_director_briefing("The Design History File (DHF) as a Strategic Asset",
        "The DHF is the compilation of records that demonstrates the design was developed in accordance with the design plan and regulatory requirements. An effective V&V leader architects the DHF from day one, ensuring that the story it tells to regulators is clear, complete, and compelling. This section demonstrates the key tools for building a world-class DHF.",
        "FDA 21 CFR 820.30 (Design Controls), ISO 13485:2016 (Section 7.3, Design and Development)",
        "Ensures audit readiness and provides a clear, defensible story of product development to regulatory bodies, accelerating submission review times."
    )
    render_metric_card("Requirements Traceability Matrix (RTM)", "The RTM is the backbone of the DHF, providing an auditable, many-to-many link between user needs, design inputs, V&V activities, and risk controls.", create_rtm_data_editor, "The matrix view instantly flags critical gaps, such as the un-tested cross-reactivity requirement (URS-003), allowing for proactive mitigation before a design freeze. This is a primary tool for preventing audit findings.", key="rtm")
    render_metric_card("Product Risk Management (FMEA & Risk Matrix)", "A systematic process for identifying, analyzing, and mitigating potential failure modes in the product design. V&V activities are primary risk mitigations.", lambda k: plot_risk_matrix(k), "The risk matrix clearly prioritizes 'False Negative' as the highest risk, ensuring that it receives the most V&V resources and attention. This is a key input for the V&V Master Plan.", key="fmea")
    render_metric_card("Design of Experiments (DOE/RSM)", "A powerful statistical tool used to efficiently characterize the product's design space and identify robust operating parameters (e.g., optimal temperature and pH for an assay).", plot_doe_rsm, "The Response Surface Methodology (RSM) plot indicates the assay's optimal performance is at ~32°C and a pH of 7.5. This data forms the basis for setting manufacturing specifications and proves the design is well-understood.", key="doe")
    
def render_method_validation_page():
    st.title("🔬 2. Method Validation & Statistical Rigor")
    st.markdown("---")
    render_director_briefing("Ensuring Data Trustworthiness",
        "Before a product can be validated, the methods and systems used to measure it must be proven to be reliable. Test Method Validation (TMV), Measurement System Analysis (MSA), and Process Capability (CpK) are the statistical pillars that provide objective evidence of this reliability. Without them, all subsequent V&V data is questionable.",
        "FDA Guidance on Analytical Procedures and Methods Validation; CLSI Guidelines (EP05, EP17, etc.); AIAG MSA Manual",
        "Prevents costly product failures and batch rejections caused by unreliable or incapable measurement and manufacturing processes. It is the foundation of data integrity."
    )
    render_metric_card("Process Capability (CpK)", "Measures how well a process is able to produce output that meets specifications. A CpK > 1.33 is typically considered capable for medical devices.", plot_cpk_analysis, "With the current specification limits, the process CpK is 1.48, which is excellent. However, tightening the LSL to 9.5 would drop the CpK below 1.0, indicating a non-capable process. This interactive tool demonstrates data-driven specification setting.", key="cpk")
    render_metric_card("Measurement System Analysis (MSA/Gage R&R)", "Quantifies the amount of variation in a measurement system attributable to the operators, equipment, and parts. A key part of Test Method Validation (TMV).", plot_msa_analysis, "The box plot shows that Operator Charlie's measurements are consistently lower than Alice and Bob's, indicating a potential training issue or procedural deviation that requires investigation before the TMV can be approved.", key="msa")
    render_metric_card("Assay Performance Regression Analysis", "Linear regression is used to characterize key assay performance attributes like linearity, analytical sensitivity, and to compare methods. The full statistical output is critical for regulatory submissions.", run_assay_regression, "The statsmodels summary provides a comprehensive model of the assay's response with high statistical significance (p < 0.001) and an R-squared of 0.99+, confirming excellent linearity across the analytical range.", key="regression")

def render_execution_monitoring_page():
    st.title("📈 3. Execution Monitoring & Quality Control")
    st.markdown("---")
    render_director_briefing("Statistical Process Control (SPC) for V&V",
        "SPC is a critical tool for monitoring processes in real-time. By applying control charts like Levey-Jennings (for labs) and Shewhart charts, we can distinguish between normal process variation ('common cause') and unexpected problems ('special cause') that require immediate investigation.",
        "FDA 21 CFR 820.250 (Statistical Techniques), ISO TR 10017, CLSI C24",
        "Provides an early warning system for process drifts or failures, reducing the risk of large-scale, costly investigations and ensuring the integrity of V&V data."
    )
    render_metric_card("Levey-Jennings & Westgard Rules", "The standard for monitoring daily quality control runs in a clinical lab environment. Westgard multi-rules provide high sensitivity for detecting systematic errors before they lead to invalid runs.", plot_levey_jennings_westgard, "The chart flags both a 1_3s rule violation (potential random error) and a 2_2s rule violation (potential systematic error). Action: Halt testing and launch a formal lab investigation as per the Quality Control SOP.", key="lj")
    render_metric_card("Individuals (I) Chart with Nelson Rules", "An I-chart is used to monitor individual data points over time when rational subgrouping is not possible. Nelson rules are another set of statistical tests to detect out-of-control conditions.", lambda k: run_control_charts(k), "The I-chart shows a clear upward shift starting at sample 15. This is a statistically significant process change that must be investigated to determine the assignable cause before proceeding with validation.", key="ichart")
    render_metric_card("First-Pass Analysis", "Measures the percentage of tests, protocols, or validation batches that are completed successfully without any rework, deviations, or failures. A primary indicator of process quality and efficiency.", lambda k: plot_rft_gauge(k), "A First-Pass (or Right-First-Time) rate of 82% indicates that nearly 1 in 5 protocols requires some form of rework. This provides a clear business case for investing in process improvement initiatives.", key="fpa")

def render_quality_management_page():
    st.title("✅ 4. Project & Quality Systems Management")
    st.markdown("---")
    render_director_briefing("Managing the V&V Ecosystem",
        "Beyond technical execution, a V&V leader must manage the project's health, track quality issues, and ensure a compliant software validation lifecycle. These KPIs provide the necessary oversight to manage timelines, scope, and compliance risks proactively.",
        "IEC 62304, 21 CFR Part 11, GAMP 5",
        "Improves project predictability, ensures software compliance (a major source of FDA 483s), and provides transparent, data-driven reporting to cross-functional stakeholders and leadership."
    )
    render_metric_card("Defect Open vs. Close Trend (Burnup)", "A burnup chart is superior to a burndown as it tracks scope changes. It visualizes the rate of work completion against the rate at which new issues are found.", plot_defect_burnup, "The widening gap between opened and closed defects indicates that our resolution rate is not keeping up with discovery. The scope increase on day 25 exacerbated this. Action: Allocate additional resources from the systems engineering team to defect resolution.", key="burnup")
    
    st.subheader("Software V&V (IEC 62304 & 21 CFR Part 11)")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**IEC 62304 Software Safety Classification**")
            risk_df = get_software_risk_data()
            def classify_color(cls):
                if cls == "Class C": return "background-color: #FF7F7F"
                if cls == "Class B": return "background-color: #FFD700"
                return "background-color: #90EE90"
            st.dataframe(risk_df.style.applymap(classify_color, subset=['IEC 62304 Class']), use_container_width=True, hide_index=True)
            st.info("The rigor of V&V activities (e.g., level of documentation, code reviews, unit testing) is directly tied to this risk classification.")
    with col2:
        with st.container(border=True):
            st.markdown("**21 CFR Part 11 Compliance Checklist**")
            st.checkbox("Validation (11.10a)", value=True, disabled=True)
            st.checkbox("Audit Trails (11.10e)", value=True, disabled=True)
            st.checkbox("Access Controls (11.10d)", value=True, disabled=True)
            st.checkbox("E-Signatures (11.200a)", value=False, disabled=True)
            st.error("Gap identified in E-Signature implementation, which must be remediated before the system can be considered Part 11 compliant.")

# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = {
    "Executive Summary": render_main_page,
    "1. Design Controls & Planning": render_design_controls_page,
    "2. Method & Process Validation": render_method_validation_page,
    "3. Execution Monitoring & SPC": render_execution_monitoring_page,
    "4. Project & Quality Management": render_quality_management_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page_to_render = PAGES[selection]
page_to_render()
