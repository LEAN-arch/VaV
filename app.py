# app.py (Final, Monolithic, World-Class Version with ALL Content and Enhancements)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

def render_metric_card(title, description, viz_function, insight, reg_context, key=""):
    """Renders a formatted container for a specific metric or visualization."""
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"*{description}*")
        st.warning(f"**Regulatory Context:** {reg_context}")
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
    fig = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis')])
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

def plot_risk_matrix(key):
    severity = [9, 10, 6, 8, 7, 5]; probability = [3, 2, 4, 3, 5, 1]
    risk_level = [s * p for s, p in zip(severity, probability)]
    text = ["False Positive", "False Negative", "Software Crash", "Contamination", "Reagent Exp.", "UI Lag"]
    fig = go.Figure(data=go.Scatter(x=probability, y=severity, mode='markers+text', text=text, textposition="top center", marker=dict(size=risk_level, sizemin=10, color=risk_level, colorscale="Reds", showscale=True, colorbar_title="Risk Score")))
    fig.update_layout(title='Risk Matrix (Severity vs. Probability)', xaxis_title='Probability of Occurrence', yaxis_title='Severity of Harm', xaxis=dict(range=[0, 6]), yaxis=dict(range=[0, 11]))
    return fig

def run_control_charts(key):
    data = [np.random.normal(10, 0.5, 5) for _ in range(20)]; data[15:] = [np.random.normal(10.8, 0.5, 5) for _ in range(5)]
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1,6)]); df['mean'] = df.mean(axis=1); df['range'] = df.max(axis=1) - df.min(axis=1)
    x_bar_cl = df['mean'].mean(); x_bar_a2 = 0.577; x_bar_ucl = x_bar_cl + x_bar_a2 * df['range'].mean(); x_bar_lcl = x_bar_cl - x_bar_a2 * df['range'].mean()
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name='Subgroup Mean', mode='lines+markers')); fig.add_hline(y=x_bar_cl, line_dash="dash", line_color="green", annotation_text="CL")
    fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL"); fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL")
    fig.update_layout(title="I-Chart (Shewhart Chart) for Process Monitoring"); return fig

def get_software_risk_data():
    return pd.DataFrame([{"Software Item": "Patient Result Algorithm", "IEC 62304 Class": "Class C"}, {"Software Item": "Database Middleware", "IEC 62304 Class": "Class B"}, {"Software Item": "UI Color Theme Module", "IEC 62304 Class": "Class A"}])

def plot_rft_gauge(key):
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = 82, title = {'text': "Right-First-Time Protocol Execution"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "cornflowerblue"}})); return fig

def run_anova_ttest_enhanced(key):
    st.info("Used to determine if there is a statistically significant difference between groups (e.g., reagent lots, instruments, or operators). This is fundamental for method transfer and comparability studies.")
    st.warning("**Regulatory Context:** FDA's guidance on Comparability Protocols; ISO 13485:2016, Section 7.5.6")
    col1, col2 = st.columns([1,2]);
    with col1:
        n_samples = st.slider("Samples per Group", 10, 100, 30, key=f"anova_n_{key}"); mean_shift = st.slider("Simulated Mean Shift in Lot B", 0.0, 5.0, 0.5, 0.1, key=f"anova_shift_{key}"); std_dev = st.slider("Group Standard Deviation", 0.5, 5.0, 2.0, 0.1, key=f"anova_std_{key}")
    group_a = np.random.normal(10, std_dev, n_samples); group_b = np.random.normal(10 + mean_shift, std_dev, n_samples)
    df = pd.melt(pd.DataFrame({'Lot A': group_a, 'Lot B': group_b}), var_name='Group', value_name='Measurement')
    with col2:
        fig = px.box(df, x='Group', y='Measurement', title="Performance Comparison with Box & Violin Plots", points='all'); fig.add_trace(go.Violin(x=df['Group'], y=df['Measurement'], box_visible=False, line_color='rgba(0,0,0,0)', fillcolor='rgba(0,0,0,0)', points=False, name='Distribution')); st.plotly_chart(fig, use_container_width=True)
    t_stat, p_value = stats.ttest_ind(group_a, group_b); st.subheader("Statistical Interpretation")
    if p_value < 0.05: st.error(f"**Actionable Insight:** The difference is statistically significant (p-value = {p_value:.4f}). Action: An investigation is required. Lot B cannot be considered comparable.")
    else: st.success(f"**Actionable Insight:** No statistically significant difference was detected (p-value = {p_value:.4f}). The lots are comparable.")
    return None

def run_regression_analysis_stat_enhanced(key):
    st.info("Linear regression is critical for verifying linearity and assessing correlation. The statsmodels output provides the detailed metrics required for a regulatory submission."); st.warning("**Regulatory Context:** CLSI EP06; FDA Guidance on Bioanalytical Method Validation")
    col1, col2 = st.columns([1,2]);
    with col1:
        noise = st.slider("Measurement Noise (Std Dev)", 0, 50, 15, key=f"regr_noise_{key}"); bias = st.slider("Systematic Bias", -20, 20, 5, key=f"regr_bias_{key}")
    conc = np.linspace(0, 400, 15); signal = 50 + 2.5 * conc + bias + np.random.normal(0, noise, 15); df = pd.DataFrame({'Concentration': conc, 'Signal': signal})
    with col2:
        fig = px.scatter(df, x='Concentration', y='Signal', trendline='ols', title="Assay Performance Regression (Linearity)"); st.plotly_chart(fig, use_container_width=True)
    X = sm.add_constant(df['Concentration']); model = sm.OLS(df['Signal'], X).fit(); st.subheader("Statistical Interpretation (Statsmodels OLS Summary)"); st.code(f"{model.summary()}"); st.success(f"**Actionable Insight:** The R-squared value of {model.rsquared:.3f} confirms excellent linearity. The p-value for the Concentration coefficient is < 0.001, proving a significant positive relationship.")
    return None

def run_descriptive_stats_stat_enhanced(key):
    st.info("The foundational analysis for any analytical validation study (e.g., LoD, Precision)."); st.warning("**Regulatory Context:** CLSI EP17 (Detection Capability); CLSI EP05-A3 (Precision)")
    data = np.random.normal(50, 2, 150); df = pd.DataFrame(data, columns=["value"]); fig = px.histogram(df, x="value", marginal="box", nbins=20, title="Descriptive Statistics for Limit of Detection (LoD) Study")
    mean, std, cv = np.mean(data), np.std(data, ddof=1), (np.std(data, ddof=1) / np.mean(data)) * 100
    ci_95 = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data)); st.plotly_chart(fig, use_container_width=True); st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4); col1.metric("Mean", f"{mean:.2f}"); col2.metric("Std Dev", f"{std:.2f}"); col3.metric("%CV", f"{cv:.2f}%"); col4.metric("95% CI for Mean", f"{ci_95[0]:.2f} - {ci_95[1]:.2f}")
    st.success("**Actionable Insight:** The low %CV and tight confidence interval provide high confidence that the LoD is reliably at 50 copies/mL, supporting the product claim for the 510(k) submission.")
    return None

def run_control_charts_stat_enhanced(key):
    st.info("X-bar & R-charts are used to monitor the mean (X-bar) and variability (R-chart) of a process when data is collected in rational subgroups (e.g., 5 measurements per batch)."); st.warning("**Regulatory Context:** FDA 21 CFR 820.250 (Statistical Techniques); ISO TR 10017")
    data = [np.random.normal(10, 0.5, 5) for _ in range(20)]; process_shift = st.checkbox("Simulate a Process Shift", key=f"spc_shift_{key}")
    if process_shift: data[15:] = [np.random.normal(10.8, 0.5, 5) for _ in range(5)]
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1,6)]); df['mean'] = df.mean(axis=1); df['range'] = df.max(axis=1) - df.min(axis=1)
    x_bar_cl = df['mean'].mean(); x_bar_a2 = 0.577; x_bar_ucl = x_bar_cl + x_bar_a2 * df['range'].mean(); x_bar_lcl = x_bar_cl - x_bar_a2 * df['range'].mean()
    r_cl = df['range'].mean(); r_d4 = 2.114; r_ucl = r_d4 * r_cl
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart (Process Mean)", "R-Chart (Process Variability)")); fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name='Subgroup Mean', mode='lines+markers'), row=1, col=1); fig.add_hline(y=x_bar_cl, line_dash="dash", line_color="green", annotation_text="CL", row=1, col=1); fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=1, col=1); fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['range'], name='Subgroup Range', mode='lines+markers', line=dict(color='orange')), row=2, col=1); fig.add_hline(y=r_cl, line_dash="dash", line_color="green", annotation_text="CL", row=2, col=1); fig.add_hline(y=r_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=2, col=1)
    fig.update_layout(height=600, title_text="X-bar and R Control Charts"); st.plotly_chart(fig, use_container_width=True)
    if process_shift: st.warning("**Actionable Insight:** A clear upward shift is detected in the X-bar chart starting at subgroup 15, while the R-chart remains stable. This indicates a special cause has shifted the process mean without affecting its variability. This requires an immediate investigation.")
    else: st.success("**Actionable Insight:** The process is in a state of statistical control. Both the mean and variability are stable and predictable, providing a solid baseline for validation.")
    return None
    
def run_kaplan_meier_stat_enhanced(key):
    st.info("Survival analysis is used to estimate the shelf-life of a product by modeling time-to-failure data, especially when some samples have not failed by the end of the study (censored data)."); st.warning("**Regulatory Context:** ICH Q1E (Evaluation of Stability Data); FDA Guidance: Q1A(R2)")
    time_to_failure = np.random.weibull(2, 50) * 24; observed = np.random.binomial(1, 0.8, 50); df = pd.DataFrame({'Months': time_to_failure, 'Status': ['Failed' if o==1 else 'Censored' for o in observed]})
    fig = px.ecdf(df, x="Months", color="Status", ecdfmode="survival", title="Kaplan-Meier Survival Plot for Shelf-Life Validation"); fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Median Survival"); st.plotly_chart(fig, use_container_width=True); st.subheader("Study Conclusion")
    st.success("**Actionable Insight:** The survival curve demonstrates the probability of a unit remaining stable over time. The point where the curve crosses the 50% line provides the estimated median shelf-life. 'Censored' data points are critical for an accurate model and must be included in the analysis.")
    return None

def run_monte_carlo_stat_enhanced(key):
    st.info("Monte Carlo simulation runs thousands of 'what-if' scenarios on a project plan with uncertain task durations to provide a probabilistic forecast."); st.warning("**Regulatory Context:** Aligned with risk-based planning principles in ISO 13485 and Project Management Body of Knowledge (PMBOK).")
    n_sims = st.slider("Number of Simulations", 1000, 10000, 5000, key=f"mc_sims_{key}")
    task1, task2, task3 = np.random.triangular(8,10,15,n_sims), np.random.triangular(15,20,30,n_sims), np.random.triangular(5,8,12,n_sims); total_times = task1 + task2 + task3
    p50 = np.percentile(total_times, 50); p90 = np.percentile(total_times, 90)
    fig = px.histogram(total_times, nbins=50, title="Monte Carlo Simulation of V&V Plan Duration"); fig.add_vline(x=p50, line_dash="dash", line_color="green", annotation_text=f"P50 (Median) = {p50:.1f} days"); fig.add_vline(x=p90, line_dash="dash", line_color="red", annotation_text=f"P90 (High Confidence) = {p90:.1f} days")
    st.plotly_chart(fig, use_container_width=True); st.subheader("Risk-Adjusted Planning")
    st.error(f"**Actionable Insight:** While the median completion time is {p50:.1f} days, there is a 10% chance the project will take **{p90:.1f} days or longer**. The P90 estimate must be communicated to the PMO as the commitment date to account for risk.")
    return None

def create_v_model_figure(key=None):
    fig = go.Figure(); fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Needs", "System Req.", "Architecture", "Module Design"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test", "Integration Test", "System V&V", "UAT"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text=None, showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig

# --- PAGE RENDERING FUNCTIONS ---

def render_main_page():
    st.title("🎯 The V&V Executive Command Center")
    st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
    st.markdown("---")
    render_director_briefing("Portfolio Objective", "This interactive application translates the core responsibilities of V&V leadership into a suite of high-density dashboards. It is designed to be an overwhelming and undeniable demonstration of the strategic, technical, and quality systems expertise required for a senior leadership role in the medical device industry.", "ISO 13485, ISO 14971, IEC 62304, 21 CFR 820, 21 CFR Part 11, CLSI Guidelines", "A well-led V&V function directly accelerates time-to-market, reduces compliance risk, lowers the cost of poor quality (COPQ), and builds a culture of data-driven excellence.")
    st.info("Please use the navigation sidebar on the left to explore each of the six core competency areas.")

def render_design_controls_page():
    st.title("🏛️ 1. Design Controls, Planning & Risk Management")
    st.markdown("---")
    render_director_briefing("The Design History File (DHF) as a Strategic Asset", "The DHF is the compilation of records that demonstrates the design was developed in accordance with the design plan and regulatory requirements. An effective V&V leader architects the DHF from day one.", "FDA 21 CFR 820.30 (Design Controls), ISO 13485:2016 (Section 7.3)", "Ensures audit readiness and provides a clear, defensible story of product development to regulatory bodies, accelerating submission review times.")
    render_metric_card("Requirements Traceability Matrix (RTM)", "The RTM is the backbone of the DHF, providing an auditable link between user needs, design inputs, V&V activities, and risk controls.", create_rtm_data_editor, "The matrix view instantly flags critical gaps, such as the un-tested cross-reactivity requirement (URS-003), allowing for proactive mitigation before a design freeze.", "FDA 21 CFR 820.30(j) - Design History File (DHF)", key="rtm")
    render_metric_card("Product Risk Management (FMEA & Risk Matrix)", "A systematic process for identifying, analyzing, and mitigating potential failure modes. V&V activities are primary risk mitigations.", plot_risk_matrix, "The risk matrix clearly prioritizes 'False Negative' as the highest risk, ensuring that it receives the most V&V resources and attention. This is a key input for the V&V Master Plan.", "ISO 14971: Application of risk management to medical devices", key="fmea")
    render_metric_card("Design of Experiments (DOE/RSM)", "A powerful statistical tool used to efficiently characterize the product's design space and identify robust operating parameters.", plot_doe_rsm, "The Response Surface Methodology (RSM) plot indicates the assay's optimal performance is at ~32°C and a pH of 7.5. This data forms the basis for setting manufacturing specifications.", "FDA Guidance on Process Validation: General Principles and Practices", key="doe")
    
def render_method_validation_page():
    st.title("🔬 2. Method Validation & Statistical Rigor")
    st.markdown("---")
    render_director_briefing("Ensuring Data Trustworthiness", "Before a product can be validated, the methods used to measure it must be proven reliable. TMV, MSA, and CpK are the statistical pillars that provide objective evidence of this reliability.", "FDA Guidance on Analytical Procedures and Methods Validation; CLSI Guidelines (EP05, EP17, etc.); AIAG MSA Manual", "Prevents costly product failures and batch rejections caused by unreliable or incapable measurement and manufacturing processes. It is the foundation of data integrity.")
    render_metric_card("Process Capability (CpK)", "Measures how well a process can produce output that meets specifications. A CpK > 1.33 is typically considered capable for medical devices.", plot_cpk_analysis, "The interactive slider shows how tightening specification limits directly impacts the CpK value, demonstrating the trade-offs between design margin and manufacturing capability.", "ISO TR 10017 - Guidance on statistical techniques", key="cpk")
    render_metric_card("Measurement System Analysis (MSA/Gage R&R)", "Quantifies the variation in a measurement system attributable to operators and equipment. A key part of Test Method Validation (TMV).", plot_msa_analysis, "The box plot shows that Operator Charlie's measurements are consistently lower than Alice and Bob's, indicating a potential training issue or procedural deviation that requires investigation.", "AIAG MSA Reference Manual", key="msa")
    render_metric_card("Assay Performance Regression Analysis", "Linear regression is used to characterize key assay performance attributes. The full statistical output is critical for regulatory submissions.", run_assay_regression, "The statsmodels summary provides a comprehensive model of the assay's response with high statistical significance (p < 0.001) and an R-squared of 0.99+, confirming excellent linearity.", "CLSI EP06 - Evaluation of the Linearity of Quantitative Measurement Procedures", key="regression")

def render_execution_monitoring_page():
    st.title("📈 3. Execution Monitoring & Quality Control")
    st.markdown("---")
    render_director_briefing("Statistical Process Control (SPC) for V&V", "SPC distinguishes between normal process variation ('common cause') and unexpected problems ('special cause') that require immediate investigation.", "FDA 21 CFR 820.250 (Statistical Techniques), ISO TR 10017, CLSI C24", "Provides an early warning system for process drifts, reducing the risk of large-scale, costly investigations and ensuring data integrity.")
    render_metric_card("Levey-Jennings & Westgard Rules", "The standard for monitoring daily quality control runs in a clinical lab. Westgard multi-rules provide high sensitivity for detecting systematic errors.", plot_levey_jennings_westgard, "The chart flags both a 1_3s rule violation (random error) and a 2_2s rule violation (systematic error). Action: Halt testing and launch a formal lab investigation.", "CLSI C24 - Statistical Quality Control for Quantitative Measurement Procedures", key="lj")
    render_metric_card("Individuals (I) Chart with Nelson Rules", "An I-chart (a type of Shewhart chart) is used to monitor individual data points over time. Nelson rules are powerful statistical tests to detect out-of-control conditions.", run_control_charts, "The I-chart shows a statistically significant upward shift at sample 15. This must be investigated to determine the assignable cause.", "FDA 21 CFR 820.250 (Statistical Techniques)", key="ichart")
    render_metric_card("First-Pass Analysis", "Measures the percentage of tests completed successfully without rework. A primary indicator of process quality.", plot_rft_gauge, "A First-Pass (RFT) rate of 82% indicates that nearly 1 in 5 protocols requires rework. This provides a clear business case for process improvement initiatives.", "Lean Six Sigma Principles", key="fpa")

def render_quality_management_page():
    st.title("✅ 4. Project & Quality Systems Management")
    st.markdown("---")
    render_director_briefing("Managing the V&V Ecosystem", "A V&V leader must manage project health, track quality issues, and ensure software compliance. These KPIs provide the necessary oversight to manage timelines, scope, and compliance risks.", "IEC 62304, 21 CFR Part 11, GAMP 5", "Improves project predictability, ensures software compliance (a major source of FDA 483s), and provides transparent reporting to stakeholders.")
    render_metric_card("Defect Open vs. Close Trend (Burnup)", "A burnup chart tracks scope changes and visualizes the rate of work completion against the rate of issue discovery.", plot_defect_burnup, "The widening gap between opened and closed defects indicates that our resolution rate is not keeping up. Action: Allocate additional resources to defect resolution.", "Agile Project Management Principles", key="burnup")
    
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
            st.dataframe(risk_df.style.map(classify_color, subset=['IEC 62304 Class']), use_container_width=True, hide_index=True)
            st.info("V&V rigor is directly tied to this risk classification.")
    with col2:
        with st.container(border=True):
            st.markdown("**21 CFR Part 11 Compliance Checklist**")
            st.checkbox("Validation (11.10a)", value=True, disabled=True)
            st.checkbox("Audit Trails (11.10e)", value=True, disabled=True)
            st.checkbox("Access Controls (11.10d)", value=True, disabled=True)
            st.checkbox("E-Signatures (11.200a)", value=False, disabled=True)
            st.error("Gap identified in E-Signature implementation.")

def render_stats_page():
    st.title("📐 5. Advanced Statistical Methods Workbench")
    st.markdown("This interactive workbench demonstrates proficiency in the specific statistical methods required for robust data analysis in a regulated V&V environment.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Performance Comparison (t-test)")
            run_anova_ttest_enhanced("anova")
        with st.container(border=True):
            st.subheader("Assay Performance (Descriptive Stats)")
            run_descriptive_stats_stat_enhanced("desc")
        with st.container(border=True):
            st.subheader("Shelf-Life & Stability (Kaplan-Meier)")
            run_kaplan_meier_stat_enhanced("km")
    with col2:
        with st.container(border=True):
            st.subheader("Risk-to-Failure Correlation (Regression)")
            run_regression_analysis_stat_enhanced("regr")
        with st.container(border=True):
            st.subheader("Process Monitoring (SPC)")
            run_control_charts_stat_enhanced("spc")
        with st.container(border=True):
            st.subheader("Project Timeline Risk (Monte Carlo)")
            run_monte_carlo_stat_enhanced("mc")

def render_strategic_command_page():
    st.title("👑 6. Strategic Command & Control")
    st.markdown("---")
    render_director_briefing("Executive-Level V&V Leadership",
        "A true V&V leader operates at the intersection of technical execution, financial reality, and cross-functional strategy. This command center demonstrates the tools and mindset required to run V&V not as a cost center, but as a strategic business partner that drives value and mitigates enterprise-level risk.",
        "ISO 13485 Section 5 (Management Responsibility) & 6 (Resource Management)",
        "Aligns V&V department with corporate financial goals, improves resource allocation, de-risks regulatory pathways, and enables scalable growth through effective talent management and partner oversight."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["💰 V&V Cost & ROI Forecaster", "🌐 Regulatory & Partner Dashboard", "🧑‍🔬 Team Competency Matrix", "🔄 ECO Impact Assessment"])

    with tab1:
        st.header("V&V Project Cost & ROI Forecaster")
        st.info("Translate technical plans into financial forecasts to justify resource allocation and demonstrate value.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Inputs")
            proj = st.selectbox("Select Project", ["ImmunoPro-A (510k)", "MolecularDX-2 (PMA)"])
            av_weeks = st.slider("Analytical V&V (Weeks)", 1, 20, 8)
            sv_weeks = st.slider("System V&V (Weeks)", 1, 20, 10)
            sw_weeks = st.slider("Software V&V (Weeks)", 1, 20, 6)
            cs_weeks = st.slider("Clinical Support (Weeks)", 1, 20, 12)
            fte_cost = st.number_input("Cost per FTE-Week ($)", value=4000)
            reagent_cost = st.number_input("Cost of Reagents per Week ($)", value=7500)
        with col2:
            st.subheader("Forecasted V&V Budget")
            total_weeks = av_weeks + sv_weeks + sw_weeks + cs_weeks
            personnel_cost = total_weeks * fte_cost * 2 # Assume 2 FTEs
            reagent_total_cost = (av_weeks + sv_weeks) * reagent_cost
            total_budget = personnel_cost + reagent_total_cost
            
            st.metric("Total Forecasted V&V Budget", f"${total_budget:,.0f}")
            
            fig = px.pie(values=[personnel_cost, reagent_total_cost], names=['Personnel', 'Reagents & Consumables'], title='V&V Budget Allocation', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.header("Regulatory Strategy & External Partner Dashboard")
        st.info("Dynamically align V&V evidence with submission requirements and manage external vendor performance.")
        sub_type = st.selectbox("Select Submission Type", ["FDA 510(k)", "FDA PMA", "EU IVDR Class D"])
        with st.container(border=True):
            st.subheader(f"Required Evidence Checklist for: {sub_type}")
            st.checkbox("Analytical Performance Studies (LoD, Precision, etc.)", value=True, disabled=True)
            if "510(k)" in sub_type: st.checkbox("Substantial Equivalence Testing Data", value=True, disabled=True)
            if "PMA" in sub_type or "IVDR" in sub_type: st.checkbox("Clinical Performance / Validation Data Support", value=True, disabled=True); st.checkbox("Stability & Robustness Data (Extended)", value=True, disabled=True)
            if "IVDR" in sub_type: st.checkbox("Scientific Validity Report", value=True, disabled=True)
        
        st.subheader("CRO Partner Performance Oversight")
        df = pd.DataFrame({'Metric': ['On-Time Delivery', 'Deviation Rate', 'Data Quality Score'], 'Internal Team': [95, 2.1, 98.5], 'CRO Partner A': [88, 4.5, 96.2]})
        fig = px.bar(df, x='Metric', y=['Internal Team', 'CRO Partner A'], barmode='group', title="Internal vs. CRO Performance")
        st.plotly_chart(fig, use_container_width=True)
        st.error("**Actionable Insight:** CRO Partner A is underperforming on OTD and has a higher deviation rate. Schedule a Quarterly Business Review (QBR) to address these performance gaps.")

    with tab3:
        st.header("Team Competency & Development Matrix")
        st.info("Move beyond simple training records to proactive talent management and strategic skill development.")
        skills = ['qPCR Method Validation', 'ELISA Development', 'GAMP 5 CSV', 'Statistical Analysis (Python)', 'ISO 14971 Risk Management']
        team = ['Alice', 'Bob', 'Charlie', 'Diana']
        data = np.random.randint(1, 4, size=(len(team), len(skills)))
        df = pd.DataFrame(data, index=team, columns=skills)

        required_skills = st.multiselect("Filter for Required Project Skills", options=skills, default=['qPCR Method Validation', 'ISO 14971 Risk Management'])
        
        def highlight_skills(s):
            return ['background-color: yellow' if s.name in required_skills else '' for i in s]
        
        st.dataframe(df.style.apply(highlight_skills, axis=1).background_gradient(cmap='RdYlGn', vmin=1, vmax=3), use_container_width=True)
        st.success("**Actionable Insight:** For the new molecular project, the team has strong qPCR skills. However, there is a critical gap as no one is an expert (Level 3) in ISO 14971. Action: Prioritize risk management training for Bob and Diana this quarter.")

    with tab4:
        st.header("Interactive ECO Impact Assessment Tool")
        st.info("A logic-driven tool to ensure a consistent, risk-based approach to V&V for post-market changes.")
        change_type = st.selectbox("Select Type of Engineering Change Order (ECO)", ["Reagent Formulation", "Software (Minor UI)", "Software (Algorithm)", "Supplier of Critical Component"])
        
        with st.container(border=True):
            st.subheader("Minimum Required V&V Activities")
            if change_type == "Reagent Formulation":
                st.error("**Full V&V Suite Required:** Analytical Performance (Precision, LoD, Linearity), Stability Studies, Clinical Bridging Study.")
                st.markdown("**Rationale:** Change directly impacts assay performance and patient results. High risk.")
            elif change_type == "Software (Minor UI)":
                st.success("**Limited V&V Required:** Software Regression Testing, Usability Assessment.")
                st.markdown("**Rationale:** Change does not impact the analytical algorithm. Low risk.")
            elif change_type == "Software (Algorithm)":
                st.error("**Full Software & Analytical V&V Required:** Full Software Validation Suite (per IEC 62304), Analytical Performance regression testing.")
                st.markdown("**Rationale:** Change to the core algorithm directly impacts patient results. Highest software risk.")
            elif change_type == "Supplier of Critical Component":
                st.warning("**Targeted V&V Required:** Component Qualification, System-level performance regression testing, limited stability run.")
                st.markdown("**Rationale:** Change introduces a new variable into the system. Medium risk requiring confirmation that system performance is unaffected.")


# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = {
    "Executive Summary": render_main_page,
    "1. Design Controls & Planning": render_design_controls_page,
    "2. Method & Process Validation": render_method_validation_page,
    "3. Execution Monitoring & SPC": render_execution_monitoring_page,
    "4. Project & Quality Management": render_quality_management_page,
    "5. Advanced Statistical Methods": render_stats_page,
    "6. Strategic Command & Control": render_strategic_command_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page_to_render = PAGES[selection]
page_to_render()
