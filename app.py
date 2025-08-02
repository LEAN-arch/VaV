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
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- UTILITY FUNCTIONS (Previously in utils.py) ---

def render_metric_summary(metric_name, description, viz_function, insight_text, reg_context=""):
    st.subheader(metric_name)
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Description:** {description}")
            st.info(f"**Director's Insight:** {insight_text}")
            if reg_context:
                st.warning(f"**Regulatory Context:** {reg_context}")
        with col2:
            fig = viz_function()
            st.plotly_chart(fig, use_container_width=True)

# --- VISUALIZATION GENERATORS ---

def plot_protocol_completion_burndown():
    days = np.arange(1, 31); planned = np.linspace(100, 0, 30); actual = np.clip(planned + np.random.randn(30).cumsum() * 1.5, 0, 100)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=days, y=planned, mode='lines', name='Planned Burndown', line=dict(dash='dash'))); fig.add_trace(go.Scatter(x=days, y=actual, mode='lines+markers', name='Actual Burndown'))
    fig.update_layout(title='Protocol Completion Burndown Chart', xaxis_title='Sprint Day', yaxis_title='% Protocols Remaining'); return fig

def plot_pass_rate_heatmap():
    df = pd.DataFrame({'Test Type': ['LoD', 'Linearity', 'Specificity', 'Precision', 'Robustness'], 'Pass Rate (%)': [95, 98, 92, 100, 88]})
    fig = px.bar(df, x='Pass Rate (%)', y='Test Type', orientation='h', title='Pass Rate by Test Type', text='Pass Rate (%)'); fig.update_traces(texttemplate='%{text}%', textposition='inside'); return fig

def plot_retest_pareto():
    df = pd.DataFrame({'Reason': ['Operator Error', 'Reagent Lot Variability', 'Instrument Drift', 'Sample Prep Issue', 'Software Glitch'], 'Count': [15, 9, 5, 3, 1]}).sort_values(by='Count', ascending=False)
    df['Cumulative Pct'] = (df['Count'].cumsum() / df['Count'].sum()) * 100; fig = go.Figure(); fig.add_trace(go.Bar(x=df['Reason'], y=df['Count'], name='Re-test Count')); fig.add_trace(go.Scatter(x=df['Reason'], y=df['Cumulative Pct'], name='Cumulative %', yaxis='y2'))
    fig.update_layout(title="Pareto Chart of Re-test Causes", yaxis2=dict(overlaying='y', side='right', title='Cumulative %')); return fig

def plot_trace_coverage_sankey():
    fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, label=["URS-01", "URS-02", "SRS-01", "DVP-01", "DVP-02", "UNCOVERED"]), link=dict(source=[0, 1, 2, 2], target=[3, 4, 3, 4], value=[8, 4, 2, 8])))
    fig.update_layout(title_text="Requirements Trace Coverage"); return fig

def plot_rpn_waterfall():
    fig = go.Figure(go.Waterfall(measure = ["relative", "relative", "total", "relative", "relative", "total"], x = ["Initial Risk (FP)", "Mitigation 1", "Subtotal", "Initial Risk (FN)", "Mitigation 2", "Final Risk Portfolio"], y = [120, -40, 0, 80, -60, 0]))
    fig.update_layout(title = "FMEA Risk Reduction (RPN Waterfall)"); return fig

def plot_validation_gantt_baseline():
    df = pd.DataFrame([dict(Task="FAT", Start='2023-01-01', Finish='2023-01-10', Type="Planned"), dict(Task="FAT", Start='2023-01-01', Finish='2023-01-12', Type="Actual"), dict(Task="SAT", Start='2023-01-15', Finish='2023-01-20', Type="Planned"), dict(Task="SAT", Start='2023-01-18', Finish='2023-01-22', Type="Actual"), dict(Task="IQ", Start='2023-01-21', Finish='2023-01-25', Type="Planned"), dict(Task="IQ", Start='2023-01-23', Finish='2023-01-25', Type="Actual")])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Type", title="Validation On-Time Rate (Gantt vs Baseline)"); return fig

def plot_sat_to_pq_violin():
    df = pd.DataFrame({'Days': np.concatenate([np.random.normal(20, 5, 20), np.random.normal(35, 8, 15)]), 'Equipment Type': ['Analyzer'] * 20 + ['Sample Prep'] * 15})
    fig = px.violin(df, y="Days", x="Equipment Type", color="Equipment Type", box=True, points="all", title="Time from SAT to PQ Approval"); return fig

def plot_protocol_review_cycle_histogram():
    fig = px.histogram(np.random.gamma(4, 2, 100), title="Protocol Review Cycle Time (Draft to Approved)", labels={'value': 'Days'}); return fig

def plot_training_donut():
    fig = go.Figure(data=[go.Pie(labels=['ISO 13485','GAMP5','21 CFR 820', 'GDP', 'Not Started'], values=[100, 95, 98, 90, 5], hole=.4, title="Training Completion Rate")]); return fig

def plot_rft_gauge():
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = 82, title = {'text': "Right-First-Time Protocol Execution"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "cornflowerblue"}})); return fig

def plot_capa_funnel():
    fig = go.Figure(go.Funnel(y = ["Identified", "Investigation", "Root Cause Analysis", "Implementation", "Effectiveness Check"], x = [100, 80, 65, 60, 55], textinfo = "value+percent initial"))
    fig.update_layout(title="CAPA Closure Effectiveness Funnel"); return fig

def run_anova_ttest(add_shift):
    group_a = np.random.normal(10, 2, 30); group_b_mean = 10.5 if not add_shift else 12.5; group_b = np.random.normal(group_b_mean, 2, 30)
    fig = px.box(pd.DataFrame({'Group A': group_a, 'Group B': group_b}), title="Performance Comparison (Lot A vs Lot B)")
    t_stat, p_value = stats.ttest_ind(group_a, group_b); result = f"**T-test Result:** p-value = {p_value:.4f}. "
    result += "**Conclusion:** Difference is statistically significant." if p_value < 0.05 else "**Conclusion:** No significant difference detected."; return fig, result

def run_regression_analysis():
    rpn = np.random.randint(20, 150, 50); failure_prob = rpn / 200; failures = np.random.binomial(1, failure_prob)
    df = pd.DataFrame({'RPN': rpn, 'Failure Occurred': failures}); df['Failure Occurred'] = df['Failure Occurred'].astype('category')
    fig = px.scatter(df, x='RPN', y='Failure Occurred', title="Correlation of Risk (RPN) to Failure Rate", marginal_y="histogram")
    return fig, "**Insight:** Higher RPN values show a clear trend towards a higher likelihood of test failure, validating the risk assessment process."

def run_descriptive_stats():
    data = np.random.normal(50, 2, 100); df = pd.DataFrame(data, columns=["LoD Measurement (copies/mL)"])
    mean, std, cv = df.iloc[:,0].mean(), df.iloc[:,0].std(), (df.iloc[:,0].std() / df.iloc[:,0].mean()) * 100
    fig = px.histogram(df, x="LoD Measurement (copies/mL)", marginal="box", title="Descriptive Statistics for LoD Study")
    return fig, f"**Mean:** {mean:.2f} | **Std Dev:** {std:.2f} | **%CV:** {cv:.2f}%"

def run_control_charts():
    data = [np.random.normal(10, 0.5, 5) for _ in range(20)]; data[15:] = [np.random.normal(10.8, 0.5, 5) for _ in range(5)]
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1,6)]); df['mean'] = df.mean(axis=1); df['range'] = df.max(axis=1) - df.min(axis=1)
    x_bar_cl = df['mean'].mean(); x_bar_ucl = x_bar_cl + 3 * (df['range'].mean() / 2.326); x_bar_lcl = x_bar_cl - 3 * (df['range'].mean() / 2.326)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name='Subgroup Mean', mode='lines+markers')); fig.add_hline(y=x_bar_cl, line_dash="dash", line_color="green", annotation_text="CL")
    fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL"); fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL")
    fig.update_layout(title="X-bar Control Chart for Process Monitoring"); return fig

def run_kaplan_meier():
    time_to_failure = np.random.weibull(2, 50) * 24; observed = np.random.binomial(1, 0.8, 50); df = pd.DataFrame({'Months': time_to_failure, 'Observed': observed}).sort_values(by='Months')
    at_risk = len(df); survival_prob = []
    for i, row in df.iterrows():
        survival = (at_risk - 1) / at_risk if row['Observed'] == 1 else 1; at_risk -= 1; survival_prob.append(survival)
    df['Survival'] = np.cumprod(survival_prob)
    fig = px.line(df, x='Months', y='Survival', title="Kaplan-Meier Survival Plot for Shelf-Life", markers=True); fig.update_yaxes(range=[0, 1.05]); return fig, "**Conclusion:** The estimated median shelf-life (time to 50% survival) is approximately 21 months."

def run_monte_carlo():
    n_sims = 5000; task1, task2, task3 = np.random.triangular(8,10,15,n_sims), np.random.triangular(15,20,30,n_sims), np.random.triangular(5,8,12,n_sims)
    total_times = task1 + task2 + task3; p90 = np.percentile(total_times, 90)
    fig = px.histogram(total_times, nbins=50, title="Monte Carlo Simulation of V&V Plan Duration"); fig.add_vline(x=p90, line_dash="dash", line_color="red", annotation_text=f"P90 = {p90:.1f} days")
    return fig, f"**Conclusion:** While the 'most likely' duration is ~38 days, there is a 10% chance the project will take **{p90:.1f} days or longer**. This P90 value should be used for risk-adjusted planning."

def create_v_model_figure():
    fig = go.Figure(); fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Needs", "System Req.", "Architecture", "Module Design"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test", "Integration Test", "System V&V", "UAT"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text=None, showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig

def get_software_risk_data():
    return pd.DataFrame([{"Software Item": "Patient Result Algorithm", "IEC 62304 Class": "Class C"}, {"Software Item": "Database Middleware", "IEC 62304 Class": "Class B"}, {"Software Item": "UI Color Theme Module", "IEC 62304 Class": "Class A"}])

def get_eco_data():
    return pd.DataFrame([{"ECO": "ECO-00451", "Change": "Reagent 2 formulation update", "Risk": "High", "V&V Impact": "Full re-validation required.", "Status": "V&V In Progress"}, {"ECO": "ECO-00488", "Change": "Update GUI software (bug fix)", "Risk": "Low", "V&V Impact": "Regression testing only.", "Status": "V&V Complete"}])


# --- PAGE RENDERING FUNCTIONS ---

def render_main_page():
    st.title("🎯 The V&V Executive Command Center")
    st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
    st.markdown("---")
    st.markdown("""
    Welcome. This application translates the core responsibilities of V&V leadership into a suite of interactive, high-density dashboards. 
    **Please use the navigation sidebar on the left to explore each of the six core competency areas.**
    """)
    st.subheader("A Framework for Compliant V&V (The V-Model)")
    st.plotly_chart(create_v_model_figure(), use_container_width=True)
    st.markdown("---")
    st.success("Built with Python & Streamlit to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")

def render_assay_metrics_page():
    st.title("🧪 Assay V&V-Specific Metrics & KPIs")
    st.markdown("---")
    st.header("1. Test Execution Metrics")
    render_metric_summary("Protocol Completion", "Tracks progress against the plan, indicating project velocity and potential delays.", plot_protocol_completion_burndown, "The team is slightly behind the ideal burndown. Action: Investigate the root cause for the slowdown.")
    render_metric_summary("Pass Rate by Test Type", "Identifies problematic test areas or assay weaknesses across projects.", plot_pass_rate_heatmap, "The low pass rate for 'Robustness' (88%) is a red flag. Action: Convene with R&D to review the robustness study design.")
    render_metric_summary("Re-test Rate & Root Cause", "Highlights sources of inefficiency and quality issues in the lab. A high re-test rate impacts timelines and cost.", plot_retest_pareto, "'Operator Error' is the primary cause of re-tests. Action: Schedule refresher training.")
    st.header("2. Design Control & Traceability")
    render_metric_summary("Requirements Trace Coverage", "The most critical metric for audit readiness, ensuring no gaps exist.", plot_trace_coverage_sankey, "The Sankey diagram clearly shows an uncovered requirement. This is a critical gap that must be closed before the design review.", "FDA 21 CFR 820.30(j) - DHF")
    st.header("3. Regulatory Compliance & Risk")
    render_metric_summary("FMEA Risk Reduction", "Demonstrates the effectiveness of V&V activities as risk mitigations.", plot_rpn_waterfall, "V&V activities successfully reduced the total risk portfolio. This provides objective evidence of building a safer product.", "ISO 14971")

def render_equipment_metrics_page():
    st.title("🏭 Equipment Validation Metrics (FAT/SAT/IQ/OQ/PQ)")
    st.markdown("---")
    st.header("1. Validation Execution Health")
    render_metric_summary("Validation On-Time Rate", "Compares planned validation timelines against actual execution, highlighting delays.", plot_validation_gantt_baseline, "The 'SAT' phase experienced a significant delay. Action: Investigate the root cause to prevent recurrence.")
    st.header("2. Readiness & Qualification")
    render_metric_summary("Time from SAT to PQ Approval", "Measures the efficiency of the on-site qualification process.", plot_sat_to_pq_violin, "'Sample Prep' equipment shows a wider and longer qualification cycle. Action: Launch a process improvement event to streamline this workflow.", "GAMP 5")

def render_team_kpis_page():
    st.title("👥 Team & Project Management KPIs")
    st.markdown("---")
    st.header("1. Productivity & Load")
    render_metric_summary("Protocol Review Cycle Time", "Measures the efficiency of the documentation workflow. Long cycle times are a bottleneck.", plot_protocol_review_cycle_histogram, "The distribution shows a long tail. Action: Implement a daily review board meeting to address aging documents.")
    st.header("2. Training & Competency")
    render_metric_summary("Training Completion Rate", "A critical quality system metric ensuring the team is competent and compliant.", plot_training_donut, "The team is lagging in GAMP5 training. Action: Schedule a GAMP5 training session for the team within the next quarter.", "ISO 13485:2016 Sec 6.2")

def render_quality_kpis_page():
    st.title("📊 Quality & Continuous Improvement KPIs")
    st.markdown("---")
    st.header("1. Right-First-Time (RFT) Metrics")
    render_metric_summary("Right-First-Time Protocol Execution", "Measures the quality of planning and execution. A low RFT rate indicates rework and delays.", plot_rft_gauge, "An RFT rate of 82% is a good starting point. Action: Set a quarterly goal to increase RFT to 90%.")
    st.header("2. CAPA & NCR Metrics")
    render_metric_summary("CAPA Closure Effectiveness", "Tracks the efficiency of the Corrective and Preventive Action (CAPA) process.", plot_capa_funnel, "There is a significant drop-off between 'Implementation' and 'Effectiveness Check'. Action: Reinforce the importance of scheduling effectiveness checks.", "FDA 21 CFR 820.100 (CAPA)")

def render_software_vv_page():
    st.title("💻 Software V&V (IEC 62304 & Part 11)")
    st.markdown("Demonstrating expertise in validating the software components of modern diagnostic systems, a critical and often-audited area.")
    st.markdown("---")
    render_metric_summary("The V-Model for Software Validation", "The V-Model is the industry-standard framework linking each development phase to a corresponding test phase.", create_v_model_figure, "This visualization demonstrates a clear understanding of the compliant software development lifecycle.", "IEC 62304")
    st.subheader("Risk-Based Testing (IEC 62304)")
    with st.container(border=True):
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("**Description:** IEC 62304 requires classifying software based on the potential harm it could cause. V&V rigor must be proportional to this risk.")
            st.info("**Director's Insight:** The 'Patient Result Algorithm' is Class C, triggering the most stringent V&V requirements.")
            st.warning("**Regulatory Context:** IEC 62304, Section 4.3: Software safety classification")
        with col2:
            risk_df = get_software_risk_data()
            def classify_color(cls):
                if cls == "Class C": return "background-color: #FF7F7F"
                if cls == "Class B": return "background-color: #FFD700"
                return "background-color: #90EE90"
            st.dataframe(risk_df.style.applymap(classify_color, subset=['IEC 62304 Class']), use_container_width=True, hide_index=True)

def render_stats_page():
    st.title("📐 Advanced Statistical Methods Workbench")
    st.markdown("This interactive workbench demonstrates proficiency in the specific statistical methods required for robust data analysis in a regulated V&V environment.")
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ANOVA / t-tests", "Regression Analysis", "Descriptive Stats", "Control Charts (SPC)", "Kaplan-Meier (Stability)", "Monte Carlo Simulation"])
    with tab1:
        st.header("Performance Comparison (t-test)"); st.info("Used to determine if there is a statistically significant difference between two groups (e.g., reagent lots).")
        add_shift = st.checkbox("Simulate a Mean Shift in Lot B's Performance"); fig, result = run_anova_ttest(add_shift); st.plotly_chart(fig, use_container_width=True); st.subheader("Statistical Interpretation"); st.markdown(result)
    with tab2:
        st.header("Risk-to-Failure Correlation (Regression)"); st.info("Used to validate risk assessments by checking if higher-risk components correlate with a higher observed failure rate.")
        fig, result = run_regression_analysis(); st.plotly_chart(fig, use_container_width=True); st.subheader("Strategic Interpretation"); st.markdown(result)
    with tab3:
        st.header("Assay Performance (Descriptive Stats)"); st.info("The foundational analysis for any analytical validation study (e.g., LoD, Precision).")
        fig, result = run_descriptive_stats(); st.plotly_chart(fig, use_container_width=True); st.subheader("Summary Statistics"); st.success(result)
    with tab4:
        st.header("Process Monitoring (Control Charts)"); st.info("X-bar charts are used to monitor the stability and variability of a process over time (e.g., daily controls).")
        fig = run_control_charts(); st.plotly_chart(fig, use_container_width=True); st.warning("**Insight:** A clear upward shift is detected around subgroup 15, indicating a special cause of variation requires investigation.")
    with tab5:
        st.header("Shelf-Life & Stability (Kaplan-Meier)"); st.info("Survival analysis is used to estimate the shelf-life of a product by modeling time-to-failure data.")
        fig, result = run_kaplan_meier(); st.plotly_chart(fig, use_container_width=True); st.subheader("Study Conclusion"); st.markdown(result)
    with tab6:
        st.header("Project Timeline Risk (Monte Carlo)"); st.info("Monte Carlo simulation runs thousands of 'what-if' scenarios on a project plan to forecast a probabilistic completion date.")
        fig, result = run_monte_carlo(); st.plotly_chart(fig, use_container_width=True); st.subheader("Risk-Adjusted Planning"); st.error(result)

# --- SIDEBAR NAVIGATION ---
PAGES = {
    "Home": render_main_page,
    "🧪 Assay V&V Metrics": render_assay_metrics_page,
    "🏭 Equipment Validation Metrics": render_equipment_metrics_page,
    "👥 Team & Project KPIs": render_team_kpis_page,
    "📊 Quality & CI KPIs": render_quality_kpis_page,
    "💻 Software V&V (IEC 62304)": render_software_vv_page,
    "📐 Advanced Statistical Methods": render_stats_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
