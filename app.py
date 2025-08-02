# app.py (Final, Monolithic, World-Class Version with ALL Content and Enhancements)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
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
# --- NEW AI/ML VISUALIZATION & DATA GENERATORS ---

def plot_multivariate_anomaly_detection(key):
    """
    Generates data and a 3D plot for multivariate anomaly detection using Isolation Forest.
    This simulates detecting subtle process deviations that univariate charts would miss.
    """
    np.random.seed(101)
    # Generate normal 'in-control' data for three process parameters
    in_control_data = np.random.multivariate_normal(
        mean=[10, 20, 5], 
        cov=[[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]], 
        size=300
    )
    # Inject subtle anomalies that are outliers in combination, but not necessarily individually
    anomalies = np.array([[11, 19, 7], [9, 21, 3]])
    data = np.vstack([in_control_data, anomalies])
    df = pd.DataFrame(data, columns=['Temp (°C)', 'Pressure (psi)', 'Flow Rate (mL/min)'])
    
    # AI/ML Model: Isolation Forest
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model.fit_predict(df)
    df['Status'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'In Control')
    
    # Visualization
    fig = px.scatter_3d(
        df, x='Temp (°C)', y='Pressure (psi)', z='Flow Rate (mL/min)',
        color='Status',
        color_discrete_map={'In Control': 'blue', 'Anomaly': 'red'},
        symbol='Status',
        size_max=10,
        title='AI-Powered Multivariate Process Monitoring'
    )
    fig.update_traces(marker=dict(size=4))
    return fig

def run_predictive_maintenance_model(key):
    """
    Simulates instrument sensor data drift over time to predict impending failure.
    Uses a Random Forest Classifier and shows feature importance.
    """
    np.random.seed(42)
    # Simulate data for 10 instruments over 100 days
    data = []
    for i in range(10):
        # Instruments 0-6 are healthy, 7-9 will fail
        will_fail = i >= 7
        for day in range(100):
            laser_drift = (day / 100) * 0.5 if will_fail else 0
            pressure_spike = (day / 100)**2 * 3 if will_fail else 0
            
            laser_intensity = np.random.normal(5 - laser_drift, 0.1)
            pump_pressure = np.random.normal(50 + pressure_spike, 0.5)
            temp_fluctuation = np.random.normal(37, 0.2 + (day/1000 if will_fail else 0))
            
            # Label failure in the last 5 days
            failure = 1 if will_fail and day > 95 else 0
            data.append([i, day, laser_intensity, pump_pressure, temp_fluctuation, failure])

    df = pd.DataFrame(data, columns=['Instrument_ID', 'Day', 'Laser_Intensity', 'Pump_Pressure', 'Temp_Fluctuation', 'Failure'])
    
    # AI/ML Model: Random Forest Classifier
    features = ['Laser_Intensity', 'Pump_Pressure', 'Temp_Fluctuation']
    X = df[features]
    y = df['Failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Display Model Performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy on Test Data", f"{accuracy:.2%}")

    # Feature Importance Plot
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Key Predictors of Instrument Failure')
    return fig

def run_nlp_topic_modeling(key):
    """
    Applies NLP Topic Modeling (LDA) to simulated free-text complaint data.
    """
    # Simulate free-text complaint data
    complaint_docs = [
        "The software froze during a run and I had to restart the instrument.",
        "Error code 503 appeared on screen, the manual is not clear on this.",
        "The reagent cartridge was leaking from the bottom seal upon opening the box.",
        "Results seem consistently higher than the previous lot, we suspect a calibration issue.",
        "The machine is making a loud grinding noise during the initial spin cycle.",
        "I cannot get the system to calibrate properly after the last software update.",
        "The touch screen is unresponsive in the top left corner.",
        "Another case of a leaky reagent pack, this is the third time this month.",
        "The instrument UI is very slow to respond after starting a new batch.",
        "Calibration failed multiple times before finally passing.",
        "The seal on the reagent pack was broken, causing a spill inside the machine."
    ] * 5 # Multiply to get more data

    # AI/ML Model: Latent Dirichlet Allocation (LDA) for Topic Modeling
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(complaint_docs)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    
    # Display topics and their keywords
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-6:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic {topic_idx+1}"] = ", ".join(top_words)

    st.write("#### Automatically Discovered Complaint Themes:")
    st.table(pd.DataFrame.from_dict(topics, orient='index', columns=["Top Keywords"]))
    return None # No chart needed, table is the output
# ================================================================================================================
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
    # FIX: Increased sample size from 7 to 8 to satisfy the statsmodels omni_normtest requirement (>=8 samples).
    conc = np.array([0, 10, 25, 50, 100, 200, 300, 400])
    signal = 50 + 2.5 * conc + np.random.normal(0, 20, 8) # Match sample size
    
    df = pd.DataFrame({'Concentration': conc, 'Signal': signal})
    fig = px.scatter(df, x='Concentration', y='Signal', trendline='ols', title="Assay Performance Regression (Linearity)")
    
    X = sm.add_constant(df['Concentration'])
    model = sm.OLS(df['Signal'], X).fit()
    
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

@st.cache_data
def get_complaint_data():
    """Generates a realistic, cached DataFrame of simulated complaint data."""
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", end="2023-12-31", freq="D"))
    complaint_types = ["False Positive", "Reagent Leak", "Instrument Error", "Software Glitch", "High CV", "No Result"]
    regions = ["AL", "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC"]
    lots = ["A2201-A", "A2201-B", "A2301-A", "A2301-B"]
    
    n_complaints = 300
    df = pd.DataFrame({
        "Complaint_ID": [f"C-{i+1:04d}" for i in range(n_complaints)],
        "Date": np.random.choice(dates, n_complaints),
        "Lot_Number": np.random.choice(lots, n_complaints, p=[0.2, 0.2, 0.3, 0.3]),
        "Region": np.random.choice(regions, n_complaints),
        "Complaint_Type": np.random.choice(complaint_types, n_complaints, p=[0.15, 0.1, 0.25, 0.1, 0.2, 0.2]),
        "Severity": np.random.choice(["Low", "Medium", "High"], n_complaints, p=[0.6, 0.3, 0.1])
    })
    
    n_signal = 15
    signal_df = pd.DataFrame({
        "Complaint_ID": [f"C-{i+301:04d}" for i in range(n_signal)],
        "Date": pd.to_datetime(pd.date_range(start="2023-11-01", periods=n_signal)),
        "Lot_Number": "A2301-B",
        "Region": "CA",
        "Complaint_Type": "False Positive",
        "Severity": "High"
    })
    
    final_df = pd.concat([df, signal_df]).sort_values("Date").reset_index(drop=True)
    return final_df

# --- PAGE RENDERING FUNCTIONS ---

def render_main_page():
    st.title("🎯 The V&V Executive Command Center")
    st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
    st.markdown("---")
    render_director_briefing("Portfolio Objective", "This interactive application translates the core responsibilities of V&V leadership into a suite of high-density dashboards. It is designed to be an overwhelming and undeniable demonstration of the strategic, technical, and quality systems expertise required for a senior leadership role in the medical device industry.", "ISO 13485, ISO 14971, IEC 62304, 21 CFR 820, 21 CFR Part 11, CLSI Guidelines", "A well-led V&V function directly accelerates time-to-market, reduces compliance risk, lowers the cost of poor quality (COPQ), and builds a culture of data-driven excellence.")
    st.info("Please use the navigation sidebar on the left to explore each of the core competency areas.")

def render_design_controls_page():
    st.title("🏛️ 1. Design Controls, Planning & Risk Management")
    st.markdown("---")
    render_director_briefing("The Design History File (DHF) as a Strategic Asset", "The DHF is the compilation of records that demonstrates the design was developed in accordance with the design plan and regulatory requirements. An effective V&V leader architects the DHF from day one.", "FDA 21 CFR 820.30 (Design Controls), ISO 13485:2016 (Section 7.3)", "Ensures audit readiness and provides a clear, defensible story of product development to regulatory bodies, accelerating submission review times.")
    
    with st.container(border=True):
        st.subheader("The V-Model: A Framework for Compliant V&V")
        st.markdown("The V-Model is the cornerstone of a structured V&V strategy, visually linking the design and development phases (left side) to the corresponding testing and validation phases (right side). This ensures that for every design input, there is a corresponding validation output, forming the basis of a complete and auditable Design History File (DHF).")
        st.plotly_chart(create_v_model_figure(), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### **Left Side: Design & Development (Building it Right)**")
            st.markdown("- **User Needs:** High-level goals from Marketing/customers.\n- **System Requirements:** Detailed functional/performance specs.\n- **Architecture:** High-level system design.\n- **Module Design:** Low-level detailed design.")
        with col2:
            st.markdown("#### **Right Side: Verification & Validation (Proving We Built the Right Thing)**")
            st.markdown("- **Unit Test:** Verifies individual code modules.\n- **Integration Test:** Verifies that modules work together.\n- **System V&V:** Verifies the complete system against requirements.\n- **User Acceptance Testing (UAT):** Validates the system against user needs.")
        st.success("**Actionable Insight:** By enforcing this model, a V&V leader prevents late-stage failures, ensures no requirements are left untested, and provides a clear, defensible V&V narrative to auditors. The horizontal lines represent the core of traceability.")
    
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
# In render_execution_monitoring_page()
# ... after the existing render_metric_card calls ...
    
    render_metric_card(
        "AI-Powered Anomaly Detection",
        "This model uses an Isolation Forest algorithm to perform multivariate anomaly detection. It identifies outlier data points based on combinations of variables, finding subtle issues that traditional single-variable control charts might miss.",
        plot_multivariate_anomaly_detection,
        "The model identified two batches that are statistically significant outliers when considering Temperature, Pressure, and Flow Rate simultaneously. Although each individual parameter may be within its spec, their combination is anomalous and requires investigation. This is a powerful tool for early warning of process drift.",
        "AIAG SPC Manual, FDA Guidance on Process Validation",
        key="anomaly_detection"
    )
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
    
    st.markdown("---")
    st.subheader("Advanced Software V&V & CSV Dashboard")
    st.info("This section provides a detailed view of Computer System Validation (CSV) for GxP systems (like LIMS) and V&V for Software in a Medical Device (SiMD), covering key metrics and compliance with GAMP 5, IEC 62304, and modern cybersecurity standards.")

    tab1, tab2, tab3 = st.tabs(["📊 Key Metrics & KPIs", "📋 GAMP 5 Compliance", "🛡️ Cybersecurity Posture"])
    with tab1:
        st.markdown("##### SiMD V&V Release-Readiness Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Requirements Coverage", "98.7%")
            st.progress(0.987)
        with col2:
            st.metric("Test Case Pass Rate (System V&V)", "99.2%")
            st.progress(0.992)
        with col1:
            st.metric("Defect Density (per KLOC)", "0.85", delta="-0.12", delta_color="inverse", help="Defects per 1,000 Lines of Code. Lower is better.")
        with col2:
            st.metric("Static Analysis Critical Warnings", "3", delta="2", delta_color="inverse", help="Number of critical issues found in automated code scans. Delta from previous build.")
        st.success("**Actionable Insight:** The high requirements coverage and test pass rates indicate strong readiness for a design freeze. The low and decreasing defect density suggests improving code quality. The 3 remaining critical warnings must be adjudicated before final release.")
    with tab2:
        st.info("**GAMP 5** provides a risk-based framework for validating GxP computerized systems (e.g., lab equipment software, LIMS, eDMS). The category determines the validation rigor required.")
        gamp_data = {
            "System": ["Instrument Control SW", "LIMS", "Statistical Analysis SW", "eDMS"],
            "GAMP 5 Category": ["Cat 4: Configured", "Cat 5: Custom", "Cat 3: Standard", "Cat 4: Configured"],
            "Validation Approach": ["Full Validation of Configured Elements", "Full Prospective Validation", "Supplier Assessment & IQ/OQ", "Risk-Based Validation"],
            "Status": ["In Progress", "Planning", "Complete", "Complete"]
        }
        df_gamp = pd.DataFrame(gamp_data)
        def gamp_color(val):
            color = "background-color: "
            if val == "Cat 5: Custom": return color + "#FF7F7F" # Red
            if val == "Cat 4: Configured": return color + "#FFD700" # Yellow
            if val == "Cat 3: Standard": return color + "#90EE90" # Green
            return ""
        st.dataframe(df_gamp.style.map(gamp_color, subset=['GAMP 5 Category']), use_container_width=True, hide_index=True)
        st.success("**Actionable Insight:** The LIMS system, as a GAMP 5 Category 5, requires a full prospective validation effort. This must be prioritized and resourced appropriately. The completed validation for the statistical software provides confidence in its use for regulatory analysis.")
    with tab3:
        st.info("A robust cybersecurity V&V strategy is non-negotiable for connected medical devices. This aligns with **FDA's Premarket Cybersecurity Guidance** and **AAMI TIR57**.")
        st.markdown("##### Cybersecurity V&V Checklist")
        st.checkbox("✅ Threat Modeling (STRIDE) Performed", value=True, disabled=True)
        st.checkbox("✅ Secure Coding Policy in place & training complete", value=True, disabled=True)
        st.checkbox("✅ Software Bill of Materials (SBOM) Generated & Reviewed", value=True, disabled=True)
        st.checkbox("❌ Penetration Testing by Third-Party Vendor", value=False, disabled=True)
        st.checkbox("✅ Vulnerability Scanning Integrated into CI/CD Pipeline", value=True, disabled=True)
        st.error("**Actionable Insight:** A critical gap exists in third-party penetration testing. This is a major finding for any regulatory submission. **Action:** Immediately engage a qualified vendor to perform penetration testing before the final code freeze.")

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
    render_director_briefing("Executive-Level V&V Leadership", "A true V&V leader operates at the intersection of technical execution, financial reality, and cross-functional strategy. This command center demonstrates the tools and mindset required to run V&V not as a cost center, but as a strategic business partner that drives value and mitigates enterprise-level risk.", "ISO 13485 Section 5 (Management Responsibility) & 6 (Resource Management)", "Aligns V&V department with corporate financial goals, improves resource allocation, de-risks regulatory pathways, and enables scalable growth through effective talent management and partner oversight." )

    tab1, tab2, tab3, tab4 = st.tabs(["💰 V&V Cost & ROI Forecaster", "🌐 Regulatory & Partner Dashboard", "🧑‍🔬 Team Competency Matrix", "🔄 ECO Impact Assessment"])

    with tab1:
        st.header("V&V Project Cost & ROI Forecaster")
        st.info("Translate technical plans into financial forecasts to justify resource allocation and demonstrate value to executive leadership.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Inputs: Project Scope & Resources")
            proj = st.selectbox("Select Project", ["ImmunoPro-A (510k)", "MolecularDX-2 (PMA)", "CardioScreen-X (De Novo)"])
            scenario = st.radio("Select Resourcing Scenario", ["Internal Team", "CRO Outsource"], horizontal=True)
            
            if scenario == "Internal Team":
                st.markdown("**Timeline Estimates**")
                av_weeks = st.slider("Analytical V&V (Weeks)", 1, 26, 8, key="int_av")
                sv_weeks = st.slider("System V&V (Weeks)", 1, 26, 10, key="int_sv")
                sw_weeks = st.slider("Software V&V (Weeks)", 1, 26, 6, key="int_sw")
                cs_weeks = st.slider("Clinical Support (Weeks)", 1, 26, 12, key="int_cs")
                
                st.markdown("**Resource Allocation**")
                fte_sci = st.slider("Number of Scientists (FTEs)", 1, 10, 2, key="int_sci")
                fte_eng = st.slider("Number of Engineers (FTEs)", 1, 10, 1, key="int_eng")

                st.markdown("**Cost Basis**")
                fte_cost = st.number_input("Fully-Burdened Cost per FTE-Week ($)", value=4000, step=100, key="int_fte_cost")
                reagent_cost_per_week = st.number_input("Cost of Reagents per Analytical/System Week ($)", value=7500, step=500, key="int_reagent")
                instrument_cost_per_week = st.number_input("Instrument Time & Maintenance per V&V Week ($)", value=1500, step=100, key="int_instr")
            else: # CRO Outsource Scenario
                st.markdown("**CRO & Internal Oversight Costs**")
                cro_contract_value = st.number_input("CRO Contract Value ($)", value=500000, step=25000)
                mgmt_fte = st.slider("Internal Management Overhead (FTEs)", 0.5, 3.0, 1.0, 0.5)
                mgmt_weeks = st.slider("Project Duration (Weeks)", 10, 52, 36)
                fte_cost = st.number_input("Fully-Burdened Cost per FTE-Week ($)", value=5000, step=100, key="cro_fte_cost")
        with col2:
            st.subheader("Forecasted V&V Budget & ROI")
            
            if scenario == "Internal Team":
                total_personnel_weeks = (av_weeks + sv_weeks + sw_weeks + cs_weeks)
                total_fte = fte_sci + fte_eng
                personnel_cost = total_personnel_weeks * total_fte * fte_cost
                reagent_total_cost = (av_weeks + sv_weeks) * reagent_cost_per_week
                instrument_total_cost = (av_weeks + sv_weeks + sw_weeks) * instrument_cost_per_week
                total_budget = personnel_cost + reagent_total_cost + instrument_total_cost
                cost_data = {'Category': ['Personnel', 'Reagents & Consumables', 'Instrument Time'], 'Cost': [personnel_cost, reagent_total_cost, instrument_total_cost]}
            else: # CRO Outsource Scenario
                personnel_cost = mgmt_fte * mgmt_weeks * fte_cost
                total_budget = cro_contract_value + personnel_cost
                cost_data = {'Category': ['CRO Contract', 'Internal Management'], 'Cost': [cro_contract_value, personnel_cost]}

            st.metric("Total Forecasted V&V Budget", f"${total_budget:,.0f}", help="Calculated based on selected scenario.")
            df_costs = pd.DataFrame(cost_data)
            fig_tree = px.treemap(df_costs, path=['Category'], values='Cost', title='V&V Budget Allocation by Category', color_discrete_map={'(?)':'#2ca02c', 'Personnel':'#1f77b4', 'Reagents & Consumables':'#ff7f0e', 'Instrument Time':'#d62728', 'CRO Contract': '#9467bd', 'Internal Management': '#8c564b'})
            st.plotly_chart(fig_tree, use_container_width=True)

            if scenario == "Internal Team":
                st.subheader("Monthly Personnel Cost Burn")
                total_weeks = av_weeks + sv_weeks + sw_weeks + cs_weeks
                monthly_cost = total_fte * fte_cost * 4.33 # Avg weeks in a month
                burn_df = pd.DataFrame({
                    "Month": pd.date_range(start="2024-01-01", periods=int(total_weeks/4.33)+1, freq="ME"),
                    "Cost": monthly_cost
                })
                fig_burn = px.bar(burn_df, x="Month", y="Cost", title="Projected Monthly Personnel Spend")
                fig_burn.update_layout(yaxis_title="Cost ($)")
                st.plotly_chart(fig_burn, use_container_width=True)

            st.subheader("Return on Investment (ROI) Estimate")
            tpp_revenue = st.number_input("TPP Forecasted 3-Year Revenue ($)", value=15_000_000, step=1_000_000, format="%d")
            if total_budget > 0:
                roi = ((tpp_revenue - total_budget) / total_budget) * 100
                st.metric("High-Level V&V ROI", f"{roi:.1f}%", help="(Forecasted Revenue - V&V Cost) / V&V Cost")
    
    with tab2:
        st.header("Regulatory Strategy & External Partner Dashboard")
        st.info("Dynamically align V&V evidence with submission requirements and manage external vendor performance.")
        sub_type = st.selectbox("Select Submission Type", ["FDA 510(k)", "FDA PMA", "EU IVDR Class C", "EU IVDR Class D"])
        with st.container(border=True):
            st.subheader(f"Dynamic Evidence Checklist for: {sub_type}")
            st.checkbox("✅ Analytical Performance Studies (LoD, Precision, Linearity, etc.)", value=True, disabled=True)
            st.checkbox("✅ Software V&V Documentation (per IEC 62304)", value=True, disabled=True)
            st.checkbox("✅ Risk Management File (per ISO 14971)", value=True, disabled=True)
            st.checkbox("✅ Stability & Shelf-Life Data", value=True, disabled=True)
            if "510(k)" in sub_type: 
                st.checkbox("✅ Substantial Equivalence Testing Data", value=True, disabled=True)
            if "PMA" in sub_type: 
                st.checkbox("🔥 Clinical Validation Data (Pivotal Study Support)", value=True, disabled=True)
                st.checkbox("🔥 PMA Module-Specific Data Packages", value=True, disabled=True)
            if "IVDR" in sub_type:
                st.checkbox("🔥 Scientific Validity Report", value=True, disabled=True)
                st.checkbox("🔥 Clinical Performance Study Report", value=True, disabled=True)
                if "Class D" in sub_type:
                    st.checkbox("🔥 Common Specifications (CS) Conformance Data", value=True, disabled=True)
                    st.checkbox("🔥 Notified Body & EURL Review Support Package", value=True, disabled=True)

        st.subheader("CRO Partner Performance Oversight")
        df_perf = pd.DataFrame({'Metric': ['On-Time Delivery (%)', 'Protocol Deviation Rate (%)', 'Data Quality Score (1-100)'], 'Internal Team': [95, 2.1, 98.5], 'CRO Partner A': [88, 4.5, 96.2]})
        fig = px.bar(df_perf, x='Metric', y=['Internal Team', 'CRO Partner A'], barmode='group', title="Quarterly Performance: Internal Team vs. CRO Partner A")
        fig.update_layout(yaxis_title="Performance Score")
        st.plotly_chart(fig, use_container_width=True)
        st.error("**Actionable Insight:** CRO Partner A is underperforming on On-Time Delivery and has more than double our internal deviation rate. This poses a significant project timeline and data integrity risk. **Action:** Schedule a Quarterly Business Review (QBR) to present this data and establish a formal Performance Improvement Plan (PIP).")

    with tab3:
        st.header("Team Competency & Development Matrix")
        st.info("Proactively manage talent, identify skill gaps for upcoming projects, and drive strategic team development.")
        skills = ['qPCR Method Validation', 'ELISA Development', 'GAMP 5 CSV', 'Statistical Analysis (Python)', 'ISO 14971 Risk Management', 'JMP/Minitab', 'Clinical Study Design']
        team = ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan']
        data = np.random.randint(1, 4, size=(len(team), len(skills)))
        df_skills = pd.DataFrame(data, index=team, columns=skills)
        df_skills.index.name = "Team Member"

        st.subheader("1. Filter for Project Needs")
        required_skills = st.multiselect("Select Required Project Skills", options=skills, default=['qPCR Method Validation', 'ISO 14971 Risk Management', 'Statistical Analysis (Python)'])
        
        st.subheader("2. Analyze Team Readiness")
        def highlight_skills(df):
            style = pd.DataFrame('', index=df.index, columns=df.columns)
            for skill in required_skills:
                if skill in df.columns:
                    style.loc[:, skill] = 'background-color: yellow'
            return style
        
        st.dataframe(df_skills.style.apply(highlight_skills, axis=None).background_gradient(cmap='RdYlGn', vmin=1, vmax=3, axis=None).set_caption("Proficiency: 1 (Novice) to 3 (Expert)"), use_container_width=True)

        st.subheader("3. Formulate Action Plan")
        missing_skills = [s for s in required_skills if s not in df_skills.columns]
        if missing_skills:
            st.error(f"**Critical Gap:** The team completely lacks the required skill(s): {', '.join(missing_skills)}.")
        
        team_readiness = df_skills[required_skills].sum(axis=1) if required_skills else pd.Series()
        if not team_readiness.empty:
            best_fit = team_readiness.idxmax()
            st.success(f"**Staffing Insight:** **{best_fit}** is the strongest individual lead for this project based on the required skills. However, for ISO 14971, no one is rated as an expert (Level 3).")
            st.warning("**Development Action:** Prioritize ISO 14971 Risk Management training for at least two team members this quarter to mitigate this single-point-of-failure risk.")
        
        csv = df_skills.to_csv().encode('utf-8')
        st.download_button(
            label="Export Full Competency Matrix (CSV)",
            data=csv,
            file_name='team_competency_matrix.csv',
            mime='text/csv',
        )

    with tab4:
        st.header("Interactive ECO Impact Assessment Tool")
        st.info("A logic-driven tool to ensure a consistent, risk-based approach to V&V for post-market changes, ensuring compliance with 21 CFR 820.")
        change_type = st.selectbox("Select Type of Engineering Change Order (ECO)", ["Reagent Formulation Change", "Software (Minor UI change)", "Software (Algorithm update)", "Supplier Change (Critical Component)", "Manufacturing Process Change"])
        
        with st.container(border=True):
            st.subheader("Minimum Required V&V Activities (per SOP-00123)")
            rationale_text = ""
            impact_text = ""
            if change_type == "Reagent Formulation Change":
                st.error("🔴 **Full V&V Suite Required**")
                st.markdown("- Analytical Performance (Precision, LoD, Linearity)\n- Stability Studies (Accelerated & Real-time)\n- Clinical Bridging Study\n- Shipping Validation")
                rationale_text = "Change directly impacts assay performance and patient results. This is a high-risk change requiring comprehensive re-validation and potentially a new regulatory filing."
                impact_text = "**URS-001** (Clinical Sensitivity), **DI-002** (Analytical Sensitivity), **DI-003** (Stability)."
            elif change_type == "Software (Minor UI change)":
                st.success("🟢 **Limited V&V Required**")
                st.markdown("- Software Regression Testing (Targeted)\n- Usability Assessment (Summative if applicable)\n- Documentation Update")
                rationale_text = "Change does not impact the analytical algorithm or patient data integrity. This is a low-risk change focused on user experience."
                impact_text = "**SRS-012** (UI Display)."
            elif change_type == "Software (Algorithm update)":
                st.error("🔴 **Full Software & Analytical V&V Required**")
                st.markdown("- Full Software Validation Suite (per IEC 62304 Class)\n- Analytical Performance regression testing using old vs. new software\n- Full Risk Management File Update")
                rationale_text = "Change to the core algorithm directly impacts patient result calculation. This is the highest software risk category and requires rigorous verification."
                impact_text = "All performance requirements (**URS-001, DI-002**) and software requirements linked to the algorithm."
            elif change_type == "Supplier Change (Critical Component)":
                st.warning("🟡 **Targeted V&V Required**")
                st.markdown("- New Component Qualification (IQC)\n- System-level performance regression testing\n- Limited stability run (bracketing)\n- Comparability Analysis")
                rationale_text = "Change introduces a new variable into the system. This is a medium-risk change requiring confirmation that system performance, reliability, and safety are unaffected."
                impact_text = "All system-level requirements and potentially stability claims (**DI-003**)."
            elif change_type == "Manufacturing Process Change":
                st.warning("🟡 **Process Re-Validation Required**")
                st.markdown("- Process Validation (IQ, OQ, PQ) for the changed step\n- Product Performance Testing on 3 new lots\n- Stability testing on 1 new lot")
                rationale_text = "Change to the manufacturing process could impact product consistency and performance. A risk-based re-validation is required to ensure continued product quality."
                impact_text = "Product specification requirements, stability claims (**DI-003**)."

            st.markdown(f"**Rationale:** {rationale_text}")
            with st.container(border=True):
                st.info(f"**Traceability Impact Analysis:** This change affects the following critical requirements in the RTM: {impact_text}")

def render_post_market_page():
    st.title("📡 7. Post-Market Intelligence & CAPA Feeder")
    render_director_briefing(
        "Closing the Quality Loop",
        "A mature V&V function extends its influence beyond product launch. This dashboard demonstrates proactive post-market surveillance, using field data to monitor real-world performance, identify emerging trends, and provide data-driven triggers for the CAPA system. This is a critical component of a robust Quality Management System.",
        "21 CFR 820.198 (Complaint files), 21 CFR 820.100 (CAPA), ISO 13485:2016 Section 8.2.2 & 8.5.2",
        "Drives continuous product improvement, reduces the risk of field actions or recalls, and demonstrates a culture of quality and patient safety to regulatory bodies."
    )
    df = get_complaint_data()
    
    capa_filter = df[(df['Lot_Number'] == 'A2301-B') & (df['Complaint_Type'] == 'False Positive')]
    if len(capa_filter) > 10:
        st.error(
            f"**🔴 CAPA Alert Triggered:** {len(capa_filter)} 'False Positive' complaints for Lot #A2301-B have been received in the last quarter, exceeding the defined threshold of 10. "
            "**Action:** Recommend initiating CAPA-2024-001. V&V to provide resources for investigation and re-validation of retained samples."
        )

    st.subheader("Post-Market Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Complaint Analysis (Pareto)**")
            complaint_counts = df['Complaint_Type'].value_counts().reset_index()
            complaint_counts.columns = ['Complaint_Type', 'Count']
            complaint_counts['Cumulative_Percentage'] = 100 * complaint_counts['Count'].cumsum() / complaint_counts['Count'].sum()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=complaint_counts['Complaint_Type'], y=complaint_counts['Count'], name='Count'), secondary_y=False)
            fig.add_trace(go.Scatter(x=complaint_counts['Complaint_Type'], y=complaint_counts['Cumulative_Percentage'], name='Cumulative %', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Pareto Chart of Complaint Types')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("**Complaint Trend (Monthly)**")
            monthly_counts = df.resample('ME', on='Date').size().reset_index(name='Count')
            fig_ts = px.line(monthly_counts, x='Date', y='Count', title='Total Complaints per Month')
            st.plotly_chart(fig_ts, use_container_width=True)

    with st.container(border=True):
        st.markdown("**Geographic Complaint Hotspots**")
        region_counts = df['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        fig_map = px.choropleth(region_counts, locations='Region', locationmode="USA-states", color='Count', scope="usa", title="Complaints by US State", color_continuous_scale="Reds")
        st.plotly_chart(fig_map, use_container_width=True)
# In render_post_market_page()
# ... after the existing content .............................................................................................................................................................................
    
    st.markdown("---")
    st.subheader("AI-Driven Predictive & Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Predictive Maintenance Model")
            st.markdown("This Random Forest model is trained on historical sensor data to predict the likelihood of an instrument failing in the near future. This enables proactive maintenance, reducing unplanned downtime and costly failed runs.")
            fig_pred = run_predictive_maintenance_model("pred_maint")
            if fig_pred:
                st.plotly_chart(fig_pred, use_container_width=True)
            st.success("**Actionable Insight:** The model shows that 'Pump_Pressure' is by far the most significant predictor of failure. The maintenance team should prioritize monitoring this parameter and consider it a leading indicator for service scheduling.")

    with col2:
        with st.container(border=True):
            st.markdown("#### NLP for Complaint Theme Discovery")
            st.markdown("This NLP model uses Latent Dirichlet Allocation (LDA) to analyze free-text from customer complaints, automatically grouping them into distinct topics. This turns unstructured data into actionable, quantifiable themes without manual review.")
            run_nlp_topic_modeling("nlp_topic")
            st.success("**Actionable Insight:** The AI has automatically identified a recurring theme related to 'leaky reagent packs'. This quantitative signal elevates the issue's priority and provides a strong justification for launching a formal investigation with the packaging engineering team.")
#===============================================================================================================================================================================
def render_dhf_hub_page():
    st.title("🗂️ 8. The Digital DHF & Workflow Hub")
    render_director_briefing(
        "Orchestrating the Design History File",
        "The DHF is not a static folder; it's a dynamic, living entity that requires active management and cross-functional alignment. This hub demonstrates the ability to manage formal QMS workflows and provides concrete examples of the key documents that V&V is responsible for authoring and maintaining, proving both procedural compliance and documentation excellence.",
        "21 CFR 820.30(j) (DHF), 21 CFR 820.40 (Document Controls), GAMP 5",
        "Ensures audit-proof documentation, accelerates review cycles by providing clear templates and expectations, and fosters seamless collaboration between V&V, R&D, Quality, and Regulatory."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("V&V Master Plan Approval Workflow")
            st.markdown("Status for `VV-MP-001_ImmunoPro-A`:")
            st.markdown("---")
            st.markdown("✔️ **V&V Lead (Self):** Approved `2024-01-15`")
            st.markdown("✔️ **R&D Project Lead:** Approved `2024-01-16`")
            st.markdown("✔️ **Quality Assurance Lead:** Approved `2024-01-17`")
            st.markdown("🟠 **Regulatory Affairs Lead:** Pending Review")
            st.markdown("⬜ **Head of Development:** Not Started")
            st.info("**Insight:** This workflow visualization provides instant status clarity for key deliverables, enabling proactive follow-up to prevent bottlenecks.")

    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer")
            st.markdown("Click to expand and view mock V&V document templates.")
            
            with st.expander("📄 View Mock V&V Protocol Template"):
                st.markdown("""
                ### V&V Protocol: AVP-LOD-01 - Analytical Sensitivity (LoD)
                **Version:** 1.0
                ---
                **1.0 Purpose:** To determine the Limit of Detection (LoD) of the ImmunoPro-A Assay, defined as the lowest concentration of analyte that can be detected with 95% probability.

                **2.0 Scope:** This protocol applies to the ImmunoPro-A Assay on the QuidelOrtho-100 platform.

                **3.0 Traceability to Requirements:**
                - **DI-002:** Analytical sensitivity (LoD) shall be <= 50 copies/mL.

                **4.0 Method/Procedure:**
                - Prepare a dilution series of the analyte standard from 100 copies/mL down to 5 copies/mL.
                - Test each dilution level with 20 replicates across 3 different reagent lots and 2 instruments.
                - Run a negative control (0 copies/mL) with 60 replicates.
                
                **5.0 Acceptance Criteria:**
                - The hit rate at the claimed LoD (50 copies/mL) must be ≥ 95%.
                - The hit rate for the negative control must be ≤ 5%.

                **6.0 Data Analysis Plan:**
                - Data will be analyzed using Probit regression to calculate the 95% detection probability concentration.
                - Results will be summarized in a table showing hit rates for each level.
                """)

            with st.expander("📋 View Mock V&V Report Template"):
                st.markdown("""
                ### V&V Report: AVR-LOD-01 - Analytical Sensitivity (LoD)
                **Version:** 1.0
                ---
                **1.0 Summary:** The LoD study was executed per protocol AVP-LOD-01. The results confirm that the ImmunoPro-A Assay meets the required analytical sensitivity.

                **2.0 Deviations:**
                - **DEV-001:** One replicate at the 25 copies/mL level on Instrument #2 was invalidated due to an operator error. The replicate was repeated successfully. Impact: None.

                **3.0 Results vs. Acceptance Criteria:**
                | Concentration (copies/mL) | Replicates | Hits | Hit Rate (%) | Acceptance Criteria | Pass/Fail |
                |---|---|---|---|---|---|
                | 50 | 60 | 59 | 98.3% | >= 95% | **PASS** |
                | 25 | 60 | 52 | 86.7% | N/A | N/A |
                | 10 | 60 | 31 | 51.7% | N/A | N/A |
                | 0 | 60 | 2 | 3.3% | <= 5% | **PASS** |

                **4.0 Conclusion:** The study successfully demonstrated a hit rate of 98.3% at 50 copies/mL, satisfying the acceptance criteria. The LoD is confirmed to be ≤ 50 copies/mL. Probit analysis estimates the 95% detection concentration at 38.5 copies/mL, providing a significant performance margin.

                **5.0 Traceability:** This report provides objective evidence fulfilling requirement **DI-002**.
                """)


# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = {
    "Executive Summary": render_main_page,
    "1. Design Controls & Planning": render_design_controls_page,
    "2. Method & Process Validation": render_method_validation_page,
    "3. Execution Monitoring & SPC": render_execution_monitoring_page,
    "4. Project & Quality Management": render_quality_management_page,
    "5. Advanced Statistical Methods": render_stats_page,
    "6. Strategic Command & Control": render_strategic_command_page,
    "7. Post-Market Surveillance": render_post_market_page,
    "8. The Digital DHF Hub": render_dhf_hub_page,
}

st.sidebar.title("V&V Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page_to_render = PAGES[selection]
page_to_render()
