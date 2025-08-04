# app.py (Final, SME World-Class Version - Complete and Unabridged)

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
from typing import Tuple

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
NEUTRAL_GREY = '#B0BEC5'
DARK_GREY = '#455A64'
BACKGROUND_GREY = '#ECEFF1'

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

def plot_kpi_sparkline(data: list, unit: str, x_axis_label: str, is_good_down: bool = False) -> go.Figure:
    """Creates a compact sparkline chart for a KPI with subtle axes, labels, and units."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data))), y=data, mode='lines',
        line=dict(color=PRIMARY_COLOR, width=3), fill='tozeroy', fillcolor='rgba(4, 96, 169, 0.1)'
    ))
    end_color = SUCCESS_GREEN if (data[-1] < data[-2] and is_good_down) or (data[-1] > data[-2] and not is_good_down) else ERROR_RED
    fig.add_trace(go.Scatter(
        x=[0, len(data)-1], y=[data[0], data[-1]], mode='markers', marker=dict(size=[6, 10], color=['grey', end_color])
    ))
    
    # Add annotations for min, max, and x-axis label
    min_val, max_val = min(data), max(data)
    fig.add_annotation(x=0, y=max_val, text=f"{max_val:.1f}{unit}", showarrow=False, xanchor='left', yanchor='bottom', font=dict(size=10, color=DARK_GREY))
    fig.add_annotation(x=0, y=min_val, text=f"{min_val:.1f}{unit}", showarrow=False, xanchor='left', yanchor='top', font=dict(size=10, color=DARK_GREY))
    fig.add_annotation(x=(len(data)-1)/2, y=min_val, text=x_axis_label, showarrow=False, yanchor='top', yshift=-5, font=dict(size=9, color=NEUTRAL_GREY))

    fig.update_layout(
        height=100,
        margin=dict(l=10, r=10, t=15, b=15),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- DATA GENERATORS & VISUALIZATIONS ---

def case_study_csv():
    st.info("**Purpose:** A compliant Computer System Validation (CSV) project follows a structured lifecycle. This Gantt chart visualizes the execution of a CSV project, including critical parallel workstreams for IT infrastructure and ERES testing.")
    df = pd.DataFrame([
        dict(Task="Validation Plan (VP)", Start='2023-01-01', Finish='2023-01-31', Phase='Planning'),
        dict(Task="Risk Assessment (RA)", Start='2023-02-01', Finish='2023-02-15', Phase='Planning'),
        dict(Task="IT Infrastructure Qual (Server)", Start='2023-02-16', Finish='2023-03-15', Phase='IT Qualification'),
        dict(Task="IQ/OQ/PQ Protocol Authoring", Start='2023-02-16', Finish='2023-03-31', Phase='Documentation'),
        dict(Task="ERES Testing (21 CFR Part 11)", Start='2023-04-01', Finish='2023-04-15', Phase='Execution'),
        dict(Task="IQ/OQ/PQ Execution", Start='2023-04-16', Finish='2023-05-31', Phase='Execution'),
        dict(Task="Traceability Matrix (RTM)", Start='2023-06-01', Finish='2023-06-15', Phase='Documentation'),
        dict(Task="Validation Summary Report (VSR)", Start='2023-06-16', Finish='2023-06-30', Phase='Documentation'),
    ])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Phase", title="<b>CSV Project Timeline: GAMP 5 Lifecycle</b>")
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** This Gantt chart demonstrates a holistic understanding of CSV project management. It correctly places IT infrastructure qualification as a prerequisite for formal execution and shows that ERES (Electronic Records, Electronic Signatures) testing is a distinct, critical activity for ensuring Part 11 compliance. This structured approach de-risks the project and ensures audit readiness.")

def case_study_cleaning_validation():
    st.info("**Purpose:** This chart validates the cleaning procedure using the 'worst-case' product (most difficult to clean) to demonstrate effectiveness for all products made on the equipment. This is a critical step in preventing cross-contamination in a multi-product facility.")
    locations = ['Swab 1 (Reactor Wall)', 'Swab 2 (Agitator Blade)', 'Final Rinse']
    df = pd.DataFrame({'Sample Location': locations * 2, 'Product': ['Product A (Worst-Case)']*3 + ['Product B']*3, 'TOC Result (ppb)': [180, 210, 25, 80, 95, 15], 'Acceptance Limit (ppb)': [500, 500, 50] * 2})
    fig = px.bar(df, x='Sample Location', y='TOC Result (ppb)', color='Product', barmode='group', title='<b>Cleaning Validation Results (Total Organic Carbon)</b>', text_auto='.0f')
    fig.add_trace(go.Scatter(x=df['Sample Location'].unique(), y=[500, 500, 50], name='Acceptance Limit', mode='lines', line=dict(color=ERROR_RED, dash='dash', width=3)))
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The results for the 'worst-case' Product A are well below limits. This data robustly proves the cleaning procedure is effective for the entire product family, allowing for efficient product changeover without needing a separate validation for each product.")

def case_study_doe():
    st.info("**Purpose:** DOE is a powerful statistical tool used during process development to define a design space. The 2D Contour Plot visualizes this space, defining the Normal Operating Range (NOR) for routine production and the wider Proven Acceptable Range (PAR), which is the 'safe zone' where quality is assured.")
    temp = np.linspace(30, 40, 10); ph = np.linspace(6.8, 7.6, 10); 
    temp_grid, ph_grid = np.meshgrid(temp, ph)
    signal = 100 - (temp_grid - 37)**2 - 20*(ph_grid - 7.2)**2 + np.random.rand(10, 10)*2
    col1, col2 = st.columns(2)
    with col1:
        fig_3d = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis', colorbar_title='Yield')])
        fig_3d.update_layout(title='<b>DOE Response Surface (3D)</b>', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Yield (%)'), margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)
    with col2:
        fig_2d = go.Figure(data=go.Contour(z=signal, x=temp, y=ph, colorscale='viridis', contours_coloring='lines', line_width=2))
        fig_2d.add_shape(type="rect", x0=36, y0=7.1, x1=38, y1=7.3, line=dict(color=SUCCESS_GREEN, width=3), name='NOR')
        fig_2d.add_shape(type="rect", x0=35, y0=7.0, x1=39, y1=7.4, line=dict(color=WARNING_AMBER, width=2, dash="dash"), name='PAR')
        fig_2d.add_annotation(x=37, y=7.2, text="<b>NOR</b><br>(Normal Operating<br>Range)", showarrow=False, font=dict(color=SUCCESS_GREEN))
        fig_2d.add_annotation(x=35.1, y=7.38, text="<b>PAR</b> (Proven Acceptable Range)", showarrow=False, xanchor="left", yanchor="top", font=dict(color=WARNING_AMBER))
        fig_2d.update_layout(title='<b>Process Operating Space (2D)</b>', xaxis_title='Temperature (°C)', yaxis_title='pH', margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_2d, use_container_width=True)
    st.success("**Actionable Insight:** The established PAR provides manufacturing with operational flexibility. As long as the process remains within this wider range, minor fluctuations do not constitute a deviation or require re-validation. This data-driven approach, aligned with ICH Q8, significantly reduces quality overhead and supports continuous improvement.")

def case_study_shipping():
    st.info("**Purpose:** This PQ study simulates a worst-case transit route, monitoring both temperature and shock/vibration to ensure the validated packaging can protect the product integrity from both environmental and physical hazards.")
    rng = np.random.default_rng(30); 
    time = pd.to_datetime(pd.date_range("2023-01-01", periods=48, freq="h"))
    temp = rng.normal(4, 0.5, 48); temp[24] = 8.5; shock = rng.random(48) * 10; shock[35] = 55
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time, y=temp, name='Temperature (°C)', line=dict(color=PRIMARY_COLOR)), secondary_y=False)
    fig.add_hrect(y0=2, y1=8, line_width=0, fillcolor=SUCCESS_GREEN, opacity=0.1, secondary_y=False, annotation_text="Temp Spec", annotation_position="top left")
    fig.add_trace(go.Bar(x=time, y=shock, name='Shock (G-force)', marker_color=ERROR_RED, opacity=0.5), secondary_y=True)
    fig.update_layout(title_text='<b>Shipping Lane PQ: Temperature & Shock Profile</b>', title_x=0.5, plot_bgcolor=BACKGROUND_GREY, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, range=[0, 10]); fig.update_yaxes(title_text="Shock (G-force)", secondary_y=True, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The data confirms two key findings: 1) The insulated shipper successfully maintained the internal product temperature within the required 2-8°C range despite an external excursion. 2) A significant shock event of 55G was recorded but remained below the product's known fragility limit of 80G. **Conclusion:** The shipping configuration is validated as robust against both thermal and physical hazards for this lane.")

def case_study_lyophilizer():
    st.info("**Context:** This plot verifies the performance of a lyophilization (freeze-drying) cycle. The OQ confirms the equipment can achieve and hold critical process parameters (shelf temperature and chamber pressure) as defined in the validated recipe. Holding a deep vacuum is essential for sublimation during primary drying.")
    time = np.arange(120); temp_set = np.concatenate([np.linspace(20, -40, 20), np.repeat(-40, 40), np.linspace(-40, 20, 60)]); temp_actual = temp_set + np.random.normal(0, 0.5, 120)
    pressure_set = np.concatenate([np.repeat(1000, 20), np.linspace(1000, 0.1, 20), np.repeat(0.1, 80)]); pressure_actual = pressure_set + np.random.normal(0, 0.02, 120)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time, y=temp_set, name='Temp Setpoint (°C)', line=dict(color=NEUTRAL_GREY, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=time, y=temp_actual, name='Temp Actual (°C)', line=dict(color=PRIMARY_COLOR)), secondary_y=False)
    fig.add_trace(go.Scatter(x=time, y=pressure_set, name='Pressure Setpoint (mbar)', line=dict(color=WARNING_AMBER, dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=time, y=pressure_actual, name='Pressure Actual (mbar)', line=dict(color=ERROR_RED)), secondary_y=True)
    fig.add_vrect(x0=0, x1=20, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text="Freezing")
    fig.add_vrect(x0=20, x1=60, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Primary Drying")
    fig.add_vrect(x0=60, x1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="Secondary Drying")
    fig.update_layout(title_text='<b>Lyophilizer OQ: Cycle Parameter Verification</b>', title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Time (minutes)"); fig.update_yaxes(title_text="Shelf Temperature (°C)", secondary_y=False); fig.update_yaxes(title_text="Chamber Pressure (mbar)", type="log", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The equipment successfully reached and held the deep vacuum (<0.2 mbar) required for sublimation during the critical Primary Drying phase, while shelf temperature remained precisely controlled. This OQ data confirms the lyophilizer is performing as specified and is ready for PQ runs with product.")

def case_study_nanoliter_dispenser():
    st.info("**Context:** This interactive analysis validates the precision and accuracy of a non-contact, acoustic liquid dispenser. Using a fluorescent dye, we measure the dispensed volume across the operating range to ensure it meets tight tolerances.")
    target_volume = st.select_slider("Select Target Dispense Volume (nL) to Analyze:", options=[2.5, 5.0, 10.0, 25.0], value=5.0)
    specs = {2.5: (0.05, 0.125), 5.0: (0.05, 0.25), 10.0: (0.07, 0.3), 25.0: (0.15, 0.75)}
    std_dev, acc_limit = specs[target_volume]
    data = np.random.normal(loc=target_volume, scale=std_dev, size=100)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, name='Distribution', marker_color=PRIMARY_COLOR, xbins=dict(size=std_dev/2)))
        fig.add_trace(go.Box(y=data, name='Summary', marker_color=WARNING_AMBER))
        fig.add_vline(x=target_volume, line_dash="dash", annotation_text="Target")
        fig.add_vrect(x0=target_volume - acc_limit, x1=target_volume + acc_limit, fillcolor=SUCCESS_GREEN, opacity=0.1, line_width=0, annotation_text="Spec Limit")
        fig.update_layout(title=f"<b>Dispense Distribution for {target_volume} nL Target</b>", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mean_val = np.mean(data); cv_val = (np.std(data) / mean_val) * 100; accuracy_val = ((mean_val - target_volume) / target_volume) * 100
        st.metric("Mean Volume (nL)", f"{mean_val:.3f}")
        st.metric("Precision (CV%)", f"{cv_val:.2f}%")
        st.metric("Accuracy", f"{accuracy_val:+.2f}%")
    st.success("**Actionable Insight:** The distribution plot confirms a normal distribution with no outliers. The statistical summary proves that for the selected target volume, the dispensed population is well within the required accuracy and precision limits. This level of granular data provides extremely high confidence in the dispenser's performance.")

def case_study_biochip_assembly():
    st.info("**Context:** For a full production line, Overall Equipment Effectiveness (OEE) is a critical PQ metric. It measures the combined impact of Availability (uptime), Performance (speed), and Quality (yield). The target for a validated line is often >85%.")
    fig = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "relative", "total"], x=["Theoretical Max Capacity", "Availability Losses (Downtime)", "Performance Losses (Speed)", "Quality Losses (Scrap)", "<b>Final OEE Output</b>"], text=["100%", "-8%", "-5%", "-2%", "85%"], y=[100, -8, -5, -2, 85], connector={"line":{"color":"rgb(63, 63, 63)"}}, totals={"marker":{"color":SUCCESS_GREEN}}, increasing={"marker":{"color":SUCCESS_GREEN}}, decreasing={"marker":{"color":ERROR_RED}}))
    fig.update_traces(textfont_size=14); fig.update_layout(title="<b>Biochip Assembly Line PQ: OEE Calculation</b>", yaxis_title="Effectiveness (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The assembly line achieved an OEE of 85%, meeting the acceptance criterion. The waterfall analysis clearly shows that **Availability Losses** (unplanned downtime) are the biggest detractor from performance. This provides a data-driven focus for the first Kaizen event: root cause analysis of the top 3 downtime reasons.")

def case_study_vision_system():
    st.info("**Context:** A confusion matrix is a key validation artifact for any AI/ML-based vision system. It challenges the system with a large set of known good and bad parts to quantify its real-world performance.")
    cm_data = np.array([[998, 2], [5, 95]])
    accuracy = (cm_data[0,0] + cm_data[1,1]) / np.sum(cm_data) * 100
    sensitivity = cm_data[1,1] / (cm_data[1,1] + cm_data[1,0]) * 100
    specificity = cm_data[0,0] / (cm_data[0,0] + cm_data[0,1]) * 100
    cm_percent = np.array([[specificity/100, (1-specificity/100)], [(1-sensitivity/100), sensitivity/100]])
    fig = px.imshow(cm_percent, text_auto=True, color_continuous_scale='Greens', labels=dict(x="Predicted Class", y="Actual Class", color="Rate"), x=['Good', 'Defect'], y=['Good', 'Defect'], title="<b>Vision System PQ: Normalized Confusion Matrix</b>")
    fig.update_traces(texttemplate="%{z:.2%}<br>(%{customdata})", customdata=cm_data)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Actionable Insight:** The system's **Sensitivity is {sensitivity:.1f}%** (it correctly identifies 95% of real defects), which meets our primary goal of preventing escapes to the customer. The 2 False Positives represent a direct cost in unnecessary scrap, while the 5 False Negatives are the most critical to investigate, as they represent potential escapes. **Action:** A review of these 5 parts is required to determine if the AI model needs retraining on a new defect type.")

def case_study_electrostatic_control():
    st.info("**Context:** For plastic biochips, uncontrolled electrostatic discharge (ESD) can damage sensitive electronics or cause latent failures. This study compares two mitigation strategies against an uncontrolled baseline.")
    df = pd.DataFrame({'Stage': ["Cassette Unmolding", "Biochip Placement", "Lid Taping", "Final Packaging"], 'Uncontrolled (V)': [1850, 2200, 2550, 1900], 'Anti-Static Coating (V)': [450, 520, 600, 480], 'Ionizer System (V)': [85, 45, 60, 50]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Uncontrolled (V)'], y=df['Stage'], name='Uncontrolled', mode='markers', marker=dict(color=ERROR_RED, size=15)))
    fig.add_trace(go.Scatter(x=df['Anti-Static Coating (V)'], y=df['Stage'], name='Anti-Static Coating', mode='markers', marker=dict(color=WARNING_AMBER, size=15)))
    fig.add_trace(go.Scatter(x=df['Ionizer System (V)'], y=df['Stage'], name='With Ionizer', mode='markers', marker=dict(color=SUCCESS_GREEN, size=15)))
    for i, row in df.iterrows():
        fig.add_shape(type="line", x0=row['Ionizer System (V)'], y0=row['Stage'], x1=row['Uncontrolled (V)'], y1=row['Stage'], line=dict(color=NEUTRAL_GREY, width=2))
    fig.add_vline(x=100, line_dash="dash", annotation_text="Acceptance Limit (<100V)")
    fig.update_layout(title_text="<b>OQ: Comparison of ESD Control Methods</b>", title_x=0.5, xaxis_title="Surface Voltage (V)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** This comparative study definitively proves the Ionizer System is the superior control method, reducing surface voltage by over 95% and staying well below the <100V damage threshold. The Anti-Static Coating is insufficient. **Decision:** The Ionizer system is a required, critical utility for this production line.")

def case_study_taping_soldering():
    st.info("**Context:** For thermal processes like heat staking or ultrasonic soldering, the OQ must prove that critical parameters are precisely and uniformly controlled across the entire operational surface to ensure a consistent seal.")
    data = np.random.normal(250.5, 0.5, (5, 5)); data[2,2] = 258.1 # Simulate a hot spot
    fig = px.imshow(data, text_auto=".1f", color_continuous_scale='Reds', aspect="auto", title="<b>OQ: Thermal Uniformity of Cassette Weld Horn (°C)</b>", labels=dict(x="Weld Point (X-axis)", y="Weld Point (Y-axis)", color="Temp (°C)"))
    fig.add_annotation(x=2, y=2, text="<b>HOT SPOT!</b>", showarrow=True, arrowhead=1, ax=0, ay=-40, font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The thermal mapping reveals a critical hot spot (+8.1°C over setpoint) in the center of the weld horn, which is outside our specification of ±5°C. This could cause material degradation and compromise the cassette seal. **Action:** A work order will be issued for the maintenance team to inspect the heating element and thermocouple in zone Y=2, X=2, and to verify the PID controller tuning for that zone before proceeding to PQ.")

def case_study_levey_jennings():
briefing_card = f"""
    <div style="border: 1px solid {BACKGROUND_GREY}; border-radius: 5px; padding: 15px; margin-bottom: 20px; background-color: #FFFFFF;">
        <p style="margin-bottom: 10px;">
            <strong style="color: {PRIMARY_COLOR};">Context:</strong> A critical QC Reagent Control Lot is run daily on a diagnostic analyzer to ensure the measurement system is stable. The target mean is 100 mg/dL with a known standard deviation (SD) of 2 mg/dL.
        </p>
        <p style="margin-bottom: 10px;">
            <strong style="color: {DARK_GREY};">Purpose:</strong> To visualize the precision and accuracy of a test system over time by plotting control values against their acceptable limits (mean ±1, 2, and 3 SD).
        </p>
        <p style="margin-bottom: 0;">
            <strong style="color: {SUCCESS_GREEN};">Reason for Use:</strong> This chart is the industry standard for lab QC. Its well-defined zones, combined with Westgard rules, provide a powerful, standardized system for detecting both random error (e.g., a 1-3s violation) and systematic error (e.g., a 2-2s violation) with high confidence.
        </p>
    </div>
    """
    st.markdown(briefing_card, unsafe_allow_html=True)
    
    # Generate data with a specific violation
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-05-01", periods=30)
    data = rng.normal(loc=100, scale=2, size=30)
    data[20] = 106.5 # 1-3s violation (random error)
    data[25:27] = [104.5, 104.8] # 2-2s violation (systematic error/bias)
    df = pd.DataFrame({'Date': dates, 'Value': data})

    mean = 100; sd = 2
    limits = {'+3sd': mean + 3*sd, '+2sd': mean + 2*sd, '+1sd': mean + 1*sd, 
              '-1sd': mean - 1*sd, '-2sd': mean - 2*sd, '-3sd': mean - 3*sd}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', name='QC Value', line=dict(color=PRIMARY_COLOR)))
    
    # Add mean and SD lines
    for limit_name, limit_val in limits.items():
        fig.add_hline(y=limit_val, line_dash='dash', line_color=DARK_GREY, 
                      annotation_text=limit_name, annotation_position="bottom right")
    fig.add_hline(y=mean, line_dash='solid', line_color=SUCCESS_GREEN, annotation_text="Mean")

    # Detect and highlight Westgard rule violations
    violations = []
    # 1-3s Rule
    violation_3s = df[df['Value'] > limits['+3sd']]
    if not violation_3s.empty:
        idx = violation_3s.index[0]
        fig.add_annotation(x=df['Date'][idx], y=df['Value'][idx], text="<b>1-3s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
        violations.append("1-3s rule violation detected (potential random error).")
        
    # 2-2s Rule
    for i in range(len(df) - 1):
        if (df['Value'][i] > limits['+2sd'] and df['Value'][i+1] > limits['+2sd']):
            fig.add_annotation(x=df['Date'][i+1], y=df['Value'][i+1], text="<b>2-2s Violation</b>", showarrow=True, arrowhead=2, ax=0, ay=40, bgcolor=WARNING_AMBER)
            violations.append("2-2s rule violation detected (potential systematic bias).")
            break
            
    fig.update_layout(title="<b>Levey-Jennings Chart for Diagnostic Control Monitoring</b>", yaxis_title="Control Value (mg/dL)", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    st.plotly_chart(fig, use_container_width=True)

    st.error(f"""
    **Actionable Insight:** The automated analysis detected **{len(violations)}** critical rule violation(s).
    - The **1-3s violation** on {violation_3s['Date'].dt.date.iloc[0]} required immediate rejection of the run and an investigation into random error (e.g., bubble in reagent, sample mix-up).
    - The **2-2s violation** indicates a systematic shift. A work order was issued to recalibrate the instrument and review the current reagent lot for degradation.
    """)

def case_study_ewma_chart():
    st.markdown("""
    **Context:** We are monitoring the critical impurity level (%) in a biologic drug substance produced by a chromatography column that is known to degrade slowly over many cycles. A small, gradual increase in impurity is a sign that the column needs repacking.
    **Purpose:** To detect small, persistent shifts in the process mean that might be missed by standard Shewhart charts.
    **Reason for Use:** The EWMA chart incorporates "memory" of past data points by giving them exponentially decreasing weights. This makes it far more sensitive to small drifts than an I-MR chart, which only considers the last one or two data points. It enables proactive intervention *before* a specification limit is breached.
    """)

    # Generate data with a small, persistent shift
    rng = np.random.default_rng(10)
    data = rng.normal(loc=0.5, scale=0.05, size=40)
    data[20:] += 0.06 # A small shift of just over 1 sigma
    df = pd.DataFrame({'Batch': range(1, 41), 'Impurity': data})
    
    # EWMA Calculation
    lam = 0.2 # Lambda (weighting factor), smaller values give more weight to past data
    mean = df['Impurity'].mean()
    sd = df['Impurity'].std()
    df['EWMA'] = df['Impurity'].ewm(span=(2/lam)-1, adjust=False).mean()
    
    # EWMA Control Limits
    n = df.index + 1
    ucl = mean + 3 * sd * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * n)))
    lcl = mean - 3 * sd * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * n)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Batch'], y=df['Impurity'], mode='markers', name='Individual Batch', marker_color=NEUTRAL_GREY))
    fig.add_trace(go.Scatter(x=df['Batch'], y=df['EWMA'], mode='lines', name='EWMA', line=dict(color=PRIMARY_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=df['Batch'], y=ucl, mode='lines', name='UCL', line=dict(color=ERROR_RED, dash='dash')))
    fig.add_trace(go.Scatter(x=df['Batch'], y=lcl, mode='lines', name='LCL', line=dict(color=ERROR_RED, dash='dash')))
    fig.add_hline(y=mean, line_dash='solid', line_color=SUCCESS_GREEN, annotation_text="Target Mean")

    # Find violation
    violation = df[df['EWMA'] > ucl].first_valid_index()
    if violation:
        fig.add_annotation(x=df['Batch'][violation], y=df['EWMA'][violation], text="<b>EWMA Signal</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))

    fig.update_layout(title="<b>EWMA Chart for Chromatography Column Degradation</b>", xaxis_title="Batch Number", yaxis_title="Impurity Level (%)", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **Actionable Insight:** The EWMA chart signaled an out-of-control condition at **Batch #{violation}**, much earlier than a standard chart would have. This early warning, triggered by the sustained small increase in impurity, allows for proactive maintenance. 
    **Decision:** A work order will be issued to repack the chromatography column at the end of the current campaign, preventing the production of any out-of-specification material and avoiding a costly deviation investigation.
    """)

def case_study_cusum_chart():
    st.markdown("""
    **Context:** We are monitoring the fill volume (in mL) of a high-speed aseptic filling line. A small, sudden clog in a filling nozzle could cause a persistent underfill that must be detected immediately to prevent an entire lot from being compromised.
    **Purpose:** To rapidly detect small, sustained shifts in the process average. The chart accumulates deviations from the target, making small shifts visually apparent very quickly.
    **Reason for Use:** While EWMA is also good for small shifts, CUSUM is often faster at detecting the *start* of the shift. This is critical in high-volume, high-cost processes like aseptic filling where minimizing the number of defective units is paramount.
    """)
    
    # Generate data with a sudden, small shift
    rng = np.random.default_rng(50)
    target = 10.0; sd = 0.05
    data = rng.normal(target, sd, 50)
    data[25:] -= 0.04 # A small, sub-sigma shift
    df = pd.DataFrame({'Sample': range(1, 51), 'Fill Volume': data})
    
    # Tabular CUSUM Calculation
    k = 0.5 * sd # "Allowance" or "slack"
    h = 5 * sd # Decision interval
    df['SH+'] = 0.0; df['SH-'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'SH+'] = max(0, df.loc[i-1, 'SH+'] + df.loc[i, 'Fill Volume'] - target - k)
        df.loc[i, 'SH-'] = max(0, df.loc[i-1, 'SH-'] + target - df.loc[i, 'Fill Volume'] - k)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Sample'], y=df['SH+'], name='CUSUM High (SH+)', mode='lines', line=dict(color=ERROR_RED, width=3)))
    fig.add_trace(go.Scatter(x=df['Sample'], y=df['SH-'], name='CUSUM Low (SH-)', mode='lines', line=dict(color=PRIMARY_COLOR, width=3)))
    fig.add_hline(y=h, line_dash='dash', line_color='black', annotation_text="Decision Limit (H)", annotation_position="top right")
    
    # Find violation
    violation = df[df['SH-'] > h].first_valid_index()
    if violation:
        fig.add_annotation(x=df['Sample'][violation], y=df['SH-'][violation], text="<b>CUSUM Signal!</b><br>Process Mean<br>has Shifted Low", showarrow=True, arrowhead=2, ax=20, ay=-60, bgcolor=PRIMARY_COLOR, font=dict(color='white'))

    fig.update_layout(title="<b>CUSUM Chart for Aseptic Fill Volume Monitoring</b>", xaxis_title="Sample Number", yaxis_title="Cumulative Sum", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"""
    **Actionable Insight:** The CUSUM chart rapidly detected a persistent downward shift in the process mean, signaling an alarm at **Sample #{violation}**. The SH- (low-side) plot crossed the decision limit (H), confirming the underfill condition.
    **Decision:** The filling line was immediately halted. An investigation traced the issue to a partial blockage in filling head #4. By detecting the issue quickly, only a small number of units required quarantine, saving the majority of the batch.
    """)

def case_study_advanced_imr():
    st.markdown("""
    **Context:** Continuous monitoring of the differential pressure (DP) between a Grade A aseptic processing area and the surrounding Grade B area. Maintaining a positive pressure gradient is a critical control parameter to prevent ingress of contaminants.
    **Purpose:** To monitor the stability and variability of a critical environmental parameter over time, detecting any special cause variation that could indicate a breach of containment.
    **Reason for Use:** The I-MR chart is the ideal tool for this application because we are dealing with individual measurements taken at regular intervals. The I-chart tracks the DP level itself, while the MR-chart tracks its short-term variability. A sudden change in variability (a spike on the MR chart) is often the first sign of a problem, even before the I-chart goes out of control.
    """)
    rng = np.random.default_rng(25)
    data = rng.normal(15, 0.5, 48) # Target: 15 Pa
    data[30] = 10 # Sudden drop - e.g., door opened
    df = pd.DataFrame({'Pressure (Pa)': data})
    df['MR'] = df['Pressure (Pa)'].diff().abs()
    
    I_CL = df['Pressure (Pa)'].mean()
    MR_CL = df['MR'].mean()
    I_UCL = I_CL + 2.66 * MR_CL; I_LCL = I_CL - 2.66 * MR_CL
    MR_UCL = 3.267 * MR_CL
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart - Differential Pressure</b>", "<b>Moving Range (MR) Chart - Variability</b>"))
    
    # I-Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['Pressure (Pa)'], name='Pressure (Pa)', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1)
    fig.add_hline(y=I_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="CL")
    fig.add_hline(y=I_UCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL")
    fig.add_hline(y=I_LCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL")

    # MR-Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['MR'], name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1)
    fig.add_hline(y=MR_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="CL")
    fig.add_hline(y=MR_UCL, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL")

    # Find violations
    i_violation = df[df['Pressure (Pa)'] < I_LCL].first_valid_index()
    mr_violation = df[df['MR'] > MR_UCL].first_valid_index()
    if i_violation: fig.add_annotation(x=i_violation, y=df['Pressure (Pa)'][i_violation], text="<b>Loss of Pressure!</b>", row=1, col=1, showarrow=True, bgcolor=ERROR_RED, font=dict(color='white'))
    if mr_violation: fig.add_annotation(x=mr_violation, y=df['MR'][mr_violation], text="<b>Variability Spike!</b>", row=2, col=1, showarrow=True, bgcolor=WARNING_AMBER)

    fig.update_layout(height=450, showlegend=False, title_text="<b>I-MR Chart for Aseptic Area Differential Pressure</b>", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    fig.update_xaxes(title_text="Time (Hourly Reading)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.error("""
    **Actionable Insight:** At hour 30, the MR chart shows a massive spike in variability, followed immediately by an out-of-control signal on the I-chart as the pressure dropped below the lower control limit. This dual alarm provides conclusive evidence of a special cause event.
    **Decision:** An immediate investigation was triggered. The Building Management System (BMS) alarm logs and security access records were reviewed, correlating the event to a service door being propped open at hour 30 for unauthorized material transfer. Corrective action involves retraining all staff on cleanroom gowning and material transfer procedures.
    """)

def case_study_zone_chart():
    st.markdown("""
    **Context:** Monitoring the seal strength (in Newtons) of a medical device pouch on a high-speed, validated heat-sealing line. The process is mature and highly capable, so we need a sensitive tool to detect early signs of drift.
    **Purpose:** To analyze process stability with greater sensitivity than a standard chart by dividing the area between control limits into sigma-based zones (A, B, C).
    **Reason for Use:** For a mature, well-understood process, simply staying within +/- 3 SD is not enough. The Zone Chart allows for the application of pattern-based rules (e.g., Westgard or Nelson rules) to detect unnatural patterns *within* the control limits, such as runs, trends, or stratification. This provides a much earlier warning of potential issues.
    """)
    rng = np.random.default_rng(33)
    mean = 20; sd = 0.5
    data = rng.normal(mean, sd, 25)
    data[15:] -= 0.4 # A small systematic shift, causing a run
    df = pd.DataFrame({'Sample': range(1, 26), 'Seal Strength (N)': data})

    fig = go.Figure()
    # Define zones
    zones = {'UCL (3s)': [mean + 2*sd, mean + 3*sd], 'Zone A': [mean + 1*sd, mean + 2*sd], 
             'Zone B': [mean, mean + 1*sd], 'Zone C (Center)': [mean - 1*sd, mean], 
             'Zone B-': [mean - 2*sd, mean - 1*sd], 'Zone A-': [mean - 3*sd, mean - 2*sd]}
    colors = {'UCL (3s)': 'rgba(211, 47, 47, 0.2)', 'Zone A': 'rgba(255, 193, 7, 0.2)',
              'Zone B': 'rgba(76, 175, 80, 0.2)', 'Zone C (Center)': 'rgba(76, 175, 80, 0.2)',
              'Zone B-': 'rgba(255, 193, 7, 0.2)', 'Zone A-': 'rgba(211, 47, 47, 0.2)'}

    for name, y_range in zones.items():
        fig.add_hrect(y0=y_range[0], y1=y_range[1], line_width=0, fillcolor=colors[name], 
                      annotation_text=name.split(' ')[0], annotation_position="top left", layer="below")
    
    fig.add_trace(go.Scatter(x=df['Sample'], y=df['Seal Strength (N)'], mode='lines+markers', name='Seal Strength', line=dict(color=PRIMARY_COLOR)))
    fig.add_hline(y=mean, line_color='black')
    
    # Detect 8-point run rule (8 consecutive points on one side of the mean)
    for i in range(len(df) - 7):
        subset = df['Seal Strength (N)'][i:i+8]
        if all(subset < mean):
            fig.add_annotation(x=i+4, y=mean - 0.7, text="<b>Westgard Rule Violation!</b><br>8 consecutive points below mean.", showarrow=False, bgcolor=WARNING_AMBER, borderpad=4)
            break
            
    fig.update_layout(title="<b>Zone Chart for Heat Sealer Performance</b>", xaxis_title="Sample Number", yaxis_title="Seal Strength (N)", title_x=0.5, plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **Actionable Insight:** Although no single point is outside the +/- 3 sigma control limits, the Zone Chart's automated rule analysis detected a run of 8 consecutive points below the center line. This is a statistically significant, non-random pattern indicating a systematic process shift.
    **Decision:** This early warning triggers a non-emergency investigation. The maintenance team will check the calibration of the sealing platen's thermocouple and inspect for heater element degradation during the next scheduled planned maintenance, addressing the drift before it can cause a failure.
    """)

def case_study_hotelling_t2():
    st.markdown("""
    **Context:** We are validating a new automated buffer preparation skid. Two critical, and inherently correlated, quality attributes are the final **Salt Concentration** and the **pH**. An error in weighing the primary salt component will affect both variables.
    **Purpose:** To simultaneously monitor the stability of two or more correlated process variables. It condenses the multivariate information into a single, easy-to-interpret value (T²).
    **Reason for Use:** This chart is vastly superior to running two separate I-MR charts for Concentration and pH. A small, simultaneous shift in both variables might not trigger an alarm on either individual chart, but the *combined state* of the system is out of control. The T² chart is specifically designed to detect these joint-shifts, preventing the release of a subtly incorrect but non-compliant buffer.
    """)
    
    # 1. Generate correlated multivariate data
    rng = np.random.default_rng(123)
    mean_vector = [150, 7.4] # Target: 150 mM Concentration, 7.4 pH
    covariance_matrix = [[1.0, 0.6], [0.6, 0.01]] # Positive correlation
    in_control_data = rng.multivariate_normal(mean_vector, covariance_matrix, size=30)
    
    # Introduce a joint shift that would be hard to catch on individual charts
    outlier_point = [151.8, 7.34] # Conc is high (+1.8s), pH is low (-1.6s)
    data = np.vstack([in_control_data[:24], outlier_point, in_control_data[24:]])
    df = pd.DataFrame(data, columns=['Concentration (mM)', 'pH'])
    
    # 2. Calculate T-squared values (for demonstration, we simulate the spike)
    # In a real scenario, this involves matrix algebra. Here we simulate the result.
    t_squared_values = rng.chisquare(2, size=len(df)) * 0.8
    ucl = 9.21 # Upper Control Limit from F-distribution for p=2, n=30, alpha=0.01
    t_squared_values[24] = 15.5 # Manually insert the spike for the outlier point

    # 3. Create the plots
    col1, col2 = st.columns([3, 2])
    with col1:
        # T-Squared Chart
        fig_t2 = go.Figure()
        fig_t2.add_trace(go.Scatter(x=df.index, y=t_squared_values, mode='lines+markers', name='T² Value', line=dict(color=PRIMARY_COLOR)))
        fig_t2.add_hline(y=ucl, line_dash='dash', line_color=ERROR_RED, annotation_text="UCL")
        
        # Highlight violation
        fig_t2.add_annotation(x=24, y=15.5, text="<b>Multivariate Anomaly!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor=ERROR_RED, font=dict(color='white'))
        fig_t2.update_layout(title="<b>Hotelling's T² Chart</b>", xaxis_title="Buffer Batch Number", yaxis_title="T² Statistic", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
        st.plotly_chart(fig_t2, use_container_width=True)

    with col2:
        # Scatter plot to visually explain the "why"
        fig_scatter = px.scatter(df, x='Concentration (mM)', y='pH', title="<b>Process Variable Correlation</b>")
        fig_scatter.add_trace(go.Scatter(x=[outlier_point[0]], y=[outlier_point[1]], mode='markers', marker=dict(color=ERROR_RED, size=12, symbol='x'), name='Anomaly'))
        fig_scatter.update_layout(plot_bgcolor=BACKGROUND_GREY)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.error("""
    **Actionable Insight:** The T² chart successfully identified a multivariate out-of-control condition at **Batch #25**. 
    The crucial insight comes from the scatter plot: while the anomalous point's Concentration was only slightly high and its pH only slightly low, its position *relative to the normal process correlation* was a significant deviation.
    **This is a classic failure mode that two separate univariate charts would likely have missed.**
    
    **Decision:** An investigation was launched. The root cause was determined to be the use of an incorrect grade of a secondary buffer salt, which subtly altered both the final concentration and the solution's buffering capacity (pH). This confirms the power of multivariate monitoring for complex formulations.
    """)

# --- END: New Helper functions ---
def display_rpn_table(key: str) -> None:
    """
    Displays an interactive, professional-grade RPN table from a pFMEA.
    """
    st.subheader("Detailed Risk Register (pFMEA)", divider='blue')
    st.info("""
    **Purpose:** While the Risk Matrix provides a visual summary, this detailed RPN table is the core risk management tool. It quantifies risk using Severity, Occurrence, and **Detectability** (the likelihood of detecting a failure before it reaches the patient). Risks exceeding the threshold (e.g., RPN > 100) require mandatory mitigation.
    """)

    # Create detailed FMEA data
    fmea_data = {
        'ID': ['FMEA-01', 'FMEA-02', 'FMEA-03', 'FMEA-04'],
        'Failure Mode': ['Incorrect Titer', 'Contamination from CIP', 'Sensor Failure (pH)', 'Software Crash'],
        'Severity (S)': [9, 10, 7, 6],
        'Occurrence (O)': [3, 2, 5, 4],
        'Detectability (D)': [5, 4, 2, 3], # Higher number = Harder to detect
        'Required Mitigation Action': [
            'Implement PAT sensor for real-time monitoring.',
            'Increase final rinse duration and add TOC sampling.',
            'Define more frequent calibration/sensor health check in maintenance SOP.',
            'Accepted risk; covered by disaster recovery plan.'
        ],
        'Status': ['In Progress', 'Complete', 'Complete', 'Accepted']
    }
    df = pd.DataFrame(fmea_data)
    df['RPN'] = df['Severity (S)'] * df['Occurrence (O)'] * df['Detectability (D)']
    
    # Reorder columns for logical flow
    df = df[['ID', 'Failure Mode', 'Severity (S)', 'Occurrence (O)', 'Detectability (D)', 'RPN', 'Required Mitigation Action', 'Status']]

    def style_status_and_rpn(df_styled):
        # Color-code RPN values
        df_styled = df_styled.background_gradient(cmap='YlOrRd', subset=['RPN'], vmin=0, vmax=200)
        
        # Color-code Status
        def style_status(val):
            color_map = {'Complete': SUCCESS_GREEN, 'In Progress': WARNING_AMBER, 'Accepted': NEUTRAL_GREY}
            bg_color = color_map.get(val, 'white')
            font_color = 'white' if val in color_map else 'black'
            return f"background-color: {bg_color}; color: {font_color};"
        
        df_styled = df_styled.map(style_status, subset=['Status'])
        return df_styled

    st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Not Started", "In Progress", "Complete", "Accepted"],
                required=True,
            ),
            "RPN": st.column_config.ProgressColumn(
                "RPN",
                help="Risk Priority Number (S x O x D). Threshold > 100 requires action.",
                min_value=0,
                max_value=200,
            ),
        },
        key=f"rpn_editor_{key}"
    )

    st.success("""
    **Actionable Insight:** The RPN analysis has correctly prioritized 'Incorrect Titer' (RPN=135) as the highest risk due to its high severity and poor detectability. The mitigation to implement a PAT sensor is 'In Progress' and is being tracked as a key deliverable for Project Atlas.
    """)

def display_revalidation_planner(key: str) -> None:
    """
    NEW: An interactive planner for developing a risk-based revalidation strategy.
    This demonstrates strategic thinking beyond simple periodic reviews.
    """
    st.subheader("Interactive Revalidation Strategy Planner", divider='blue')
    st.info("""
    **Purpose:** This tool moves beyond a fixed periodic review schedule to a dynamic, risk-based revalidation strategy. By simulating the impact of potential future events (like major software patches or process drifts), we can proactively plan and budget for necessary revalidation efforts.
    """)
    
    review_data = {"System": ["Bioreactor C", "Purification A", "WFI System"], 
                   "Risk Level": ["High", "High", "High"],
                   "Base Reval Effort (Days)": [20, 15, 10]}
    df = pd.DataFrame(review_data)

    st.markdown("##### Select a Potential Future Change Event:")
    
    # --- FIX for Accessibility Warning ---
    # Provide a descriptive label for screen readers and hide it visually.
    change_event = st.selectbox(
        label="Select a potential future change event to forecast its revalidation impact.", 
        options=["No Significant Change", "Major Software Patch (OS Update)", "Minor Process Drift Detected (SPC Trend)", "New Raw Material Supplier"],
        label_visibility="collapsed"
    )

    # Define risk multipliers for different events
    multipliers = {
        "No Significant Change": {'High': 0.1, 'Medium': 0.05, 'Low': 0},
        "Major Software Patch (OS Update)": {'High': 0.8, 'Medium': 0.5, 'Low': 0.2},
        "Minor Process Drift Detected (SPC Trend)": {'High': 0.5, 'Medium': 0.2, 'Low': 0.1},
        "New Raw Material Supplier": {'High': 1.0, 'Medium': 0.7, 'Low': 0.3}
    }
    
    selected_multiplier = multipliers[change_event]
    
    df['Impact Multiplier'] = df['Risk Level'].map(selected_multiplier)
    df['Forecasted Revalidation Effort (Days)'] = df['Base Reval Effort (Days)'] * df['Impact Multiplier']
    
    total_effort = df['Forecasted Revalidation Effort (Days)'].sum()
    
    st.markdown("##### Forecasted Revalidation Workload")
    st.dataframe(df.round(1), use_container_width=True, hide_index=True)
    
    st.metric("Total Forecasted Revalidation Effort for this Scenario (Person-Days)", f"{total_effort:.1f}")
    
    st.success(f"""
    **Actionable Insight for '{change_event}':** This scenario forecasts **{total_effort:.1f} person-days** of revalidation work. This data can be used to proactively secure contractor budget or adjust project timelines *before* the change occurs, ensuring we remain in a state of control.
    """)
def run_rft_prediction_model(key: str) -> None:
    """
    NEW: An interactive model to predict the probability of a protocol succeeding on the first try.
    This demonstrates a proactive, data-driven approach to ensuring quality.
    """
    st.subheader("Predictive Right-First-Time (RFT) Modeler", divider='blue')
    st.info("""
    **Purpose:** This model uses historical data to predict the likelihood that a new validation protocol will be executed without deviations (i.e., "Right First Time"). It allows us to proactively de-risk complex validation activities *before* execution begins.
    """)

    # Simulate historical data and train a simple model
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({
        '# of Test Cases': rng.integers(10, 100, 20),
        'Vendor Doc Quality (%)': rng.integers(80, 101, 20),
        'Team Experience (Avg. Yrs)': rng.uniform(1, 5, 20)
    })
    y_train = (X_train['Vendor Doc Quality (%)'] > 90) & (X_train['Team Experience (Avg. Yrs)'] > 2.5) & (X_train['# of Test Cases'] < 50)
    model = LogisticRegression().fit(X_train, y_train)

    col1, col2, col3 = st.columns(3)
    with col1:
        test_cases = st.slider("Number of Test Cases", 10, 100, 65, key=f"rft_tc_{key}")
    with col2:
        vendor_quality = st.slider("Vendor Doc Quality (%)", 80, 100, 85, key=f"rft_vendor_{key}")
    with col3:
        team_exp = st.slider("Team Experience (Avg. Yrs)", 1.0, 5.0, 1.5, step=0.5, key=f"rft_exp_{key}")

    new_protocol_data = pd.DataFrame([[test_cases, vendor_quality, team_exp]], columns=X_train.columns)
    prediction_proba = model.predict_proba(new_protocol_data)[0][1] # Probability of 'True' (Success)
    
    color = SUCCESS_GREEN if prediction_proba > 0.7 else WARNING_AMBER if prediction_proba > 0.4 else ERROR_RED
    
    st.markdown(f"""
    <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">
        <h4 style="color: white;">Predicted RFT Probability: {prediction_proba:.1%}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if prediction_proba < 0.7:
        st.error("""
        **Automated Alert: High Risk of Protocol Deviation**
        
        **Actionable Insight:** The model predicts a high probability of failure for this protocol execution. The low team experience is the likely root cause.
        **Recommendation:** Assign a senior engineer (e.g., S. Smith from the Skills Matrix) to mentor the junior team member executing this protocol. Conduct a dry run of the most complex test cases to mitigate risk.
        """)
    else:
        st.success("**Insight:** The model predicts a high likelihood of a successful, deviation-free execution.")
        
def display_vendor_dashboard(key: str) -> None:
    """
    NEW: Displays a dynamic Vendor Risk & Opportunity Dashboard.
    This demonstrates proactive supply chain management and strategic partnership.
    """
    st.subheader("Vendor Risk & Opportunity Dashboard", divider='blue')
    st.info("""
    **Purpose:** An effective manager proactively manages the supply chain. This dashboard moves beyond simple performance scoring to assess future risks and identify strategic partnership opportunities, ensuring project success and long-term value.
    """)
    
    vendor_data = {
        'Vendor': ['Vendor A (Automation)', 'Vendor B (Components)', 'Vendor C (Software)'],
        'Overall Score': [91, 85, 94],
        'Anticipated Risk': ['Low', 'Medium', 'Low'],
        'Risk Driver': ['None', 'Sole Source for Key Part', 'None'],
        'Strategic Opportunity': ['Partner for next-gen platform', 'Second Source Qualification', 'Integrate for data analytics']
    }
    df = pd.DataFrame(vendor_data)

    def style_risk(val: str) -> str:
        color = {'Low': SUCCESS_GREEN, 'Medium': WARNING_AMBER, 'High': ERROR_RED}.get(val, 'white')
        font_color = 'white' if val in ['Low', 'Medium', 'High'] else 'black'
        return f"background-color: {color}; color: {font_color};"

    styled_df = df.style.map(style_risk, subset=['Anticipated Risk'])\
                       .background_gradient(cmap='Greens', subset=['Overall Score'], vmin=70, vmax=100)\
                       .set_properties(**{'text-align': 'center'})
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.success("""
    **Actionable Insight:** Vendor B, despite an acceptable performance score, is flagged as a 'Medium' risk due to being a sole-source supplier. 
    **Recommendation:** Initiate a project to qualify a second source for their key component to mitigate supply chain risk for future projects. Vendor C's high score and software expertise make them a prime candidate for a strategic data analytics partnership.
    """)

def display_team_skill_matrix(key: str) -> None:
    """
    Displays a team skill matrix to demonstrate strategic talent management.
    This directly addresses the leadership and mentorship requirements of the role.
    """
    st.subheader("Team Skill Matrix & Development Plan", divider='blue')
    st.info("""
    **Purpose:** A skills matrix is a critical tool for a manager to visualize team capabilities, identify skill gaps, and plan targeted development. 
    It ensures that resource allocation for new projects is backed by data and that individual growth aligns with the department's strategic needs.
    *Proficiency Scale: 1 (Novice) to 5 (SME)*
    """)
    
    skill_data = {
        'Team Member': ['J. Doe (Lead)', 'S. Smith (Eng. II)', 'A. Wong (Spec. I)', 'B. Zeller (Eng. I)'],
        'CSV & Part 11': [5, 3, 2, 2],
        'Cleaning Validation': [4, 4, 3, 1],
        'Statistics (Cpk/DOE)': [5, 3, 2, 3],
        'Project Management': [5, 2, 1, 1],
        'Development Goal': [
            'Mentor team on advanced stats', 
            'Lead CSV for Project Beacon', 
            'Cross-train on Cleaning Val', 
            'Achieve PMP certification'
        ]
    }
    df = pd.DataFrame(skill_data)
    
    skill_cols = df.columns.drop(['Team Member', 'Development Goal'])
    styled_df = df.style.background_gradient(
        cmap='Greens', subset=skill_cols, vmin=1, vmax=5
    ).set_properties(**{'text-align': 'center'}, subset=skill_cols)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=215)
    
    st.success("""
    **Actionable Insight:** The matrix identifies a potential bottleneck in Project Management skills among junior staff. 
    Based on B. Zeller's development goal, the budget for PMP certification training will be approved. 
    Furthermore, A. Wong will be assigned to the 'Atlas' Cleaning Validation PQ to gain hands-on experience, supported by S. Smith.
    """)
    
def plot_predictive_compliance_risk() -> go.Figure:
    """Creates a gauge chart for a predictive compliance risk score."""
    risk_score = (1 * 50) + (3 * 10) + (0 * 5) # (overdue_reviews * w1 + open_capas * w2 + high_utilization * w3)
    normalized_score = min(risk_score / 150, 1) * 100

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = normalized_score,
        title = {'text': "<b>Predictive Compliance Risk Score</b>"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': PRIMARY_COLOR},
            'steps' : [
                {'range': [0, 50], 'color': SUCCESS_GREEN},
                {'range': [50, 80], 'color': WARNING_AMBER},
                {'range': [80, 100], 'color': ERROR_RED}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20))
    return fig
    
def create_process_map_diagram() -> go.Figure:
    """
    Digitally renders the Equipment Validation Scheme as a professional-grade process map
    using clean, content-aware sized boxes with enhanced font sizes.
    """
    fig = go.Figure()
    
    # Define nodes with manually tuned w/h properties for a balanced, professional layout
    nodes = {
        'sys_desc': {
            'pos': [2.5, 9.2], 'w': 2.3, 'h': 0.8,
            'text': '<b>System Description</b><br> • Specifications<br> • Functional/Performance Requirements', 
            'color': DARK_GREY, 'shape': 'terminator'},
        'fat_sat': {
            'pos': [2.5, 7.5], 'w': 2.3, 'h': 1.0,
            'text': '<b>FAT/SAT</b><br><i>Instrument/Manufacturer Related</i><br> • Instrument Components P&ID, electrical<br> • Instrument Performance, CV, repeatability', 
            'color': PRIMARY_COLOR, 'shape': 'terminator'},
        'val_activities': {
            'pos': [2.5, 4.2], 'w': 2.5, 'h': 2.8,
            'text': '''<b>Validation Activities</b><br><br><u>Installation Qualification (IQ)</u><br> • Meet manufacturer’s specifications<br> • Manuals, maintenance plans<br><u>Operational Qualification (OQ)</u><br> • Test accuracy, precision and repeatability<br> • Confirm instrument resolution<br><u>Performance Qualification (PQ)</u><br><i>Production scale testing (define batch numbers, procure<br>material, align w/R&D)</i>''', 
            'color': NEUTRAL_GREY, 'shape': 'terminator'},
        'sample_size': {
            'pos': [2.5, 0.8], 'w': 2.3, 'h': 0.6,
            'text': '<b>Sample Size</b><br>Determined by “Binomial Power Analysis”<br>or AQL table', 
            'color': '#D35400', 'shape': 'process'},
        'acceptance': {
            'pos': [6.5, 5.8], 'w': 1.8, 'h': 0.9,
            'text': '<b>Acceptance Criteria</b><br> • IQ Acceptance Criteria<br> • OQ Acceptance Criteria<br> • PQ Acceptance Criteria', 
            'color': PRIMARY_COLOR, 'shape': 'process'},
        'docs': {
            'pos': [8.5, 8.0], 'w': 2.5, 'h': 1.1,
            'text': '<b>Documentation</b><br> • IQ, OQ, PQ Documentation<br> • Master Validation Plan (MVP)<br> • Design History File (DHF)<br> • Device Master Record (DMR)', 
            'color': PRIMARY_COLOR, 'shape': 'terminator'},
        'onboarding': {
            'pos': [6.5, 3.8], 'w': 1.8, 'h': 0.8,
            'text': '<b>Equipment Onboarding</b><br> • Defining calibration<br>   points/frequency<br> • Maintenance schedule', 
            'color': PRIMARY_COLOR, 'shape': 'process'},
        'sops': {
            'pos': [9.5, 3.8], 'w': 1.5, 'h': 0.8,
            'text': '<b>Protocols & SOPs</b><br> • Personnel Training<br> • <span style="color:red">SOPs/Protocols</span>', 
            'color': PRIMARY_COLOR, 'shape': 'terminator'},
        'change_control': {
            'pos': [6.5, 1.8], 'w': 1.8, 'h': 1.1,
            'text': '<b>Change Control: ECOs/DCOs</b><br> • Change Request<br> • Impact Assessment<br> • Revalidation Requirement<br> • Engineering/Docs', 
            'color': PRIMARY_COLOR, 'shape': 'process'},
    }

    # Add Shapes and Annotations for each node
    for key, node in nodes.items():
        x_c, y_c, w, h = node['pos'][0], node['pos'][1], node['w'], node['h']
        align = 'left' if key == 'val_activities' else 'center'
        shape_type = node.get('shape')
        
        # --- ENHANCEMENT: Increased Font Size for better readability ---
        font_size = 15
        if key == 'val_activities':
            font_size = 15.5 # Slightly larger for the main box
        
        path = ""
        # Define SVG paths for each shape type to ensure correct rendering
        if shape_type == 'process': # Standard Rectangle
            path = f"M {x_c-w},{y_c-h} L {x_c+w},{y_c-h} L {x_c+w},{y_c+h} L {x_c-w},{y_c+h} Z"
        elif shape_type == 'terminator': # Rounded Rectangle using SVG Arc paths
            r = 0.4  # Radius of corners
            path = (f"M {x_c-w+r},{y_c-h} L {x_c+w-r},{y_c-h} A {r},{r} 0 0 1 {x_c+w},{y_c-h+r} "
                    f"L {x_c+w},{y_c+h-r} A {r},{r} 0 0 1 {x_c+w-r},{y_c+h} "
                    f"L {x_c-w+r},{y_c+h} A {r},{r} 0 0 1 {x_c-w},{y_c+h-r} "
                    f"L {x_c-w},{y_c-h+r} A {r},{r} 0 0 1 {x_c-w+r},{y_c-h} Z")
        
        fig.add_shape(type="path", path=path, line=dict(color="Black"), fillcolor=node['color'], opacity=0.95, layer="below")
        fig.add_annotation(
            x=x_c, y=y_c, text=node['text'], showarrow=False, align=align, 
            font=dict(color='white' if node['color'] not in [NEUTRAL_GREY] else 'black', size=font_size)
        )

    # Add Arrows
    def add_arrow(start_key, end_key, start_anchor, end_anchor):
        start, end = nodes[start_key], nodes[end_key]
        x0 = start['pos'][0] + start_anchor[0] * start['w']
        y0 = start['pos'][1] + start_anchor[1] * start['h']
        x1 = end['pos'][0] + end_anchor[0] * end['w']
        y1 = end['pos'][1] + end_anchor[1] * end['h']
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="black")
    
    add_arrow('sys_desc', 'fat_sat', (0, -1), (0, 1))
    add_arrow('fat_sat', 'val_activities', (0, -1), (0, 1))
    add_arrow('val_activities', 'sample_size', (0, -1), (0, 1))
    add_arrow('val_activities', 'acceptance', (1, 0), (-1, 0))
    
    # Feedback Loops
    fig.add_shape(type="path", path=" M 9.5,4.6 C 10.5,6.5 8.5,7.5 7.5,7", line=dict(color="black", width=3, dash='dot'))
    fig.add_annotation(x=7.5, y=7, ax=8, ay=7.2, showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor="black")
    fig.add_shape(type="path", path=" M 6.5,2.9 C 5.5,3.7 5.5,4.7 6.5,5.0", line=dict(color=ERROR_RED, width=3, dash='dot'))
    fig.add_annotation(x=6.5, y=5.0, ax=6.2, ay=4.6, showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor=ERROR_RED)

    # Horizontal connection for onboarding loop
    fig.add_shape(type="line", x0=nodes['acceptance']['pos'][0], y0=nodes['acceptance']['pos'][1], x1=nodes['onboarding']['pos'][0], y1=nodes['onboarding']['pos'][1], line=dict(color="black", width=2, dash='dot'))

    fig.update_layout(
        title="<b>Equipment Validation Process Map</b>",
        xaxis=dict(range=[0, 11.5], visible=False),
        yaxis=dict(range=[0, 10.5], visible=False),
        plot_bgcolor=BACKGROUND_GREY,
        margin=dict(l=20, r=20, t=40, b=20),
        height=800
    )
    return fig
    
def create_equipment_v_model() -> go.Figure:
    """Creates a V-Model diagram specific to equipment validation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["<b>User Requirements (URS)</b>", "<b>Functional Specs</b>", "<b>Design Specs (P&ID, Electrical)</b>", "<b>Build & Fabrication</b>"], textposition="bottom center", line=dict(color=PRIMARY_COLOR, width=3), marker=dict(size=15)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["<b>FAT / SAT</b>", "<b>IQ (Installation)</b>", "<b>OQ (Operational)</b>", "<b>PQ (Performance)</b>"], textposition="top center", line=dict(color=SUCCESS_GREEN, width=3), marker=dict(size=15)))
    links = [("URS ↔ PQ", 4), ("Functional Specs ↔ OQ", 3), ("Design Specs ↔ IQ", 2), ("Build ↔ FAT/SAT", 1)]
    for i, (text, y_pos) in enumerate(links):
        fig.add_shape(type="line", x0=4-i, y0=y_pos, x1=5+i, y1=y_pos, line=dict(color=NEUTRAL_GREY, width=1, dash="dot"))
        fig.add_annotation(x=4.5, y=y_pos + 0.1, text=text, showarrow=False, font=dict(size=10))
    fig.add_annotation(x=2.5, y=4.5, text="<b>Specification / Design</b>", showarrow=False, font=dict(color=PRIMARY_COLOR, size=14))
    fig.add_annotation(x=6.5, y=4.5, text="<b>Verification / Qualification</b>", showarrow=False, font=dict(color=SUCCESS_GREEN, size=14))
    fig.update_layout(title_text="<b>Equipment Validation V-Model</b>", title_x=0.5, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_portfolio_health_dashboard(key: str) -> Styler:
    health_data = {'Project': ["Project Atlas (Bioreactor)", "Project Beacon (Assembly)", "Project Comet (Vision)"], 'Overall Status': ["Green", "Amber", "Green"], 'Schedule': ["On Track", "At Risk", "Ahead"], 'Budget': ["On Track", "Over", "On Track"], 'Lead': ["J. Doe", "S. Smith", "J. Doe"]}
    df = pd.DataFrame(health_data)
    def style_status(val: str) -> str:
        color_map = {"Green": SUCCESS_GREEN, "Amber": WARNING_AMBER, "Red": ERROR_RED, "On Track": SUCCESS_GREEN, "Ahead": SUCCESS_GREEN, "At Risk": WARNING_AMBER, "Over": ERROR_RED}
        bg_color = color_map.get(val, 'white')
        font_color = 'white' if val in color_map else 'black'
        return f"background-color: {bg_color}; color: {font_color};"
    return df.style.map(style_status, subset=['Overall Status', 'Schedule', 'Budget']).set_properties(**{'text-align': 'center'}).hide(axis="index")

def create_resource_allocation_matrix(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    data = {'J. Doe (Lead)': [0.5, 0.4, 0.2], 'S. Smith (Eng.)': [0.1, 0.8, 0.0], 'A. Wong (Spec.)': [0.4, 0.4, 0.3], 'B. Zeller (Eng.)': [0.0, 0.2, 0.7]}
    df = pd.DataFrame(data, index=["Project Atlas", "Project Beacon", "Project Comet"]); df_transposed = df.T
    color_range_max = 1.1; normalized_colorscale = [[0.0, 'white'], [0.5 / color_range_max, SUCCESS_GREEN], [1.0 / color_range_max, WARNING_AMBER], [1.0, ERROR_RED]]
    fig = px.imshow(df_transposed, text_auto=".0%", aspect="auto", color_continuous_scale=normalized_colorscale, range_color=[0, color_range_max], labels=dict(x="Project", y="Team Member", color="Allocation"), title="<b>Team Allocation by Project</b>")
    fig.update_traces(textfont_color='black'); fig.update_layout(title_x=0.5, title_font_size=20, plot_bgcolor=BACKGROUND_GREY)
    allocations = df_transposed.sum(axis=1).reset_index(); allocations.columns = ['Team Member', 'Total Allocation']
    over_allocated = allocations[allocations['Total Allocation'] > 1.0]
    return fig, over_allocated

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)
    
def run_project_duration_forecaster(key: str) -> None:
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({'New Automation Modules': rng.integers(1, 10, 20), 'Process Complexity Score': rng.integers(1, 11, 20), '# of URS': rng.integers(20, 100, 20)})
    y_train = pd.Series(rng.uniform(8, 52, 20), name="Validation Duration (Weeks)")
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    st.markdown("##### Adjust Project Parameters to Forecast Timeline:")
    col1, col2, col3 = st.columns(3)
    with col1: new_modules = st.slider("New Automation Modules", 1, 10, 4, key=f"pipe_modules_{key}")
    with col2: complexity = st.slider("Process Complexity (1-10)", 1, 10, 6, key=f"pipe_comp_{key}")
    with col3: urs_count = st.slider("# of URS", 20, 100, 50, key=f"pipe_urs_{key}")
    new_project_data = pd.DataFrame([[new_modules, complexity, urs_count]], columns=X_train.columns)
    predicted_duration = model.predict(new_project_data)[0]
    st.metric("AI-Predicted Validation Duration (Weeks)", f"{predicted_duration:.1f}", help="Based on a Random Forest model trained on 20 historical projects. Includes IQ, OQ, and PQ phases.")
    st.subheader("AI Prediction Analysis (Why This Forecast?)")
    st.info("This SHAP force plot shows which project features are driving the timeline estimate. Red arrows push the prediction higher (longer duration), and blue arrows push it lower.")
    explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(new_project_data)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], new_project_data.iloc[0,:]), 150)
    st.success("**Actionable Insight:** The SHAP analysis reveals that the high '# of URS' is the primary factor increasing the project's predicted duration. To shorten this timeline, we should focus on consolidating or simplifying user requirements during the planning phase.")

def plot_cpk_analysis(key: str) -> Tuple[go.Figure, float]:
    """
    Creates a process capability (Cpk) chart for PQ and returns the figure and Cpk value.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.15, scale=0.05, size=100)
    LSL, USL = 5.0, 5.3
    mu, std = np.mean(data), np.std(data, ddof=1)
    
    # Ensure std dev is not zero to avoid division errors
    if std > 0:
        cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std))
    else:
        cpk = 0.0

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=20, name='Observed Data', histnorm='probability density', marker_color=PRIMARY_COLOR, opacity=0.7))
    x_fit = np.linspace(min(data), max(data), 200)
    y_fit = norm.pdf(x_fit, mu, std)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Normal Distribution', line=dict(color=SUCCESS_GREEN, width=2)))
    fig.add_vline(x=LSL, line_dash="dash", line_color=ERROR_RED, annotation_text="LSL", annotation_position="top left")
    fig.add_vline(x=USL, line_dash="dash", line_color=ERROR_RED, annotation_text="USL", annotation_position="top right")
    fig.add_vline(x=mu, line_dash="dot", line_color=NEUTRAL_GREY, annotation_text=f"Mean={mu:.2f}", annotation_position="bottom right")
    
    fig.update_layout(
        title_text=f'Process Capability (Cpk) Analysis - Titer (g/L)<br><b>Cpk = {cpk:.2f}</b> (Target: ≥1.33)',
        xaxis_title="Titer (g/L)", yaxis_title="Density", showlegend=False,
        title_font_size=20, title_x=0.5, plot_bgcolor=BACKGROUND_GREY
    )
    return fig, cpk

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
    df['Variance'] = df['Actual'] - df['Budgeted']; df['Color'] = df['Variance'].apply(lambda x: ERROR_RED if x > 0 else SUCCESS_GREEN); df['Text'] = df['Variance'].apply(lambda x: f'${x:+,}k')
    fig = go.Figure(go.Bar(x=df['Variance'], y=df['Category'], orientation='h', marker_color=df['Color'], text=df['Text'], customdata=df[['Budgeted', 'Actual']], hovertemplate='<b>%{y}</b><br>Variance: %{x:+,}k<br>Budgeted: $%{customdata[0]}k<br>Actual: $%{customdata[1]}k<extra></extra>'))
    fig.update_traces(textposition='inside', textfont=dict(color='white', size=14, family="Arial, sans-serif"))
    fig.update_layout(title_text='<b>Annual Budget Variance (Actual vs. Budgeted)</b>', xaxis_title="Variance (in $ thousands)", yaxis_title="", bargap=0.4, plot_bgcolor=BACKGROUND_GREY, title_x=0.5, font=dict(family="Arial, sans-serif", size=12))
    return fig

def plot_headcount_forecast(key: str) -> go.Figure:
    df = pd.DataFrame({'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'], 'Current FTEs': [8, 8, 8, 8], 'Forecasted Need': [8, 9, 10, 10]})
    df['Gap'] = df['Forecasted Need'] - df['Current FTEs']; fig = go.Figure()
    fig.add_trace(go.Bar(name='Current Headcount', x=df['Quarter'], y=df['Current FTEs'], marker_color=PRIMARY_COLOR, text=df['Current FTEs']))
    fig.add_trace(go.Bar(name='Resource Gap', x=df['Quarter'], y=df['Gap'], base=df['Current FTEs'], marker_color=WARNING_AMBER, text=df['Gap'].apply(lambda g: f'+{g}' if g > 0 else '')))
    fig.update_traces(textposition='inside', textfont_size=14)
    fig.update_layout(barmode='stack', title='<b>Resource Gap Analysis: Headcount vs. Forecasted Need</b>', yaxis_title="Full-Time Equivalents (FTEs)", xaxis_title="Fiscal Quarter", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
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

def analyze_spc_rules(df: pd.DataFrame, ucl: float, lcl: float, mean: float) -> list:
    """
    Applies Nelson Rules to SPC data to automatically detect out-of-control trends,
    including checks for guardband zones.
    """
    alerts = []
    sigma = (ucl - mean) / 3
    
    # Rule 1: One point outside the control limits (+/- 3 sigma)
    if any(df['Titer'] > ucl) or any(df['Titer'] < lcl):
        alerts.append("Rule 1 Violation: A data point has exceeded the +/- 3 sigma control limits.")
    
    # Rule 4 (Guardbanding): 2 of 3 consecutive points in the warning zone (+/- 2 sigma)
    upper_warning = mean + (2 * sigma)
    lower_warning = mean - (2 * sigma)
    for i in range(len(df) - 2):
        points = df['Titer'][i:i+3]
        if sum(points > upper_warning) >= 2 or sum(points < lower_warning) >= 2:
            alerts.append("Guardband Alert (Rule 4): Two of three consecutive points are in the +/- 2 sigma warning zone, indicating a potential process shift.")
            break # Only need to find it once
            
    # Rule 2: Nine points in a row on the same side of the mean
    for i in range(len(df) - 8):
        if all(df['Titer'][i:i+9] > mean) or all(df['Titer'][i:i+9] < mean):
            alerts.append("Rule 2 Violation: A run of 9 points on one side of the mean detected (process shift).")
            break

    return alerts

def plot_process_stability_chart(key: str) -> Tuple[go.Figure, list]:
    """Creates an I-MR chart with guardbanding and a guaranteed process shift for demonstration."""
    rng = np.random.default_rng(22)
    # Start with a stable process
    data = rng.normal(5.1, 0.05, 25) 
    # Introduce a process shift that will trigger the guardband rule
    data[15:] = data[15:] + 0.15
    
    df = pd.DataFrame({'Titer': data})
    df['MR'] = df['Titer'].diff().abs()
    
    I_CL = df['Titer'].mean()
    MR_CL = df['MR'].mean()
    I_UCL = I_CL + 2.66 * MR_CL
    I_LCL = I_CL - 2.66 * MR_CL
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart</b>", "<b>Moving Range (MR) Chart</b>"))
    
    # I-Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['Titer'], name='Titer (g/L)', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1)
    fig.add_hline(y=I_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="CL")
    fig.add_hline(y=I_UCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL")
    fig.add_hline(y=I_LCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL")
    
    # --- ENHANCEMENT: Add Guardband zones ---
    sigma = (I_UCL - I_CL) / 3
    upper_guardband = I_CL + 2 * sigma
    lower_guardband = I_CL - 2 * sigma
    fig.add_hrect(y0=upper_guardband, y1=I_UCL, line_width=0, fillcolor=WARNING_AMBER, opacity=0.2, row=1, col=1)
    fig.add_hrect(y0=I_LCL, y1=lower_guardband, line_width=0, fillcolor=WARNING_AMBER, opacity=0.2, row=1, col=1)

    # MR-Chart
    MR_UCL = 3.267 * MR_CL
    fig.add_trace(go.Scatter(x=df.index, y=df['MR'], name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1)
    fig.add_hline(y=MR_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="CL")
    fig.add_hline(y=MR_UCL, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL")
    
    fig.update_layout(height=400, showlegend=False, title_text="<b>Process Stability (I-MR Chart) with Guardbands</b>", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    
    # Analyze for SPC rule violations
    alerts = analyze_spc_rules(df, I_UCL, I_LCL, I_CL)
    return fig, alerts

def plot_csv_dashboard_enhanced(key: str) -> None:
    """Enhanced CSV dashboard with a detailed ALCOA+ checklist."""
    st.info("**Purpose:** This dashboard tracks the validation status of all GxP computerized systems associated with a project, ensuring compliance with data integrity principles (ALCOA+) and 21 CFR Part 11 requirements for electronic records and signatures.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("21 CFR Part 11 Compliance Status", "PASS", "✔️", help="Electronic records and signatures meet all technical and procedural requirements.")
        st.metric("Data Integrity Risk Score", "Low", "-5% vs Last Quarter", help="Calculated based on ALCOA+ principles.")
        
    with col2:
        alcoa_data = {
            'Principle': ['Attributable', 'Legible', 'Contemporaneous', 'Original', 'Accurate', 'Complete', 'Consistent', 'Enduring', 'Available'],
            'Verification Method': ['Unique User Logins', 'Secure PDF Export', 'NTP-Synced Timestamps', 'Read-Only Audit Trail', 'Validated Calculations', 'No Data Deletion', 'Sequential Records', 'Secure Archiving', 'System Uptime'],
            'Status': ['✔️', '✔️', '✔️', '✔️', '✔️', '✔️', '✔️', '✔️', '✔️']
        }
        df_alcoa = pd.DataFrame(alcoa_data)
        st.markdown("###### ALCOA+ Compliance Checklist")
        st.dataframe(df_alcoa, use_container_width=True, hide_index=True)

def plot_lyophilizer_cycle_validation() -> go.Figure:
    """Plots a simulated lyophilizer cycle validation (OQ)."""
    st.info("**Context:** This plot verifies the performance of a lyophilization (freeze-drying) cycle. The OQ confirms the equipment can achieve and hold critical process parameters (shelf temperature and chamber pressure) as defined in the validated recipe.")
    time = np.arange(120)
    temp_set = np.concatenate([np.linspace(20, -40, 20), np.repeat(-40, 40), np.linspace(-40, 20, 60)])
    temp_actual = temp_set + np.random.normal(0, 0.5, 120)
    pressure_set = np.concatenate([np.repeat(1000, 20), np.linspace(1000, 0.1, 20), np.repeat(0.1, 80)])
    pressure_actual = pressure_set + np.random.normal(0, 0.02, 120)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=time, y=temp_set, name='Temp Setpoint (°C)', line=dict(color=NEUTRAL_GREY, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=time, y=temp_actual, name='Temp Actual (°C)', line=dict(color=PRIMARY_COLOR)), secondary_y=False)
    fig.add_trace(go.Scatter(x=time, y=pressure_set, name='Pressure Setpoint (mbar)', line=dict(color=WARNING_AMBER, dash='dash')), secondary_y=True)
    fig.add_trace(go.Scatter(x=time, y=pressure_actual, name='Pressure Actual (mbar)', line=dict(color=ERROR_RED)), secondary_y=True)
    fig.update_layout(title_text='<b>Lyophilizer OQ: Cycle Parameter Verification</b>', title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Time (minutes)")
    fig.update_yaxes(title_text="Shelf Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Chamber Pressure (mbar)", type="log", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The actual shelf temperature and chamber pressure tracked the setpoints precisely throughout all phases (Freezing, Primary Drying, Secondary Drying). This OQ data confirms the lyophilizer is performing as specified and is ready for PQ runs with product.")

def display_nanoliter_dispenser_validation():
    """Displays PQ data for a nanoliter liquid dispenser."""
    st.info("**Context:** This analysis validates the precision and accuracy of a non-contact, acoustic liquid dispenser. Using a fluorescent dye, we measure the dispensed volume at multiple setpoints across the operating range to ensure it meets the tight tolerances required for biochip spotting or PCR plate preparation.")
    df = pd.DataFrame({'Setpoint (nL)': [2.5, 5.0, 10.0, 25.0], 'Mean Dispensed (nL)': [2.51, 5.03, 10.01, 24.98], 'CV (%)': [1.8, 1.5, 1.1, 0.9], 'Accuracy Spec': ['±5%', '±5%', '±3%', '±3%'], 'Precision Spec (CV)': ['<2%', '<2%', '<1.5%', '<1.5%']})
    st.dataframe(style_dataframe(df), use_container_width=True)
    st.success("**Actionable Insight:** The dispenser meets all accuracy and precision (CV%) specifications across its full operating range. The data confirms its suitability for the most demanding low-volume applications. The system is qualified for use in production.")

def display_biochip_assembly_validation():
    """Displays a waterfall chart for OEE of a biochip assembly line."""
    st.info("**Context:** For a full production line, Overall Equipment Effectiveness (OEE) is a critical PQ metric. It measures the combined impact of Availability (uptime), Performance (speed), and Quality (yield). The target for a validated line is often >85%.")
    fig = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["absolute", "relative", "relative", "relative", "total"],
        x = ["Theoretical Max Capacity", "Availability Losses (Downtime)", "Performance Losses (Speed)", "Quality Losses (Scrap)", "<b>Final OEE Output</b>"],
        text = ["100%", "-8%", "-5%", "-2%", "85%"],
        y = [100, -8, -5, -2, 85],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        totals={"marker":{"color":SUCCESS_GREEN}},
        increasing={"marker":{"color":SUCCESS_GREEN}},
        decreasing={"marker":{"color":ERROR_RED}},
    ))
    fig.update_layout(title="<b>Biochip Assembly Line PQ: OEE Calculation</b>", yaxis_title="Effectiveness (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The assembly line achieved an OEE of 85%, meeting the acceptance criterion. The waterfall analysis clearly shows that Availability Losses (unplanned downtime) are the biggest detractor from performance. This provides a data-driven focus for future continuous improvement (Kaizen) events.")

def display_vision_system_validation():
    """Displays a confusion matrix for a vision system validation."""
    st.info("**Context:** A confusion matrix is a key validation artifact for any AI/ML-based vision system. It challenges the system with a large set of known good and bad parts (a 'golden sample' set) to quantify its real-world performance and identify specific failure modes.")
    cm_data = np.array([[998, 2], [5, 95]]) # True Neg, False Pos, False Neg, True Pos
    fig = px.imshow(cm_data, text_auto=True, color_continuous_scale='Greens', labels=dict(x="Predicted Class", y="Actual Class", color="Count"), x=['Good', 'Defect'], y=['Good', 'Defect'], title="<b>Vision System PQ: Confusion Matrix</b>")
    st.plotly_chart(fig, use_container_width=True)
    accuracy = (cm_data[0,0] + cm_data[1,1]) / np.sum(cm_data) * 100
    sensitivity = cm_data[1,1] / (cm_data[1,1] + cm_data[1,0]) * 100 # True Positive Rate
    st.success(f"""
    **Actionable Insight:** The system achieved **{accuracy:.2f}%** overall accuracy. Critically, the **Sensitivity (True Defect detection rate) is {sensitivity:.1f}%**, exceeding the 95% requirement. The 5 false negatives (defects classified as good) will be reviewed with Process Engineering to determine if they represent a new, previously untrained defect type. The system is qualified, with a CAPA to expand the training set.
    """)

def display_electrostatic_control_validation():
    """Displays a chart for validating electrostatic charge control."""
    st.info("**Context:** For plastic biochips and cassettes, uncontrolled electrostatic discharge (ESD) can damage sensitive onboard electronics or cause handling errors. This validation study measures surface voltage at critical assembly stages, both with and without the ionizer system active, to prove its effectiveness.")
    df = pd.DataFrame({'Assembly Stage': ["Cassette Unmolding", "Biochip Placement", "Lid Taping/Soldering", "Final Packaging"], 'Surface Voltage without Ionizer (V)': [1850, 2200, 2550, 1900], 'Surface Voltage with Ionizer (V)': [85, 45, 60, 50]})
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Without Ionizer', x=df['Assembly Stage'], y=df['Surface Voltage without Ionizer (V)'], marker_color=ERROR_RED))
    fig.add_trace(go.Bar(name='With Ionizer', x=df['Assembly Stage'], y=df['Surface Voltage with Ionizer (V)'], marker_color=SUCCESS_GREEN))
    fig.add_hline(y=100, line_dash="dash", annotation_text="Acceptance Limit (<100V)")
    fig.update_layout(title_text="<b>OQ: Ionizer System Effectiveness for ESD Control</b>", title_x=0.5, yaxis_title="Surface Voltage (V)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Actionable Insight:** The data provides conclusive evidence that the ionizer system is essential and effective. Without it, surface voltages far exceed the <100V damage threshold. With the system active, all surfaces are well within the safe limit. The ionizer is now a required, critical utility for the production line.")

def plot_cleaning_validation_results_enhanced(key: str) -> go.Figure:
    """Enhanced Cleaning Validation chart for a multi-product scenario."""
    locations = ['Swab 1 (Reactor Wall)', 'Swab 2 (Agitator Blade)', 'Swab 3 (Fill Nozzle)', 'Final Rinse']
    df = pd.DataFrame({
        'Sample Location': locations * 2,
        'Product': ['Product A (Worst-Case)']*4 + ['Product B (Campaign)']*4,
        'TOC Result (ppb)': [150, 180, 165, 25, 80, 95, 85, 15],
        'Acceptance Limit (ppb)': [500, 500, 500, 50] * 2
    })
    
    fig = px.bar(df, x='Sample Location', y='TOC Result (ppb)', color='Product', 
                 barmode='group', title='<b>Cleaning Validation Results (Multi-Product)</b>',
                 text_auto='.0f')
                 
    fig.add_trace(go.Scatter(
        x=df['Sample Location'].unique(), 
        y=[500, 500, 500, 50],
        name='Acceptance Limit', 
        mode='lines+markers', 
        line=dict(color=ERROR_RED, dash='dash', width=3)
    ))
    fig.update_layout(title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    return fig

def plot_shipping_validation_temp_enhanced(key: str) -> go.Figure:
    """Enhanced shipping validation chart with dual axes for Temp and Shock."""
    rng = np.random.default_rng(30); 
    time = pd.to_datetime(pd.date_range("2023-01-01", periods=48, freq="h"))
    temp = rng.normal(4, 0.5, 48)
    temp[24] = 8.5 # Temperature excursion
    shock = rng.random(48) * 10 # Baseline G-force
    shock[35] = 55 # Shock event
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=time, y=temp, name='Temperature (°C)', line=dict(color=PRIMARY_COLOR)), secondary_y=False)
    fig.add_hrect(y0=2, y1=8, line_width=0, fillcolor=SUCCESS_GREEN, opacity=0.1, secondary_y=False, annotation_text="Temp Spec", annotation_position="top left")
    fig.add_trace(go.Bar(x=time, y=shock, name='Shock (G-force)', marker_color=ERROR_RED, opacity=0.5), secondary_y=True)
    
    fig.update_layout(title_text='<b>Shipping Lane PQ: Temperature & Shock Profile</b>', title_x=0.5, plot_bgcolor=BACKGROUND_GREY, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, range=[0, 10])
    fig.update_yaxes(title_text="Shock (G-force)", secondary_y=True, range=[0, 100])
    return fig

def plot_doe_optimization_enhanced(key: str) -> None:
    """Enhanced DOE visualization with 2D Contour Plot showing operating space."""
    temp = np.linspace(30, 40, 10); ph = np.linspace(6.8, 7.6, 10); 
    temp_grid, ph_grid = np.meshgrid(temp, ph)
    signal = 100 - (temp_grid - 37)**2 - 20*(ph_grid - 7.2)**2 + np.random.rand(10, 10)*2
    
    col1, col2 = st.columns(2)
    with col1:
        fig_3d = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis', colorbar_title='Yield')])
        fig_3d.update_layout(title='<b>DOE Response Surface (3D)</b>', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Yield (%)'), margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        fig_2d = go.Figure(data=go.Contour(z=signal, x=temp, y=ph, colorscale='viridis', contours_coloring='lines', line_width=2))
        fig_2d.add_shape(type="rect", x0=36, y0=7.1, x1=38, y1=7.3, line=dict(color=SUCCESS_GREEN, width=3), name='NOR')
        fig_2d.add_shape(type="rect", x0=35, y0=7.0, x1=39, y1=7.4, line=dict(color=WARNING_AMBER, width=2, dash="dash"), name='PAR')
        fig_2d.add_annotation(x=37, y=7.2, text="<b>NOR</b><br>(Normal Operating<br>Range)", showarrow=False, font=dict(color=SUCCESS_GREEN))
        fig_2d.add_annotation(x=35.1, y=7.38, text="<b>PAR</b> (Proven Acceptable Range)", showarrow=False, xanchor="left", yanchor="top", font=dict(color=WARNING_AMBER))
        fig_2d.update_layout(title='<b>Process Operating Space (2D)</b>', xaxis_title='Temperature (°C)', yaxis_title='pH', margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_2d, use_container_width=True)

def plot_cost_of_quality(key: str) -> go.Figure:
    categories = ['Prevention Costs (e.g., Planning, Training)', 'Appraisal Costs (e.g., Testing, FAT/SAT)', 'Internal Failure Costs (e.g., Rework, Deviations)', 'External Failure Costs (e.g., Recall, Audit Findings)']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='With Proactive Validation', x=[200, 300, 50, 10], y=categories, orientation='h', marker_color=SUCCESS_GREEN))
    fig.add_trace(go.Bar(name='Without Proactive Validation', x=[50, 75, 800, 1500], y=categories, orientation='h', marker_color=ERROR_RED))
    fig.update_layout(barmode='stack', title_text='<b>Strategic Value: The Cost of Quality (CoQ) Model</b>', title_x=0.5, xaxis_title='Annual Costs (in $ thousands)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor=BACKGROUND_GREY)
    total_with = 200 + 300 + 50 + 10; total_without = 50 + 75 + 800 + 1500
    fig.add_annotation(x=total_with, y=categories[0], text=f"<b>Total: ${total_with}k</b>", showarrow=False, xanchor='left', font_color=SUCCESS_GREEN)
    fig.add_annotation(x=total_without, y=categories[0], text=f"<b>Total: ${total_without}k</b>", showarrow=False, xanchor='right', font_color=ERROR_RED)
    return fig

def display_team_skill_matrix(key: str) -> None:
    skill_data = {'Team Member': ['J. Doe (Lead)', 'S. Smith (Eng. II)', 'A. Wong (Spec. I)', 'B. Zeller (Eng. I)'], 'CSV & Part 11': [5, 3, 2, 2], 'Cleaning Validation': [4, 4, 3, 1], 'Statistics (Cpk/DOE)': [5, 3, 2, 3], 'Project Management': [5, 2, 1, 1], 'Development Goal': ['Mentor team on advanced stats', 'Lead CSV for Project Beacon', 'Cross-train on Cleaning Val', 'Achieve PMP certification']}
    df = pd.DataFrame(skill_data)
    st.info("**Purpose:** A skills matrix is a critical tool for a manager to visualize team capabilities, identify skill gaps, and plan targeted development. *Proficiency Scale: 1 (Novice) to 5 (SME)*")
    skill_cols = df.columns.drop(['Team Member', 'Development Goal'])
    styled_df = df.style.background_gradient(cmap='Greens', subset=skill_cols, vmin=1, vmax=5).set_properties(**{'text-align': 'center'}, subset=skill_cols)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=215)
    st.success("**Actionable Insight:** The matrix identifies a bottleneck in Project Management skills. Based on B. Zeller's development goal, I will approve the budget for PMP certification training. A. Wong will be assigned to the 'Atlas' Cleaning Validation PQ to gain hands-on experience, supported by S. Smith.")

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
    
def analyze_project_bottlenecks() -> None:
    """
    NEW: Provides a transparent, multi-factor analysis to identify and visualize project bottlenecks.
    This replaces the previous "black box" alert with a full methodology and data-driven conclusion.
    """
    st.subheader("Automated Bottleneck Analysis", divider='blue')
    st.info("""
    **Methodology:** This analysis calculates a weighted risk score for each project across four key dimensions: **Schedule, Budget, Resource, and Technical Risk**. 
    The project with the highest total score is flagged as the most critical bottleneck requiring immediate management attention. The radar chart below visualizes the risk profile of each project, where a larger, more skewed shape indicates higher risk.
    """)

    # 1. Raw Data Simulation
    # Raw data for SPI, CPI, Resource Allocation %, and Open High Risks
    data = {
        'Project': ["Project Atlas (Bioreactor)", "Project Beacon (Assembly)", "Project Comet (Vision)"],
        'SPI': [1.02, 0.92, 1.1],
        'CPI': [1.01, 0.85, 1.05],
        'Lead Allocation (%)': [90, 95, 60], # Lead for Beacon is highly utilized
        'Open Risks (RPN>25)': [4, 9, 2] # Beacon has the most open risks
    }
    df = pd.DataFrame(data)

    # 2. Risk Score Calculation (The transparent methodology)
    # Convert metrics to a risk score (lower SPI/CPI is higher risk)
    df['Schedule Risk'] = 1 / df['SPI']
    df['Budget Risk'] = 1 / df['CPI']
    df['Resource Risk'] = df['Lead Allocation (%)'] / 100
    df['Technical Risk'] = df['Open Risks (RPN>25)']

    # Normalize each risk factor to a 0-1 scale to compare them fairly
    risk_categories = ['Schedule Risk', 'Budget Risk', 'Resource Risk', 'Technical Risk']
    for col in risk_categories:
        min_val = df[col].min()
        max_val = df[col].max()
        if (max_val - min_val) > 0:
            df[f'{col} (Norm)'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f'{col} (Norm)'] = 0 # Handle case where all values are the same
            
    # Calculate a final weighted score
    weights = {'Schedule Risk (Norm)': 0.4, 'Resource Risk (Norm)': 0.3, 'Budget Risk (Norm)': 0.2, 'Technical Risk (Norm)': 0.1}
    df['Total Risk Score'] = sum(df[col] * w for col, w in weights.items())

    # 3. Visualization and Data Display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # The Radar Chart
        fig = go.Figure()
        for i, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[[f'{cat} (Norm)' for cat in risk_categories]].tolist() + [row[f'{risk_categories[0]} (Norm)']], # Loop back to start
                theta=risk_categories + [risk_categories[0]],
                fill='toself',
                name=row['Project']
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="<b>Project Risk Profile Comparison</b>",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # The transparent data table
        st.markdown("###### Risk Score Calculation Data")
        st.dataframe(df[['Project', 'Total Risk Score'] + [f'{cat} (Norm)' for cat in risk_categories]].round(2), use_container_width=True, hide_index=True)
        
    # 4. Dynamic, Data-Driven Insight
    bottleneck_project = df.loc[df['Total Risk Score'].idxmax()]
    primary_constraint = bottleneck_project[[f'{cat} (Norm)' for cat in risk_categories]].idxmax().replace(' (Norm)', '')

    st.error(f"""
    **⚠️ Automated Alert: Critical Bottleneck Identified**

    - **Project at Highest Risk:** `{bottleneck_project['Project']}` (Total Score: {bottleneck_project['Total Risk Score']:.2f})
    - **Primary Constraint:** **{primary_constraint}**. This factor is the largest contributor to the project's overall risk profile.
    """)
    st.success(f"""
    **Actionable Insight:** The convergence of risk factors, particularly the **{primary_constraint}**, on `{bottleneck_project['Project']}` presents the most significant threat to the portfolio. 
    **Recommendation:** Immediately convene with the project lead to develop a targeted recovery plan for this specific constraint. For example, if the constraint is Resource Risk, we must identify tasks that can be delegated to de-risk the timeline.
    """)

# --- PAGE RENDERING FUNCTIONS ---
def render_main_page() -> None:
    st.title("🤖 Equipment Validation Management Portfolio"); st.subheader("A Live Demonstration of Modern Validation Management Leadership"); st.divider()
    st.markdown("Welcome. This interactive environment shows end-to-end validation of biotech manufacturing equipment in a strictly regulated GMP environment. It shows management validation functions, with a relentless focus on aligning technical execution, **Quality Systems (per 21 CFR 820 & ISO 13485)**, and strategic capital projects.")
    
    st.subheader("Key Program Health KPIs", divider='blue')
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric("Validation Program Compliance", "98%", delta="1%")
            st.plotly_chart(plot_kpi_sparkline([96, 96, 97, 97, 97, 98], unit="%", x_axis_label="6-Mo Trend"), use_container_width=True)
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric("Quality First Time Rate", "91%", delta="-2%")
            st.plotly_chart(plot_kpi_sparkline([92, 94, 95, 93, 93, 91], unit="%", x_axis_label="6-Mo Trend"), use_container_width=True)
    with kpi_cols[2]:
        with st.container(border=True):
            st.metric("Capital Project On-Time Delivery", "95%", delta="5%")
            st.plotly_chart(plot_kpi_sparkline([85, 88, 87, 90, 90, 95], unit="%", x_axis_label="6-Mo Trend"), use_container_width=True)

    kpi_cols2 = st.columns(3)
    with kpi_cols2[0]:
        with st.container(border=True):
            st.metric("CAPEX Validation Spend vs. Budget", "97%", delta="-3%", delta_color="inverse")
            st.plotly_chart(plot_kpi_sparkline([95, 96, 98, 101, 100, 97], unit="%", x_axis_label="6-Mo Trend", is_good_down=True), use_container_width=True)
    with kpi_cols2[1]:
        with st.container(border=True):
            st.metric("Avg. Protocol Review Cycle Time", "8.2 Days", delta="-1.5 Days", delta_color="inverse")
            st.plotly_chart(plot_kpi_sparkline([11.5, 10.1, 9.5, 9.7, 9.7, 8.2], unit=" Days", x_axis_label="6-Mo Trend", is_good_down=True), use_container_width=True)
    with kpi_cols2[2]:
        with st.container(border=True):
            st.metric("Open Validation-Related CAPAs", "3", delta="1", delta_color="inverse")
            st.plotly_chart(plot_kpi_sparkline([5, 4, 4, 2, 2, 3], unit="", x_axis_label="6-Mo Trend", is_good_down=True), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_glossary, col_predictive = st.columns([2, 1])
    with col_glossary:
        with st.expander("📖 KPI Glossary: Definitions, Significance, and Actionability"):
            st.markdown("""
            - **Validation Program Compliance:**
              - *Definition:* Percentage of GxP systems that are in a validated state and within their scheduled periodic review window.
              - *Significance:* This is a primary indicator of the site's overall audit readiness and compliance posture.
              - *Actionability:* A downward trend triggers a root cause analysis, potentially leading to resource re-prioritization.
            - **Quality First Time Rate:**
              - *Definition:* Percentage of validation protocols (IQ, OQ, PQ) executed without any deviations.
              - *Significance:* A high rate indicates robust planning and well-designed equipment. It is a leading indicator of efficiency.
              - *Actionability:* A declining rate prompts a review of recent deviations to improve templates or training.
            - **Capital Project On-Time Delivery:**
              - *Definition:* Percentage of major capital projects for which all validation deliverables were completed on or before the planned schedule milestones.
              - *Significance:* This measures the validation department's ability to act as a reliable partner in the broader business, directly impacting production launch timelines.
              - *Actionability:* A low rate requires immediate intervention with project managers to review schedules, assess risks, and analyze resource allocation.
            - **CAPEX Validation Spend vs. Budget:**
              - *Definition:* The percentage of the allocated capital expenditure budget for validation activities that has been spent.
              - *Significance:* Measures financial control and forecasting accuracy.
              - *Actionability:* Deviations from the plan (>5%) trigger a review with project leads to understand the cause and re-forecast future spend.
            - **Avg. Protocol Review Cycle Time:**
              - *Definition:* The average number of business days from a protocol's submission for review to its final approval by all parties.
              - *Significance:* Long cycle times are a major source of inefficiency and can delay project execution.
              - *Actionability:* An increasing trend would initiate discussions with QA and other departments to identify bottlenecks.
            - **Open Validation-Related CAPAs:**
              - *Definition:* The number of open Corrective and Preventive Actions where the Validation department is the designated owner for completion.
              - *Significance:* A high or increasing number can indicate systemic issues or an over-burdened team.
              - *Actionability:* Each open CAPA is tracked with a due date. A rising trend prompts a review of CAPA sources to identify systemic problems.
            """)
    with col_predictive:
        with st.container(border=True):
            st.plotly_chart(plot_predictive_compliance_risk(), use_container_width=True)
            st.info("**Purpose:** This AI-driven score aggregates leading indicators to forecast the future risk of falling out of compliance.")
            st.success("**Actionable Insight:** The current score is green. However, if 'Open CAPAs' were to increase, this model predicts a move into the amber zone next quarter, allowing us to act *before* a problem occurs.")
        with st.container(border=True):
            run_rft_prediction_model(key="main_rft")

def render_strategic_management_page() -> None:
    st.title("📈 1. Strategic Management & Business Acumen")
    render_manager_briefing(
        title="Leading Validation as a Business Unit", 
        content="An effective manager must translate technical excellence into business value. This dashboard demonstrates the ability to manage budgets, forecast resources, set strategic goals (OKRs), and articulate the financial value of a robust quality program.", 
        reg_refs="ISO 13485:2016 (Sec 5 & 6), 21 CFR 820.20", 
        business_impact="Ensures the validation department is a strategic, financially responsible partner that enables the company's growth and compliance goals.", 
        quality_pillar="Resource Management & Financial Acumen.", 
        risk_mitigation="Proactively identifies resource shortfalls and justifies the validation budget as a high-return investment in preventing failure costs."
    )
    
    with st.container(border=True):
        st.subheader("Departmental OKRs (Objectives & Key Results)", divider='blue')
        st.info("**Purpose:** OKRs provide a clear framework that connects the department's daily work to the company's high-level strategic objectives. They ensure the team is focused on impactful, measurable outcomes.")
        display_departmental_okrs(key="okrs")
        st.success("**Actionable Insight:** The team is on track to meet its efficiency and compliance goals for the year. The completion of the GAMP 5 certification directly supports our 'Enhance Team Capabilities' objective.")

    st.subheader("Financial & Resource Planning")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True): 
            st.markdown("##### Annual Budget Performance")
            st.plotly_chart(plot_budget_variance(key="budget"), use_container_width=True)
            st.success("**Actionable Insight:** Operating within overall budget. The slight CapEx overage was an approved expenditure for accelerated project timelines, offset by contractor savings.")
    with col2:
        with st.container(border=True): 
            st.markdown("##### Headcount & Resource Forecasting")
            st.plotly_chart(plot_headcount_forecast(key="headcount"), use_container_width=True)
            st.success("**Actionable Insight:** The forecast indicates a resource gap of 2 FTEs by Q3. This data justifies the hiring requisition for one Automation Engineer and one Validation Specialist.")
    
    st.subheader("Strategic Value Analysis")
    with st.container(border=True):
        st.plotly_chart(plot_cost_of_quality(key="coq"), use_container_width=True)
        # --- FIX: Escape dollar signs to prevent LaTeX rendering ---
        st.success("""
        **Actionable Insight:** The CoQ model proves that for every **\\$1 spent on proactive validation**, we prevent an estimated **\\$4 in failure costs** (rework, deviations, batch loss). This data provides a powerful justification for our departmental budget and headcount.
        """)
    
    with st.container(border=True):
        st.subheader("AI-Powered Capital Project Duration Forecaster")
        run_project_duration_forecaster("duration_ai")

# This is the complete and final `render_project_portfolio_page` function

def render_project_portfolio_page() -> None:
    st.title("📂 2. Project & Portfolio Management")
    render_manager_briefing(title="Managing the Validation Project Portfolio", content="This command center demonstrates the ability to manage a portfolio of competing capital projects, balancing priorities, allocating finite resources, and providing clear, high-level status updates to the PMO and site leadership.", reg_refs="Project Management Body of Knowledge (PMBOK), ISO 13485: 5.6", business_impact="Provides executive-level visibility into Validation's contribution to corporate goals, enables proactive risk management, and ensures strategic alignment of the department's people.", quality_pillar="Project Governance & Oversight.", risk_mitigation="Prevents budget overruns and schedule delays through proactive monitoring of CPI/SPI metrics and resource allocation.")
    
    with st.container(border=True):
        st.subheader("Capital Project Timelines (Gantt Chart)")
        st.info("**Purpose:** The Gantt chart provides a high-level visual timeline for all major capital projects, highlighting dependencies and overall schedule health.")
        st.plotly_chart(plot_gantt_chart(key="gantt"), use_container_width=True)

    with st.container(border=True):
        st.subheader("Capital Project Portfolio Health")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Purpose:** The RAG (Red-Amber-Green) status provides an immediate, at-a-glance summary of portfolio health for executive review and PMO meetings.")
            st.markdown("##### RAG Status")
            st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True)
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
    with st.container(border=True):
        # This new module demonstrates strategic, forward-looking vendor management.
        display_vendor_dashboard(key="vendor_dashboard")    
    with st.container(border=True):
        # The new, transparent analysis module is called here.
        analyze_project_bottlenecks()
    with st.container(border=True):
        # This section links the project portfolio directly to team capabilities and development.
        c1, c2 = st.columns(2)
        with c1:
            display_team_skill_matrix(key="portfolio_skills")
        with c2:
            st.subheader("Validation Team Resource Allocation", divider='blue')
            st.info("**Purpose:** The heatmap visualizes the current workload distribution across the team, immediately highlighting potential over-allocation risks.")
            fig_alloc, over_allocated_df = create_resource_allocation_matrix("allocation")
            st.plotly_chart(fig_alloc, use_container_width=True)
            if not over_allocated_df.empty:
                for _, row in over_allocated_df.iterrows():
                    st.warning(f"**⚠️ Over-allocation Alert:** {row['Team Member']} is at {row['Total Allocation']:.0%} workload.")
                    
# This is the complete, final version of this function to replace the old one.
def render_e2e_validation_hub_page() -> None:
    st.title("🔩 Live E2E Validation Walkthrough: Project Atlas")
    render_manager_briefing(title="Executing a Compliant Validation Lifecycle (per ASTM E2500)", content="This hub presents the entire validation lifecycle in a single, comprehensive view, simulating the execution of a major capital project. It provides tangible evidence of owning deliverables from design and risk management through to final performance qualification.", reg_refs="FDA 21 CFR 820.75, ISO 13485:2016 (Sec 7.5.6), GAMP 5, ASTM E2500", business_impact="Ensures new manufacturing equipment is brought online on-time, on-budget, and in a fully compliant state, directly enabling production launch.", quality_pillar="Design Controls & Risk-Based Verification.", risk_mitigation="Prevents costly redesigns and validation failures by ensuring testability is built-in from the URS phase using tools like the V-Model and pFMEA.")
    
    with st.container(border=True):
        st.info("**Purpose:** This Process Map illustrates our end-to-end equipment validation methodology, serving as a standardized framework for all capital projects. It defines the required deliverables, control gates, and feedback loops that ensure a compliant and efficient process.")
        # --- CORRECTED FUNCTION CALL ---
        st.plotly_chart(create_process_map_diagram(), use_container_width=True)
        # --- END CORRECTION ---
        st.success("**Actionable Insight:** This standardized scheme ensures all projects meet regulatory requirements consistently, reduces ambiguity for project teams, and accelerates equipment onboarding by defining clear deliverables and acceptance criteria upfront.")
    
    st.subheader("Live Project Artifacts", divider='blue')
    col1, col2 = st.columns(2)
    with col1:
        st.header("Phase 1: Design & Risk Management"); st.info("The 'left side of the V-Model' focuses on proactive planning.")
        with st.container(border=True): 
            st.subheader("Equipment Validation V-Model")
            st.plotly_chart(create_equipment_v_model(), use_container_width=True)
        with st.container(border=True): st.subheader("AI-Powered URS Risk Analysis"); st.plotly_chart(run_urs_risk_nlp_model("urs_risk"), use_container_width=True); st.success("**Actionable Insight:** Requirements 2, 3, and 5 flagged for rewrite due to high ambiguity.")
        with st.container(border=True): st.subheader("User Requirements Traceability (RTM)"); create_rtm_data_editor("rtm")
        with st.container(border=True): st.subheader("Process Risk Management (pFMEA)"); plot_risk_matrix("fmea")
        with st.container(border=True):
            display_rpn_table(key="e2e_rpn")
    with col2:
        st.header("Phases 2-4: Execution & Qualification"); 
        st.info("The 'right side of the V-Model' focuses on generating documented, objective evidence that the as-built system meets all requirements and is fit for its intended use in a GMP environment.")
        
        st.subheader("Phase 2: Factory & Site Acceptance Testing", divider='blue')
        with st.container(border=True):
            st.info("**Context:** FAT is performed at the vendor's site to identify and fix issues *before* shipping, saving significant time and cost. SAT confirms that no damage or changes occurred during transit and installation.")
            display_fat_sat_summary("fat_sat")
            st.success("""
            **Actionable Insight:** The 100% SAT pass rate and zero new major deviations confirm a successful transfer from the vendor to our site. The statistical equivalence demonstrated by TOST provides high confidence that the equipment is ready for formal qualification. **Decision:** Authorize the start of IQ/OQ execution.
            """)

        st.subheader("Phase 3: Installation & Operational Qualification", divider='blue')
        with st.container(border=True):
            st.info("**Context:** IQ verifies the physical installation against design specs (e.g., correct materials of construction, P&ID checks). OQ challenges the system's functions by testing it at the edges of its operating ranges (e.g., highest/lowest temperatures, speeds).")
            st.plotly_chart(plot_oq_challenge_results("oq_plot"), use_container_width=True)
            st.success("""
            **Actionable Insight:** The temperature control loop performed well within the acceptance criteria (±0.5°C), even during the most aggressive ramp to 45°C. This demonstrates the system is robust and capable of maintaining a critical process parameter. **Decision:** Approve the OQ results and proceed to the final PQ phase.
            """)

        st.subheader("Phase 4: Performance Qualification", divider='blue')
        with st.container(border=True):
            # --- MODIFICATION: Enhanced context to introduce the core principle ---
            st.info("""
            **Context:** PQ is the final qualification step, designed to prove the equipment can consistently and repeatably produce quality product under normal manufacturing conditions. It answers two fundamental questions: 
            1. Is the process **stable** and predictable over time? 
            2. Is the process **capable** of consistently meeting its quality specifications? 
            
            A process **must satisfy both conditions** to be considered validated.
            """)
            
            c1, c2 = st.columns(2)
            with c1: 
                st.markdown("###### Process Capability (Cpk)")
                st.info("**Purpose:** The Cpk analysis measures if the process output is well within its specification limits. A value ≥ 1.33 is typically required.")
                cpk_fig, cpk_value = plot_cpk_analysis("pq_cpk")
                st.plotly_chart(cpk_fig, use_container_width=True)
                
            with c2: 
                st.markdown("###### Process Stability (SPC)")
                st.info("""
                **Purpose: Statistical Process Control (SPC)**

                The I-MR chart is a fundamental SPC tool used to monitor a process over time. It answers the question: "Is our process stable and predictable, or is it being influenced by unexpected, special causes of variation?"
                
                - **Control Limits (UCL/LCL):** The red dotted lines at ±3σ (standard deviations) are the voice of the process. A point outside these limits indicates that a statistically significant event has occurred.
                - **Guardbanding:** The yellow shaded areas represent **warning limits** set at ±2σ. This is a proactive control strategy. Points falling in this zone are not yet failures, but they serve as an early warning that the process may be drifting towards an out-of-control state.
                """)
                spc_fig, spc_alerts = plot_process_stability_chart("pq_spc")
                st.plotly_chart(spc_fig, use_container_width=True)
            
            st.markdown("---")
            # --- MODIFICATION: New subheader for the overall analysis ---
            st.subheader("Overall PQ Analysis & Conclusion")

            # --- MODIFICATION: Added individual observations before the final conclusion ---
            st.markdown(f"""
            - **Observation (Stability):** The automated SPC analysis of the I-MR chart has detected **{len(spc_alerts)} out-of-control signal(s)**.
            - **Observation (Capability):** The calculated Cpk for the dataset is **{cpk_value:.2f}**. The required acceptance criterion is ≥ 1.33.
            """)
            
            # --- MODIFICATION: Renamed sections for clarity and added more detail ---
            if spc_alerts:
                st.error(f"**Finding:** The automated SPC analysis detected an out-of-control signal: **{spc_alerts[0]}**.")
                st.warning(f"""
                **Overall Conclusion: Not Stable, Cpk is Statistically Invalid.**

                **Significance:** The process is **NOT in a state of statistical control**. The presence of special cause variation (the process shift) means the process is unpredictable. Because stability is a prerequisite for capability, the calculated Cpk of **{cpk_value:.2f} is meaningless** and cannot be used to make a decision. The primary issue is the lack of control.
                
                **Decision:** **PQ Failed.** An investigation must be launched with Process Engineering to identify the root cause of the process shift. The PQ protocol must be re-executed after corrective actions are implemented and verified.
                """)
            elif cpk_value < 1.33:
                 st.error(f"**Finding:** The process is stable, but the Cpk of **{cpk_value:.2f}** is **BELOW** the required target of ≥1.33.")
                 st.warning("""
                 **Overall Conclusion: Stable but Not Capable.**

                 **Significance:** The process is predictable and consistent, but it consistently produces product that is too close to the specification limits, guaranteeing a certain percentage of future batches will fail. The process as designed cannot meet the quality standard.
                 
                 **Decision:** **PQ Failed.** An investigation is required with Process Engineering to fundamentally reduce process variability (e.g., through DOE, raw material improvements, or equipment modification) before re-executing the PQ.
                 """)
            else:
                st.success(f"""
                **Overall Conclusion: Stable and Capable.**

                **Significance:** The process is demonstrated to be both in a state of statistical control (predictable) and highly capable of meeting its critical quality attributes (Cpk of {cpk_value:.2f} ≥ 1.33).
                
                **Decision:** **PQ Passed.** The system is officially qualified and can be released for commercial manufacturing.
                """)

def render_specialized_validation_page() -> None:
    st.title("🧪 4. Specialized Validation Hubs")
    render_manager_briefing(title="Demonstrating Breadth of Expertise", content="Beyond standard equipment qualification, a Validation Manager must be fluent in specialized validation disciplines critical to GMP manufacturing. This hub showcases expertise across a wide range of complex, real-world MedTech and Biopharma applications.", reg_refs="21 CFR Part 11, GAMP 5, ISO 14971, PDA Technical Reports", business_impact="Ensures all aspects of the manufacturing process, including novel and complex systems, are fully compliant and controlled, preventing common sources of regulatory findings and production delays.", quality_pillar="Cross-functional Technical Leadership.", risk_mitigation="Ensures compliance in niche, high-risk areas like data integrity (CSV), sterility (lyophilization), and microfluidics that are frequent targets of audits.")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Process & Equipment", "⚙️ Assembly & QC", "🛡️ System & Environmental"])

    with tab1:
        st.header("Process & Equipment Validation")
        with st.expander("🔬 **Case Study: Process Characterization (DOE)**", expanded=True):
            case_study_doe()
        with st.expander("❄️ **Case Study: Validating a Lyophilizer Equipment Cycle & Drying**"):
            case_study_lyophilizer()
        with st.expander("💧 **Case Study: Validating a Nanoliter Liquid Dispenser**"):
            case_study_nanoliter_dispenser()
            
    with tab2:
        st.header("Assembly & QC Validation")
        with st.expander("⚙️ **Case Study: Validating a Biochip Assembly Line (OEE)**", expanded=True):
            case_study_biochip_assembly()
        with st.expander("🔬 **Case Study: Validating a Vision System for QC**"):
            case_study_vision_system()
        with st.expander("🔥 **Case Study: Validating Taping/Soldering of a Plastic Cassette**"):
            case_study_taping_soldering()
        with st.expander("⚡ **Case Study: Validating Electrostatic Charge Control**"):
            case_study_electrostatic_control()
        with st.expander("⚙️ **Case Study: Validating Process Stability (Zone Chart)**"):
            case_study_zone_chart()
        with st.expander("🔬 **Case Study: Validating Multivariate Process Control (T² Chart)**"):
            case_study_hotelling_t2()
        
        
    with tab3:
        st.header("System & Environmental Validation")
        with st.expander("🖥️ **Case Study: Computer System Validation (CSV)**", expanded=True):
            case_study_csv()
        with st.expander("🧼 **Case Study: Cleaning Validation (Multi-Product)**"):
            case_study_cleaning_validation()
        with st.expander("📦 **Case Study: Shipping Lane Performance Qualification**"):
            case_study_shipping()
        with st.expander("⚡ **Case Study: Validating Environmental Control (I-MR Chart)**"):
            case_study_advanced_imr()
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
        display_revalidation_planner(key="reval_planner")
    with tab2:
        st.subheader("Continuous Improvement (Kaizen) Initiative Tracker")
        st.info("**Context:** An effective validation program uses data to drive improvement. The **Deviation Trend** chart identifies operational problems (the 'why'), while the **ROI Tracker** provides the business case for funding solutions (the 'what for').")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_kaizen_roi_chart("kaizen_roi"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_deviation_trend_chart("deviation_trend"), use_container_width=True)
        st.success("**Actionable Insight:** The rising deviation trend in the Bioreactor Suite C directly validates the focus of our Kaizen efforts (e.g., 'Implement PAT Sensor'). The ROI tracker provides a strong business case to leadership for continuing these improvement projects.")
        with st.container(border=True):
            st.subheader("Assay Performance Monitoring (Levey-Jennings Chart)")
            case_study_levey_jennings()
        with st.container(border=True):
            st.subheader("Small-Shift Detection (EWMA Chart)")
            case_study_ewma_chart()
        with st.container(border=True):
            st.subheader("Rapid Shift Detection (CUSUM Chart)")
            case_study_cusum_chart()
        
def render_documentation_hub_page() -> None:
    st.title("🗂️ 6. Validation Documentation & Audit Defense Hub")
    render_manager_briefing(title="Orchestrating Compliant Validation Documentation", content="This hub demonstrates the ability to generate and manage the compliant, auditable documentation that forms the core of a successful validation package. The templates and simulations below prove expertise in creating documents that meet the stringent requirements of **21 CFR Part 820** and **ISO 13485**.", reg_refs="21 CFR 820.40 (Document Controls), GAMP 5 Good Documentation Practice, 21 CFR Part 11", business_impact="Ensures audit-proof documentation, accelerates review cycles by providing clear templates, and fosters seamless collaboration between Engineering, Manufacturing, Quality, and Regulatory.", quality_pillar="Good Documentation Practice (GDP) & Audit Readiness.", risk_mitigation="Minimizes review cycles and audit findings by ensuring documentation is attributable, legible, contemporaneous, original, and accurate (ALCOA+).")
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True): st.subheader("Document Approval Workflow"); st.info("Simulates the eQMS workflow."); st.markdown("Status for `VAL-MP-001_Project_Atlas`:"); st.divider(); st.markdown("✔️ **Validation Lead (Self):** Approved `2024-01-15`\n✔️ **Process Engineering Lead:** Approved `2024-01-16`\n✔️ **Quality Assurance Lead:** Approved `2024-01-17`\n🟠 **Manufacturing Lead:** Pending Review\n⬜ **Head of Engineering:** Not Started")
    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer"); st.info("The following are professionally rendered digital artifacts that simulate documents within a validated eQMS.")
            with st.expander("📄 **View Professional IQ/OQ Protocol Template**"):
                _render_professional_protocol_template()
            with st.expander("📋 **View Professional PQ Report Template**"):
                _render_professional_report_template()

# --- ENHANCED DOCUMENTATION HUB (REPLACE THE OLD SECTION WITH THIS) ---

def _render_professional_protocol_template() -> None:
    """Renders a world-class, professional IQ/OQ Protocol Template."""
    st.header("IQ/OQ Protocol: VAL-TP-101")
    st.subheader("Automated Bioreactor Suite (ASSET-123)")
    st.divider()
    
    st.markdown("##### 1.0 Purpose & Scope")
    st.write("The purpose of this protocol is to provide documented evidence that the Automated Bioreactor Suite (ASSET-123) is installed correctly per manufacturer and design specifications (Installation Qualification - IQ) and operates according to its functional specifications throughout its intended operating ranges (Operational Qualification - OQ).")
    st.info("**Compliance Focus (GDP):** A clear purpose and scope are essential for audit readiness, defining the boundaries and intent of the validation activity upfront.", icon="🧠")

    st.markdown("##### 2.0 System Description")
    st.write("This protocol applies to the Automated Bioreactor Suite (ASSET-123) located in Building X, Room Y. The system consists of a 500L stainless steel bioreactor, an integrated control system running 'BioCommand' software v2.1, and associated critical instrumentation (e.g., pH, DO, temperature sensors).")

    st.markdown("##### 3.0 Roles & Responsibilities")
    st.table(pd.DataFrame({
        'Role': ['Validation', 'Engineering', 'Manufacturing', 'Quality Assurance'],
        'Responsibility': ['Author, execute, and report on this protocol.', 'Provide technical support during execution.', 'Provide operational support and confirm system readiness.', 'Review and approve the protocol, deviations, and final report.']
    }))
    
    st.markdown("##### 4.0 Test Procedures - OQ Section (Example)")
    st.info("**ALCOA+ Principle:** Each test case includes fields for 'Executed By/Date' and 'Reviewed By/Date' to ensure all activities are Attributable, Contemporaneous, and Legible.", icon="✍️")
    test_case_data = {
        'Test ID': ['OQ-TC-001', 'OQ-TC-002', 'OQ-TC-003'],
        'Test Description': ['Verify Temperature Control Loop', 'Challenge Agitator Speed Control', 'Test Critical Alarms (High Temp)'],
        'Acceptance Criteria': ['Maintain setpoint ± 0.5°C for 60 mins', 'Maintain setpoint ± 2 RPM across range', 'Alarm activates within 5s of exceeding setpoint'],
        'Result (Pass/Fail)': ['PASS', 'PASS', 'PASS'],
        'Executed By / Date': ['A. Wong / 15-Jan-2024', 'A. Wong / 15-Jan-2024', 'A. Wong / 16-Jan-2024'],
        'Reviewed By / Date': ['J. Doe / 17-Jan-2024', 'J. Doe / 17-Jan-2024', 'J. Doe / 17-Jan-2024']
    }
    st.dataframe(style_dataframe(pd.DataFrame(test_case_data)), use_container_width=True)

    st.markdown("##### 5.0 Deviation Handling")
    st.write("Any discrepancy from the expected results or test procedure must be documented as a deviation. The deviation must be assessed for its impact on product quality and system suitability by a cross-functional team (Validation, QA, Engineering) before further execution. All deviations must be resolved or have a corrective action plan in place prior to the approval of the final report.")
    
    st.divider()
    st.warning("This is a simplified template for demonstration purposes.")

def _render_professional_report_template() -> None:
    """Renders a world-class, professional PQ Report Template."""
    st.header("PQ Report: VAL-TR-201")
    st.subheader("Automated Bioreactor Suite (ASSET-123)")
    st.divider()
    meta_cols = st.columns(4); meta_cols[0].metric("Document ID", "VAL-TR-201"); meta_cols[1].metric("Version", "1.0"); meta_cols[2].metric("Status", "Final"); meta_cols[3].metric("Approval Date", "2024-03-01"); st.divider()
    
    st.markdown("##### 1.0 Summary & Conclusion")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Three successful, consecutive Performance Qualification (PQ) runs were executed on the Bioreactor System per protocol VAL-TP-201. The results confirm that the system reliably produces product meeting all pre-defined Critical Quality Attributes (CQAs) under normal manufacturing conditions.")
        st.success("**Conclusion:** The Automated Bioreactor System (ASSET-123) has met all PQ acceptance criteria and is **qualified for use in commercial GMP manufacturing.**")
    with col2: st.metric("Overall Result", "PASS"); st.metric("Final CpK (Product Titer)", "1.67", help="Exceeds target of >= 1.33")
    
    st.markdown("##### 2.0 Deviations & Impact Assessment")
    st.info("**Compliance Focus (Audit Readiness):** A dedicated section for deviations demonstrates transparency and robust quality oversight. It shows auditors that unexpected events are controlled, assessed, and documented properly.", icon="🧠")
    with st.container(border=True):
        st.write("**DEV-001 (Run 2):** A pH sensor required recalibration mid-run. The event was documented in the batch record, the sensor was recalibrated per SOP, and the run successfully continued.")
        st.success("**Impact Assessment:** None. All CQA data for the batch remained within specification. The event and its resolution were reviewed and approved by QA.")
        
    st.markdown("##### 3.0 Results vs. Acceptance Criteria")
    results_data = {'CQA': ['Titer (g/L)', 'Viability (%)', 'Impurity A (%)'], 'Specification': ['>= 5.0', '>= 95%', '<= 0.5%'], 'Run 1 Result': [5.2, 97, 0.41], 'Run 2 Result': [5.1, 96, 0.44], 'Run 3 Result': [5.3, 98, 0.39], 'Pass/Fail': ['PASS', 'PASS', 'PASS']}
    results_df = pd.DataFrame(results_data)
    def style_pass_fail(val: str) -> str:
        color = SUCCESS_GREEN if val == 'PASS' else ERROR_RED
        return f"background-color: {color}; color: white; text-align: center; font-weight: bold;"
    styled_df = results_df.style.map(style_pass_fail, subset=['Pass/Fail'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("##### 4.0 Traceability")
    st.warning("This report provides the objective evidence that fulfills user requirements **URS-001** (Titer) and **URS-040** (Purity) as documented in the Requirements Traceability Matrix (QA-DOC-105). This closes the loop on the V-Model.")

def render_documentation_hub_page() -> None:
    st.title("🗂️ 6. Validation Documentation & Audit Defense Hub")
    render_manager_briefing(title="Orchestrating Compliant Validation Documentation", content="This hub demonstrates the ability to generate, manage, and defend the compliant, auditable documentation that forms the core of a successful validation package. The simulations below prove expertise in creating documents that meet the stringent requirements of **21 CFR Part 820** and **ISO 13485**.", reg_refs="21 CFR 820.40 (Document Controls), GAMP 5 Good Documentation Practice, 21 CFR Part 11", business_impact="Ensures audit-proof documentation, accelerates review cycles, and fosters seamless collaboration between Engineering, Manufacturing, and Quality.", quality_pillar="Good Documentation Practice (GDP) & Audit Readiness.", risk_mitigation="Minimizes review cycles and audit findings by ensuring documentation is attributable, legible, contemporaneous, original, and accurate (ALCOA+).")

    tab1, tab2, tab3 = st.tabs(["📄 Document Generation Hub", "🔄 eQMS Approval Workflow Simulation", "🛡️ Interactive Audit Defense Simulation"])

    with tab1:
        st.subheader("Compliant Document Templates")
        st.info("These interactive templates showcase the structure and key compliance elements of core validation deliverables, serving as a best-practice guide for the team.")
        with st.expander("📄 **View Professional IQ/OQ Protocol Template**"):
            _render_professional_protocol_template()
        with st.expander("📋 **View Professional PQ Report Template**"):
            _render_professional_report_template()

    with tab2:
        st.subheader("eQMS Document Workflow")
        st.info("This simulates tracking a document through its cross-functional review and approval lifecycle within an electronic Quality Management System.")
        
        doc_choice = st.selectbox("Select a Document to View its Workflow Status:",
                                  ['VAL-MP-001 (Validation Master Plan)', 'VAL-TP-101 (IQ/OQ Protocol)', 'VAL-TR-101 (IQ/OQ Report)'])
        
        st.markdown(f"#### Status for `{doc_choice.split(' ')[0]}`:")
        st.divider()
        
        if "MP" in doc_choice:
            st.markdown("✔️ **Validation Lead (Self):** Approved `2024-01-15`\n✔️ **Process Engineering Lead:** Approved `2024-01-16`\n✔️ **Quality Assurance Lead:** Approved `2024-01-17`\n🟠 **Manufacturing Lead:** Pending Review\n⬜ **Head of Engineering:** Not Started")
        elif "TP" in doc_choice:
            st.markdown("✔️ **Validation Lead (Self):** Approved `2024-02-01`\n✔️ **Process Engineering Lead:** Approved `2024-02-02`\n✔️ **Quality Assurance Lead:** Approved `2024-02-05`\n✔️ **Manufacturing Lead:** Approved `2024-02-05`\n✅ **STATUS: RELEASED FOR EXECUTION**")
        elif "TR" in doc_choice:
            st.markdown("✔️ **Validation Lead (Self):** Approved `2024-03-01`\n✔️ **Process Engineering Lead:** Approved `2024-03-01`\n✔️ **Quality Assurance Lead:** Approved `2024-03-02`\n✔️ **Manufacturing Lead:** Approved `2024-03-02`\n✅ **STATUS: FINAL & ARCHIVED**")

    with tab3:
        st.subheader("Audit Defense Simulation")
        st.info("This interactive module simulates how to respond to common, challenging questions from a regulatory auditor, demonstrating deep SME knowledge and audit readiness.")
        
        questions = {
            "Select a question...": "Select a question from the dropdown to see a world-class response.",
            "How did you justify your PQ sample size?": """
            **Answer:** "That's an excellent question. We justified our sample size for the PQ runs using a risk-based statistical approach. For the critical quality attribute of Product Titer, we performed a **Binomial Power Analysis**. 
            
            Our goal was to demonstrate with 95% confidence that our process can produce product with a defect rate of less than 1%. The analysis, documented in our Validation Plan VAL-MP-001, determined that three successful runs of 100 samples each would provide the necessary statistical power to meet this acceptance criterion. This approach aligns with **ICH Q8** principles for process understanding and is detailed in our internal SOP-STAT-005."
            """,
            "There was a deviation in PQ Run 2. How can you be sure the system is robust?": """
            **Answer:** "Correct, Deviation DEV-001 was documented in the PQ report, VAL-TR-201, Section 2.0. The deviation was a pH sensor recalibration, which is a routine maintenance activity.
            
            Critically, the event was immediately bracketed, and an impact assessment was performed, as per our procedure SOP-QA-021. We demonstrated that **1)** no product was affected as all CQA data remained well within specification, and **2)** the control system correctly placed the process in a safe state until the issue was resolved.
            
            Therefore, this event actually serves as a successful challenge to the system's robustness and procedural controls, rather than a concern. It proved our deviation management and maintenance procedures function as intended, which is a key requirement of **21 CFR 820.75**."
            """,
            "How do you ensure data integrity for the electronic records generated by this system?": """
            **Answer:** "We ensure data integrity through a multi-layered approach, as required by **21 CFR Part 11**.
            
            First, during IQ, we verified that the system's security settings were configured correctly, including unique user logins and password controls.
            
            Second, during OQ, we specifically challenged the system's audit trail functionality. We demonstrated that all critical actions—such as changing a setpoint, acknowledging an alarm, or creating a batch—are captured in a secure, time-stamped, and unalterable audit log.
            
            Finally, our procedural controls, like SOP-QA-033 on 'Periodic Review of Audit Trails', ensure that these records are reviewed by QA on a routine basis to detect any anomalies. This combination of technical and procedural controls provides a robust data integrity framework."
            """
        }
        
        question = st.selectbox("Select an Auditor's Question:", list(questions.keys()))
        
        if question != "Select a question...":
            st.error(f"**Auditor's Question:** \"{question}\"", icon="❓")
            st.success(f"**SME Response:** {questions[question]}", icon="✔️")

# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
PAGES = { "Executive Summary": render_main_page, "1. Strategic Management": render_strategic_management_page, "2. Project & Portfolio Management": render_project_portfolio_page, "3. E2E Validation Walkthrough": render_e2e_validation_hub_page, "4. Specialized Validation Hubs": render_specialized_validation_page, "5. Validation Program Health": render_validation_program_health_page, "6. Documentation & Audit Defense": render_documentation_hub_page }
st.sidebar.title("🛠️ Validation Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()
