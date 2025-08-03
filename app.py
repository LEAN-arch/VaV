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
    using clean, content-aware sized boxes.
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
        font_size = 11
        
        fig.add_shape(
            type="rect", 
            x0=x_c-w, y0=y_c-h, x1=x_c+w, y1=y_c+h, 
            line=dict(color="Black"), 
            fillcolor=node['color'], 
            opacity=0.95, 
            layer="below",
            cornerradius=20 if shape_type == 'terminator' else 0 
        )
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

def plot_cpk_analysis(key: str) -> go.Figure:
    rng = np.random.default_rng(42); data = rng.normal(loc=5.15, scale=0.05, size=100); LSL, USL = 5.0, 5.3; mu, std = np.mean(data), np.std(data, ddof=1)
    cpk = min((USL - mu) / (3 * std), (mu - LSL) / (3 * std)); fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=20, name='Observed Data', histnorm='probability density', marker_color=PRIMARY_COLOR, opacity=0.7))
    x_fit = np.linspace(min(data), max(data), 200); y_fit = norm.pdf(x_fit, mu, std)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Normal Distribution', line=dict(color=SUCCESS_GREEN, width=2)))
    fig.add_vline(x=LSL, line_dash="dash", line_color=ERROR_RED, annotation_text="LSL", annotation_position="top left"); fig.add_vline(x=USL, line_dash="dash", line_color=ERROR_RED, annotation_text="USL", annotation_position="top right"); fig.add_vline(x=mu, line_dash="dot", line_color=NEUTRAL_GREY, annotation_text=f"Mean={mu:.2f}", annotation_position="bottom right")
    fig.update_layout(title_text=f'Process Capability (Cpk) Analysis - Titer (g/L)<br><b>Cpk = {cpk:.2f}</b> (Target: ≥1.33)', xaxis_title="Titer (g/L)", yaxis_title="Density", showlegend=False, title_font_size=20, title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
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
    alerts = []
    if any(df['Titer'] > ucl) or any(df['Titer'] < lcl): alerts.append("Rule 1 Violation: A data point has exceeded the control limits.")
    for i in range(len(df) - 8):
        if all(df['Titer'][i:i+9] > mean) or all(df['Titer'][i:i+9] < mean):
            alerts.append("Rule 2 Violation: A run of 9 points on one side of the mean detected (process shift)."); break
    if len(df)>2 and df['Titer'].iloc[-1] > mean + (2 * (ucl-mean)/3) and df['Titer'].iloc[-2] > mean + (2 * (ucl-mean)/3): alerts.append("Rule 3 (Simulated) Violation: Two of three consecutive points are in Zone A (potential loss of control).")
    return alerts

def plot_process_stability_chart(key: str) -> Tuple[go.Figure, list]:
    rng = np.random.default_rng(22); data = rng.normal(5.2, 0.25, 25); data[15:] = data[15:] + 0.3 
    df = pd.DataFrame({'Titer': data}); df['MR'] = df['Titer'].diff().abs()
    I_CL = df['Titer'].mean(); MR_CL = df['MR'].mean(); I_UCL = I_CL + 2.66 * MR_CL; I_LCL = I_CL - 2.66 * MR_CL; MR_UCL = 3.267 * MR_CL
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Individuals (I) Chart</b>", "<b>Moving Range (MR) Chart</b>"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Titer'], name='Titer (g/L)', mode='lines+markers', marker_color=PRIMARY_COLOR), row=1, col=1)
    fig.add_hline(y=I_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=1, col=1, annotation_text="CL"); fig.add_hline(y=I_UCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="UCL"); fig.add_hline(y=I_LCL, line_dash="dot", line_color=ERROR_RED, row=1, col=1, annotation_text="LCL")
    fig.add_trace(go.Scatter(x=df.index, y=df['MR'], name='Moving Range', mode='lines+markers', marker_color=WARNING_AMBER), row=2, col=1)
    fig.add_hline(y=MR_CL, line_dash="dash", line_color=SUCCESS_GREEN, row=2, col=1, annotation_text="CL"); fig.add_hline(y=MR_UCL, line_dash="dot", line_color=ERROR_RED, row=2, col=1, annotation_text="UCL")
    fig.update_layout(height=400, showlegend=False, title_text="<b>Process Stability (I-MR Chart) for PQ Run 1 Titer</b>", title_x=0.5, plot_bgcolor=BACKGROUND_GREY)
    alerts = analyze_spc_rules(df, I_UCL, I_LCL, I_CL)
    return fig, alerts

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
    st.title("🤖 Automated Equipment Validation Portfolio"); st.subheader("A Live Demonstration of Modern Validation Management Leadership"); st.divider()
    st.markdown("Welcome. This interactive environment provides **undeniable proof of expertise in the end-to-end validation of automated manufacturing equipment** in a strictly regulated GMP environment. It simulates how an effective leader manages a validation function, with a relentless focus on aligning technical execution, **Quality Systems (per 21 CFR 820 & ISO 13485)**, and strategic capital projects.")
    
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
        st.success("**Actionable Insight:** The CoQ model proves that for every **$1 spent on proactive validation**, we prevent an estimated **$4 in failure costs** (rework, deviations, batch loss). This data provides a powerful justification for our departmental budget and headcount.")
    
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
    with col2:
        st.header("Phases 2-4: Execution & Qualification"); st.info("The 'right side of the V-Model' focuses on generating objective evidence.")
        st.subheader("Phase 2: Factory & Site Acceptance Testing", divider='blue')
        with st.container(border=True): st.markdown("Purpose: To execute acceptance testing at the vendor's facility (FAT) and our site (SAT). The goal is to catch as many issues as possible *before* formal qualification begins."); display_fat_sat_summary("fat_sat")
        st.subheader("Phase 3: Installation & Operational Qualification", divider='blue')
        with st.container(border=True): st.markdown("Purpose: The IQ provides documented evidence of correct installation. The OQ challenges the equipment's functions to prove it operates as intended throughout its specified operating ranges."); st.plotly_chart(plot_oq_challenge_results("oq_plot"), use_container_width=True)
        st.subheader("Phase 4: Performance Qualification", divider='blue')
        with st.container(border=True):
            st.markdown("Purpose: The PQ is the final step, providing documented evidence that the equipment can consistently produce quality product under normal, real-world manufacturing conditions.")
            c1, c2 = st.columns(2)
            with c1: st.subheader("Process Capability"); st.plotly_chart(plot_cpk_analysis("pq_cpk"), use_container_width=True)
            with c2: 
                st.subheader("Process Stability")
                spc_fig, spc_alerts = plot_process_stability_chart("pq_spc")
                st.plotly_chart(spc_fig, use_container_width=True)
                if spc_alerts:
                    st.error(f"**🚨 Automated SPC Alert Detected:** {spc_alerts[0]}")
                    st.success("**Actionable Insight:** The automated rule check has detected a process shift. This would trigger an immediate investigation with Process Engineering to identify the root cause before qualifying the equipment.")


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
