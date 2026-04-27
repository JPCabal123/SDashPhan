import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ==========================================
# 1. PAGE CONFIG & DARK THEME
# ==========================================
st.set_page_config(page_title="Radiological Phantom Explorer", layout="wide")

# Custom CSS mimicking the layout in the provided images
st.markdown("""
    <style>
    .stApp { background-color: #0F1116; color: white; }
    .kpi-container { display: flex; gap: 20px; margin-bottom: 20px; }
    .kpi-box { flex: 1; background-color: #1A1D24; padding: 15px; border-radius: 5px; border: 1px solid #2A2D35; }
    .kpi-title { font-size: 10px; color: #8A8D93; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { font-size: 24px; color: #4A90E2; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA WORKFLOW: CALIBRATION & INTERPOLATION
# ==========================================
@st.cache_data
def process_phantom_data():
    # Load dataset
    df = pd.read_csv('Results2.csv')
    
    # Clean and extract columns
    df_clean = df[['Material', 'Thickness (mm)', 'Mean']].copy()
    df_clean['Thickness (mm)'] = pd.to_numeric(df_clean['Thickness (mm)'], errors='coerce')
    df_clean['Mean'] = pd.to_numeric(df_clean['Mean'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Material', 'Thickness (mm)', 'Mean'])

    # 1. Calibration using User's References
    I_air = 0.812  # Air-Background
    I_bone = 0.476 # HAAP-Bone Equivalent

    # Calculate slope and intercept for HU calibration
    slope = 2000 / (I_bone - I_air)
    intercept = 1000 - slope * I_bone

    # Convert Mean to HU
    df_clean['HU'] = slope * df_clean['Mean'] + intercept

    # Create interpolation models
    materials = [m for m in df_clean['Material'].unique() if m and 'Air' not in m and 'Bone' not in m]
    interp_models = {}
    
    for mat in materials:
        # Group by thickness to get mean HU per thickness
        mat_data = df_clean[df_clean['Material'] == mat].groupby('Thickness (mm)')['HU'].mean().reset_index()
        mat_data = mat_data.dropna()
        
        if len(mat_data) >= 2:
            x = mat_data['Thickness (mm)'].values
            y = mat_data['HU'].values
            
            # Ensure x is strictly increasing for interp1d
            sort_idx = np.argsort(x)
            x_sorted, y_sorted = x[sort_idx], y[sort_idx]
            
            # Use quadratic interpolation for smooth curves if we have >= 3 points, else linear
            interp_kind = 'quadratic' if len(x_sorted) >= 3 else 'linear'
            interp_models[mat] = interp1d(x_sorted, y_sorted, kind=interp_kind, fill_value='extrapolate')

    return df_clean, interp_models, I_air

df_processed, material_models, i0_ref = process_phantom_data()
available_materials = list(material_models.keys())

# Define clinical targets globally so all charts can use them
clinical_targets = {"Cancellous Bone": 500, "Soft Tissue": 50, "Fat": -100, "Lung": -700}

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Explorer Settings")

def multiselect_with_all(label, options):
    """Helper function to add Select All/Deselect All logic to multiselect."""
    display_options = ["Select All", "Deselect All"] + options
    selected = st.sidebar.multiselect(label, display_options, default=["Select All"])
    
    if "Deselect All" in selected:
        return []
    elif "Select All" in selected:
        return options
    else:
        return [opt for opt in selected if opt not in ["Select All", "Deselect All"]]

selected_mats = multiselect_with_all("Active Materials", available_materials)

st.sidebar.divider()

mat_focus_list = multiselect_with_all("Simulate Specific Material(s)", selected_mats if selected_mats else available_materials)
sim_thick = st.sidebar.slider("Simulated Thickness (mm)", 0.0, 40.0, 10.0)

# Filter dataset to selected materials
filtered_df = df_processed[df_processed['Material'].isin(selected_mats)]
max_hu = int(filtered_df['HU'].max()) if not filtered_df.empty else 0

# ==========================================
# 4. DASHBOARD HEADER & KPIs
# ==========================================
st.title("Radiological Phantom Explorer")

st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box">
            <div class="kpi-title">Active Materials</div>
            <div class="kpi-value">{len(selected_mats)}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Air Background (I0)</div>
            <div class="kpi-value">{i0_ref:.3f}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Max HU Estimate</div>
            <div class="kpi-value">{max_hu} HU</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 5. VISUALIZATION LAYOUT
# ==========================================
col_left, col_right = st.columns([1, 1])

# --- TOP LEFT: SMOOTH CURVES (TARGET HU vs THICKNESS) ---
with col_left:
    st.markdown("#### Target HU vs Thickness")
    fig_curve = go.Figure()
    
    # Add clinical reference dashed lines
    for t_name, t_hu in clinical_targets.items():
        fig_curve.add_hline(y=t_hu, line_dash="dash", line_color="gray", opacity=0.5, 
                             annotation_text=t_name, annotation_position="top left", 
                             annotation_font=dict(color="gray", size=10))

    for mat in selected_mats:
        if mat in material_models:
            mat_raw = filtered_df[filtered_df['Material'] == mat]
            mat_scatter = mat_raw.groupby('Thickness (mm)')['HU'].mean().reset_index()
            
            # 1. Plot the continuous smooth extrapolated line
            x_max = mat_scatter['Thickness (mm)'].max()
            x_plot = np.linspace(0, x_max + 10, 100) # Extrapolate a bit past max
            y_plot = material_models[mat](x_plot)
            
            fig_curve.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f"{mat}", 
                                           line=dict(width=3), line_shape='spline'))
            
            # 2. Plot the distinct marker points on top
            fig_curve.add_trace(go.Scatter(x=mat_scatter['Thickness (mm)'], y=mat_scatter['HU'], 
                                           mode='markers', name=f"{mat} Data", 
                                           marker=dict(size=8, line=dict(width=1, color='black')), showlegend=False))

    fig_curve.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Thickness (mm)", yaxis_title="Target HU",
                            yaxis=dict(range=[-1200, 1600], gridcolor='#2A2D35'), 
                            xaxis=dict(gridcolor='#2A2D35'), height=450)
    st.plotly_chart(fig_curve, use_container_width=True)

# --- TOP RIGHT: TISSUE EQUIVALENCY ANALYSIS ---
with col_right:
    st.markdown("#### Tissue Equivalency Analysis")
    fig_tissue = go.Figure()
    
    # Add clinical reference dashed lines
    for t_name, t_hu in clinical_targets.items():
        fig_tissue.add_hline(y=t_hu, line_dash="dash", line_color="white", opacity=0.7, 
                             annotation_text=t_name, annotation_position="top right", 
                             annotation_font=dict(color="white", size=11))
    
    # Add dynamic markers for ALL simulated selections
    for m_focus in mat_focus_list:
        if m_focus in material_models:
            sim_hu = material_models[m_focus](sim_thick)
            fig_tissue.add_trace(go.Scatter(
                x=[m_focus], y=[sim_hu], mode="markers+text", 
                text=[f"{int(sim_hu)} HU"], textposition="bottom center",
                marker=dict(size=18, symbol="diamond"),
                name=m_focus
            ))

    fig_tissue.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                             yaxis=dict(range=[-1200, 1600], gridcolor='#2A2D35'), 
                             showlegend=False, height=450)
    st.plotly_chart(fig_tissue, use_container_width=True)

# --- MIDDLE: GLOBAL RADIODENSITY INDEX ---
st.divider()
st.markdown("#### Global Radiodensity Index (All Materials)")
st.markdown("<span style='font-size: 12px; color: #888;'>Cumulative HU Performance</span>", unsafe_allow_html=True)

fig_global = go.Figure()

# Add clinical reference dashed lines
for t_name, t_hu in clinical_targets.items():
    fig_global.add_hline(y=t_hu, line_dash="dash", line_color="gray", opacity=0.5, 
                         annotation_text=t_name, annotation_position="top left", 
                         annotation_font=dict(color="gray", size=10))

for mat in selected_mats:
    if mat in material_models:
        x_plot = np.linspace(5, 85, 25) 
        y_plot = material_models[mat](x_plot)
        fig_global.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines+markers', name=mat, 
                                        line_shape='spline', marker=dict(size=4)))

fig_global.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         xaxis_title="Thickness (mm)", yaxis_title="HU",
                         xaxis=dict(gridcolor='#2A2D35'), yaxis=dict(gridcolor='#2A2D35'), height=350)
st.plotly_chart(fig_global, use_container_width=True)

# --- BOTTOM: REQUIRED THICKNESS BAR CHART ---
st.divider()
st.markdown("#### Required Thickness per Target Tissue")
st.markdown("<span style='font-size: 12px; color: #888;'>Interpolated material thickness to match clinical Hounsfield Units</span>", unsafe_allow_html=True)

# Added Dense Bone back in just for the required thickness calculation
targets_for_bar = {**clinical_targets, "Dense Bone": 1000}
interp_res = []

for mat in selected_mats:
    if mat in material_models:
        x_fine = np.linspace(0, 100, 1000)
        y_hu_fine = material_models[mat](x_fine)
        
        for t_name, t_hu in targets_for_bar.items():
            # np.interp requires strictly increasing x-values
            if y_hu_fine[0] > y_hu_fine[-1]:
                req_t = np.interp(t_hu, y_hu_fine[::-1], x_fine[::-1])
            else:
                req_t = np.interp(t_hu, y_hu_fine, x_fine)
            
            # Filter out crazy extrapolations
            if 0 < req_t <= 100:
                interp_res.append({"Material": mat, "Tissue": t_name, "Thickness (mm)": req_t})

if interp_res:
    df_bar = pd.DataFrame(interp_res)
    # Order the X-axis logically by tissue density
    tissue_order = ["Air (Lung)", "Fat", "Soft Tissue", "Cancellous Bone", "Dense Bone"]
    
    fig_bar = px.bar(df_bar, x="Tissue", y="Thickness (mm)", color="Material", 
                     barmode="group", text_auto='.1f', category_orders={"Tissue": tissue_order})
    
    fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(gridcolor='#2A2D35'), yaxis=dict(gridcolor='#2A2D35'), 
                          legend_title="", height=450)
    fig_bar.update_traces(textposition='outside', textfont_color='white')
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No materials reached the clinical target HUs within the 0-100mm interpolation bounds.")