import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import folium
from streamlit_folium import st_folium
import json
# joblib removed - using HardcodedScaler instead
from pathlib import Path
import datetime

# --- Premium UI Custom CSS ---
st.set_page_config(page_title="AI Epidemiological Warning", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
    /* Premium Typography */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    
    /* Light Professional Theme (HPSC Style) */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    .stMarkdown, .stText, p {
        color: #334155 !important;
    }
    
    /* Fix for faint label colors in Streamlit Light Mode */
    label, .stSelectbox label, .stSlider label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    h1, h2, h3 {
        color: #0f766e !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    
    .metric-card h4 {
        color: #1e40af !important;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* Custom Info Box */
    .scenario-info-box {
        background-color: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 4px;
        color: #0369a1;
        font-weight: 500;
        font-size: 0.9em;
        margin-top: 10px;
    }
    
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Model Definitions ---
class EarlyWarningGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.4):
        super(EarlyWarningGRU, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# --- Config & Data Loading ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "secondary_data" / "model_input" / "feature_matrix_hse.csv"
GEOJSON_FILE = BASE_DIR / "secondary_data" / "spatial" / "hse_regions.geojson"
MODEL_PATH = BASE_DIR / "secondary_data" / "models" / "gru_best.pt"
# SCALER_PATH removed - using HardcodedScaler instead

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_geojson():
    with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_model(input_size, disease_name):
    model = EarlyWarningGRU(input_size=input_size)
    safe_disease = disease_name.replace(" ", "_").replace("/", "_")
    model_path = BASE_DIR / "secondary_data" / "models" / f"gru_best_{safe_disease}.pt"
    
    # Fallback if specific model doesn't exist (e.g. for some reason), fallback to original gru_best.pt
    if not model_path.exists():
        model_path = BASE_DIR / "secondary_data" / "models" / "gru_best.pt"
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

class HardcodedScaler:
    def __init__(self):
        self.scale_ = np.array([0.07413394142653074, 0.010590563060932963, 0.07413394142653074, 0.010590563060932963, 0.07413394142653074, 0.010590563060932963, 0.07413394142653074, 0.010590563060932963, 0.07413394142653074, 0.010590563060932963, 0.05655613533075239, 5.851415833652605, 0.2393980848153215, 0.0431486161622388, 0.05655613533075239, 5.851415833652605, 0.2393980848153215, 0.0431486161622388, 0.05655613533075239, 5.851415833652605, 0.2393980848153215, 0.0431486161622388, 0.05655613533075239, 5.851415833652605, 0.2393980848153215, 0.0431486161622388, 0.05655613533075239, 5.851415833652605, 0.2393980848153215, 0.0431486161622388, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.min_ = np.array([-1.8431090409904525e-05, -1.843109040991088e-05, -1.8431090409904525e-05, -1.843109040991088e-05, -1.8431090409904525e-05, -1.843109040991088e-05, -1.8431090409904525e-05, -1.843109040991088e-05, -1.8431090409904525e-05, -1.843109040991088e-05, -0.1799360377040902, -1.4562014558879863, -0.0506155950752394, -0.05029895826912409, -0.1799360377040902, -1.4562014558879863, -0.0506155950752394, -0.05029895826912409, -0.1799360377040902, -1.4562014558879863, -0.0506155950752394, -0.05029895826912409, -0.1799360377040902, -1.4562014558879863, -0.0506155950752394, -0.05029895826912409, -0.1799360377040902, -1.4562014558879863, -0.0506155950752394, -0.05029895826912409, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return X * self.scale_ + self.min_

@st.cache_resource
def load_scaler_v2():
    return HardcodedScaler()


def prepare_features(df):
    df_encoded = pd.get_dummies(df, columns=['HSE_Region'], prefix='Region')
    region_cols = [c for c in df_encoded.columns if c.startswith('Region_')]
    feature_cols = [
        'weekly_mean_rain', 'weekly_sum_rain',
        'rain_mean_lag1', 'rain_sum_lag1',
        'rain_mean_lag2', 'rain_sum_lag2',
        'rain_mean_lag3', 'rain_sum_lag3',
        'rain_mean_lag4', 'rain_sum_lag4',
        'sin_month', 'cos_month',
        'temperature', 'soil_moisture', 'et0', 'radiation',
        'temp_lag1', 'temp_lag2', 'temp_lag3', 'temp_lag4',
        'soil_lag1', 'soil_lag2', 'soil_lag3', 'soil_lag4',
        'et0_lag1', 'et0_lag2', 'et0_lag3', 'et0_lag4',
        'rad_lag1', 'rad_lag2', 'rad_lag3', 'rad_lag4'
    ] + region_cols
    return df_encoded, feature_cols, region_cols

# --- Main UI ---
st.markdown("<h1>AI Waterborne Disease Early Warning System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1em; color: #64748b; margin-bottom: 5px;'>Advanced Deep Learning (GRU) Spatial-Temporal Forecasting for Environmental Pathogens</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.0em; color: #0f766e; font-weight: 500; margin-top: 0px;'>Lead Researcher & Developer: <strong>Elleigh Jiyu Kim</strong> (5th Year, The Institute of Education)</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

df_raw = load_data()
df_encoded, feature_cols, region_cols = prepare_features(df_raw)
scaler = load_scaler_v2()
geojson_data = load_geojson()

DISEASES = ['Cryptosporidiosis', 'Giardiasis', 'Salmonellosis', 'Shigellosis', 'Verotoxigenic Escherichia coli infection']
selected_disease = st.selectbox("Select Target Disease", DISEASES)

# Dynamically load the model for the selected disease
model = load_model(input_size=len(feature_cols), disease_name=selected_disease)

# --- Sidebar Controls ---
st.sidebar.markdown("### 1. Temporal Anchor (Current Week 't')")
st.sidebar.markdown("<p style='font-size: 0.85em; color: #94a3b8;'>Select the 'present' week. The AI uses the 8 weeks leading up to this date to forecast outbreaks 2 weeks into the future (t+2).</p>", unsafe_allow_html=True)

available_dates_str = sorted(df_raw['date'].dt.strftime('%Y-%m-%d').unique())
default_date_str = '2023-09-04'
if default_date_str not in available_dates_str:
    default_date_str = available_dates_str[-1]

selected_date_str = st.sidebar.select_slider(
    "Select Current Week (t)",
    options=available_dates_str,
    value=default_date_str
)
selected_date = pd.to_datetime(selected_date_str)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Climate Simulation (Scenario Modeling)")
st.sidebar.markdown("<p style='font-size: 0.85em; color: #94a3b8;'>Inject a hypothetical extreme weather event into the current week (t) to observe causality and trigger future warnings.</p>", unsafe_allow_html=True)

rain_delta_mm = st.sidebar.slider("Extreme Rainfall Addition (mm/week)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
if rain_delta_mm > 0:
    st.sidebar.markdown(f"""
    <div class="scenario-info-box">
        <strong>Simulated Event:</strong><br>
        +{rain_delta_mm} mm of rain injected across all regions this week.
    </div>
    """, unsafe_allow_html=True)

# --- Prediction Logic ---
SEQ_LEN = 8
predictions = []

regions = df_raw['HSE_Region'].unique()
for region in regions:
    region_mask = df_raw['HSE_Region'] == region
    region_data = df_encoded[region_mask].copy()
    region_data = region_data.sort_values('date')
    
    past_data = region_data[region_data['date'] <= selected_date]
    
    if len(past_data) < SEQ_LEN:
        predictions.append({'HSE_Region': region, 'Risk_Prob': 0, 'Status': 'Insufficient Data', 'Base_Rain': 0})
        continue
        
    seq_df = past_data.tail(SEQ_LEN).copy()
    base_rain = seq_df.iloc[-1]['weekly_sum_rain']
    
    # Apply Simulation
    if rain_delta_mm > 0:
        seq_df.iloc[-1, seq_df.columns.get_loc('weekly_sum_rain')] += rain_delta_mm
        # Proportionally adjust mean rain (assuming constant rainy days)
        ratio = (base_rain + rain_delta_mm) / (base_rain + 1e-6)
        seq_df.iloc[-1, seq_df.columns.get_loc('weekly_mean_rain')] *= ratio
    
    X_raw = seq_df[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.sigmoid(logits).item()
        
    predictions.append({'HSE_Region': region, 'Risk_Prob': prob, 'Base_Rain': base_rain})

pred_df = pd.DataFrame(predictions)
target_forecast_date = selected_date + pd.Timedelta(days=14)

# --- Layout ---
col_map, col_info = st.columns([5, 5])

with col_map:
    st.markdown(f"### Live Forecast Map (Target: {target_forecast_date.strftime('%Y-%m-%d')})")
    
    m = folium.Map(location=[53.3, -7.7], zoom_start=6, tiles="CartoDB positron")
    
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        data=pred_df,
        columns=['HSE_Region', 'Risk_Prob'],
        key_on='feature.properties.HSE_Region',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name='Outbreak Probability (%)'
    ).add_to(m)
    
    for feature in geojson_data['features']:
        reg_name = feature['properties']['HSE_Region']
        prob = pred_df[pred_df['HSE_Region'] == reg_name]['Risk_Prob'].values
        base_rain = pred_df[pred_df['HSE_Region'] == reg_name]['Base_Rain'].values
        if len(prob) > 0:
            feature['properties']['Risk_Prob'] = f"{prob[0]*100:.1f}%"
            feature['properties']['Rainfall'] = f"{base_rain[0]:.1f} mm (+{rain_delta_mm} mm sim)"
        else:
            feature['properties']['Risk_Prob'] = "N/A"
            feature['properties']['Rainfall'] = "N/A"

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': '#ffffff',
            'color': '#38bdf8',
            'weight': 1.5,
            'fillOpacity': 0.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['HSE_Region', 'Rainfall', 'Risk_Prob'],
            aliases=['Region:', 'Current Rain:', 't+2 Outbreak Risk:'],
            style=("background-color: #1e293b; color: #f8fafc; font-family: Inter; font-size: 13px; padding: 10px; border-radius: 8px; border: 1px solid #334155;")
        )
    ).add_to(m)
    st_folium(m, width="100%", height=550, returned_objects=[])

with col_info:
    st.markdown("### Regional Risk Breakdown")
    
    # Sort by risk
    pred_df = pred_df.sort_values('Risk_Prob', ascending=False).reset_index(drop=True)
    
    for _, row in pred_df.iterrows():
        risk_level = "High" if row['Risk_Prob'] > 0.5 else "Medium" if row['Risk_Prob'] > 0.2 else "Low"
        color = "#ef4444" if risk_level == "High" else "#f59e0b" if risk_level == "Medium" else "#10b981"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <h4 style="margin:0; padding:0;">{row['HSE_Region']}</h4>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span>Probability: <strong style="color:{color};">{row['Risk_Prob']*100:.1f}%</strong></span>
                <span style="color:#94a3b8; font-size:0.9em;">Total Rain: {row['Base_Rain'] + rain_delta_mm:.1f} mm</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("## About the AI Early Warning System")

col_faq1, col_faq2 = st.columns(2)

with col_faq1:
    st.markdown("""
    <div class="metric-card">
        <h4>Q1. How does the Live Forecasting work?</h4>
        <p>The AI model ingests the <strong>cumulative meteorological sequence of the past 8 weeks</strong> to forecast the outbreak risk <strong>2 weeks into the future (t+2)</strong>. 
        This 2-week horizon is biologically calibrated to the typical incubation and reporting delay of waterborne pathogens like <i>Cryptosporidium</i>. 
        By adjusting the <strong>'Extreme Rainfall Addition'</strong> slider, you are simulating a live meteorological anomaly occurring this week, allowing health authorities to deploy preventative warnings up to half a month before the clinical cases peak.</p>
    </div>
    """, unsafe_allow_html=True)

with col_faq2:
    st.markdown("""
    <div class="metric-card">
        <h4>Q2. Were confounding variables (e.g., sanitation, population density) controlled?</h4>
        <p>Yes. While directly collecting and merging demographic data is one approach, this system employs a mathematically rigorous econometric technique known as <strong>Region-Specific Fixed Effects (One-Hot Encoding)</strong> embedded directly into the Deep Learning architecture. 
        Because variables like infrastructure quality, agricultural intensity, and baseline population demographics are <strong>Time-Invariant</strong> (they do not change week-to-week), the Neural Network learns a unique <i>Baseline Risk Intercept</i> for each of the 6 HSE Regions. This effectively absorbs and controls for all unobserved spatial confounders, successfully preventing the Ecological Fallacy without requiring external census datasets.</p>
    </div>
    """, unsafe_allow_html=True)
