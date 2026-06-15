import streamlit as st
from src.ui.dashboard import render_dashboard
from src.ui.predict import render_predict

# 1. Layout & Structure
st.set_page_config(page_title="ChurnInspect AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Sidebar and Light Main Panel
st.markdown("""
<style>
    /* Dark Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E2E;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    
    /* Main Panel Light Theme */
    .stApp {
        background-color: #F8F9FA;
        color: #333333;
    }
    
    /* Global Component Styling for White/Light mode */
    [data-testid="stHeader"] {
        background-color: #F8F9FA;
    }
    .css-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E9ECEF;
    }
    
    /* Metrics Styling */
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #2B2B2B;
        margin: 0;
    }
    .metric-label {
        font-size: 14px;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-indicator-up { color: #28A745; font-size: 13px; font-weight: 600;}
    .metric-indicator-down { color: #DC3545; font-size: 13px; font-weight: 600;}
    
    /* Badge & Alert */
    .badge-orange {
        background-color: #FFF3CD;
        color: #856404;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .recommendation-box {
        background-color: #F8F9FA;
        border-left: 4px solid #007BFF;
        padding: 15px;
        border-radius: 4px;
        margin-top: 15px;
    }
    
    /* Main Panel Default Text colors fix */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label {
        color: #333333 !important;
    }
    /* Exclude sidebar text from becoming black */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Change sidebar radio select color */
    .stRadio > div {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ChurnInspect")
    st.markdown("---")
    st.markdown("**MAIN**")
    nav = st.radio("Navigation", ["Dashboard", "Predict (Inference)"], label_visibility="collapsed")

# 2. Page Routing
if nav == "Dashboard":
    render_dashboard()
elif nav == "Predict (Inference)":
    render_predict()
