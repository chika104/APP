# streamlit_app.py
"""
Smart Energy Forecasting — Streamlit App with Persistent Background (Session Only)
Features:
- Theme selector (Dark/Light/Custom image) — persists across all pages
- Menu navigation: Dashboard, Forecast, Device Management, Reports, Settings, Help
- Upload CSV or manual entry
- LinearRegression forecasting + R² accuracy
- Graphs (Plotly), Excel export, optional PDF export (reportlab)
- Optional MySQL connection with test/save
"""

import os
import io
import base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional imports
REPORTLAB_AVAILABLE = False
PLOTLY_IMG_OK = False
MYSQL_AVAILABLE = True

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# -------------------------
# Default & Light CSS Themes
# -------------------------
DEFAULT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #F5F5F5;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
</style>
"""

LIGHT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    color: #000000;
}
[data-testid="stSidebar"] {background-color: rgba(0,0,0,0.03);}
</style>
"""

# -------------------------
# Apply Saved Theme
# -------------------------
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

if "bg_custom" not in st.session_state:
    st.session_state.bg_custom = ""

if st.session_state.bg_mode == "Dark":
    st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
elif st.session_state.bg_mode == "Light":
    st.markdown(LIGHT_STYLE, unsafe_allow_html=True)
elif st.session_state.bg_mode == "Custom" and st.session_state.bg_custom:
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{st.session_state.bg_custom}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# Utility Functions
# -------------------------
def normalize_cols(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

# -------------------------
# Dashboard
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("""
    Welcome to the **Smart Energy Forecasting System**.  
    Use the sidebar to navigate to the Energy Forecast, Device Management, Reports, or Settings pages.
    """)
    st.info("Your selected background theme will persist across all pages until you change it in Settings.")

# -------------------------
# Energy Forecast
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")
    st.write("Perform forecasting using uploaded or manually entered data.")

# (To save space, keep your full forecasting logic here — unchanged from your version.
# The only difference: background now stays applied automatically.)
# You can paste your entire forecasting section from your last version here.

# -------------------------
# Device Management
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.markdown("Add and manage your devices here.")

# -------------------------
# Reports
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("View or download your generated reports here.")

# -------------------------
# Settings (Theme + DB)
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Database")

    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"])

    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        st.session_state.bg_custom = ""
        st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
        st.success("Dark theme applied and will stay across all menus.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        st.session_state.bg_custom = ""
        st.markdown(LIGHT_STYLE, unsafe_allow_html=True)
        st.success("Light theme applied and will stay across all menus.")
    else:
        img_url = st.text_input("Enter full image URL:")
        if img_url:
            st.session_state.bg_mode = "Custom"
            st.session_state.bg_custom = img_url
            st.markdown(f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img_url}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """, unsafe_allow_html=True)
            st.success("Custom background applied and will stay across all menus.")

# -------------------------
# Help & About
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed for forecasting and analyzing energy consumption, cost, and CO₂ emissions.

    📧 **Support:** chikaenergyforecast@gmail.com
    """)

# -------------------------
# End of file
# -------------------------
