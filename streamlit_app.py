# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App (Railway-ready)
- Default Railway DB proxy host/port/user/database set (replace password in Settings or env var RAILWAY_DB_PASSWORD)
- Session persistence for data and theme so switching menus doesn't lose state
- Excel export + PDF export (optional)
- Save historical, factors, forecast to MySQL (Railway)
- PDF history stored in Reports menu
"""
import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional PDF support
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Plotly -> PNG support for embedding in PDF
PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

# MySQL connector
MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Session defaults and theme persistence
# -------------------------
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"  # default dark
if "bg_image_url" not in st.session_state:
    st.session_state.bg_image_url = ""

# default Railway DB config in session (user supplies password)
if "db_host" not in st.session_state:
    st.session_state.db_host = "switchback.proxy.rlwy.net"
if "db_port" not in st.session_state:
    st.session_state.db_port = 55398
if "db_user" not in st.session_state:
    st.session_state.db_user = "root"
if "db_password" not in st.session_state:
    st.session_state.db_password = "<YOUR_RAILWAY_PASSWORD>"
if "db_database" not in st.session_state:
    st.session_state.db_database = "railway"

# data persistence
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "df_factors" not in st.session_state:
    st.session_state.df_factors = pd.DataFrame()
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "report_history" not in st.session_state:
    st.session_state.report_history = []  # list of dicts: {filename, bytes, generated_at}
if "devices" not in st.session_state:
    st.session_state.devices = []

# -------------------------
# Utility functions
# -------------------------
def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def try_get_plot_png(fig):
    if PLOTLY_IMG_OK:
        try:
            return fig.to_image(format="png", width=900, height=540, scale=2)
        except Exception:
            return None
    return None

def make_pdf_bytes(title_text, summary_lines, table_blocks, image_bytes_list=None, logo_bytes=None):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    if logo_bytes:
        try:
            logo_buf = io.BytesIO(logo_bytes)
            img = RLImage(logo_buf, width=80, height=80)
            elements.append(img)
        except Exception:
            pass
    elements.append(Paragraph(title_text, styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    for line in summary_lines:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 12))
    if image_bytes_list:
        for im_bytes in image_bytes_list:
            try:
                imgbuf = io.BytesIO(im_bytes)
                img = RLImage(imgbuf, width=450, height=280)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
    for title, df in table_blocks:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        elements.append(Spacer(1, 6))
        data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# Database helpers
# -------------------------
def get_db_config():
    cfg = {}
    cfg['host'] = st.session_state.get("db_host") or os.environ.get("DB_HOST")
    cfg['user'] = st.session_state.get("db_user") or os.environ.get("DB_USER")
    cfg['password'] = st.session_state.get("db_password") or os.environ.get("DB_PASSWORD") or os.environ.get("RAILWAY_DB_PASSWORD")
    cfg['database'] = st.session_state.get("db_database") or os.environ.get("DB_DATABASE")
    cfg['port'] = int(st.session_state.get("db_port") or os.environ.get("DB_PORT") or 3306)
    return cfg

def connect_db(timeout=10):
    cfg = get_db_config()
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql.connector not installed in this environment.")
    if not cfg['host'] or not cfg['user'] or not cfg['database']:
        raise ValueError("Database host/user/database must be set in Settings (or as environment variables).")
    conn = mysql.connector.connect(
        host=cfg['host'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        port=cfg['port'],
        connection_timeout=timeout
    )
    return conn

def init_db_tables(conn):
    cursor = conn.cursor()
    # Create energy_data, energy_factors if not exists
    t1 = """
    CREATE TABLE IF NOT EXISTS energy_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        year INT NOT NULL,
        consumption DOUBLE,
        baseline_cost DOUBLE,
        fitted DOUBLE,
        adjusted DOUBLE,
        baseline_cost_rm DOUBLE,
        adjusted_cost_rm DOUBLE,
        baseline_co2_kg DOUBLE,
        adjusted_co2_kg DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    t2 = """
    CREATE TABLE IF NOT EXISTS energy_factors (
        id INT AUTO_INCREMENT PRIMARY KEY,
        device VARCHAR(128),
        units INT,
        hours_per_year INT,
        action VARCHAR(32),
        kwh_per_year DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    cursor.execute(t1)
    cursor.execute(t2)
    conn.commit()
    cursor.close()

def save_results_to_db(conn, historical_df, factors_df, forecast_df):
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO energy_data 
    (year, consumption, baseline_cost, fitted, adjusted, baseline_cost_rm, adjusted_cost_rm, baseline_co2_kg, adjusted_co2_kg)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    # historical rows
    for _, row in historical_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row['consumption']) if not pd.isna(row['consumption']) else None,
            float(row.get('baseline_cost', None)) if not pd.isna(row.get('baseline_cost', None)) else None,
            float(row.get('fitted', None)) if not pd.isna(row.get('fitted', None)) else None,
            float(row.get('adjusted', None)) if not pd.isna(row.get('adjusted', None)) else None,
            float(row.get('baseline_cost', None)) if not pd.isna(row.get('baseline_cost', None)) else None,
            float(row.get('adjusted_cost', None)) if not pd.isna(row.get('adjusted_cost', None)) else None,
            float(row.get('baseline_co2_kg', None)) if not pd.isna(row.get('baseline_co2_kg', None)) else None,
            float(row.get('adjusted_co2_kg', None)) if not pd.isna(row.get('adjusted_co2_kg', None)) else None
        ))
    # forecast rows
    for _, row in forecast_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row.get('baseline_consumption_kwh', None)) if not pd.isna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('baseline_consumption_kwh', None)) if not pd.isna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('adjusted_consumption_kwh', None)) if not pd.isna(row.get('adjusted_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('adjusted_cost_rm', None)) if not pd.isna(row.get('adjusted_cost_rm', None)) else None,
            float(row.get('baseline_co2_kg', None)) if not pd.isna(row.get('baseline_co2_kg', None)) else None,
            float(row.get('adjusted_co2_kg', None)) if not pd.isna(row.get('adjusted_co2_kg', None)) else None
        ))
    # factors
    insert_f = """
    INSERT INTO energy_factors (device, units, hours_per_year, action, kwh_per_year)
    VALUES (%s,%s,%s,%s,%s)
    """
    for _, r in factors_df.iterrows():
        cursor.execute(insert_f, (str(r['device']), int(r['units']), int(r['hours_per_year']), str(r['action']), float(r['kwh_per_year'])))
    conn.commit()
    cursor.close()

# -------------------------
# Page config & apply theme (every run applies what's in session_state)
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")
DEFAULT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
</style>
"""
LIGHT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
[data-testid="stSidebar"] {background-color: rgba(0,0,0,0.03);}
</style>
"""
def apply_theme():
    if st.session_state.bg_mode == "Dark":
        st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
    elif st.session_state.bg_mode == "Light":
        st.markdown(LIGHT_STYLE, unsafe_allow_html=True)
    elif st.session_state.bg_mode == "Custom" and st.session_state.bg_image_url:
        # custom image
        custom_style = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{st.session_state.bg_image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
        st.markdown(custom_style, unsafe_allow_html=True)
apply_theme()

# Sidebar and navigation
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# DASHBOARD
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("""
    **Welcome** — use the left menu to go to the Energy Forecast module, manage devices, or download reports.
    - Forecast energy and cost, compare baseline vs adjusted scenarios.
    - Export formal PDF & Excel reports.
    """)
    st.info("Tip: Use Settings to change background (Dark / Light / Custom image) or set Database credentials.")

# -------------------------
# ENERGY FORECAST
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Quick reset button
    if st.button("🔄 Reset Factors & Forecast"):
        st.session_state.df_factors = pd.DataFrame()
        st.session_state.forecast_df = pd.DataFrame()
        st.success("Factors and forecast cleared. Enter new data.")
