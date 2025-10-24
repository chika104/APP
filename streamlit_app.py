# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App with optional MySQL saving
Features included in this single-file app:
- Theme selector (Dark/Light/Custom image)
- Menu navigation: Dashboard, Energy Forecast, Device Management, Reports, Settings, Help & About
- Input: Upload CSV or Manual entry
- Adjustment factors, forecast (LinearRegression)
- Model accuracy (R^2)
- Graphs, Excel export, optional PDF export (reportlab)
- Optional MySQL connection: configure in Settings, test connection, Save results to DB
"""
import os
import io
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# model & metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# try to import reportlab for PDF (optional)
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# try to import plotly image backend (kaleido) for embedding charts into PDF
PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

# try to import mysql connector
MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Page config and default theme
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

DEFAULT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
</style>
"""
st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

# -------------------------
# Utility functions
# -------------------------
def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

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

# PDF builder (optional)
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
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# MySQL helper functions (optional)
# -------------------------
def get_db_config():
    cfg = {}
    cfg['host'] = st.session_state.get("db_host") or os.environ.get("DB_HOST")
    cfg['user'] = st.session_state.get("db_user") or os.environ.get("DB_USER")
    cfg['password'] = st.session_state.get("db_password") or os.environ.get("DB_PASSWORD")
    cfg['database'] = st.session_state.get("db_database") or os.environ.get("DB_DATABASE")
    cfg['port'] = int(st.session_state.get("db_port") or os.environ.get("DB_PORT") or 3306)
    return cfg

def connect_db(timeout=10):
    cfg = get_db_config()
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql.connector not installed on this environment.")
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
    for _, row in historical_df.iterrows():
        fitted = row.get("fitted", None)
        adjusted = row.get("adjusted", None)
        baseline_cost_rm = row.get("baseline_cost", None)
        adjusted_cost_rm = row.get("adjusted_cost", None)
        baseline_co2_kg = row.get("baseline_co2_kg", None)
        adjusted_co2_kg = row.get("adjusted_co2_kg", None)
        cursor.execute(insert_sql, (
            int(row['year']), float(row['consumption']), 
            float(baseline_cost_rm) if baseline_cost_rm is not None else None,
            float(fitted) if pd.notna(fitted) else None,
            float(adjusted) if pd.notna(adjusted) else None,
            float(baseline_cost_rm) if baseline_cost_rm is not None else None,
            float(adjusted_cost_rm) if adjusted_cost_rm is not None else None,
            float(baseline_co2_kg) if baseline_co2_kg is not None else None,
            float(adjusted_co2_kg) if adjusted_co2_kg is not None else None
        ))
    for _, row in forecast_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']), float(row['baseline_consumption_kwh']),
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('baseline_consumption_kwh', None)) if not pd.isna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('adjusted_consumption_kwh', None)) if not pd.isna(row.get('adjusted_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('adjusted_cost_rm', None)) if not pd.isna(row.get('adjusted_cost_rm', None)) else None,
            float(row.get('baseline_co2_kg', None)) if not pd.isna(row.get('baseline_co2_kg', None)) else None,
            float(row.get('adjusted_co2_kg', None)) if not pd.isna(row.get('adjusted_co2_kg', None)) else None
        ))
    insert_f = """
    INSERT INTO energy_factors (device, units, hours_per_year, action, kwh_per_year)
    VALUES (%s,%s,%s,%s,%s)
    """
    for _, r in factors_df.iterrows():
        cursor.execute(insert_f, (str(r['device']), int(r['units']), int(r['hours_per_year']), str(r['action']), float(r['kwh_per_year'])))
    conn.commit()
    cursor.close()

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

    # Step 1: Input
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (needs 'year' & 'consumption' columns)", type=["csv","xlsx"])
        if uploaded:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)
            if "year" not in df_raw.columns or not any(c for c in df_raw.columns if "consum" in c or "kwh" in c or "energy" in c):
                st.error("CSV must contain 'year' and a consumption column (e.g. 'consumption', 'kwh').")
                st.stop()
            year_col = "year"
            cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum","kwh","energy"])][0]
            df = pd.DataFrame({
                "year": df_raw[year_col].astype(int),
                "consumption": pd.to_numeric(df_raw[cons_col], errors="coerce")
            })
            cost_cols = [c for c in df_raw.columns if "cost" in c]
            if cost_cols:
                df["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
            else:
                df["baseline_cost"] = np.nan
    else:
        rows = st.number_input("Number of historical rows:", min_value=1, max_value=20, value=5)
        data = []
        for i in range(int(rows)):
            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i, key=f"year_{i}")
            with c2:
                cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, 10000.0, key=f"cons_{i}")
            with c3:
                cost = st.number_input(f"Baseline cost RM ({y}) (optional)", 0.0, 10_000_000.0, 0.0, key=f"cost_{i}")
            data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
        df = pd.DataFrame(data)

    if df is None or df.empty:
        st.warning("Please upload or enter data to continue.")
        st.stop()

    df["year"] = df["year"].astype(int)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0.0)
    if "baseline_cost" not in df.columns:
        df["baseline_cost"] = np.nan
    df["baseline_cost"] = pd.to_numeric(df["baseline_cost"], errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)
    st.subheader("Loaded baseline data")
    st.dataframe(df)

    # Step 2: Factors
    st.header("Step 2 — Adjustment factors (additions or reductions)")
    st.markdown("Enter device-level adjustments. Hours are per YEAR.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}
    n_factors = st.number_input("How many factor rows to add?", min_value=1, max_value=10, value=1)
    factor_rows = []
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", options=["Lamp - LED","Lamp - CFL","Lamp - Fluorescent","Computer","Lab Equipment"], key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units", min_value=0, value=0, step=1, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR", min_value=0, max_value=8760, value=0, step=1, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action", options=["Addition","Reduction"], key=f"action_{i}")
        if device.startswith("Lamp"):
            subtype = device.split(" - ")[1]
            watt = WATT[subtype]
            dev_name = f"{subtype} Lamp"
        else:
            dev_key = device
            watt = WATT[dev_key]
            dev_name = dev_key
        kwh_per_year = (watt * int(units) * int(hours)) / 1000.0
        if action == "Reduction":
            kwh_per_year = -abs(kwh_per_year)
        else:
            kwh_per_year = abs(kwh_per_year)
        factor_rows.append({
            "device": dev_name,
            "units": int(units),
            "hours_per_year": int(hours),
            "action": action,
            "kwh_per_year": kwh_per_year
        })
    df_factors = pd.DataFrame(factor_rows)
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)

    # site-level change
    st.markdown("General site-level hours change (positive = add load, negative = reduce load)")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=0)
    general_avg_load_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1)
    general_kwh = float(general_avg_load_kw) * float(general_hours) if general_hours != 0 else 0.0
    total_net_adjust_kwh = df_factors["kwh_per_year"].sum() + general_kwh
    if total_net_adjust_kwh > 0:
        st.info(f"Net adjustment (additional): {total_net_adjust_kwh:,.2f} kWh/year")
    elif total_net_adjust_kwh < 0:
        st.info(f"Net adjustment (reduction): {abs(total_net_adjust_kwh):,.2f} kWh/year")
    else:
        st.info("Net adjustment: 0 kWh/year")

    # Step 3: Forecast settings
    st.header("Step 3 — Forecast settings & compute")
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value
