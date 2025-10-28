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

    # --- Step 1: Input ---
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (needs 'year' & a consumption column)", type=["csv", "xlsx"])
        if uploaded:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
            df_raw = normalize_cols(df_raw)
            # find columns
            year_candidates = [c for c in df_raw.columns if "year" in c]
            cons_candidates = [c for c in df_raw.columns if any(k in c for k in ["consum", "kwh", "energy"])]
            if not year_candidates or not cons_candidates:
                st.error("CSV must contain 'year' and a consumption column (e.g. 'consumption' or 'kwh').")
                st.stop()
            year_col = year_candidates[0]
            cons_col = cons_candidates[0]
            # coerce to numeric and drop invalid
            df_raw[year_col] = pd.to_numeric(df_raw[year_col], errors="coerce")
            df_raw[cons_col] = pd.to_numeric(df_raw[cons_col], errors="coerce")
            before = len(df_raw)
            df_raw = df_raw.dropna(subset=[year_col, cons_col])
            after = len(df_raw)
            if after < before:
                st.warning(f"{before - after} rows removed due to invalid year/consumption.")
            df_loaded = pd.DataFrame({
                "year": df_raw[year_col].astype(int),
                "consumption": df_raw[cons_col]
            })
            cost_cols = [c for c in df_raw.columns if "cost" in c]
            if cost_cols:
                df_loaded["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
            else:
                df_loaded["baseline_cost"] = np.nan
            st.session_state.df = df_loaded.sort_values("year").reset_index(drop=True)

    else:
        # Manual entry only if session df is empty
        if st.session_state.df is None or st.session_state.df.empty:
            rows = st.number_input("Number of historical rows:", min_value=1, max_value=20, value=5)
            data = []
            for i in range(int(rows)):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    y = st.number_input(f"Year {i+1}", 2000, 2100, 2020 + i, key=f"year_{i}")
                with c2:
                    cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, 10000.0, key=f"cons_{i}")
                with c3:
                    cost = st.number_input(f"Baseline cost RM ({y}) (optional)", 0.0, 10_000_000.0, 0.0, key=f"cost_{i}")
                data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost > 0 else np.nan})
            st.session_state.df = pd.DataFrame(data).sort_values("year").reset_index(drop=True)

    # show loaded data
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("No historical data available yet.")
        st.stop()
    df = st.session_state.df.copy()
    st.subheader("Loaded baseline data")
    st.dataframe(df)
# -------------------------
# Step 2 — Device / Factors input
# -------------------------
st.header("Step 2 — Device / Energy Factors")
with st.expander("Manage devices / factors"):
    df_factors = st.session_state.df_factors.copy()
    if df_factors.empty:
        df_factors = pd.DataFrame(columns=["device", "units", "hours_per_year", "action", "kwh_per_year"])
    # Add new row
    add_new = st.checkbox("Add new device/factor row")
    if add_new:
        dev = st.text_input("Device name", key="new_dev")
        units = st.number_input("Units", 1, 1000, 1, key="new_units")
        hpy = st.number_input("Hours per year", 1, 8760, 8760, key="new_hpy")
        action = st.selectbox("Action", ["None", "Reduce", "Increase"], key="new_action")
        if st.button("Add device/factor"):
            kwh_per_year = units * hpy * (0.1 if action != "None" else 0.0)  # dummy calc
            df_factors.loc[len(df_factors)] = [dev, units, hpy, action, kwh_per_year]
            st.session_state.df_factors = df_factors
    # Display editable table
    st.dataframe(df_factors)

# -------------------------
# Step 3 — Forecasting
# -------------------------
st.header("Step 3 — Forecasting & Results")

years = df["year"].values.reshape(-1, 1)
cons = df["consumption"].values

# Fit simple linear regression (baseline)
model = LinearRegression()
model.fit(years, cons)
predicted = model.predict(years)
df["fitted"] = predicted

# Apply device adjustment factor
if not df_factors.empty:
    total_kwh_adj = df_factors["kwh_per_year"].sum()
else:
    total_kwh_adj = 0.0
df["adjusted"] = df["fitted"] - total_kwh_adj

# cost & CO2 calculations (dummy multipliers)
df["baseline_cost_rm"] = df["baseline_cost"].fillna(df["fitted"] * 0.5)
df["adjusted_cost_rm"] = df["baseline_cost_rm"] * (df["adjusted"] / df["fitted"])
df["baseline_co2_kg"] = df["fitted"] * 0.55
df["adjusted_co2_kg"] = df["adjusted"] * 0.55

st.subheader("Historical + Adjusted Results")
st.dataframe(df)

# -------------------------
# Step 4 — Monthly Forecast Table & Line Charts
# -------------------------
st.subheader("Monthly Forecast / Trend Analysis")

# Generate forecast next 12 months
last_year = df["year"].max()
months = pd.date_range(start=f"{last_year+1}-01-01", periods=12, freq="MS")
monthly_forecast = pd.DataFrame({
    "month": months,
    "baseline_consumption_kwh": np.repeat(df["fitted"].iloc[-1]/12, 12),
    "adjusted_consumption_kwh": np.repeat(df["adjusted"].iloc[-1]/12, 12)
})
monthly_forecast["baseline_cost_rm"] = monthly_forecast["baseline_consumption_kwh"] * 0.5
monthly_forecast["adjusted_cost_rm"] = monthly_forecast["adjusted_consumption_kwh"] * 0.5
monthly_forecast["baseline_co2_kg"] = monthly_forecast["baseline_consumption_kwh"] * 0.55
monthly_forecast["adjusted_co2_kg"] = monthly_forecast["adjusted_consumption_kwh"] * 0.55

st.session_state.forecast_df = monthly_forecast

st.markdown("**Monthly Forecast Table**")
st.dataframe(monthly_forecast)

# Line chart — Historical vs Forecast
st.markdown("**Line Chart — Historical vs Adjusted Forecast**")
line_df = pd.concat([
    df[["year","fitted","adjusted"]].rename(columns={"year":"time","fitted":"baseline","adjusted":"adjusted"}),
    pd.DataFrame({
        "time": monthly_forecast["month"].dt.year + monthly_forecast["month"].dt.month/12,
        "baseline": monthly_forecast["baseline_consumption_kwh"],
        "adjusted": monthly_forecast["adjusted_consumption_kwh"]
    })
], ignore_index=True)

fig = px.line(line_df, x="time", y=["baseline","adjusted"], labels={"value":"kWh","time":"Year"}, title="Energy Consumption Trend")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Monthly Line Chart per Forecast Month
# -------------------------
st.subheader("Monthly Breakdown — Next Year Forecast")
fig2 = px.line(monthly_forecast, x="month", y=["baseline_consumption_kwh","adjusted_consumption_kwh"],
               labels={"value":"kWh","month":"Month"}, title="Monthly Forecast Next Year")
st.plotly_chart(fig2, use_container_width=True)
# -------------------------
# Step 5 — Export & Reports
# -------------------------
st.header("Step 5 — Export & Reports")

# Excel export
excel_bytes = df_to_excel_bytes({
    "historical": df,
    "forecast": st.session_state.forecast_df,
    "factors": df_factors
})
st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes,
                   file_name="energy_forecast.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# PDF export (if reportlab + kaleido available)
if REPORTLAB_AVAILABLE:
    images = []
    for fig_plot in [fig, fig2]:
        img_bytes = try_get_plot_png(fig_plot)
        if img_bytes:
            images.append(img_bytes)
    table_blocks = [
        ("Historical Data", df),
        ("Device Factors", df_factors),
        ("Monthly Forecast", st.session_state.forecast_df)
    ]
    summary_lines = [f"Forecast year: {last_year+1}", f"Total adjustment kWh: {total_kwh_adj:.2f}"]
    pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
    if pdf_bytes:
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                           file_name=f"energy_forecast_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                           mime="application/pdf")

# -------------------------
# Step 6 — MySQL save
# -------------------------
st.header("Step 6 — Save to MySQL (Optional)")
if MYSQL_AVAILABLE:
    st.markdown("Set DB credentials in Settings first. Then test connection & save.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test DB connection"):
            try:
                conn = connect_db()
                init_db_tables(conn)
                conn.close()
                st.success("DB connection successful, tables verified/created.")
            except Exception as e:
                st.error(f"DB connection failed: {e}")
    with col2:
        if st.button("Save data to DB"):
            try:
                conn = connect_db()
                init_db_tables(conn)
                save_results_to_db(conn, df, df_factors, st.session_state.forecast_df)
                conn.close()
                st.success("Data saved to DB successfully.")
            except Exception as e:
                st.error(f"Error saving to DB: {e}")
else:
    st.info("MySQL connector not installed — install 'mysql-connector-python' to enable DB features.")

# -------------------------
# Settings (Theme & DB)
# -------------------------
    if menu == "⚙️ Settings":
     st.title("⚙️ Settings — Appearance & Database")
    st.markdown("Theme and Database configuration (session-persistent).")

    # Theme selection
    choice = st.radio("Background / Theme:", ["Dark", "Light", "Custom Image"], index=0)
    if choice == "Dark":
        st.session_state.bg_mode = "Dark"
        st.session_state.bg_image_url = ""
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        st.session_state.bg_image_url = ""
    else:
        img_url = st.text_input("Enter image URL for background", value=st.session_state.bg_image_url)
        if img_url:
            st.session_state.bg_mode = "Custom"
            st.session_state.bg_image_url = img_url
    apply_theme()

    st.markdown("---")
    st.subheader("Database configuration (optional)")
    db_host = st.text_input("DB Host", value=st.session_state.get("db_host", "switchback.proxy.rlwy.net"))
    db_port = st.text_input("DB Port", value=str(st.session_state.get("db_port", 55398)))
    db_user = st.text_input("DB User", value=st.session_state.get("db_user", "root"))
    db_password = st.text_input("DB Password", value=st.session_state.get("db_password", "<YOUR_RAILWAY_PASSWORD>"), type="password")
    db_database = st.text_input("DB Database", value=st.session_state.get("db_database", "railway"))
    if st.button("Save DB settings to session"):
        st.session_state["db_host"] = db_host.strip()
        st.session_state["db_port"] = int(db_port)
        st.session_state["db_user"] = db_user.strip()
        st.session_state["db_password"] = db_password
        st.session_state["db_database"] = db_database.strip()
        st.success("DB settings saved to session.")

# -------------------------
# Help & About
# -------------------------
    elif menu == "❓ Help & About":
        st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Forecast and compare historical vs adjusted energy consumption, cost, and CO₂ emissions.

    **Support / Report issues:**  
    📧 chikaenergyforecast@gmail.com

   Note: Historical data comes from uploaded CSV / manual entry. No hardware required.
  """)