# streamlit_app.py — Smart Energy Forecasting (CO₂ graph removed)
"""
Smart Energy Forecasting — Full Streamlit App (CO₂ graph removed)
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
    st.session_state.bg_mode = "Dark"
if "bg_image_url" not in st.session_state:
    st.session_state.bg_image_url = ""

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

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "df_factors" not in st.session_state:
    st.session_state.df_factors = pd.DataFrame()
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "report_history" not in st.session_state:
    st.session_state.report_history = []
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
    (year, consumption, baseline_cost, fitted, adjusted, baseline_cost_rm, adjusted_cost_rm)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    """
    for _, row in historical_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row['consumption']) if not pd.isna(row['consumption']) else None,
            float(row.get('baseline_cost', None)) if not pd.isna(row.get('baseline_cost', None)) else None,
            float(row.get('fitted', None)) if not pd.isna(row.get('fitted', None)) else None,
            float(row.get('adjusted', None)) if not pd.isna(row.get('adjusted', None)) else None,
            float(row.get('baseline_cost', None)) if not pd.isna(row.get('baseline_cost', None)) else None,
            float(row.get('adjusted_cost', None)) if not pd.isna(row.get('adjusted_cost', None)) else None
        ))
    for _, row in forecast_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row.get('baseline_consumption_kwh', None)) if not pd.isna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('baseline_consumption_kwh', None)) if not pd.isna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('adjusted_consumption_kwh', None)) if not pd.isna(row.get('adjusted_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if not pd.isna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('adjusted_cost_rm', None)) if not pd.isna(row.get('adjusted_cost_rm', None)) else None
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
# Page config & theme
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

# Sidebar
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# DASHBOARD
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("""
    **Welcome** — use the left menu to go to Energy Forecast module, manage devices, or download reports.
    - Forecast energy and cost, compare baseline vs adjusted scenarios.
    - Export formal PDF & Excel reports.
    """)
    st.info("Tip: Use Settings to change theme or set Database credentials.")

# -------------------------
# ENERGY FORECAST
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Step 1: Input
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
            year_candidates = [c for c in df_raw.columns if "year" in c]
            cons_candidates = [c for c in df_raw.columns if any(k in c for k in ["consum", "kwh", "energy"])]
            if not year_candidates or not cons_candidates:
                st.error("CSV must contain 'year' and a consumption column (e.g. 'consumption' or 'kwh').")
                st.stop()
            year_col = year_candidates[0]
            cons_col = cons_candidates[0]
            df_raw[year_col] = pd.to_numeric(df_raw[year_col], errors="coerce")
            df_raw[cons_col] = pd.to_numeric(df_raw[cons_col], errors="coerce")
            df_raw = df_raw.dropna(subset=[year_col, cons_col])
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

    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("No historical data available yet.")
        st.stop()
    df = st.session_state.df.copy()
    st.subheader("Loaded baseline data")
    st.dataframe(df)

    # Step 2: Factors
    st.header("Step 2 — Adjustment factors")
    st.markdown("Enter device-level adjustments. Hours are per YEAR.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}

    if st.session_state.df_factors is None or st.session_state.df_factors.empty:
        st.session_state.df_factors = pd.DataFrame([{"device": "LED Lamp", "units": 0, "hours_per_year": 0, "action": "Addition", "kwh_per_year": 0.0}])

    n_factors = st.number_input("How many factor rows to show/edit?", min_value=1, max_value=10, value=max(1, len(st.session_state.df_factors)), key="n_factors")
    factors_edit = []
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        prev = st.session_state.df_factors.iloc[i].to_dict() if i < len(st.session_state.df_factors) else {}
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", options=["Lamp - LED", "Lamp - CFL", "Lamp - Fluorescent", "Computer", "Lab Equipment"], index=0, key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units", min_value=0, value=int(prev.get("units", 0)), step=1, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR", min_value=0, max_value=8760, value=int(prev.get("hours_per_year", 0)), step=1, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action", options=["Addition", "Reduction"], index=0 if prev.get("action", "Addition") == "Addition" else 1, key=f"action_{i}")
        if device.startswith("Lamp"):
            subtype = device.split(" - ")[1]
            watt = WATT.get(subtype, 10)
            dev_name = f"{subtype} Lamp"
        else:
            dev_key = device
            watt = WATT.get(dev_key, 100)
            dev_name = dev_key
        kwh_per_year = (watt * int(units) * int(hours)) / 1000.0
        if action == "Reduction":
            kwh_per_year = -abs(kwh_per_year)
        factors_edit.append({"device": dev_name, "units": int(units), "hours_per_year": int(hours), "action": action, "kwh_per_year": kwh_per_year})
    st.session_state.df_factors = pd.DataFrame(factors_edit)
    st.subheader("Adjustment factors table")
    st.dataframe(st.session_state.df_factors)

    # Step 3: Forecast
    st.header("Step 3 — Forecast computation")

    if st.button("Compute Forecast"):
        df_hist = st.session_state.df.copy()
        df_hist["baseline_cost"] = df_hist.get("baseline_cost", np.nan)
        df_hist["year_numeric"] = df_hist["year"].astype(int)
        X = df_hist[["year_numeric"]]
        y = df_hist["consumption"]
        model = LinearRegression()
        model.fit(X, y)
        df_hist["fitted"] = model.predict(X)
        last_year = df_hist["year"].max()
        years_future = [last_year + i for i in range(1, 6)]
        X_future = pd.DataFrame({"year_numeric": years_future})
        future_baseline = model.predict(X_future)
        kwh_adjust_total = st.session_state.df_factors["kwh_per_year"].sum()
        future_adjusted = future_baseline + kwh_adjust_total

        forecast_df = pd.DataFrame({
            "year": years_future,
            "baseline_consumption_kwh": future_baseline,
            "adjusted_consumption_kwh": future_adjusted
        })
        # compute costs if baseline_cost exists
        avg_unit_cost = df_hist["baseline_cost"].sum() / df_hist["consumption"].sum() if df_hist["consumption"].sum() > 0 else 0.0
        forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * avg_unit_cost
        forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * avg_unit_cost

        st.session_state.forecast_df = forecast_df

        st.subheader("Forecast Table (Next 5 years)")
        st.dataframe(forecast_df)

        st.subheader("Forecast Graphs")
        fig_future = px.line(forecast_df, x="year", y=["baseline_consumption_kwh", "adjusted_consumption_kwh"],
                             labels={"value": "kWh", "variable": "Scenario"}, title="Energy Forecast (kWh)")
        st.plotly_chart(fig_future, use_container_width=True)

        fig_cost = px.line(forecast_df, x="year", y=["baseline_cost_rm", "adjusted_cost_rm"],
                           labels={"value": "RM", "variable": "Scenario"}, title="Forecast Cost (RM)")
        st.plotly_chart(fig_cost, use_container_width=True)

        # PDF / Excel export
        st.subheader("Export Options")
        images = []
        for fig in (fig_future, fig_cost):
            img = try_get_plot_png(fig)
            if img:
                images.append(img)

        if st.button("Export PDF Report") and REPORTLAB_AVAILABLE:
            pdf_bytes = make_pdf_bytes(
                title_text="Energy Forecast Report",
                summary_lines=[
                    f"Historical years: {df_hist['year'].min()} - {df_hist['year'].max()}",
                    f"Average unit cost RM/kWh: {avg_unit_cost:.3f}",
                    f"Total adjustment kWh/year: {kwh_adjust_total:.2f}"
                ],
                table_blocks=[("Forecast Table", forecast_df)],
                image_bytes_list=images
            )
            if pdf_bytes:
                st.download_button("Download PDF Report", data=pdf_bytes, file_name="energy_forecast.pdf", mime="application/pdf")

        if st.button("Export Excel Report"):
            xlsx_bytes = df_to_excel_bytes({"Historical": df_hist, "Forecast": forecast_df, "Factors": st.session_state.df_factors})
            st.download_button("Download Excel Report", data=xlsx_bytes, file_name="energy_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# DEVICE MANAGEMENT
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.markdown("Manage device types and adjustment factors here.")
    st.dataframe(st.session_state.df_factors)

# -------------------------
# REPORTS
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.info("Reports will include only energy & cost. CO₂ data removed.")

# -------------------------
# SETTINGS
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.radio("Theme", options=["Dark", "Light", "Custom"], key="bg_mode", on_change=apply_theme)
    if st.session_state.bg_mode == "Custom":
        st.text_input("Background image URL:", key="bg_image_url", on_change=apply_theme)
    st.subheader("Database")
    st.text_input("Host", key="db_host")
    st.number_input("Port", key="db_port", value=55398)
    st.text_input("User", key="db_user")
    st.text_input("Password", key="db_password", type="password")
    st.text_input("Database", key="db_database")

# -------------------------
# HELP / ABOUT
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting** helps you:
    - Forecast energy usage & cost
    - Apply device-level adjustments
    - Export reports (PDF & Excel)

    **Changelog:** CO₂ visualization removed in this version.
    """)

