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

    # Step 2: Factors
    st.header("Step 2 — Adjustment factors")
    st.markdown("Enter device-level adjustments. Hours are per YEAR.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}

    # initialize default if empty
    if st.session_state.df_factors is None or st.session_state.df_factors.empty:
        st.session_state.df_factors = pd.DataFrame([{"device": "LED Lamp", "units": 0, "hours_per_year": 0, "action": "Addition", "kwh_per_year": 0.0}])

    n_factors = st.number_input("How many factor rows to show/edit?", min_value=1, max_value=10, value=max(1, len(st.session_state.df_factors)), key="n_factors")
    factors_edit = []
    for i in range(n_factors):
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
        with c1:
            dev = st.text_input(f"Device {i+1}", value=st.session_state.df_factors.iloc[i]["device"] if i < len(st.session_state.df_factors) else "", key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units {i+1}", min_value=0, max_value=1000, value=int(st.session_state.df_factors.iloc[i]["units"]) if i < len(st.session_state.df_factors) else 0, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours/year {i+1}", min_value=0, max_value=8760, value=int(st.session_state.df_factors.iloc[i]["hours_per_year"]) if i < len(st.session_state.df_factors) else 0, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action {i+1}", ["Addition", "Reduction"], index=0 if i >= len(st.session_state.df_factors) else (0 if st.session_state.df_factors.iloc[i]["action"]=="Addition" else 1), key=f"action_{i}")
        with c5:
            default_watt = WATT.get(dev.split()[0], 100) if dev else 100
            kwh = units * hours * default_watt / 1000
            st.text(f"≈ {kwh:.1f} kWh/year")
        factors_edit.append({"device": dev, "units": units, "hours_per_year": hours, "action": action, "kwh_per_year": kwh})
    df_factors = pd.DataFrame(factors_edit)
    st.session_state.df_factors = df_factors

    # Step 3: Forecast parameters
    st.header("Step 3 — Forecast settings")
    next_years = st.number_input("Forecast how many years ahead?", 1, 20, 5)
    st.session_state.forecast_horizon = int(next_years)

    # Step 4: Compute forecast
    st.header("Step 4 — Forecast computation")
    X = df[["year"]]
    y = df["consumption"]
    model = LinearRegression()
    model.fit(X, y)
    fitted = model.predict(X)
    df["fitted"] = fitted
    r2 = r2_score(y, fitted)
    st.write(f"Linear regression R² = {r2:.3f}")

    # baseline forecast
    last_year = df["year"].max()
    forecast_years = np.arange(last_year + 1, last_year + 1 + next_years)
    forecast_baseline = model.predict(forecast_years.reshape(-1, 1))
    baseline_cost_per_kwh = (df["baseline_cost"].sum() / df["consumption"].sum()) if df["consumption"].sum() > 0 else 0.5
    forecast_baseline_cost = forecast_baseline * baseline_cost_per_kwh

    # compute adjusted forecast
    total_factor_kwh = df_factors.apply(lambda r: r["kwh_per_year"] if r["action"]=="Addition" else -r["kwh_per_year"], axis=1).sum()
    forecast_adjusted = forecast_baseline + total_factor_kwh
    forecast_adjusted_cost = forecast_adjusted * baseline_cost_per_kwh
    # CO2 reduction approx (kg)
    baseline_co2_kg = forecast_baseline * 0.85  # arbitrary factor
    adjusted_co2_kg = forecast_adjusted * 0.85
    saving_co2_kg = baseline_co2_kg - adjusted_co2_kg

    forecast_df = pd.DataFrame({
        "year": forecast_years,
        "baseline_consumption_kwh": forecast_baseline,
        "adjusted_consumption_kwh": forecast_adjusted,
        "baseline_cost_rm": forecast_baseline_cost,
        "adjusted_cost_rm": forecast_adjusted_cost,
        "baseline_co2_kg": baseline_co2_kg,
        "adjusted_co2_kg": adjusted_co2_kg,
        "saving_kwh": forecast_baseline - forecast_adjusted,
        "saving_cost_rm": forecast_baseline_cost - forecast_adjusted_cost,
        "saving_co2_kg": saving_co2_kg
    })
    st.session_state.forecast_df = forecast_df

    # Step 5 — Show forecast tables
    st.subheader("Forecast results — Detailed")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh": "{:.0f}",
        "adjusted_consumption_kwh": "{:.0f}",
        "baseline_cost_rm": "{:.2f}",
        "adjusted_cost_rm": "{:.2f}",
        "saving_kwh": "{:.0f}",
        "saving_cost_rm": "{:.2f}",
        "saving_co2_kg": "{:.0f}"
    }))

    # Prepare export-ready table
    forecast_df_export = forecast_df.rename(columns={
        "adjusted_consumption_kwh": "total energy consumption kwh",
        "adjusted_cost_rm": "total cost energy consumption",
        "saving_co2_kg": "CO2 Carbon Emission Reduction"
    })[["year", "total energy consumption kwh", "total cost energy consumption", "CO2 Carbon Emission Reduction"]]
    st.subheader("Forecast results — Export-ready")
    st.dataframe(forecast_df_export.style.format({
        "total energy consumption kwh": "{:.0f}",
        "total cost energy consumption": "{:.2f}",
        "CO2 Carbon Emission Reduction": "{:.0f}"
    }))

    # Step 6 — Export buttons
    st.header("Step 6 — Export / Save")
    excel_bytes = df_to_excel_bytes({
        "Historical": df[["year", "consumption", "baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}),
        "Factors": df_factors[["device","units","hours_per_year","action","kwh_per_year"]],
        "Forecast": forecast_df_export
    })
    st.download_button("💾 Download Excel", data=excel_bytes, file_name="energy_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if REPORTLAB_AVAILABLE:
        # plot figure
        fig = px.line(forecast_df, x="year", y=["baseline_consumption_kwh","adjusted_consumption_kwh"], markers=True,
                      labels={"value":"kWh","year":"Year"}, title="Baseline vs Adjusted Forecast")
        st.plotly_chart(fig)
        plot_bytes = try_get_plot_png(fig)
        pdf_bytes = make_pdf_bytes(
            title_text="Smart Energy Forecast Report",
            summary_lines=[f"Forecast generated for {next_years} years ahead."],
            table_blocks=[("Forecast", forecast_df_export)],
            image_bytes_list=[plot_bytes] if plot_bytes else None
        )
        if pdf_bytes:
            st.download_button("📄 Download PDF", data=pdf_bytes, file_name="energy_forecast.pdf", mime="application/pdf")
        else:
            st.warning("PDF generation unavailable (ReportLab or Plotly Kaleido missing)")

    # Step 7 — Save to DB (optional)
    if st.button("💾 Save results to Database"):
        if MYSQL_AVAILABLE:
            try:
                conn = connect_db()
                init_db_tables(conn)
                save_results_to_db(conn, df, df_factors, forecast_df)
                conn.close()
                st.success("Results saved to database successfully.")
            except Exception as e:
                st.error(f"Error saving to DB: {e}")
        else:
            st.warning("MySQL connector not installed; cannot save to DB.")

# -------------------------
# DEVICE MANAGEMENT
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.info("Currently handled via 'Energy Forecast → Adjustment Factors'. Future: CRUD device management.")
    st.dataframe(st.session_state.df_factors)

# -------------------------
# REPORTS
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.info("Reports will show previously generated PDF or Excel data.")

# -------------------------
# SETTINGS
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    bg_mode = st.radio("Background theme:", ["Dark","Light","Custom"], index=["Dark","Light","Custom"].index(st.session_state.bg_mode))
    st.session_state.bg_mode = bg_mode
    if bg_mode == "Custom":
        st.session_state.bg_image_url = st.text_input("Background image URL:", st.session_state.bg_image_url)
    apply_theme()

    st.subheader("Database Credentials (optional)")
    st.session_state.db_host = st.text_input("Host:", st.session_state.db_host)
    st.session_state.db_port = st.number_input("Port:", value=st.session_state.db_port)
    st.session_state.db_user = st.text_input("User:", st.session_state.db_user)
    st.session_state.db_password = st.text_input("Password:", st.session_state.db_password, type="password")
    st.session_state.db_database = st.text_input("Database:", st.session_state.db_database)

# -------------------------
# HELP & ABOUT
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting App**
    - Built with Streamlit, Pandas, Scikit-learn, Plotly
    - Forecast energy usage and cost, compute CO2 reduction
    - Export to Excel & PDF, optional MySQL save
    - Theme: Dark / Light / Custom
    """)
