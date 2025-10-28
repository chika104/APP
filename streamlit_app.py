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
    st.session_state.db_password = "polrwgDJZnGLaungxPtGkOTaduCuolEj"
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
    uploaded = st.file_uploader("Upload CSV or Excel (needs 'year', 'consumption' and optional 'CO2' column)", type=["csv", "xlsx"])
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
        co2_candidates  = [c for c in df_raw.columns if any(k in c for k in ["co2", "carbon", "emission"])]

        if not year_candidates or not cons_candidates:
            st.error("CSV must contain 'year' and a consumption column (e.g. 'consumption' or 'kwh').")
            st.stop()
        year_col = year_candidates[0]
        cons_col = cons_candidates[0]
        co2_col = co2_candidates[0] if co2_candidates else None

        # coerce to numeric and drop invalid
        df_raw[year_col] = pd.to_numeric(df_raw[year_col], errors="coerce")
        df_raw[cons_col] = pd.to_numeric(df_raw[cons_col], errors="coerce")
        if co2_col:
            df_raw[co2_col] = pd.to_numeric(df_raw[co2_col], errors="coerce")

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
        df_loaded["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce") if cost_cols else np.nan

        # optional CO2 column
        if co2_col:
            df_loaded["baseline_co2_kg"] = df_raw[co2_col]
        else:
            df_loaded["baseline_co2_kg"] = np.nan  # default nan kalau CSV tak ada

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
    df_factors = st.session_state.df_factors.copy()
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)

    # site-level change
    st.markdown("General site-level hours change")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=0, key="general_hours")
    general_avg_load_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1, key="general_avg_load_kw")
    general_kwh = float(general_avg_load_kw) * float(general_hours) if general_hours != 0 else 0.0
    total_net_adjust_kwh = df_factors["kwh_per_year"].sum() + general_kwh
    if total_net_adjust_kwh > 0:
        st.info(f"Net adjustment (additional): {total_net_adjust_kwh:,.2f} kWh/year")
    elif total_net_adjust_kwh < 0:
        st.info(f"Net adjustment (reduction): {abs(total_net_adjust_kwh):,.2f} kWh/year")
    else:
        st.info("Net adjustment: 0 kWh/year")

    # Step 3: Forecast settings & compute
    st.header("Step 3 — Forecast settings & compute")
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01, key="tariff")
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01, key="co2_factor")
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1, key="n_years_forecast")

    df["baseline_cost"] = df.get("baseline_cost", np.nan)
    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    # model
    model = LinearRegression()
    X_hist = df[["year"]].values
    y_hist = df["consumption"].values
    if len(X_hist) >= 2:
        model.fit(X_hist, y_hist)
        df["fitted"] = model.predict(X_hist)
        r2 = r2_score(y_hist, df["fitted"])
    else:
        df["fitted"] = df["consumption"]
        r2 = 1.0

    last_year = int(df["year"].max())
    future_years = [last_year + i for i in range(1, int(n_years_forecast) + 1)]
    future_X = np.array(future_years).reshape(-1, 1)
    future_baseline_forecast = model.predict(future_X) if len(X_hist) >= 2 else np.array([df["consumption"].iloc[-1]] * len(future_years))
    adjusted_forecast = future_baseline_forecast + total_net_adjust_kwh

    forecast_df = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": future_baseline_forecast,
        "adjusted_consumption_kwh": adjusted_forecast
    })
    forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
    forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
    forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * co2_factor
    forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * co2_factor
    forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
    forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
    forecast_df["saving_co2_kg"] = forecast_df["baseline_co2_kg"] - forecast_df["adjusted_co2_kg"]

    # persist results in session
    st.session_state.df = df
    st.session_state.df_factors = df_factors
    st.session_state.forecast_df = forecast_df

    # Step 4: Visualizations
    st.header("Step 4 — Visual comparisons & model accuracy")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_baseline = px.line(df, x="year", y="consumption", markers=True, title="Baseline kWh (historical)", labels={"consumption": "kWh"})
        st.plotly_chart(fig_baseline, use_container_width=True)

        plot_all = pd.concat([
            pd.DataFrame({"year": df["year"], "baseline": df["consumption"], "fitted": df["fitted"]}),
            pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"], "fitted": forecast_df["adjusted_consumption_kwh"]})
        ], ignore_index=True)
        fig_both = px.line(plot_all.sort_values("year"), x="year", y=["baseline", "fitted"], markers=True, labels={"value": "kWh", "variable": "Series"}, title="Baseline vs Forecast (kWh)")
        st.plotly_chart(fig_both, use_container_width=True)

        fig_future = px.line(forecast_df, x="year", y=["baseline_consumption_kwh", "adjusted_consumption_kwh"], markers=True, labels={"value": "kWh", "variable": "Series"}, title="Future Forecast (kWh)")
        st.plotly_chart(fig_future, use_container_width=True)

        fig_cost = px.bar(forecast_df, x="year", y=["baseline_cost_rm", "adjusted_cost_rm"], barmode="group", title="Baseline vs Forecast Cost (RM)")
        st.plotly_chart(fig_cost, use_container_width=True)

        fig_co2 = px.bar(forecast_df, x="year", y=["baseline_co2_kg", "adjusted_co2_kg"], barmode="group", title="CO₂ Forecast (kg)")
        st.plotly_chart(fig_co2, use_container_width=True)

    with col2:
        st.subheader("Model performance")
        st.markdown(f"**R²:** `{r2:.4f}`")
        if r2 >= 0.8:
            st.success("Model accuracy: High")
        elif r2 >= 0.6:
            st.warning("Model accuracy: Moderate")
        else:
            st.error("Model accuracy: Low — consider more history or features")

        st.markdown("**Totals over forecast period**")
        total_baseline_kwh = forecast_df["baseline_consumption_kwh"].sum()
        total_adjusted_kwh = forecast_df["adjusted_consumption_kwh"].sum()
        total_kwh_saving = total_baseline_kwh - total_adjusted_kwh
        total_cost_saving = forecast_df["saving_cost_rm"].sum()
        total_co2_saving = forecast_df["saving_co2_kg"].sum()

        st.metric("Baseline kWh (forecast period)", f"{total_baseline_kwh:,.0f} kWh")
        st.metric("Adjusted kWh (forecast period)", f"{total_adjusted_kwh:,.0f} kWh")
        st.metric("Total energy saving (kWh)", f"{total_kwh_saving:,.0f} kWh")
        st.metric("Total cost saving (RM)", f"RM {total_cost_saving:,.2f}")
        st.metric("Total CO₂ reduction (kg)", f"{total_co2_saving:,.0f} kg")

    # Step 5: Tables
    st.header("Step 5 — Forecast tables")
    st.subheader("Historical (baseline)")
    st.dataframe(df[["year", "consumption", "baseline_cost"]].rename(columns={"consumption": "consumption_kwh", "baseline_cost": "baseline_cost_rm"}))
    st.subheader("Forecast results")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh": "{:.0f}",
        "adjusted_consumption_kwh": "{:.0f}",
        "baseline_cost_rm": "{:.2f}",
        "adjusted_cost_rm": "{:.2f}",
        "saving_kwh": "{:.0f}",
        "saving_cost_rm": "{:.2f}",
        "saving_co2_kg": "{:.0f}"
    }))

    # Step 6: Export & Save
    st.header("Step 6 — Export results & Save")
    excel_bytes = df_to_excel_bytes({"historical": df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    images = []
    for fig in (fig_both, fig_future, fig_cost, fig_co2):
        img = try_get_plot_png(fig)
        if img:
            images.append(img)

    summary_lines = [
        f"Forecast period: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
        f"Net adjustment (kWh/year): {total_net_adjust_kwh:.2f}",
        f"Total energy saving (kWh): {total_kwh_saving:,.2f}",
        f"Total cost saving (RM): RM {total_cost_saving:,.2f}",
        f"Total CO₂ reduction (kg): {total_co2_saving:,.2f}",
        f"Model R²: {r2:.4f}"
    ]

    table_blocks = [
        ("Historical (baseline)", df[["year", "consumption", "baseline_cost"]].rename(columns={"consumption": "consumption_kwh", "baseline_cost": "baseline_cost_rm"})),
        ("Factors (kWh/year)", df_factors[["device", "units", "hours_per_year", "action", "kwh_per_year"]]),
        ("Forecast results", forecast_df)
    ]

    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
    if pdf_bytes:
        filename = f"energy_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.session_state.report_history.append({
            "filename": filename,
            "bytes": pdf_bytes,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.download_button("📄 Download formal PDF report", data=pdf_bytes, file_name=filename, mime="application/pdf")
    else:
        st.info("PDF export not available (reportlab not installed). Excel export available.")

    # MySQL save UI
    st.markdown("---")
    st.subheader("Optional: Save results to MySQL database")
    if not MYSQL_AVAILABLE:
        st.info("MySQL support not installed. Install 'mysql-connector-python' to enable DB features.")
    else:
        st.markdown("Set DB credentials in Settings → Database (host, port, user, password, database). Then test connection and press 'Save to DB'.")
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    conn.close()
                    st.success("DB connection successful and tables verified/created.")
                except Exception as e:
                    st.error(f"DB connection failed: {e}")
        with colB:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_results_to_db(conn, df, df_factors, forecast_df)
                    conn.close()
                    st.success("Saved historical, forecast, and factors to DB successfully.")
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")

# -------------------------
# Device Management
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.markdown("Add and manage device types used in forecasts.")
    with st.form("add_device"):
        d_name = st.text_input("Device name (e.g. 'LED 10W')", value="")
        d_watt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        d_note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Add device")
        if submitted and d_name:
            st.session_state.devices.append({"name": d_name, "watt": d_watt, "note": d_note})
            st.success("Device added.")
    if st.session_state.devices:
        st.table(pd.DataFrame(st.session_state.devices))

# -------------------------
# Reports (history)
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("PDF reports you generated are listed below. Click 'Download' to retrieve them again.")
    if not st.session_state.report_history:
        st.info("No PDF reports generated yet. Generate a PDF from Energy Forecast → Step 6.")
    else:
        # show most recent first
        for idx, r in enumerate(reversed(st.session_state.report_history)):
            st.markdown(f"**{r['filename']}** — generated at {r['generated_at']}")
            st.download_button(f"⬇️ Download {r['filename']}", data=r["bytes"], file_name=r["filename"], mime="application/pdf", key=f"dl_{idx}")

# -------------------------
# Settings (Theme + DB)
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Database")
    st.markdown("Theme and Database configuration. Values are stored in session (persist while app runs).")

    # Theme selection; start index according to session
    def idx_for_mode(mode):
        return 0 if mode == "Dark" else (1 if mode == "Light" else 2)
    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"], index=idx_for_mode(st.session_state.bg_mode))

    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        st.session_state.bg_image_url = ""
        st.success("Applied Dark theme.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        st.session_state.bg_image_url = ""
        st.success("Applied Light theme.")
    else:
        img_url = st.text_input("Enter a full image URL to use as background:", value=st.session_state.bg_image_url)
        if img_url:
            st.session_state.bg_mode = "Custom"
            st.session_state.bg_image_url = img_url
            st.success("Applied custom background image.")
    # immediately apply
    apply_theme()

    st.markdown("---")
    st.subheader("Database configuration (optional)")
    st.markdown("Defaults point to your Railway proxy. Enter your Railway DB password here or set env var `RAILWAY_DB_PASSWORD` on the host.")

    db_host = st.text_input("DB host", value=st.session_state.get("db_host", "switchback.proxy.rlwy.net"))
    db_port = st.text_input("DB port", value=str(st.session_state.get("db_port", 55398)))
    db_user = st.text_input("DB user", value=st.session_state.get("db_user", "root"))
    db_password = st.text_input("DB password", value=st.session_state.get("db_password", "<YOUR_RAILWAY_PASSWORD>"), type="password")
    db_database = st.text_input("DB database", value=st.session_state.get("db_database", "railway"))
    if st.button("Save DB settings to session"):
        st.session_state["db_host"] = db_host.strip()
        try:
            st.session_state["db_port"] = int(db_port)
        except Exception:
            st.session_state["db_port"] = 55398
        st.session_state["db_user"] = db_user.strip()
        st.session_state["db_password"] = db_password
        st.session_state["db_database"] = db_database.strip()
        st.success("DB settings saved to session. Use Test/Save in Energy Forecast page.")

    st.markdown("---")
    st.markdown("**PDF Export:** Formal PDF generation requires `reportlab`. Embedding charts into PDF requires `kaleido` for Plotly image export.")

# -------------------------
# Help & About
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed for forecasting and scenario comparison of energy consumption, cost and CO₂.

    **Support / Report issues:**  
    📧 chikaenergyforecast@gmail.com

    Note: This app uses offline historical data you upload or enter manually — no hardware (IoT) is required.
    """)

# End of file

