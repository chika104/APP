# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App with Login/Register + optional MySQL saving.

Notes:
- Users table: app_users(id INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(100) UNIQUE, password VARCHAR(255))
- Passwords are stored hashed with bcrypt.
- Background choice stored in session_state (persists per active session).
- Forecast & factors stored in session_state so switching menu won't reset inputs/results.
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

# Optional libs
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

# mysql connector (optional)
MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

# password hashing
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    BCRYPT_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Page config & initial CSS
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide", initial_sidebar_state="expanded")

# default background applied via session_state-stored CSS
if "bg_style" not in st.session_state:
    # default dark background
    st.session_state.bg_style = {
        "mode": "Dark",
        "css": """
        <style>
        /* App background */
        [data-testid="stAppViewContainer"] {background-color: #0E1117 !important; color: #F5F5F5;}
        /* Sidebar / menu black bar for mobile/responsive */
        [data-testid="stSidebar"] {background-color: #000000 !important; color: #FFFFFF;}
        /* Make sidebar title white */
        .css-1d391kg { color: #FFFFFF; }
        /* Top header transparent */
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}
        </style>
        """
    }

# helper to apply current bg CSS
def apply_bg_css():
    st.markdown(st.session_state.bg_style["css"], unsafe_allow_html=True)

apply_bg_css()

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
# keep 6 menu items per user earlier request (Dashboard, Energy Forecast, Device Management, Reports, Settings, Help)
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# Persistence helpers
# -------------------------
def ensure_session_vars():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    if "forecast_inputs" not in st.session_state:
        st.session_state.forecast_inputs = None
    if "factors_df" not in st.session_state:
        st.session_state.factors_df = pd.DataFrame()
    if "forecast_df" not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if "historical_df" not in st.session_state:
        st.session_state.historical_df = pd.DataFrame()
    if "bg_mode" not in st.session_state:
        st.session_state.bg_mode = st.session_state.bg_style.get("mode", "Dark")
    if "app_users_checked" not in st.session_state:
        st.session_state["app_users_checked"] = False

ensure_session_vars()

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
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# MySQL helpers
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
        raise RuntimeError("mysql.connector not installed in this environment.")
    if not cfg['host'] or not cfg['user'] or not cfg['database']:
        raise ValueError("DB host/user/database not set in Settings.")
    conn = mysql.connector.connect(
        host=cfg['host'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        port=cfg['port'],
        connection_timeout=timeout
    )
    return conn

def ensure_app_users_table(conn):
    cur = conn.cursor()
    create_users = """
    CREATE TABLE IF NOT EXISTS app_users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE,
        password VARCHAR(255)
    ) ENGINE=InnoDB;
    """
    cur.execute(create_users)
    conn.commit()
    cur.close()

def init_db_tables_standard(conn):
    """Create energy_data and energy_factors if missing."""
    cur = conn.cursor()
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
    cur.execute(t1)
    cur.execute(t2)
    conn.commit()
    cur.close()

def save_results_to_db(conn, historical_df, factors_df, forecast_df):
    cur = conn.cursor()
    insert_sql = """
    INSERT INTO energy_data 
    (year, consumption, baseline_cost, fitted, adjusted, baseline_cost_rm, adjusted_cost_rm, baseline_co2_kg, adjusted_co2_kg)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    # historical
    for _, row in historical_df.iterrows():
        fitted = row.get("fitted", None)
        adjusted = row.get("adjusted", None)
        baseline_cost_rm = row.get("baseline_cost", None)
        adjusted_cost_rm = row.get("adjusted_cost", None)
        baseline_co2_kg = row.get("baseline_co2_kg", None)
        adjusted_co2_kg = row.get("adjusted_co2_kg", None)
        cur.execute(insert_sql, (
            int(row['year']), float(row['consumption']),
            float(baseline_cost_rm) if baseline_cost_rm is not None else None,
            float(fitted) if pd.notna(fitted) else None,
            float(adjusted) if pd.notna(adjusted) else None,
            float(baseline_cost_rm) if baseline_cost_rm is not None else None,
            float(adjusted_cost_rm) if adjusted_cost_rm is not None else None,
            float(baseline_co2_kg) if baseline_co2_kg is not None else None,
            float(adjusted_co2_kg) if adjusted_co2_kg is not None else None
        ))
    # forecast rows
    for _, row in forecast_df.iterrows():
        cur.execute(insert_sql, (
            int(row['year']), float(row['baseline_consumption_kwh']),
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
        cur.execute(insert_f, (str(r['device']), int(r['units']), int(r['hours_per_year']), str(r['action']), float(r['kwh_per_year'])))
    conn.commit()
    cur.close()

# -------------------------
# Authentication (register/login)
# -------------------------
def hash_password(pw: str) -> str:
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    else:
        # fallback (not recommended): simple reversible encoding (we avoid plain text)
        return base64.b64encode(pw.encode("utf-8")).decode("utf-8")

def check_password(pw: str, hashed: str) -> bool:
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    else:
        return base64.b64encode(pw.encode("utf-8")).decode("utf-8") == hashed

def register_user_db(username: str, password: str):
    """Register user into app_users table. Returns (ok,msg)."""
    if not MYSQL_AVAILABLE:
        return False, "MySQL driver not available on server."
    try:
        conn = connect_db()
    except Exception as e:
        return False, f"DB connection failed: {e}"
    try:
        ensure_app_users_table(conn)
        cur = conn.cursor()
        pw_hash = hash_password(password)
        cur.execute("INSERT INTO app_users (username, password) VALUES (%s, %s)", (username, pw_hash))
        conn.commit()
        cur.close()
        conn.close()
        return True, "Registered successfully."
    except mysql.connector.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Registration failed: {e}"

def login_user_db(username: str, password: str):
    """Attempt login. Returns (ok,msg)."""
    if not MYSQL_AVAILABLE:
        return False, "MySQL driver not available on server."
    try:
        conn = connect_db()
    except Exception as e:
        return False, f"DB connection failed: {e}"
    try:
        ensure_app_users_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT password FROM app_users WHERE username=%s", (username,))
        res = cur.fetchone()
        cur.close()
        conn.close()
        if not res:
            return False, "Username not found."
        stored = res[0]
        if check_password(password, stored):
            return True, "Logged in."
        else:
            return False, "Incorrect username or password."
    except Exception as e:
        return False, f"Login failed: {e}"

# -------------------------
# Login/Register UI
# -------------------------
def login_register_ui():
    st.title("🔐 Login / Register")
    st.markdown("Please login to access the Smart Energy Forecasting dashboard.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        uname = st.text_input("Username", key="login_uname")
        upass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            ok, msg = login_user_db(uname.strip(), upass)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user = uname.strip()
                st.success("Login successful.")
                # no experimental_rerun - simply show success and app will continue (state updated)
            else:
                st.error(f"Login failed: {msg}")
    with col2:
        st.subheader("Register")
        r_uname = st.text_input("Choose username", key="reg_uname")
        r_pass = st.text_input("Choose password", type="password", key="reg_pass")
        if st.button("Register"):
            ok, msg = register_user_db(r_uname.strip(), r_pass)
            if ok:
                st.success("Registered. Please login using your credentials.")
            else:
                st.error(f"Gagal register: {msg}")

# If user not logged in, show login page only
if not st.session_state.logged_in:
    login_register_ui()
    st.stop()

# -------------------------
# Main app (user is logged in)
# -------------------------
apply_bg_css()  # re-apply CSS in case changed

st.markdown(f"**Logged in as:** `{st.session_state.user}`")
st.markdown("---")

# -------------------------
# DASHBOARD
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard — Smart Energy Forecasting")
    st.markdown("""
    Welcome to the Smart Energy Forecasting dashboard.
    Use the left menu to go to Energy Forecast, Device Management, Reports, Settings, or Help.
    """)
    st.markdown("Quick summary of latest forecast (if any):")
    if not st.session_state.forecast_df.empty:
        st.dataframe(st.session_state.forecast_df.head())
    else:
        st.info("No forecast available yet. Go to ⚡ Energy Forecast to run a forecast.")

# -------------------------
# ENERGY FORECAST
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Step 1: Input baseline
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    # if previously loaded, show and keep
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (needs 'year' & 'consumption' columns)", type=["csv","xlsx"], key="uploader")
        if uploaded:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
                df_raw = normalize_cols(df_raw)
                if "year" not in df_raw.columns or not any(c for c in df_raw.columns if "consum" in c or "kwh" in c or "energy" in c):
                    st.error("CSV must contain 'year' and a consumption column.")
                else:
                    year_col = "year"
                    cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum","kwh","energy"])][0]
                    df = pd.DataFrame({"year": df_raw[year_col].astype(int),
                                       "consumption": pd.to_numeric(df_raw[cons_col], errors="coerce")})
                    cost_cols = [c for c in df_raw.columns if "cost" in c]
                    if cost_cols:
                        df["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
                    else:
                        df["baseline_cost"] = np.nan
                    st.session_state.historical_df = df.sort_values("year").reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to load file: {e}")
    else:
        # Manual entry; preserve previously entered values saved in session_state.forecast_inputs
        rows = st.number_input("Number of historical rows:", min_value=1, max_value=50, value=st.session_state.get("manual_rows", 5), key="manual_rows_input")
        st.session_state["manual_rows"] = rows
        data = []
        if "historical_df" in st.session_state and not st.session_state.historical_df.empty:
            # show existing and offer edit? For simplicity show existing and allow re-enter
            st.info("Existing historical data loaded (from previous session). You can edit by re-entering below.")
        for i in range(int(rows)):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                y = st.number_input(f"Year {i+1}", 2000, 2100, value=int(st.session_state.historical_df['year'].iloc[i]) if (not st.session_state.historical_df.empty and i < len(st.session_state.historical_df)) else 2020 + i, key=f"m_year_{i}")
            with c2:
                cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, value=float(st.session_state.historical_df['consumption'].iloc[i]) if (not st.session_state.historical_df.empty and i < len(st.session_state.historical_df)) else 10000.0, key=f"m_cons_{i}")
            with c3:
                cost = st.number_input(f"Baseline cost RM ({y}) (optional)", 0.0, 10_000_000.0, value=float(st.session_state.historical_df['baseline_cost'].iloc[i]) if (not st.session_state.historical_df.empty and i < len(st.session_state.historical_df)) else 0.0, key=f"m_cost_{i}")
            data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost > 0 else np.nan})
        st.session_state.historical_df = pd.DataFrame(data)

    # display loaded historical
    if st.session_state.historical_df is None or st.session_state.historical_df.empty:
        st.warning("Please upload or enter historical data to continue.")
        st.stop()

    st.subheader("Loaded baseline data")
    st.dataframe(st.session_state.historical_df)

    # Step 2: Factors
    st.header("Step 2 — Adjustment factors (additions or reductions)")
    st.markdown("Device-level adjustments. Hours per YEAR.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}
    n_factors = st.number_input("How many factor rows to add?", min_value=1, max_value=10, value=st.session_state.get("n_factors", 1), key="n_factors_ui")
    st.session_state["n_factors"] = n_factors
    factor_rows = []
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", options=["Lamp - LED", "Lamp - CFL", "Lamp - Fluorescent", "Computer", "Lab Equipment"], key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units", min_value=0, value=int(st.session_state.get(f"units_{i}", 0)), key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR", min_value=0, max_value=8760, value=int(st.session_state.get(f"hours_{i}", 0)), key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action", options=["Addition", "Reduction"], index=0 if st.session_state.get(f"action_{i}", "Addition") == "Addition" else 1, key=f"action_{i}")
        # compute
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
        factor_rows.append({"device": dev_name, "units": int(units), "hours_per_year": int(hours), "action": action, "kwh_per_year": kwh_per_year})
    df_factors = pd.DataFrame(factor_rows)
    st.session_state.factors_df = df_factors
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)

    # general site-level change
    st.markdown("General site-level hours change (positive = add load, negative = reduce load)")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=st.session_state.get("general_hours", 0), key="general_hours_ui")
    general_avg_load_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=float(st.session_state.get("general_avg_load_kw", 2.0)), step=0.1, key="general_avg_load_kw_ui")
    st.session_state.general_hours = general_hours
    st.session_state.general_avg_load_kw = general_avg_load_kw
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
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=float(st.session_state.get("tariff", 0.52)), step=0.01, key="tariff_ui")
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=float(st.session_state.get("co2_factor", 0.75)), step=0.01, key="co2_ui")
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=int(st.session_state.get("n_years_forecast", 3)), step=1, key="n_forecast_ui")
    st.session_state.tariff = tariff
    st.session_state.co2_factor = co2_factor
    st.session_state.n_years_forecast = n_years_forecast

    # prepare historical df
    df = st.session_state.historical_df.copy()
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

    # store in session so it persists when switching pages
    st.session_state.forecast_df = forecast_df
    st.session_state.factors_df = df_factors
    st.session_state.historical_df = df

    # Step 4: Visualizations (7 graphs)
    st.header("Step 4 — Visual comparisons & model accuracy")
    # color palette: bright vs dark contrasts
    color_seq = ["#FF4136", "#001f3f", "#2ECC40", "#FF851B", "#B10DC9", "#FFDC00", "#AAAAAA"]  # red, dark blue, green, orange, purple, yellow, gray

    c1, c2 = st.columns([2, 1])
    with c1:
        # 1. Baseline only (historical baseline)
        st.subheader("1) Baseline (historical)")
        fig1 = px.line(df, x="year", y="consumption", markers=True, title="Baseline (historical)", color_discrete_sequence=[color_seq[1]])
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Baseline vs Forecast (baseline series + forecast baseline)
        st.subheader("2) Baseline vs Forecast")
        plot_all = pd.concat([
            pd.DataFrame({"year": df["year"], "baseline": df["consumption"], "series": "Historical Baseline"}),
            pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"], "series": "Forecast Baseline"})
        ], ignore_index=True)
        fig2 = px.line(plot_all.sort_values("year"), x="year", y="baseline", color="series", markers=True,
                       color_discrete_sequence=[color_seq[1], color_seq[0]], title="Baseline vs Forecast")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Adjusted vs Forecast vs Baseline
        st.subheader("3) Adjusted vs Forecast vs Baseline")
        plot3 = pd.concat([
            pd.DataFrame({"year": df["year"], "value": df["consumption"], "series": "Historical Baseline"}),
            pd.DataFrame({"year": forecast_df["year"], "value": forecast_df["baseline_consumption_kwh"], "series": "Forecast Baseline"}),
            pd.DataFrame({"year": forecast_df["year"], "value": forecast_df["adjusted_consumption_kwh"], "series": "Forecast Adjusted"})
        ], ignore_index=True)
        fig3 = px.line(plot3.sort_values(["year", "series"]), x="year", y="value", color="series", markers=True,
                       color_discrete_sequence=[color_seq[1], color_seq[0], color_seq[2]], title="Adjusted vs Forecast vs Baseline")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.subheader("Model performance & totals")
        st.markdown(f"**R²:** `{r2:.4f}`")
        if r2 >= 0.8:
            st.success("Model accuracy: High")
        elif r2 >= 0.6:
            st.warning("Model accuracy: Moderate")
        else:
            st.error("Model accuracy: Low — consider more history or features")

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

    # Additional graphs below
    st.subheader("4) Baseline cost (historical & forecast)")
    # Historical baseline cost
    df_hist_cost = df[["year", "baseline_cost"]].rename(columns={"baseline_cost": "baseline_cost_rm"})
    fig4 = px.line(pd.concat([df_hist_cost, forecast_df[["year", "baseline_cost_rm"]].rename(columns={"baseline_cost_rm":"baseline_cost_rm"})], ignore_index=True).sort_values("year"),
                   x="year", y="baseline_cost_rm", markers=True, title="Baseline cost (RM)", color_discrete_sequence=[color_seq[5]])
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("5) Forecast cost vs Baseline cost (forecast period)")
    fig5 = px.bar(forecast_df, x="year", y=["baseline_cost_rm", "adjusted_cost_rm"], barmode="group", title="Forecast cost vs Baseline cost", color_discrete_sequence=[color_seq[0], color_seq[2]])
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("6) CO₂ baseline (historical)")
    fig6 = px.line(df, x="year", y="baseline_co2_kg", markers=True, title="CO₂ baseline (kg)", color_discrete_sequence=[color_seq[6]])
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("7) CO₂ baseline vs CO₂ forecast")
    df_co2_plot = pd.concat([
        pd.DataFrame({"year": df["year"], "co2": df["baseline_co2_kg"], "series": "Historical CO2"}),
        pd.DataFrame({"year": forecast_df["year"], "co2": forecast_df["baseline_co2_kg"], "series": "Forecast CO2 Baseline"}),
        pd.DataFrame({"year": forecast_df["year"], "co2": forecast_df["adjusted_co2_kg"], "series": "Forecast CO2 Adjusted"})
    ], ignore_index=True)
    fig7 = px.line(df_co2_plot.sort_values(["year", "series"]), x="year", y="co2", color="series", markers=True,
                   color_discrete_sequence=[color_seq[6], color_seq[0], color_seq[2]], title="CO₂ baseline vs CO₂ forecast")
    st.plotly_chart(fig7, use_container_width=True)

    # Step 5: Tables
    st.header("Step 5 — Tables")
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

    # Step 6: Export & optional DB save
    st.header("Step 6 — Export & Save")
    excel_bytes = df_to_excel_bytes({"historical": df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF (optional)
    images = []
    for fig in (fig2, fig3, fig5, fig7):
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
        ("Historical (baseline)", df[["year", "consumption", "baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"})),
        ("Factors (kWh/year)", df_factors[["device","units","hours_per_year","action","kwh_per_year"]]),
        ("Forecast results", forecast_df)
    ]
    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
    if pdf_bytes:
        st.download_button("📄 Download formal PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export not available (reportlab not installed). Excel export available.")

    st.markdown("---")
    st.subheader("Save results to MySQL (optional)")
    if not MYSQL_AVAILABLE:
        st.info("MySQL not installed on server. Install 'mysql-connector-python' to enable DB features.")
    else:
        colA, colB = st.columns(2)
        with colA:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_db_tables_standard(conn)
                    ensure_app_users_table(conn)
                    conn.close()
                    st.success("DB connection OK. Tables verified/created.")
                except Exception as e:
                    st.error(f"Gagal sambung DB: {e}")
        with colB:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    init_db_tables_standard(conn)
                    save_results_to_db(conn, df, df_factors, forecast_df)
                    conn.close()
                    st.success("Saved historical, factors & forecast to DB.")
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")

# -------------------------
# Device Management
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.markdown("Add and manage device types used in forecasts. (Session-only registry)")
    if "devices" not in st.session_state:
        st.session_state.devices = []
    with st.form("add_device"):
        d_name = st.text_input("Device name (e.g. 'LED 10W')", value="")
        d_watt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        d_note = st.text_input("Note (optional)", value="")
        submitted = st.form_submit_button("Add device")
        if submitted and d_name:
            st.session_state.devices.append({"name": d_name, "watt": d_watt, "note": d_note})
            st.success("Device added.")
    if st.session_state.devices:
        st.table(pd.DataFrame(st.session_state.devices))

# -------------------------
# Reports
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("Use the Energy Forecast screen to export Excel / PDF reports. Downloads are via browser; files are not stored server-side.")

# -------------------------
# Settings (Theme + DB)
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Database")
    st.markdown("Background & DB configuration. Background selection will persist in this session until changed.")

    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"], index=0 if st.session_state.bg_mode == "Dark" else (1 if st.session_state.bg_mode == "Light" else 2))
    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #0E1117 !important; color: #F5F5F5;}
        [data-testid="stSidebar"] {background-color: #000000 !important; color: #FFFFFF;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}
        </style>
        """
        st.session_state.bg_style = {"mode": "Dark", "css": css}
        apply_bg_css()
        st.success("Applied Dark theme for this session.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF !important; color: #000000;}
        [data-testid="stSidebar"] {background-color: #111111 !important; color: #FFFFFF;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}
        </style>
        """
        st.session_state.bg_style = {"mode": "Light", "css": css}
        apply_bg_css()
        st.success("Applied Light theme for this session.")
    else:
        img_url = st.text_input("Enter a full image URL to use as background:", value=st.session_state.bg_style.get("custom_url", ""))
        if st.button("Apply custom background"):
            if img_url:
                css = f"""
                <style>
                [data-testid="stAppViewContainer"] {{
                    background-image: url("{img_url}");
                    background-size: cover;
                    background-position: center;
                }}
                [data-testid="stSidebar"] {{background-color: #000000 !important; color: #FFFFFF;}}
                </style>
                """
                st.session_state.bg_style = {"mode": "Custom", "css": css, "custom_url": img_url}
                apply_bg_css()
                st.success("Applied custom background for this session.")
            else:
                st.error("Please provide a valid image URL.")

    st.markdown("---")
    st.subheader("Database configuration (optional)")
    st.markdown("Enter MySQL connection details here (or set env vars DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE, DB_PORT).")
    db_host = st.text_input("DB host", value=st.session_state.get("db_host", ""))
    db_port = st.text_input("DB port", value=str(st.session_state.get("db_port", "3306")))
    db_user = st.text_input("DB user", value=st.session_state.get("db_user", ""))
    db_password = st.text_input("DB password", value=st.session_state.get("db_password", ""), type="password")
    db_database = st.text_input("DB database", value=st.session_state.get("db_database", ""))
    if st.button("Save DB settings to session"):
        st.session_state["db_host"] = db_host.strip()
        try:
            st.session_state["db_port"] = int(db_port)
        except Exception:
            st.session_state["db_port"] = 3306
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
    📧 **Email:** chikaenergyforecast@gmail.com

    Note: This app uses historical data you upload or enter manually — no physical hardware is required.
    """)

# End of file
