# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App with Login/Register + MySQL (optional)
Features:
- Login / Register (users table in MySQL)
- Theme selector (Dark/Light/Custom image) and background persistent in session
- Menu navigation: Dashboard, Energy Forecast, Device Management, Reports, Settings, Help & About
- Input: Upload CSV or Manual entry (data persisted in session)
- Adjustment factors, forecast (LinearRegression)
- Model accuracy (R^2)
- 7 graphs (colors set bright vs dark)
- Export Excel, optional PDF (reportlab)
- Optional MySQL saving (historical, factors, forecast) and users table for login
"""
import os
import io
from datetime import datetime
import hashlib
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# model & metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# optional libs
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

# MySQL connector optional
MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

# bcrypt optional (fallback to sha256)
USE_BCRYPT = False
try:
    import bcrypt
    USE_BCRYPT = True
except Exception:
    USE_BCRYPT = False

EXCEL_ENGINE = "xlsxwriter"

# ---------- Helpers ----------
def apply_background_style():
    """Apply stored background/theme (persist in session_state)."""
    mode = st.session_state.get("bg_mode", "Dark")
    if mode == "Dark":
        style = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        [data-testid="stSidebar"] {background-color: #0b0c0d;}
        </style>
        """
    elif mode == "Light":
        style = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
        [data-testid="stSidebar"] {background-color: #f2f2f2;}
        </style>
        """
    else:
        img = st.session_state.get("bg_image_url", "")
        if img:
            style = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img}");
                background-size: cover;
                background-position: center;
            }}
            [data-testid="stSidebar"] {{background-color: rgba(0,0,0,0.5);}}
            </style>
            """
        else:
            style = """
            <style>
            [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
            [data-testid="stSidebar"] {background-color: #0b0c0d;}
            </style>
            """
    st.markdown(style, unsafe_allow_html=True)

def hash_password(password: str) -> str:
    if USE_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    else:
        # fallback - salted sha256 (less secure than bcrypt)
        salt = os.environ.get("PW_SALT", "static_salt_please_change")
        return hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(password: str, pw_hash: str) -> bool:
    if USE_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode(), pw_hash.encode())
        except Exception:
            return False
    else:
        salt = os.environ.get("PW_SALT", "static_salt_please_change")
        return hashlib.sha256((salt + password).encode()).hexdigest() == pw_hash

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

# ---------- DB helpers ----------
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
        raise RuntimeError("mysql.connector not installed.")
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

def init_user_table(conn):
    cursor = conn.cursor()
    q = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE,
        password_hash VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    cursor.execute(q)
    conn.commit()
    cursor.close()

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
    # historical
    for _, row in historical_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']), float(row['consumption']),
            float(row.get('baseline_cost')) if not pd.isna(row.get('baseline_cost')) else None,
            float(row.get('fitted')) if not pd.isna(row.get('fitted')) else None,
            None,
            float(row.get('baseline_cost')) if not pd.isna(row.get('baseline_cost')) else None,
            None,
            float(row.get('baseline_co2_kg')) if not pd.isna(row.get('baseline_co2_kg')) else None,
            None
        ))
    # forecast
    for _, row in forecast_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']), float(row['baseline_consumption_kwh']),
            float(row.get('baseline_cost_rm')) if not pd.isna(row.get('baseline_cost_rm')) else None,
            float(row.get('baseline_consumption_kwh')) if not pd.isna(row.get('baseline_consumption_kwh')) else None,
            float(row.get('adjusted_consumption_kwh')) if not pd.isna(row.get('adjusted_consumption_kwh')) else None,
            float(row.get('baseline_cost_rm')) if not pd.isna(row.get('baseline_cost_rm')) else None,
            float(row.get('adjusted_cost_rm')) if not pd.isna(row.get('adjusted_cost_rm')) else None,
            float(row.get('baseline_co2_kg')) if not pd.isna(row.get('baseline_co2_kg')) else None,
            float(row.get('adjusted_co2_kg')) if not pd.isna(row.get('adjusted_co2_kg')) else None
        ))
    insert_f = """
    INSERT INTO energy_factors (device, units, hours_per_year, action, kwh_per_year)
    VALUES (%s,%s,%s,%s,%s)
    """
    for _, r in factors_df.iterrows():
        cursor.execute(insert_f, (str(r['device']), int(r['units']), int(r['hours_per_year']), str(r['action']), float(r['kwh_per_year'])))
    conn.commit()
    cursor.close()

# ---------- init session ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "bg_mode" not in st.session_state:
    st.session_state["bg_mode"] = "Dark"   # Dark/Light/Custom
if "bg_image_url" not in st.session_state:
    st.session_state["bg_image_url"] = ""
# persist data frames so switching menu doesn't reset
for k in ("df", "df_factors", "forecast_df"):
    if k not in st.session_state:
        st.session_state[k] = None

apply_background_style()

# ---------- Top-level login/register ----------
def login_register_ui():
    st.markdown("<h1 style='color:white;'>🔐 Please login to access the dashboard</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("🔑 Login")
        uname = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            if MYSQL_AVAILABLE:
                try:
                    conn = connect_db()
                    init_user_table(conn)
                    cur = conn.cursor(dictionary=True)
                    cur.execute("SELECT * FROM users WHERE username=%s", (uname,))
                    row = cur.fetchone()
                    cur.close()
                    conn.close()
                    if row and verify_password(pwd, row["password_hash"]):
                        st.session_state["logged_in"] = True
                        st.session_state["user"] = uname
                        st.success("Login successful.")
                    else:
                        st.error("Nama pengguna atau kata laluan salah!")
                except Exception as e:
                    st.error(f"Gagal sambung DB untuk login: {e}")
            else:
                # fallback: local session users (for dev only)
                loc_users = st.session_state.get("local_users", {})
                if uname in loc_users and verify_password(pwd, loc_users[uname]):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = uname
                    st.success("Login successful (local).")
                else:
                    st.error("Nama pengguna atau kata laluan salah!")
    with col2:
        st.subheader("📝 Register")
        r_uname = st.text_input("New username", key="reg_user")
        r_pwd = st.text_input("New password", type="password", key="reg_pwd")
        if st.button("Register"):
            if not r_uname or not r_pwd:
                st.error("Isi username & password.")
            else:
                pw_hash = hash_password(r_pwd)
                if MYSQL_AVAILABLE:
                    try:
                        conn = connect_db()
                        init_user_table(conn)
                        cur = conn.cursor()
                        cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (r_uname, pw_hash))
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success("Registered successfully. You can login now.")
                    except mysql.connector.IntegrityError:
                        st.error("Username already exists.")
                    except Exception as e:
                        st.error(f"Gagal register ke DB: {e}")
                else:
                    # local fallback
                    if "local_users" not in st.session_state:
                        st.session_state["local_users"] = {}
                    if r_uname in st.session_state["local_users"]:
                        st.error("Username exists (local).")
                    else:
                        st.session_state["local_users"][r_uname] = pw_hash
                        st.success("Registered locally. Use Login to proceed.")

# If not logged in -> show login/register and stop
if not st.session_state.get("logged_in", False):
    login_register_ui()
    st.stop()

# ---------- Sidebar + Menu ----------
st.sidebar.markdown("<div style='background:#000; padding:12px; border-radius:6px;'><h3 style='color:#fff;'>🔹 Smart Energy Forecasting</h3></div>", unsafe_allow_html=True)
menu = st.sidebar.radio("Navigate:", ["Dashboard", "Energy Forecast", "Device Management", "Reports", "Settings", "Help & About"])

# show user info and logout
st.sidebar.write(f"**User:** {st.session_state.get('user')}")
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.experimental_rerun()

# ---------- Color palette for graphs ----------
COLORS = {
    "red": "#FF4C4C",
    "blue": "#0050A0",
    "green": "#00B050",
    "orange": "#FFA500",
    "purple": "#8000FF",
    "yellow": "#FFD700",
    "grey": "#808080"
}

# ---------- DASHBOARD ----------
if menu == "Dashboard":
    st.title("🏠 Dashboard — Smart Energy Forecasting")
    st.markdown("""
    Welcome — use the left menu to go to Energy Forecast, manage devices, or download reports.
    - Forecast energy & cost, compare baseline vs adjusted scenarios.
    - Exports: Excel & optional PDF.
    """)
    st.markdown("**Quick actions:**")
    c1, c2, c3 = st.columns(3)
    c1.button("Go to Energy Forecast", key="goto_forecast")
    c2.button("Device Management", key="goto_devices")
    c3.button("Download last Excel", key="dl_last")
    st.markdown("---")
    st.markdown("**Current session data**")
    st.write("Historical rows:", None if st.session_state["df"] is None else len(st.session_state["df"]))
    st.write("Factors rows:", None if st.session_state["df_factors"] is None else len(st.session_state["df_factors"]))
    if st.session_state["df"] is not None:
        st.dataframe(st.session_state["df"])

# ---------- ENERGY FORECAST ----------
elif menu == "Energy Forecast":
    st.title("⚡ Energy Forecast")
    # Step 1: Input (persist in session)
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))
    df = st.session_state.get("df")
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (needs 'year' & 'consumption' columns)", type=["csv", "xlsx"])
        if uploaded is not None:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
            if "year" not in df_raw.columns or not any("consum" in c or "kwh" in c or "energy" in c for c in df_raw.columns):
                st.error("CSV must contain 'year' and a consumption column.")
            else:
                year_col = "year"
                cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum","kwh","energy"])][0]
                df = pd.DataFrame({"year": df_raw[year_col].astype(int), "consumption": pd.to_numeric(df_raw[cons_col], errors="coerce")})
                cost_cols = [c for c in df_raw.columns if "cost" in c]
                if cost_cols:
                    df["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
                else:
                    df["baseline_cost"] = np.nan
                st.session_state["df"] = df.copy()
    else:
        if st.session_state.get("df") is None:
            rows = st.number_input("Number of historical rows:", min_value=1, max_value=50, value=5)
            data = []
            for i in range(int(rows)):
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i, key=f"m_year_{i}")
                with c2:
                    cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, 10000.0, key=f"m_cons_{i}")
                with c3:
                    cost = st.number_input(f"Baseline cost RM ({y}) (optional)", 0.0, 10_000_000.0, 0.0, key=f"m_cost_{i}")
                data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
            df = pd.DataFrame(data)
            st.session_state["df"] = df.copy()
        else:
            st.info("Using data from session. Change in Settings or reload to clear.")
            df = st.session_state["df"]
    if df is None or df.empty:
        st.warning("Please upload or enter data to continue.")
        st.stop()

    # Ensure types & sorting
    df["year"] = df["year"].astype(int)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0.0)
    if "baseline_cost" not in df.columns:
        df["baseline_cost"] = np.nan
    df["baseline_cost"] = pd.to_numeric(df["baseline_cost"], errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)
    st.subheader("Loaded baseline data")
    st.dataframe(df)
    st.session_state["df"] = df.copy()

    # Step 2: Factors (persist)
    st.header("Step 2 — Adjustment factors (additions or reductions)")
    st.markdown("Enter device-level adjustments. Hours are per YEAR.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}
    n_factors = st.number_input("How many factor rows to add?", min_value=1, max_value=10, value=1, key="f_n")
    factor_rows = []
    # if session has saved factors and n_factors matches, reuse
    saved_factors = st.session_state.get("df_factors")
    if saved_factors is not None and len(saved_factors) == n_factors:
        df_factors = saved_factors.copy()
    else:
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
            factor_rows.append({"device": dev_name, "units": int(units), "hours_per_year": int(hours), "action": action, "kwh_per_year": kwh_per_year})
        df_factors = pd.DataFrame(factor_rows)
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)
    st.session_state["df_factors"] = df_factors.copy()

    # general site change
    st.markdown("General site-level hours change (positive = add load, negative = reduce load)")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=0, key="gen_hours")
    general_avg_load_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1, key="gen_kw")
    general_kwh = float(general_avg_load_kw) * float(general_hours) if general_hours != 0 else 0.0
    total_net_adjust_kwh = df_factors["kwh_per_year"].sum() + general_kwh
    if total_net_adjust_kwh > 0:
        st.info(f"Net adjustment (additional): {total_net_adjust_kwh:,.2f} kWh/year")
    elif total_net_adjust_kwh < 0:
        st.info(f"Net adjustment (reduction): {abs(total_net_adjust_kwh):,.2f} kWh/year")
    else:
        st.info("Net adjustment: 0 kWh/year")

    # Step 3: Forecast compute (persist forecast_df)
    st.header("Step 3 — Forecast settings & compute")
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01, key="tariff")
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01, key="co2")
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1, key="fy")

    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

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
    future_years = [last_year + i for i in range(1, int(n_years_forecast)+1)]
    future_X = np.array(future_years).reshape(-1,1)
    future_baseline_forecast = model.predict(future_X) if len(X_hist) >= 2 else np.array([df["consumption"].iloc[-1]]*len(future_years))

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

    st.session_state["forecast_df"] = forecast_df.copy()
    st.session_state["df"] = df.copy()
    st.session_state["df_factors"] = df_factors.copy()

    # Step 4: Visualizations (7 graphs)
    st.header("Step 4 — Visual comparisons & model accuracy")
    col_main, col_side = st.columns([2,1])
    with col_main:
        # 1 Baseline only (historical)
        st.subheader("1) Baseline (historical)")
        fig1 = px.line(df, x="year", y="consumption", title="Baseline (historical)", markers=True)
        fig1.update_traces(line=dict(color=COLORS["blue"]))
        st.plotly_chart(fig1, use_container_width=True)

        # 2 Baseline vs Forecast
        st.subheader("2) Baseline vs Forecast")
        plot_all = pd.concat([
            pd.DataFrame({"year": df["year"], "baseline": df["consumption"]}),
            pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"]})
        ], ignore_index=True)
        fig2 = px.line(plot_all.sort_values("year"), x="year", y=["baseline"], title="Baseline (historical + forecast overlay)", markers=True)
        fig2.update_traces(line=dict(color=COLORS["blue"]))
        # overlay forecast adjusted as separate line in same chart
        fig2.add_scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"], mode="lines+markers", name="Baseline forecast", line=dict(color=COLORS["grey"], dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

        # 3 Adjusted vs Forecast vs Baseline
        st.subheader("3) Adjusted vs Forecast vs Baseline")
        fig3 = px.line(forecast_df, x="year", y=["baseline_consumption_kwh","adjusted_consumption_kwh"], markers=True,
                       labels={"value":"kWh","variable":"Series"}, title="Baseline vs Adjusted (forecast period)")
        fig3.update_traces(selector=dict(name="baseline_consumption_kwh"), line=dict(color=COLORS["blue"]))
        fig3.update_traces(selector=dict(name="adjusted_consumption_kwh"), line=dict(color=COLORS["red"]))
        st.plotly_chart(fig3, use_container_width=True)

        # 4 Baseline cost (historical + forecast baseline)
        st.subheader("4) Baseline cost")
        # baseline historical cost
        df_cost_hist = pd.DataFrame({"year": df["year"], "cost": df["baseline_cost"]})
        fig4 = px.line(df_cost_hist.sort_values("year"), x="year", y="cost", title="Baseline cost (historical)", markers=True)
        fig4.update_traces(line=dict(color=COLORS["green"]))
        st.plotly_chart(fig4, use_container_width=True)

        # 5 Forecast cost vs Baseline cost
        st.subheader("5) Forecast cost vs Baseline cost")
        fig5 = px.bar(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], barmode="group", title="Forecast cost vs Baseline cost")
        st.plotly_chart(fig5, use_container_width=True)

        # 6 CO2 baseline (historical)
        st.subheader("6) CO₂ baseline (historical)")
        fig6 = px.line(df, x="year", y="baseline_co2_kg", title="CO₂ baseline (historical)", markers=True)
        fig6.update_traces(line=dict(color=COLORS["orange"]))
        st.plotly_chart(fig6, use_container_width=True)

        # 7 CO2 baseline vs CO2 forecast
        st.subheader("7) CO₂ baseline vs CO₂ forecast")
        # create series for historical baseline and forecast baseline
        hist_co2 = pd.DataFrame({"year": df["year"], "co2": df["baseline_co2_kg"]})
        fc_co2 = pd.DataFrame({"year": forecast_df["year"], "co2_baseline": forecast_df["baseline_co2_kg"], "co2_adjusted": forecast_df["adjusted_co2_kg"]})
        fig7 = px.line(fc_co2, x="year", y=["co2_baseline","co2_adjusted"], markers=True, title="CO₂ baseline vs CO₂ forecast")
        fig7.update_traces(selector=dict(name="co2_baseline"), line=dict(color=COLORS["purple"]))
        fig7.update_traces(selector=dict(name="co2_adjusted"), line=dict(color=COLORS["yellow"]))
        st.plotly_chart(fig7, use_container_width=True)

    with col_side:
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

    # Step 5: Tables and exports
    st.header("Step 5 — Tables & Exports")
    st.subheader("Baseline table (historical)")
    st.dataframe(df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}))
    st.subheader("Forecast results table")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh":"{:.0f}",
        "adjusted_consumption_kwh":"{:.0f}",
        "baseline_cost_rm":"{:.2f}",
        "adjusted_cost_rm":"{:.2f}",
        "saving_kwh":"{:.0f}",
        "saving_cost_rm":"{:.2f}",
        "saving_co2_kg":"{:.0f}"
    }))

    # Exports
    excel_bytes = df_to_excel_bytes({"historical": df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    images = []
    for fig in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        png = try_get_plot_png(fig)
        if png:
            images.append(png)
    if REPORTLAB_AVAILABLE:
        summary_lines = [
            f"Forecast period: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
            f"Net adjustment (kWh/year): {total_net_adjust_kwh:.2f}",
            f"Total energy saving (kWh): {total_kwh_saving:,.2f}",
            f"Total cost saving (RM): RM {total_cost_saving:,.2f}",
            f"Total CO₂ reduction (kg): {total_co2_saving:,.2f}",
            f"Model R²: {r2:.4f}"
        ]
        table_blocks = [
            ("Historical (baseline)", df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"})),
            ("Factors (kWh/year)", df_factors[["device","units","hours_per_year","action","kwh_per_year"]]),
            ("Forecast results", forecast_df)
        ]
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
        if pdf_bytes:
            st.download_button("📄 Download formal PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export not available (reportlab not installed). Excel export available.")

    # Optional DB save (init tables if Test DB successful)
    st.markdown("---")
    st.subheader("Optional: Save results to MySQL (Railway)")
    if not MYSQL_AVAILABLE:
        st.info("MySQL support not installed on host. Install mysql-connector-python to enable.")
    else:
        colA, colB = st.columns(2)
        with colA:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_user_table(conn)
                    init_db_tables(conn)
                    conn.close()
                    st.success("DB connection OK and tables ready.")
                except Exception as e:
                    st.error(f"DB connection failed: {e}")
        with colB:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_results_to_db(conn, df, df_factors, forecast_df)
                    conn.close()
                    st.success("Saved data to DB.")
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")

# ---------- Device Management ----------
elif menu == "Device Management":
    st.title("💡 Device Management")
    st.markdown("Add and manage device types used in forecasts.")
    if "devices" not in st.session_state:
        st.session_state["devices"] = []
    with st.form("add_device", clear_on_submit=True):
        d_name = st.text_input("Device name (e.g. 'LED 10W')", value="")
        d_watt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        d_note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Add device")
        if submitted and d_name:
            st.session_state["devices"].append({"name":d_name,"watt":d_watt,"note":d_note})
            st.success("Device added.")
    if st.session_state["devices"]:
        st.table(pd.DataFrame(st.session_state["devices"]))

# ---------- Reports ----------
elif menu == "Reports":
    st.title("📊 Reports")
    st.markdown("Exports from Energy Forecast are available via downloads. Use Settings to adjust PDF/Excel options.")
    # show last exported summary
    if st.session_state.get("forecast_df") is not None:
        st.subheader("Last forecast snapshot")
        st.dataframe(st.session_state["forecast_df"])

# ---------- Settings ----------
elif menu == "Settings":
    st.title("⚙️ Settings — Appearance & Database")
    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"])
    if choice == "Dark (default)":
        st.session_state["bg_mode"] = "Dark"
        apply_background_style()
        st.success("Applied Dark theme.")
    elif choice == "Light":
        st.session_state["bg_mode"] = "Light"
        apply_background_style()
        st.success("Applied Light theme.")
    else:
        img_url = st.text_input("Enter image URL to use as background:", value=st.session_state.get("bg_image_url",""))
        if st.button("Apply image background"):
            st.session_state["bg_mode"] = "Custom"
            st.session_state["bg_image_url"] = img_url
            apply_background_style()
            st.success("Custom background applied and saved in session.")

    st.markdown("---")
    st.subheader("Database configuration (optional)")
    st.markdown("Enter MySQL connection details here (or set env vars DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE, DB_PORT).")
    db_host = st.text_input("DB host", value=st.session_state.get("db_host",""))
    db_port = st.text_input("DB port", value=str(st.session_state.get("db_port","3306")))
    db_user = st.text_input("DB user", value=st.session_state.get("db_user",""))
    db_password = st.text_input("DB password", value=st.session_state.get("db_password",""), type="password")
    db_database = st.text_input("DB database", value=st.session_state.get("db_database",""))
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
    st.markdown("**Security:** Password hashing uses bcrypt if available; otherwise SHA256 is used as fallback. For production, install bcrypt and set a secure PW_SALT in env.")

# ---------- Help & About ----------
elif menu == "Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed for forecasting and scenario comparison of energy consumption, cost and CO₂.

    **Support / Report issues:**  
    📧 **Email:** chikaenergyforecast@gmail.com

    Notes:
    - This app persists UI choices (background, theme) in session.
    - Use Settings to configure DB connection; after configure use Test DB and Save to DB on Energy Forecast page.
    """)
