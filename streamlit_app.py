# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App with optional MySQL saving
Features:
- Login/Register (users table) with auto-detect hash/plaintext
- Persistent theme/background (session_state)
- Sidebar menu (black) responsive and persistent
- Energy Forecast: Upload CSV or Manual, adjustable factors, LinearRegression forecast
- 7 graphs (colors bright vs dark)
- Export Excel, optional PDF (reportlab), optional MySQL save (Railway)
- Persist form data in session_state so switching menus won't lose inputs
"""
import os
import io
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional libs
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    BCRYPT_AVAILABLE = False

# PDF libs (optional)
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# plotly image
PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Page config and default theme
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide", initial_sidebar_state="expanded")

# initialize session defaults
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"
if "bg_custom_url" not in st.session_state:
    st.session_state.bg_custom_url = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
# persist data across menus
if "hist_df" not in st.session_state:
    st.session_state.hist_df = None
if "factors_df" not in st.session_state:
    st.session_state.factors_df = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None

# default CSS: black sidebar, persistent background container
def apply_theme_css():
    # sidebar black and solid for mobile
    sidebar_style = """
    <style>
    /* App container background */
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #F5F5F5;
        transition: background 0.3s ease;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        color: #fff;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] .css-1d391kg { /* label styling fallback */
        color: #fff;
    }
    /* Make sidebar full-height and fixed look */
    .css-1d391kg { color: #fff; }
    /* Ensure header transparent */
    [data-testid="stHeader"] {background: rgba(0,0,0,0); }
    /* Avoid overlay issues on mobile: sidebar items text white and icons remain visible */
    .block-container { padding-top: 1rem; }
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

# apply persistent background (dark, light, or custom image)
def apply_background():
    if st.session_state.bg_mode == "Dark":
        apply_theme_css()
        # background already dark via CSS above
    elif st.session_state.bg_mode == "Light":
        light_style = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #111111;}
        [data-testid="stSidebar"] {background-color: #f7f7f7 !important; color: #111;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        </style>
        """
        st.markdown(light_style, unsafe_allow_html=True)
    else:
        # custom image
        url = st.session_state.get("bg_custom_url","")
        if url:
            custom = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{url}");
                background-size: cover;
                background-position: center;
                color: #fff;
            }}
            [data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.55) !important; }}
            </style>
            """
            st.markdown(custom, unsafe_allow_html=True)
        else:
            apply_theme_css()

apply_background()

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

# password utilities: registration uses bcrypt if available, else sha256
def hash_password(pw: str) -> str:
    if BCRYPT_AVAILABLE:
        ph = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode()
        return ph
    else:
        return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def check_password(stored: str, provided: str) -> bool:
    """
    Auto-detect:
    - if stored starts with $2 (bcrypt), use bcrypt
    - elif stored length == 64 -> sha256 hex
    - else compare plaintext equality
    Also accept plaintext match if stored equals provided.
    """
    if stored is None:
        return False
    try:
        # direct plaintext match (support legacy)
        if stored == provided:
            return True
        # bcrypt
        if BCRYPT_AVAILABLE and (stored.startswith("$2a$") or stored.startswith("$2b$") or stored.startswith("$2y$")):
            return bcrypt.checkpw(provided.encode("utf-8"), stored.encode("utf-8"))
        # sha256 hex detection
        if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored.lower()):
            return hashlib.sha256(provided.encode("utf-8")).hexdigest() == stored
        # fallback: compare stored == provided
        return stored == provided
    except Exception:
        return False

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
        raise RuntimeError("mysql.connector not installed.")
    if not cfg['host'] or not cfg['user'] or not cfg['database']:
        raise ValueError("DB host/user/database not configured.")
    conn = mysql.connector.connect(
        host=cfg['host'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        port=cfg['port'],
        connection_timeout=timeout
    )
    return conn

def init_users_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE,
        password_hash VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """)
    conn.commit()
    cur.close()

def create_user_db(conn, username, password_hash):
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username, password_hash) VALUES (%s,%s)", (username, password_hash))
    conn.commit()
    cur.close()

def get_user_record(conn, username):
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username, password_hash FROM users WHERE username=%s", (username,))
    row = cur.fetchone()
    cur.close()
    return row

# -------------------------
# Authentication UI (login/register)
# -------------------------
def login_register_page():
    st.title("🔐 Please login to access the dashboard")
    st.markdown("**Secure Login**")
    col1, col2 = st.columns([1,1])
    with col1:
        uname = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            # try DB if configured else check local simplistic register (session only)
            if MYSQL_AVAILABLE and st.session_state.get("db_host"):
                try:
                    conn = connect_db()
                    init_users_table(conn)
                    user = get_user_record(conn, uname)
                    conn.close()
                    if user and check_password(user.get("password_hash",""), pwd):
                        st.session_state.logged_in = True
                        st.session_state.user = uname
                        st.success(f"Logged in as {uname}")
                    else:
                        st.error("Nama pengguna atau kata laluan salah!")
                except Exception as e:
                    st.error(f"Gagal sambung DB: {e}")
            else:
                # fallback: check session-state 'local_users'
                local = st.session_state.get("local_users", {})
                stored = local.get(uname)
                if stored and check_password(stored, pwd):
                    st.session_state.logged_in = True
                    st.session_state.user = uname
                    st.success(f"Logged in as {uname} (local)")
                else:
                    st.error("Nama pengguna atau kata laluan salah!")
    with col2:
        st.markdown("**Register (create new account)**")
        runame = st.text_input("New username", key="reg_user")
        rpass = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Register"):
            if not runame or not rpass:
                st.error("Please provide username & password")
            else:
                ph = hash_password(rpass)
                if MYSQL_AVAILABLE and st.session_state.get("db_host"):
                    try:
                        conn = connect_db()
                        init_users_table(conn)
                        # try create
                        create_user_db(conn, runame, ph)
                        conn.close()
                        st.success("User created in DB. You can now login.")
                    except mysql.connector.IntegrityError:
                        st.error("Username already exists.")
                    except Exception as e:
                        st.error(f"DB error: {e}")
                else:
                    # store locally (session only)
                    local = st.session_state.get("local_users", {})
                    if runame in local:
                        st.error("Username already exists (local).")
                    else:
                        local[runame] = ph
                        st.session_state["local_users"] = local
                        st.success("User created locally (session only).")

# if not logged in show login page only
if not st.session_state.logged_in:
    login_register_page()
    st.stop()

# -------------------------
# Sidebar / Navigation (after login)
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["Dashboard", "Energy Forecast", "Device Management",
                                     "Reports", "Settings", "Help & About"], index=0)

# quick logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.experimental_rerun = getattr(st, "experimental_rerun", None)
    # If experimental_rerun exists, call it; else just reload the page by setting stop
    if callable(st.experimental_rerun):
        st.experimental_rerun()
    else:
        st.stop()

# -------------------------
# Global color palette for graphs (bright vs dark)
# -------------------------
COLORS = {
    "red": "#FF4C4C",
    "navy": "#002f6c",
    "green": "#00B050",
    "orange": "#FFA500",
    "purple": "#8000FF",
    "yellow": "#FFD700",
    "grey": "#808080"
}

# -------------------------
# Pages
# -------------------------
if menu == "Dashboard":
    st.title("🏠 Dashboard")
    st.markdown(f"Hello **{st.session_state.user}** — welcome to Smart Energy Forecasting.")
    st.markdown("- Use **Energy Forecast** to upload historical data (CSV) or enter manually, adjust factors and run forecasts.")
    st.markdown("- Settings → save DB credentials to enable Railway MySQL saving.")
    st.markdown("Quick summary (if available):")
    if st.session_state.hist_df is not None:
        st.metric("Historical rows", len(st.session_state.hist_df))
    else:
        st.info("No historical data loaded yet (Energy Forecast).")

# -------------------------
# ENERGY FORECAST
# -------------------------
elif menu == "Energy Forecast":
    st.title("⚡ Energy Forecast")
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    # load saved session or new
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (must contain year + consumption)", type=["csv","xlsx"])
        if uploaded:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)
            # check columns
            if "year" not in df_raw.columns or not any("consum" in c or "kwh" in c or "energy" in c for c in df_raw.columns):
                st.error("CSV must contain 'year' and a consumption column.")
                st.stop()
            cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum","kwh","energy"])][0]
            df = pd.DataFrame({"year": df_raw["year"].astype(int), "consumption": pd.to_numeric(df_raw[cons_col], errors="coerce")})
            cost_cols = [c for c in df_raw.columns if "cost" in c]
            if cost_cols:
                df["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
            else:
                df["baseline_cost"] = np.nan
            st.session_state.hist_df = df.sort_values("year").reset_index(drop=True)
    else:
        # Manual entry - use session saved values to avoid losing when change menu
        rows = st.number_input("Number of historical rows:", min_value=1, max_value=20, value=st.session_state.get("manual_rows",5))
        st.session_state["manual_rows"] = int(rows)
        data = []
        # if existing session hist_df exists and matches length, prefill
        existing = st.session_state.get("hist_df")
        for i in range(int(rows)):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                default_year = int(existing["year"].iloc[i]) if (existing is not None and i < len(existing)) else (2020 + i)
                y = st.number_input(f"Year {i+1}", 2000, 2100, value=default_year, key=f"m_year_{i}")
            with c2:
                default_cons = float(existing["consumption"].iloc[i]) if (existing is not None and i < len(existing)) else 10000.0
                cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, value=default_cons, key=f"m_cons_{i}")
            with c3:
                default_cost = float(existing["baseline_cost"].iloc[i]) if (existing is not None and i < len(existing) and not np.isnan(existing["baseline_cost"].iloc[i])) else 0.0
                cost = st.number_input(f"Baseline cost RM ({y}) (0 to compute)", 0.0, 10_000_000.0, value=default_cost, key=f"m_cost_{i}")
            data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
        st.session_state.hist_df = pd.DataFrame(data)

    # show loaded data
    if st.session_state.hist_df is None or st.session_state.hist_df.empty:
        st.warning("Please load or enter baseline data to continue.")
        st.stop()

    df = st.session_state.hist_df.copy()
    df["year"] = df["year"].astype(int)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0.0)
    df["baseline_cost"] = pd.to_numeric(df.get("baseline_cost", np.nan), errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)
    st.subheader("Loaded baseline data")
    st.dataframe(df)

    # Step 2: Factors
    st.header("Step 2 — Adjustment factors (additions or reductions)")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}
    n_factors = st.number_input("How many factor rows to add?", min_value=1, max_value=10, value=st.session_state.get("n_factors",1))
    st.session_state["n_factors"] = int(n_factors)
    factor_rows = []
    existing_factors = st.session_state.get("factors_df")
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", options=["Lamp - LED","Lamp - CFL","Lamp - Fluorescent","Computer","Lab Equipment"], key=f"dev_{i}")
        with c2:
            default_units = int(existing_factors["units"].iloc[i]) if (existing_factors is not None and i < len(existing_factors)) else 0
            units = st.number_input(f"Units", min_value=0, value=default_units, step=1, key=f"units_{i}")
        with c3:
            default_hours = int(existing_factors["hours_per_year"].iloc[i]) if (existing_factors is not None and i < len(existing_factors)) else 0
            hours = st.number_input(f"Hours per YEAR", min_value=0, max_value=8760, value=default_hours, step=1, key=f"hours_{i}")
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
    st.session_state.factors_df = df_factors
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)

    # site-level change
    st.markdown("General site-level hours change (positive = add load, negative = reduce load)")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=st.session_state.get("general_hours",0))
    st.session_state["general_hours"] = int(general_hours)
    general_avg_load_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=float(st.session_state.get("general_avg_load_kw",2.0)), step=0.1)
    st.session_state["general_avg_load_kw"] = float(general_avg_load_kw)
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
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=st.session_state.get("tariff",0.52), step=0.01)
    st.session_state["tariff"] = float(tariff)
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=st.session_state.get("co2_factor",0.75), step=0.01)
    st.session_state["co2_factor"] = float(co2_factor)
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=st.session_state.get("n_years_forecast",3), step=1)
    st.session_state["n_years_forecast"] = int(n_years_forecast)

    # prepare baseline cost and co2
    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    # linear regression model
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
    future_X = np.array(future_years).reshape(-1,1)
    if len(X_hist) >= 2:
        future_baseline_forecast = model.predict(future_X)
    else:
        future_baseline_forecast = np.array([df["consumption"].iloc[-1]] * len(future_years))

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

    # persist in session
    st.session_state.hist_df = df.copy()
    st.session_state.factors_df = df_factors.copy()
    st.session_state.forecast_df = forecast_df.copy()

    # Step 4: Visualizations (7 graphs)
    st.header("Step 4 — Visual comparisons & model accuracy")
    colA, colB = st.columns([2,1])
    with colA:
        # 1. Baseline only (historical)
        fig1 = px.line(df, x="year", y="consumption", markers=True, title="Baseline (historical) - Consumption (kWh)")
        fig1.update_traces(line=dict(color=COLORS["navy"]))

        # 2. Baseline vs Forecast (baseline historical + baseline forecast)
        combined = pd.concat([
            df[["year","consumption"]].rename(columns={"consumption":"baseline_kwh"}),
            pd.DataFrame({"year": forecast_df["year"], "baseline_kwh": forecast_df["baseline_consumption_kwh"]})
        ], ignore_index=True).sort_values("year")
        fig2 = px.line(combined, x="year", y="baseline_kwh", markers=True, title="Baseline (historical + forecast)")
        fig2.update_traces(line=dict(color=COLORS["navy"]))

        # 3. Adjusted vs Forecast vs Baseline (future)
        df_plot3 = forecast_df[["year","baseline_consumption_kwh","adjusted_consumption_kwh"]]
        fig3 = px.line(df_plot3, x="year", y=["baseline_consumption_kwh","adjusted_consumption_kwh"], markers=True,
                       title="Baseline vs Adjusted (forecast period)")
        fig3.update_traces(selector=dict(name="baseline_consumption_kwh"), line=dict(color=COLORS["navy"]))
        fig3.update_traces(selector=dict(name="adjusted_consumption_kwh"), line=dict(color=COLORS["red"]))

        # 4. Baseline cost (historical + forecast baseline)
        hist_cost = pd.DataFrame({"year": df["year"], "baseline_cost_rm": df["baseline_cost"]})
        future_cost = pd.DataFrame({"year": forecast_df["year"], "baseline_cost_rm": forecast_df["baseline_cost_rm"]})
        cost_comb = pd.concat([hist_cost, future_cost], ignore_index=True).sort_values("year")
        fig4 = px.line(cost_comb, x="year", y="baseline_cost_rm", markers=True, title="Baseline cost (RM)")
        fig4.update_traces(line=dict(color=COLORS["purple"]))

        # 5. Forecast cost vs Baseline cost (forecast period)
        fig5 = px.bar(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], barmode="group", title="Forecast cost vs Baseline cost")
        # pick colors
        fig5.update_traces(selector=dict(name="baseline_cost_rm"), marker_color=COLORS["grey"])
        fig5.update_traces(selector=dict(name="adjusted_cost_rm"), marker_color=COLORS["orange"])

        # 6. CO2 baseline (historical)
        fig6 = px.line(df, x="year", y="baseline_co2_kg", markers=True, title="CO₂ baseline (kg)")
        fig6.update_traces(line=dict(color=COLORS["green"]))

        # 7. CO2 baseline vs CO2 forecast
        co2_comb = pd.DataFrame({"year": df["year"], "baseline_co2": df["baseline_co2_kg"]})
        co2_future = pd.DataFrame({"year": forecast_df["year"], "baseline_co2": forecast_df["baseline_co2_kg"], "forecast_co2": forecast_df["adjusted_co2_kg"]})
        co2_plot = pd.concat([co2_comb, co2_future[["year","baseline_co2","forecast_co2"]].rename(columns={"forecast_co2":"adjusted_co2"})], ignore_index=True).sort_values("year")
        # To plot both series, create fig with separate traces
        fig7 = px.line(co2_plot.sort_values("year"), x="year", y=["baseline_co2","adjusted_co2"], markers=True, title="CO₂: Baseline vs Forecast")
        fig7.update_traces(selector=dict(name="baseline_co2"), line=dict(color=COLORS["green"]))
        fig7.update_traces(selector=dict(name="adjusted_co2"), line=dict(color=COLORS["navy"]))

        # show charts stacked (responsive)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.plotly_chart(fig6, use_container_width=True)
        st.plotly_chart(fig7, use_container_width=True)

    with colB:
        st.subheader("Model performance & totals")
        st.markdown(f"**R²:** `{r2:.4f}`")
        if r2 >= 0.8:
            st.success("Model accuracy: High")
        elif r2 >= 0.6:
            st.warning("Model accuracy: Moderate")
        else:
            st.error("Model accuracy: Low — consider more data or more features")

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
    st.header("Step 5 — Tables")
    st.subheader("Historical (baseline)")
    st.dataframe(df.rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}))
    st.subheader("Forecast results (table)")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh":"{:.0f}",
        "adjusted_consumption_kwh":"{:.0f}",
        "baseline_cost_rm":"{:.2f}",
        "adjusted_cost_rm":"{:.2f}",
        "saving_kwh":"{:.0f}",
        "saving_cost_rm":"{:.2f}",
        "saving_co2_kg":"{:.0f}"
    }))
    # comparison summary
    summary = pd.DataFrame([{
        "metric":"Total baseline kWh (forecast period)",
        "value": f"{total_baseline_kwh:,.0f}"
    },{
        "metric":"Total adjusted kWh (forecast period)",
        "value": f"{total_adjusted_kwh:,.0f}"
    },{
        "metric":"Total energy saving (kWh)",
        "value": f"{total_kwh_saving:,.0f}"
    },{
        "metric":"Total cost saving (RM)",
        "value": f"RM {total_cost_saving:,.2f}"
    },{
        "metric":"Total CO2 reduction (kg)",
        "value": f"{total_co2_saving:,.0f}"
    }])
    st.subheader("Comparison summary")
    st.table(summary)

    # Step 6: Export & optional DB save
    st.header("Step 6 — Export & Save")
    excel_bytes = df_to_excel_bytes({"historical": df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    images = []
    for fig in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        png = try_get_plot_png(fig)
        if png:
            images.append(png)

    # PDF generation (optional)
    if REPORTLAB_AVAILABLE:
        # simplified PDF builder
        def build_pdf_bytes():
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = [Paragraph("Smart Energy Forecasting Report", styles["Title"]),
                        Spacer(1,8),
                        Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}", styles["Normal"]),
                        Spacer(1,12)]
            for line in summary["value"].tolist():
                elements.append(Paragraph(line, styles["Normal"]))
                elements.append(Spacer(1,6))
            # tables
            data_tbl = [list(df.rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}).columns)]
            data_tbl += df.rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}).astype(str).values.tolist()
            tbl = Table(data_tbl, repeatRows=1)
            tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25, colors.grey)]))
            elements.append(tbl)
            try:
                doc.build(elements)
                return buf.getvalue()
            except Exception:
                return None
        pdf_bytes = build_pdf_bytes()
        if pdf_bytes:
            st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
        else:
            st.info("PDF report generation failed.")
    else:
        st.info("PDF export not available (reportlab not installed).")

    # DB Save
    st.markdown("---")
    st.subheader("Optional: Save results to MySQL database")
    if not MYSQL_AVAILABLE:
        st.info("MySQL connector not installed in environment. Install 'mysql-connector-python' to enable DB features.")
    else:
        st.markdown("Set DB credentials in Settings then Test/Save.")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_users_table(conn)  # ensure users exists for auth; also create energy tables below
                    # create energy tables if not exists
                    cur = conn.cursor()
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS energy_data (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            year INT,
                            consumption DOUBLE,
                            baseline_cost DOUBLE,
                            fitted DOUBLE,
                            adjusted DOUBLE,
                            baseline_cost_rm DOUBLE,
                            adjusted_cost_rm DOUBLE,
                            baseline_co2_kg DOUBLE,
                            adjusted_co2_kg DOUBLE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ) ENGINE=InnoDB;""")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS energy_factors (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            device VARCHAR(128),
                            units INT,
                            hours_per_year INT,
                            action VARCHAR(32),
                            kwh_per_year DOUBLE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ) ENGINE=InnoDB;""")
                    conn.commit(); cur.close(); conn.close()
                    st.success("DB connection ok and tables ensured.")
                except Exception as e:
                    st.error(f"DB connection failed: {e}")
        with c2:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    # insert energy_data & factors simple append
                    cur = conn.cursor()
                    insert_ed = ("INSERT INTO energy_data (year, consumption, baseline_cost, fitted, adjusted, baseline_cost_rm, adjusted_cost_rm, baseline_co2_kg, adjusted_co2_kg) "
                                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                    for _, row in df.iterrows():
                        cur.execute(insert_ed, (int(row["year"]), float(row["consumption"]), float(row.get("baseline_cost",0)), 
                                                float(row.get("fitted", None)) if not pd.isna(row.get("fitted", None)) else None,
                                                None, float(row.get("baseline_cost",0)), None,
                                                float(row.get("baseline_co2_kg", None)) if not pd.isna(row.get("baseline_co2_kg", None)) else None,
                                                None))
                    for _, row in forecast_df.iterrows():
                        cur.execute(insert_ed, (int(row["year"]), float(row["baseline_consumption_kwh"]), None,
                                                float(row.get("baseline_consumption_kwh", None)), float(row.get("adjusted_consumption_kwh", None)),
                                                float(row.get("baseline_cost_rm", None)), float(row.get("adjusted_cost_rm", None)),
                                                float(row.get("baseline_co2_kg", None)), float(row.get("adjusted_co2_kg", None))))
                    insert_f = "INSERT INTO energy_factors (device, units, hours_per_year, action, kwh_per_year) VALUES (%s,%s,%s,%s,%s)"
                    for _, r in df_factors.iterrows():
                        cur.execute(insert_f, (r["device"], int(r["units"]), int(r["hours_per_year"]), r["action"], float(r["kwh_per_year"])))
                    conn.commit(); cur.close(); conn.close()
                    st.success("Saved results to DB.")
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")

# -------------------------
# Device Management
# -------------------------
elif menu == "Device Management":
    st.title("💡 Device Management")
    st.markdown("Add common device types. (Stored in session only for now.)")
    if "devices" not in st.session_state:
        st.session_state.devices = []
    with st.form("device_form"):
        d_name = st.text_input("Device name (e.g. LED 10W)")
        d_watt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        d_note = st.text_input("Note")
        if st.form_submit_button("Add device"):
            if d_name:
                st.session_state.devices.append({"name": d_name, "watt": d_watt, "note": d_note})
                st.success("Device added.")
    if st.session_state.devices:
        st.table(pd.DataFrame(st.session_state.devices))

# -------------------------
# Reports
# -------------------------
elif menu == "Reports":
    st.title("📊 Reports")
    st.markdown("Use Energy Forecast → Export to download Excel/PDF. The app does not persist files server-side.")

# -------------------------
# Settings
# -------------------------
elif menu == "Settings":
    st.title("⚙️ Settings — Appearance & Database")
    choice = st.radio("Background / Theme:", ["Dark", "Light", "Custom image URL"], index=0 if st.session_state.bg_mode=="Dark" else (1 if st.session_state.bg_mode=="Light" else 2))
    if choice == "Dark":
        st.session_state.bg_mode = "Dark"
        st.session_state.bg_custom_url = ""
        st.success("Applied Dark theme.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        st.session_state.bg_custom_url = ""
        st.success("Applied Light theme.")
    else:
        url = st.text_input("Enter full image URL for background (will persist in session)", value=st.session_state.get("bg_custom_url",""))
        if st.button("Apply custom background"):
            if url:
                st.session_state.bg_mode = "Custom"
                st.session_state.bg_custom_url = url
                st.success("Custom background applied.")
            else:
                st.error("Please input a valid image URL.")
    # re-apply after change
    apply_background()

    st.markdown("---")
    st.subheader("Database configuration (optional)")
    st.markdown("Enter MySQL (Railway) details here or set environment variables DB_HOST/DB_USER/DB_PASSWORD/DB_DATABASE/DB_PORT.")
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
        st.success("DB settings saved to session. Use Test/Save on Energy Forecast page.")

# -------------------------
# Help & About
# -------------------------
elif menu == "Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    - Forecast energy & cost using historical data and a simple Linear Regression model.  
    - Export Excel; PDF optional (reportlab).  
    - Optional MySQL saving (Railway).  
    **Support:** chikaenergyforecast@gmail.com
    """)
