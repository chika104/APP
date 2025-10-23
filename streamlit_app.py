# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App
Features:
- Login & register (MySQL optional)
- Black responsive sidebar (mobile friendly)
- Background selection (Dark/Light/Custom) persisted in session
- Input: CSV or Manual (data persisted in session when switching menu)
- Adjustment factors
- LinearRegression forecast, R^2 metric
- 7 graphs (colors: red, darkblue, green, orange, purple, gray, cyan)
- Export Excel, optional save to MySQL
"""

import os
import io
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Try import mysql (optional)
MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

# --------- Defaults (Railway sample from you) ----------
DEFAULT_DB = {
    "host": os.environ.get("DB_HOST", "switchback.proxy.rlwy.net"),
    "port": int(os.environ.get("DB_PORT", 55398)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "polrwgDJZnGLaungxPtGkOTaduCuolEj"),
    "database": os.environ.get("DB_DATABASE", "railway"),
}

EXCEL_ENGINE = "xlsxwriter"

# --------- Page config ----------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide", initial_sidebar_state="expanded")

# --------- Session defaults ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"   # Dark / Light / Custom
if "bg_css" not in st.session_state:
    # store the last applied CSS so it persists across menu changes
    st.session_state.bg_css = None
if "historical_df" not in st.session_state:
    st.session_state.historical_df = None
if "factors_df" not in st.session_state:
    st.session_state.factors_df = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "db_settings" not in st.session_state:
    st.session_state.db_settings = DEFAULT_DB.copy()

# --------- Helper utilities ----------
def apply_background_css():
    """Apply background according to session state (persist)."""
    mode = st.session_state.get("bg_mode", "Dark")
    if mode == "Dark":
        css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        </style>
        """
    elif mode == "Light":
        css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
        [data-testid="stHeader"] {background: rgba(255,255,255,0);}
        </style>
        """
    else:
        # custom stored in bg_css (if provided)
        css = st.session_state.get("bg_css") or """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def set_custom_background_image(url):
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{url}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.session_state.bg_css = css
    st.session_state.bg_mode = "Custom"
    apply_background_css()

def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def try_plot_png(fig):
    # Optional: attempt to export png (requires kaleido installed on host)
    try:
        import plotly.io as pio
        pio.kaleido.scope.default_format = "png"
        return fig.to_image(format="png", width=900, height=540, scale=2)
    except Exception:
        return None

# --------- Database helpers ----------
def get_db_config():
    cfg = st.session_state.get("db_settings", DEFAULT_DB.copy())
    # ensure keys
    return {
        "host": cfg.get("host"),
        "port": int(cfg.get("port", 3306)),
        "user": cfg.get("user"),
        "password": cfg.get("password"),
        "database": cfg.get("database"),
    }

def connect_db(timeout=10):
    cfg = get_db_config()
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector not installed in this environment.")
    if not cfg["host"] or not cfg["user"] or not cfg["database"]:
        raise ValueError("DB host/user/database not configured.")
    conn = mysql.connector.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        port=cfg["port"],
        connection_timeout=timeout
    )
    return conn

def init_db_tables(conn):
    cur = conn.cursor()
    # create users table (simple), energy_data, energy_factors
    t_users = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(128) UNIQUE,
        password VARCHAR(256),
        pref_bg_mode VARCHAR(32),
        pref_bg_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    t_data = """
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
    ) ENGINE=InnoDB;
    """
    t_factors = """
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
    cur.execute(t_users)
    cur.execute(t_data)
    cur.execute(t_factors)
    conn.commit()
    cur.close()

def save_results_to_db(conn, historical_df, factors_df, forecast_df):
    cur = conn.cursor()
    ins = """
    INSERT INTO energy_data
    (year, consumption, baseline_cost, fitted, adjusted, baseline_cost_rm, adjusted_cost_rm, baseline_co2_kg, adjusted_co2_kg)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    # historical
    for _, r in historical_df.iterrows():
        cur.execute(ins, (
            int(r['year']),
            float(r['consumption']),
            float(r.get('baseline_cost', None)) if not pd.isna(r.get('baseline_cost', None)) else None,
            float(r.get('fitted', None)) if r.get('fitted', None) is not None else None,
            float(r.get('adjusted', None)) if r.get('adjusted', None) is not None else None,
            float(r.get('baseline_cost', None)) if r.get('baseline_cost', None) is not None else None,
            float(r.get('adjusted_cost', None)) if r.get('adjusted_cost', None) is not None else None,
            float(r.get('baseline_co2_kg', None)) if r.get('baseline_co2_kg', None) is not None else None,
            float(r.get('adjusted_co2_kg', None)) if r.get('adjusted_co2_kg', None) is not None else None
        ))
    # forecast
    for _, r in forecast_df.iterrows():
        cur.execute(ins, (
            int(r['year']),
            float(r['baseline_consumption_kwh']),
            float(r.get('baseline_cost_rm', None)) if not pd.isna(r.get('baseline_cost_rm', None)) else None,
            float(r.get('baseline_consumption_kwh', None)) if r.get('baseline_consumption_kwh', None) is not None else None,
            float(r.get('adjusted_consumption_kwh', None)) if r.get('adjusted_consumption_kwh', None) is not None else None,
            float(r.get('baseline_cost_rm', None)) if r.get('baseline_cost_rm', None) is not None else None,
            float(r.get('adjusted_cost_rm', None)) if r.get('adjusted_cost_rm', None) is not None else None,
            float(r.get('baseline_co2_kg', None)) if r.get('baseline_co2_kg', None) is not None else None,
            float(r.get('adjusted_co2_kg', None)) if r.get('adjusted_co2_kg', None) is not None else None
        ))
    # factors
    ins_f = "INSERT INTO energy_factors (device, units, hours_per_year, action, kwh_per_year) VALUES (%s,%s,%s,%s,%s)"
    for _, r in factors_df.iterrows():
        cur.execute(ins_f, (str(r['device']), int(r['units']), int(r['hours_per_year']), str(r['action']), float(r['kwh_per_year'])))
    conn.commit()
    cur.close()

# --------- Authentication ----------
def check_credentials_db(username, password):
    if not MYSQL_AVAILABLE:
        return False
    try:
        conn = connect_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user is not None
    except Exception:
        return False

def register_user_db(username, password):
    if not MYSQL_AVAILABLE:
        return False, "MySQL support not available."
    try:
        conn = connect_db()
        init_db_tables(conn)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        cur.close()
        conn.close()
        return True, "Registered."
    except Exception as e:
        return False, str(e)

# --------- Layout: sidebar (black solid) ----------
def render_sidebar():
    # ensure sidebar solid black and text readable on mobile
    sidebar_css = """
    <style>
    /* Make sidebar background solid black and text white */
    [data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        color: #FFFFFF !important;
    }
    /* Sidebar items color */
    .css-1o0k8ji { color: #FFFFFF !important; }
    /* Make sidebar have fixed high z-index to avoid content overlay on mobile */
    .css-1d391kg { z-index: 9999; }
    /* Tweak main container to have spacing from left */
    .appview-container .main .block-container { padding-left: 2rem; padding-right: 2rem; }
    </style>
    """
    st.markdown(sidebar_css, unsafe_allow_html=True)

    st.sidebar.markdown("## 🔹 Smart Energy Forecasting")
    nav = st.sidebar.radio("", [
        "🏠 Dashboard",
        "⚡ Energy Forecast",
        "💡 Device Management",
        "📊 Reports",
        "⚙️ Settings",
        "❓ Help & About"
    ], index=0)
    return nav

# --------- Login page (no experimental_rerun) ----------
def login_page():
    st.markdown("<h1 style='text-align:center; color:white;'>🔐 Please login to access the dashboard</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        uname = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pass")
    with col2:
        st.write("")  # spacer
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login"):
            # prefer DB auth if available; else simple in-session check (for demo)
            ok = False
            if MYSQL_AVAILABLE:
                try:
                    ok = check_credentials_db(uname, pwd)
                except Exception:
                    ok = False
            # fallback: simple in-memory demo credentials (username=admin password=admin)
            if not ok and uname == "admin" and pwd == "admin":
                ok = True
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.success("Login berjaya.")
                # safe rerun: try experimental_rerun, else set query params
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.experimental_set_query_params(_rerun=int(time.time()))
                    except Exception:
                        pass
            else:
                st.error("Nama pengguna atau kata laluan salah!")
    with c2:
        if st.button("Register (local)"):
            # Try register to DB if available, else register to a simple session store (not persistent)
            if MYSQL_AVAILABLE:
                ok, msg = register_user_db(uname, pwd)
                if ok:
                    st.success("Akaun didaftar di DB. Sila login.")
                else:
                    st.error("Gagal daftar: " + msg)
            else:
                # store in session (temporary)
                if "local_users" not in st.session_state:
                    st.session_state.local_users = {}
                if uname in st.session_state.local_users:
                    st.warning("Username sudah wujud (temp).")
                else:
                    st.session_state.local_users[uname] = pwd
                    st.success("Akaun temp didaftar. Gunakan login segera.")

# --------- Core: Energy Forecast functions ----------
def prepare_history_from_upload_or_manual(df):
    # normalize expected cols
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # choose year and consumption columns
    if "year" not in df.columns:
        raise ValueError("Data must contain 'year' column")
    # find consumption-like column
    cons_cols = [c for c in df.columns if any(k in c for k in ("consum", "kwh", "energy"))]
    if len(cons_cols) == 0:
        raise ValueError("Data must contain consumption column (e.g. 'consumption' or 'kwh')")
    cons_col = cons_cols[0]
    out = pd.DataFrame({
        "year": df["year"].astype(int),
        "consumption": pd.to_numeric(df[cons_col], errors="coerce")
    })
    # optional cost
    cost_cols = [c for c in df.columns if "cost" in c]
    if cost_cols:
        out["baseline_cost"] = pd.to_numeric(df[cost_cols[0]], errors="coerce")
    else:
        out["baseline_cost"] = np.nan
    return out

def compute_forecast(historical_df, factors_df, tariff=0.52, co2_factor=0.75, n_years=3):
    df = historical_df.copy().reset_index(drop=True)
    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    model = LinearRegression()
    X = df[["year"]].values
    y = df["consumption"].values
    if len(X) >= 2:
        model.fit(X, y)
        df["fitted"] = model.predict(X)
        r2 = r2_score(y, df["fitted"])
    else:
        df["fitted"] = df["consumption"]
        r2 = 1.0

    last_year = int(df["year"].max())
    future_years = [last_year + i for i in range(1, int(n_years) + 1)]
    future_X = np.array(future_years).reshape(-1, 1)
    future_baseline = model.predict(future_X) if len(X) >= 2 else np.array([df["consumption"].iloc[-1]] * len(future_years))

    # compute net adjustment from factors_df
    total_net_adjust = 0.0
    if factors_df is not None and not factors_df.empty:
        total_net_adjust = factors_df["kwh_per_year"].sum()
    # adjusted baseline by adding net adjustment (applies equally per forecast year)
    adjusted = future_baseline + total_net_adjust

    fx = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": future_baseline,
        "adjusted_consumption_kwh": adjusted
    })
    fx["baseline_cost_rm"] = fx["baseline_consumption_kwh"] * tariff
    fx["adjusted_cost_rm"] = fx["adjusted_consumption_kwh"] * tariff
    fx["baseline_co2_kg"] = fx["baseline_consumption_kwh"] * co2_factor
    fx["adjusted_co2_kg"] = fx["adjusted_consumption_kwh"] * co2_factor
    fx["saving_kwh"] = fx["baseline_consumption_kwh"] - fx["adjusted_consumption_kwh"]
    fx["saving_cost_rm"] = fx["baseline_cost_rm"] - fx["adjusted_cost_rm"]
    fx["saving_co2_kg"] = fx["baseline_co2_kg"] - fx["adjusted_co2_kg"]

    return df, fx, r2, total_net_adjust

# --------- Graph generation (7 graphs) ----------
def plot_7_graphs(hist_df, forecast_df):
    # colors
    c_baseline = "red"           # baseline primary
    c_forecast = "darkblue"      # forecast
    c_adjusted = "green"         # adjusted
    c_forecast_orange = "orange" # alt
    c_purple = "purple"
    c_gray = "gray"
    c_cyan = "cyan"

    # 1 Baseline only
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hist_df["year"], y=hist_df["consumption"], mode="lines+markers", name="Baseline", line=dict(color=c_baseline)))
    fig1.update_layout(title="Baseline only (kWh)", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # 2 Baseline vs Forecast (plot forecast on forecast_df)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist_df["year"], y=hist_df["consumption"], mode="lines+markers", name="Baseline (hist)", line=dict(color=c_baseline)))
    fig2.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"], mode="lines+markers", name="Baseline (forecast)", line=dict(color=c_forecast)))
    fig2.update_layout(title="Baseline vs Baseline-Forecast (kWh)", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # 3 Adjusted vs Forecast vs Baseline
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["adjusted_consumption_kwh"], mode="lines+markers", name="Adjusted forecast", line=dict(color=c_adjusted)))
    fig3.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"], mode="lines+markers", name="Baseline forecast", line=dict(color=c_forecast)))
    # include historical baseline for continuity
    fig3.add_trace(go.Scatter(x=hist_df["year"], y=hist_df["consumption"], mode="lines+markers", name="Baseline (hist)", line=dict(color=c_baseline, dash="dot")))
    fig3.update_layout(title="Adjusted vs Forecast vs Baseline", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    # 4 Baseline cost (historical)
    fig4 = go.Figure()
    if "baseline_cost" in hist_df.columns:
        fig4.add_trace(go.Bar(x=hist_df["year"], y=hist_df["baseline_cost"], name="Baseline cost RM", marker_color=c_purple))
    else:
        # estimate
        est = hist_df["consumption"] * 0.52
        fig4.add_trace(go.Bar(x=hist_df["year"], y=est, name="Baseline cost (est)", marker_color=c_purple))
    fig4.update_layout(title="Baseline Cost (RM) - historical", template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    # 5 Forecast cost vs Baseline cost
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=forecast_df["year"], y=forecast_df["baseline_cost_rm"], name="Baseline cost (forecast)", marker_color=c_baseline))
    fig5.add_trace(go.Bar(x=forecast_df["year"], y=forecast_df["adjusted_cost_rm"], name="Adjusted cost (forecast)", marker_color=c_forecast))
    fig5.update_layout(barmode="group", title="Forecast cost vs Baseline cost", template="plotly_dark")
    st.plotly_chart(fig5, use_container_width=True)

    # 6 CO2 baseline (historical)
    fig6 = go.Figure()
    if "baseline_co2_kg" in hist_df.columns:
        fig6.add_trace(go.Line(x=hist_df["year"], y=hist_df["baseline_co2_kg"], name="CO₂ baseline", line=dict(color=c_gray)))
    else:
        est_co2 = hist_df["consumption"] * 0.75
        fig6.add_trace(go.Line(x=hist_df["year"], y=est_co2, name="CO₂ baseline (est)", line=dict(color=c_gray)))
    fig6.update_layout(title="CO₂ Baseline (kg)", template="plotly_dark")
    st.plotly_chart(fig6, use_container_width=True)

    # 7 CO2 baseline vs CO2 forecast
    fig7 = go.Figure()
    fig7.add_trace(go.Line(x=forecast_df["year"], y=forecast_df["baseline_co2_kg"], name="CO₂ baseline (forecast)", line=dict(color=c_gray)))
    fig7.add_trace(go.Line(x=forecast_df["year"], y=forecast_df["adjusted_co2_kg"], name="CO₂ adjusted (forecast)", line=dict(color=c_cyan)))
    fig7.update_layout(title="CO₂ Baseline vs CO₂ Forecast (kg)", template="plotly_dark")
    st.plotly_chart(fig7, use_container_width=True)

# --------- Main UI ----------
# apply background CSS first
apply_background_css()
render_sidebar()
# If not logged in -> show login only
if not st.session_state.logged_in:
    login_page()
    st.stop()

# If logged in, show main app
nav = st.sidebar.radio("", [
    "🏠 Dashboard",
    "⚡ Energy Forecast",
    "💡 Device Management",
    "📊 Reports",
    "⚙️ Settings",
    "❓ Help & About"
], index=0)

# main top bar title
st.markdown(f"<h2 style='color: #ffffff;'>🔹 Smart Energy Forecasting — Welcome, {st.session_state.username}</h2>", unsafe_allow_html=True)

# ---------- DASHBOARD ----------
if nav == "🏠 Dashboard":
    st.header("🏠 Dashboard Overview")
    st.markdown("Quick summary & graphs")
    if st.session_state.historical_df is None:
        st.info("No historical data loaded yet. Go to 'Energy Forecast' to upload or enter data.")
    else:
        hist = st.session_state.historical_df
        fx = st.session_state.forecast_df if st.session_state.forecast_df is not None else pd.DataFrame()
        st.subheader("Historical data (preview)")
        st.dataframe(hist)
        if not fx.empty:
            st.subheader("Forecast (preview)")
            st.dataframe(fx)
        st.subheader("Visual summary")
        if not fx.empty:
            plot_7_graphs(hist, fx)
        else:
            st.info("Please compute forecast in Energy Forecast page to see full graphs.")

# ---------- ENERGY FORECAST ----------
elif nav == "⚡ Energy Forecast":
    st.header("⚡ Energy Forecast")

    # Step 1 Input method
    st.subheader("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (must include 'year' and consumption column)", type=["csv", "xlsx"])
        if uploaded is not None:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)
                df_hist = prepare_history_from_upload_or_manual(raw)
                st.session_state.historical_df = df_hist.sort_values("year").reset_index(drop=True)
                st.success("Data loaded.")
            except Exception as e:
                st.error("Error parsing file: " + str(e))
    else:
        # manual entry - persist values by using session
        rows = st.number_input("How many historical rows?", min_value=1, max_value=40, value=5, key="hist_rows")
        manual_data = []
        cols = st.columns([1,1,1])
        # create inputs grouped, store to session to avoid clear on menu flip
        for i in range(int(rows)):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                y = st.number_input(f"Year {i+1}", min_value=1900, max_value=2100, value=2020+i, key=f"m_year_{i}")
            with c2:
                cons = st.number_input(f"Consumption (kWh) {i+1}", min_value=0.0, value=10000.0, key=f"m_cons_{i}")
            with c3:
                cost = st.number_input(f"Baseline cost RM (optional) {i+1}", min_value=0.0, value=0.0, key=f"m_cost_{i}")
            manual_data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
        if st.button("Load Manual Data"):
            st.session_state.historical_df = pd.DataFrame(manual_data).sort_values("year").reset_index(drop=True)
            st.success("Manual data saved in session.")

    # show loaded data
    if st.session_state.historical_df is not None:
        st.subheader("Loaded baseline data")
        st.dataframe(st.session_state.historical_df)

    # Step 2 Factors
    st.subheader("Step 2 — Adjustment factors")
    st.markdown("Device factors: units, hours/year, action. Wattage defaults preloaded.")
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}
    n = st.number_input("Number of factor rows", min_value=1, max_value=8, value=1, key="n_factors_input")
    factor_rows = []
    for i in range(int(n)):
        st.markdown(f"**Factor {i+1}**")
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        with c1:
            device = st.selectbox(f"Device (factor {i+1})", ["Lamp - LED", "Lamp - CFL", "Lamp - Fluorescent", "Computer", "Lab Equipment"], key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units", min_value=0, value=0, step=1, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR", min_value=0, max_value=8760, value=0, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action", ["Addition", "Reduction"], key=f"action_{i}")
        # compute kwh
        if device.startswith("Lamp"):
            subtype = device.split(" - ")[1]
            watt = WATT[subtype]
            dname = subtype + " Lamp"
        else:
            dname = device
            watt = WATT[device]
        kwh_per_year = (watt * int(units) * int(hours)) / 1000.0
        if action == "Reduction":
            kwh_per_year = -abs(kwh_per_year)
        factor_rows.append({"device": dname, "units": int(units), "hours_per_year": int(hours), "action": action, "kwh_per_year": kwh_per_year})

    df_factors = pd.DataFrame(factor_rows)
    st.session_state.factors_df = df_factors
    st.subheader("Factors summary (kWh/year)")
    st.dataframe(df_factors)

    # site-level general hours change
    st.markdown("General site-level operating hours change (kW * hours)")
    general_hours = st.number_input("General extra/reduced hours per year (neg = reduce)", min_value=-8760, max_value=8760, value=0, key="general_hours")
    general_kw = st.number_input("Average site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1, key="general_kw")
    general_kwh = float(general_hours) * float(general_kw)
    # include general_kwh into a pseudo factor for display
    if general_kwh != 0:
        st.info(f"General site-level change contributes {general_kwh:.2f} kWh/year")

    # Step 3 Forecast settings & compute
    st.subheader("Step 3 — Forecast settings & compute")
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)
    n_years = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1)

    if st.button("Compute forecast"):
        if st.session_state.historical_df is None:
            st.warning("Sila muat naik atau masukkan data sejarah dahulu.")
        else:
            # add general kwh to factors_df as single extra (non-persistent row)
            df_factors = st.session_state.factors_df.copy()
            if general_kwh != 0:
                extra = pd.DataFrame([{"device": "General site change", "units": 0, "hours_per_year": int(general_hours),
                                       "action": "General", "kwh_per_year": float(general_kwh)}])
                df_factors = pd.concat([df_factors, extra], ignore_index=True)
            hist, fx, r2, total_adj = compute_forecast(st.session_state.historical_df, df_factors, tariff=tariff, co2_factor=co2_factor, n_years=n_years)
            st.session_state.historical_df = hist
            st.session_state.factors_df = df_factors
            st.session_state.forecast_df = fx
            st.session_state.last_r2 = r2
            st.session_state.last_total_adjust = total_adj
            st.success("Forecast computed.")

    # Show results if available
    if st.session_state.forecast_df is not None:
        st.subheader("Forecast results")
        st.dataframe(st.session_state.forecast_df)
        st.markdown(f"**Model R²:** {st.session_state.get('last_r2', 0):.4f}")
        st.markdown(f"**Net adjustment (kWh/year):** {st.session_state.get('last_total_adjust', 0):,.2f}")
        # graphs
        plot_7_graphs(st.session_state.historical_df, st.session_state.forecast_df)

    # Step 4 Export & DB save
    st.subheader("Export & Save")
    if st.session_state.historical_df is not None and st.session_state.forecast_df is not None:
        excel_data = df_to_excel_bytes({
            "historical": st.session_state.historical_df,
            "factors": st.session_state.factors_df,
            "forecast": st.session_state.forecast_df
        })
        st.download_button("⬇️ Download Excel (.xlsx)", data=excel_data, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.subheader("Optional: Save to MySQL")
    if not MYSQL_AVAILABLE:
        st.info("MySQL connector not installed in environment. DB features disabled.")
    else:
        colA, colB = st.columns(2)
        with colA:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    conn.close()
                    st.success("DB connection OK and tables ready.")
                except Exception as e:
                    st.error("DB connection failed: " + str(e))
        with colB:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_results_to_db(conn, st.session_state.historical_df, st.session_state.factors_df, st.session_state.forecast_df)
                    conn.close()
                    st.success("Saved to DB.")
                except Exception as e:
                    st.error("Save failed: " + str(e))

# ---------- Device Management ----------
elif nav == "💡 Device Management":
    st.header("💡 Device Management")
    if "devices" not in st.session_state:
        st.session_state.devices = []
    with st.form("device_form", clear_on_submit=False):
        dname = st.text_input("Device name (e.g. LED 10W)", value="")
        dwatt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        dnote = st.text_input("Note")
        if st.form_submit_button("Add device"):
            st.session_state.devices.append({"name": dname, "watt": float(dwatt), "note": dnote})
            st.success("Device added (session).")
    if st.session_state.devices:
        st.table(pd.DataFrame(st.session_state.devices))

# ---------- Reports ----------
elif nav == "📊 Reports":
    st.header("📊 Reports")
    st.markdown("Use the Export button in Energy Forecast to download Excel. DB persistent exports appear in your DB if saved.")
    if st.session_state.historical_df is not None:
        st.subheader("Download previously computed Excel")
        excel_data = df_to_excel_bytes({
            "historical": st.session_state.historical_df,
            "factors": st.session_state.factors_df if st.session_state.factors_df is not None else pd.DataFrame(),
            "forecast": st.session_state.forecast_df if st.session_state.forecast_df is not None else pd.DataFrame()
        })
        st.download_button("⬇️ Download Excel (.xlsx)", data=excel_data, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Settings ----------
elif nav == "⚙️ Settings":
    st.header("⚙️ Settings — Appearance & Database")
    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"])
    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        apply_background_css()
        st.success("Applied Dark theme (session).")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        apply_background_css()
        st.success("Applied Light theme (session).")
    else:
        url = st.text_input("Enter image URL for background (full URL):")
        if st.button("Apply background image URL"):
            if url:
                set_custom_background_image(url)
                st.success("Applied custom background (session).")
            else:
                st.warning("Please enter a valid image URL.")

    st.markdown("---")
    st.subheader("Database settings")
    st.markdown("Set DB credentials (these are saved to session only). You can also set environment variables on host.")
    db_host = st.text_input("DB host", value=st.session_state.db_settings.get("host", ""))
    db_port = st.text_input("DB port", value=str(st.session_state.db_settings.get("port", 3306)))
    db_user = st.text_input("DB user", value=st.session_state.db_settings.get("user", ""))
    db_pass = st.text_input("DB password", value=st.session_state.db_settings.get("password", ""), type="password")
    db_name = st.text_input("DB database", value=st.session_state.db_settings.get("database", ""))
    if st.button("Save DB settings to session"):
        try:
            st.session_state.db_settings["host"] = db_host.strip()
            st.session_state.db_settings["port"] = int(db_port)
            st.session_state.db_settings["user"] = db_user.strip()
            st.session_state.db_settings["password"] = db_pass
            st.session_state.db_settings["database"] = db_name.strip()
            st.success("DB settings saved to session.")
        except Exception as e:
            st.error("Invalid DB port or input: " + str(e))

# ---------- Help & About ----------
elif nav == "❓ Help & About":
    st.header("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed as a software-only forecasting tool.  
    **Support / report issues:** chikaenergyforecast@gmail.com
    """)
    st.markdown("Note: background selection and data are persisted in the current browser session. To persist per-user preferences across logins requires storing preferences in DB (can be added).")

# End of file
