# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App
- 6 Menus: Dashboard, Energy Forecast, Database, Reports, Settings, Help & About
- Login / Register (session-based, optional DB)
- Theme selector (Dark / Light / Custom image) — selection persists across menus
- Energy Forecast with 7 graphs, factors, manual/CSV input, export Excel/PDF
- Optional MySQL saving (configure in Settings)
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

MYSQL_AVAILABLE = True
try:
    import mysql.connector
    from mysql.connector import errorcode
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Page config and apply persistent CSS (theme)
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# initialize persistent session keys
if "bg_css" not in st.session_state:
    # default dark background CSS
    st.session_state.bg_css = """
    <style>
    [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    [data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
    /* sidebar radio background fix for mobile */
    .css-1oe6wyx .stRadio { background-color: rgba(0,0,0,0.0); }
    </style>
    """

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "users" not in st.session_state:
    # simple user store: list of dicts {'username':..., 'password':...}
    st.session_state.users = []

# persist forecast data so switching menu won't clear inputs
for key in ("hist_df", "factors_df", "forecast_df", "last_model_r2"):
    if key not in st.session_state:
        st.session_state[key] = None

# apply background CSS (always)
st.markdown(st.session_state.bg_css, unsafe_allow_html=True)

# -------------------------
# Utility helpers
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

# PDF helper (minimal)
def make_pdf_bytes(title_text, summary_lines, table_blocks, image_bytes_list=None, logo_bytes=None):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title_text, styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    for line in summary_lines:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 12))
    if image_bytes_list:
        for im in image_bytes_list:
            try:
                imgbuf = io.BytesIO(im)
                img = RLImage(imgbuf, width=450, height=280)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
    for title, df in table_blocks:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                                 ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d91")),
                                 ("TEXTCOLOR", (0,0), (-1,0), colors.white)]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# MySQL helpers (optional)
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
        raise ValueError("DB host/user/database must be set in Settings.")
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
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row['consumption']),
            float(row.get('baseline_cost', None)) if pd.notna(row.get('baseline_cost', None)) else None,
            float(row.get('fitted', None)) if pd.notna(row.get('fitted', None)) else None,
            float(row.get('adjusted', None)) if pd.notna(row.get('adjusted', None)) else None,
            float(row.get('baseline_cost', None)) if pd.notna(row.get('baseline_cost', None)) else None,
            float(row.get('adjusted_cost', None)) if pd.notna(row.get('adjusted_cost', None)) else None,
            float(row.get('baseline_co2_kg', None)) if pd.notna(row.get('baseline_co2_kg', None)) else None,
            float(row.get('adjusted_co2_kg', None)) if pd.notna(row.get('adjusted_co2_kg', None)) else None
        ))
    # forecast rows
    for _, row in forecast_df.iterrows():
        cursor.execute(insert_sql, (
            int(row['year']),
            float(row['baseline_consumption_kwh']) if pd.notna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if pd.notna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('baseline_consumption_kwh', None)) if pd.notna(row.get('baseline_consumption_kwh', None)) else None,
            float(row.get('adjusted_consumption_kwh', None)) if pd.notna(row.get('adjusted_consumption_kwh', None)) else None,
            float(row.get('baseline_cost_rm', None)) if pd.notna(row.get('baseline_cost_rm', None)) else None,
            float(row.get('adjusted_cost_rm', None)) if pd.notna(row.get('adjusted_cost_rm', None)) else None,
            float(row.get('baseline_co2_kg', None)) if pd.notna(row.get('baseline_co2_kg', None)) else None,
            float(row.get('adjusted_co2_kg', None)) if pd.notna(row.get('adjusted_co2_kg', None)) else None
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
# AUTH (simple)
# -------------------------
def login_form():
    st.subheader("Login")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            for u in st.session_state.users:
                if u["username"] == uname and u["password"] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.user = uname
                    st.success(f"Logged in as {uname}")
                    st.experimental_rerun()
            st.error("Invalid credentials.")
    with col2:
        if st.button("Register"):
            if not uname or not pwd:
                st.error("Enter username & password to register.")
            else:
                if any(u["username"] == uname for u in st.session_state.users):
                    st.error("Username exists.")
                else:
                    st.session_state.users.append({"username": uname, "password": pwd})
                    st.success("Registered. Now login.")
    st.info("Tip: For production, use proper authentication & store users in DB.")

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.experimental_rerun()

# -------------------------
# Sidebar & Navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
if st.session_state.logged_in:
    st.sidebar.markdown(f"**User:** {st.session_state.user}")
    if st.sidebar.button("Logout"):
        logout()

menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "🗄️ Database", "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# If not logged in, show login page only
if not st.session_state.logged_in:
    st.title("🔐 Please login to access the dashboard")
    login_form()
    st.stop()

# -------------------------
# MENU: Dashboard
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard — Smart Energy Forecasting")
    st.markdown("""
    **Overview**: quick summary and shortcuts.
    """)
    # quick metrics from session
    hist = st.session_state.get("hist_df")
    forecast_df = st.session_state.get("forecast_df")
    if hist is not None:
        st.metric("Historical rows", len(hist))
    else:
        st.metric("Historical rows", "No data")
    if forecast_df is not None:
        st.metric("Forecast years", len(forecast_df))
    else:
        st.metric("Forecast years", "No forecast")
    st.markdown("---")
    st.markdown("Use **Energy Forecast** to load data and run forecasting. Use **Settings** to change theme or DB configs.")

# -------------------------
# MENU: Energy Forecast (main)
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast — 7 Graphs & Save")
    # load persisted data if any
    hist_df = st.session_state.get("hist_df")
    df_factors = st.session_state.get("factors_df")
    forecast_df = st.session_state.get("forecast_df")

    # Step 1: Input data
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"), index=0)

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (columns: year, consumption[, baseline_cost])", type=["csv","xlsx"])
        if uploaded:
            if str(uploaded.name).lower().endswith(".csv"):
                raw = pd.read_csv(uploaded)
            else:
                raw = pd.read_excel(uploaded)
            raw = normalize_cols(raw)
            if "year" not in raw.columns or not any("consum" in c or "kwh" in c or "energy" in c for c in raw.columns):
                st.error("CSV must include 'year' and a consumption column.")
            else:
                cons_col = [c for c in raw.columns if any(k in c for k in ("consum","kwh","energy"))][0]
                base_cost_cols = [c for c in raw.columns if "cost" in c]
                hist_df = pd.DataFrame({
                    "year": raw["year"].astype(int),
                    "consumption": pd.to_numeric(raw[cons_col], errors="coerce")
                })
                if base_cost_cols:
                    hist_df["baseline_cost"] = pd.to_numeric(raw[base_cost_cols[0]], errors="coerce")
                else:
                    hist_df["baseline_cost"] = np.nan
                st.session_state.hist_df = hist_df.sort_values("year").reset_index(drop=True)
                st.success("Loaded CSV to historical data (saved in session).")
    else:
        # Manual entry form (persist rows in session if exist)
        with st.form("manual_hist_form", clear_on_submit=False):
            rows = st.number_input("Number of historical rows:", min_value=1, max_value=20, value=5)
            manual_data = []
            for i in range(int(rows)):
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i, key=f"m_year_{i}")
                with c2:
                    cons = st.number_input(f"Consumption kWh ({y})", 0.0, 1e8, 10000.0, key=f"m_cons_{i}")
                with c3:
                    cost = st.number_input(f"Baseline cost RM ({y})", 0.0, 1e9, 0.0, key=f"m_cost_{i}")
                manual_data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
            submitted = st.form_submit_button("Save historical data")
            if submitted:
                hist_df = pd.DataFrame(manual_data).sort_values("year").reset_index(drop=True)
                st.session_state.hist_df = hist_df
                st.success("Saved manual historical data in session.")

    # show current historical
    hist_df = st.session_state.get("hist_df")
    if hist_df is None or hist_df.empty:
        st.warning("No historical data loaded yet. Upload CSV or use Manual Entry.")
        st.stop()

    st.subheader("Loaded historical data")
    st.dataframe(hist_df)

    # Step 2: Factors (persist)
    st.header("Step 2 — Adjustment factors")
    st.markdown("Device-level additions/reductions (hours per year).")
    WATT = {"LED":10, "CFL":15, "Fluorescent":40, "Computer":150, "Lab Equipment":500}
    n_factors = st.number_input("How many factor rows to add?", min_value=1, max_value=10, value=1, key="n_factors_factors")
    factor_rows = []
    for i in range(int(n_factors)):
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", ["Lamp - LED","Lamp - CFL","Lamp - Fluorescent","Computer","Lab Equipment"], key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units (factor {i+1})", min_value=0, value=0, step=1, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR (factor {i+1})", min_value=0, max_value=8760, value=0, step=1, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action", ["Addition","Reduction"], key=f"action_{i}")
        if device.startswith("Lamp"):
            subtype = device.split(" - ")[1]
            watt = WATT[subtype]
            dev_name = f"{subtype} Lamp"
        else:
            dev_name = device
            watt = WATT.get(device, 100)
        kwh_per_year = (watt * int(units) * int(hours))/1000.0
        if action == "Reduction":
            kwh_per_year = -abs(kwh_per_year)
        factor_rows.append({"device":dev_name, "units":int(units), "hours_per_year":int(hours), "action":action, "kwh_per_year":kwh_per_year})
    df_factors = pd.DataFrame(factor_rows)
    st.session_state.factors_df = df_factors
    st.subheader("Factors summary")
    st.dataframe(df_factors)

    # general site-level
    st.markdown("General site-level change")
    general_hours = st.number_input("Extra/reduced hours per year (positive add, negative reduce)", min_value=-8760, max_value=8760, value=0)
    general_avg_kw = st.number_input("Avg site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1)
    general_kwh = float(general_hours) * float(general_avg_kw) if general_hours != 0 else 0.0
    total_net_adjust = df_factors["kwh_per_year"].sum() + general_kwh
    st.info(f"Net adjustment (kWh/year): {total_net_adjust:,.2f}")

    # Step 3: Forecast settings
    st.header("Step 3 — Forecast compute")
    tariff = st.number_input("Tariff (RM/kWh)", min_value=0.0, value=0.52, step=0.01)
    co2_factor = st.number_input("CO2 factor (kg CO2 / kWh)", min_value=0.0, value=0.75, step=0.01)
    n_years = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1)

    # ensure baseline cost present
    hist_df["baseline_cost"] = hist_df.get("baseline_cost", np.nan)
    hist_df["baseline_cost"] = hist_df["baseline_cost"].fillna(hist_df["consumption"] * tariff)
    hist_df["baseline_co2_kg"] = hist_df["consumption"] * co2_factor

    # Linear regression model
    model = LinearRegression()
    X = hist_df[["year"]].values
    y = hist_df["consumption"].values
    if len(X) >= 2:
        model.fit(X, y)
        hist_df["fitted"] = model.predict(X)
        r2 = r2_score(y, hist_df["fitted"])
    else:
        hist_df["fitted"] = hist_df["consumption"]
        r2 = 1.0
    st.session_state.last_model_r2 = r2
    last_year = int(hist_df["year"].max())
    future_years = [last_year + i for i in range(1, int(n_years)+1)]
    future_X = np.array(future_years).reshape(-1,1)
    if len(X) >= 2:
        future_baseline = model.predict(future_X)
    else:
        future_baseline = np.array([hist_df["consumption"].iloc[-1]] * len(future_years))
    # apply net adjustment equally to future years
    adjusted_forecast = future_baseline + total_net_adjust

    forecast_df = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": future_baseline,
        "adjusted_consumption_kwh": adjusted_forecast
    })
    forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
    forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
    forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * co2_factor
    forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * co2_factor
    forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
    forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
    forecast_df["saving_co2_kg"] = forecast_df["baseline_co2_kg"] - forecast_df["adjusted_co2_kg"]

    st.session_state.hist_df = hist_df
    st.session_state.factors_df = df_factors
    st.session_state.forecast_df = forecast_df

    # Step 4: Visualizations (7 graphs)
    st.header("Step 4 — Visualizations (7 graphs)")
    # combine for plotting baseline history + fitted and forecast
    hist_plot = pd.DataFrame({
        "year": hist_df["year"],
        "baseline": hist_df["consumption"],
        "fitted": hist_df["fitted"]
    })
    combined_plot = pd.concat([
        hist_plot,
        pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"], "fitted": forecast_df["adjusted_consumption_kwh"]})
    ], ignore_index=True).sort_values("year")

    # graph 1: Baseline forecast (just forecast baseline)
    fig1 = px.line(forecast_df, x="year", y="baseline_consumption_kwh", title="Baseline Forecast (kWh)", markers=True)
    # graph 2: Baseline vs Forecast (history baseline + forecast baseline)
    fig2 = px.line(combined_plot, x="year", y=["baseline","fitted"], title="Baseline vs Forecast (baseline history & fitted/forecast)", markers=True)
    # graph 3: Adjusted vs Forecast vs Baseline (three-series): use baseline history, forecast baseline, adjusted forecast
    df_three = pd.DataFrame({
        "year": list(hist_df["year"]) + list(forecast_df["year"]),
    })
    df_three = df_three.drop_duplicates(subset=["year"])
    # build series mapping
    ser_baseline_hist = pd.Series(data=list(hist_df["consumption"]), index=list(hist_df["year"]))
    ser_baseline_fore = pd.Series(data=list(forecast_df["baseline_consumption_kwh"]), index=list(forecast_df["year"]))
    ser_adjusted_fore = pd.Series(data=list(forecast_df["adjusted_consumption_kwh"]), index=list(forecast_df["year"]))
    df_plot3 = pd.DataFrame({
        "year": sorted(set(list(ser_baseline_hist.index) + list(ser_baseline_fore.index))),
    })
    df_plot3["baseline"] = df_plot3["year"].map(ser_baseline_hist).fillna(df_plot3["year"].map(ser_baseline_fore))
    df_plot3["adjusted"] = df_plot3["year"].map(ser_adjusted_fore)
    fig3 = px.line(df_plot3.sort_values("year"), x="year", y=["baseline","adjusted"], title="Adjusted vs Forecast vs Baseline (kWh)", markers=True)

    # graph 4: Baseline cost trend (historical + baseline forecast)
    cost_hist = pd.DataFrame({"year": hist_df["year"], "baseline_cost": hist_df["baseline_cost"]})
    cost_fore = pd.DataFrame({"year": forecast_df["year"], "baseline_cost": forecast_df["baseline_cost_rm"]})
    cost_combined = pd.concat([cost_hist.rename(columns={"baseline_cost":"baseline_cost_rm"}), cost_fore], ignore_index=True).sort_values("year")
    fig4 = px.line(cost_combined, x="year", y="baseline_cost_rm", title="Baseline Cost Trend (RM)", markers=True)

    # graph 5: Forecast cost vs Baseline cost (forecast period)
    fig5 = px.bar(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], barmode="group", title="Forecast Cost vs Baseline Cost (RM)")

    # graph 6: CO2 Baseline (historical baseline co2)
    co2_hist = pd.DataFrame({"year": hist_df["year"], "baseline_co2_kg": hist_df["baseline_co2_kg"]})
    fig6 = px.line(co2_hist, x="year", y="baseline_co2_kg", title="CO₂ Baseline (kg)", markers=True)

    # graph 7: CO2 Baseline vs CO2 Forecast
    co2_comb = pd.concat([co2_hist, forecast_df[["year","baseline_co2_kg","adjusted_co2_kg"]].rename(columns={"adjusted_co2_kg":"co2_adjusted", "baseline_co2_kg":"co2_baseline"})], ignore_index=True).sort_values("year")
    # for readability, create a frame with proper columns
    co2_plot = pd.DataFrame({
        "year": list(hist_df["year"]) + list(forecast_df["year"]),
    }).drop_duplicates().sort_values("year")
    co2_plot["co2_baseline"] = co2_plot["year"].map(dict(zip(hist_df["year"], hist_df["baseline_co2_kg"]))).fillna(co2_plot["year"].map(dict(zip(forecast_df["year"], forecast_df["baseline_co2_kg"]))))
    co2_plot["co2_forecast"] = co2_plot["year"].map(dict(zip(forecast_df["year"], forecast_df["adjusted_co2_kg"])))
    fig7 = px.line(co2_plot, x="year", y=["co2_baseline","co2_forecast"], title="CO₂ Baseline vs CO₂ Forecast (kg)", markers=True)

    # render graphs (arranged)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.plotly_chart(fig7, use_container_width=True)

    # model metrics & totals
    st.subheader("Model & Totals")
    st.markdown(f"**R²:** `{st.session_state.last_model_r2:.4f}`")
    st.metric("Total baseline (forecast period) kWh", f"{forecast_df['baseline_consumption_kwh'].sum():,.0f} kWh")
    st.metric("Total adjusted (forecast period) kWh", f"{forecast_df['adjusted_consumption_kwh'].sum():,.0f} kWh")
    st.metric("Total cost saving (RM)", f"RM {forecast_df['saving_cost_rm'].sum():,.2f}")

    # Step 5: Tables
    st.header("Step 5 — Tables & Export")
    st.subheader("Historical (baseline)")
    st.dataframe(hist_df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}))
    st.subheader("Forecast results")
    st.dataframe(forecast_df)

    excel_bytes = df_to_excel_bytes({"historical": hist_df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF export if available
    images = []
    for fig in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        img = try_get_plot_png(fig)
        if img:
            images.append(img)
    summary_lines = [
        f"Forecast period: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
        f"Net adjustment (kWh/year): {total_net_adjust:.2f}",
        f"R²: {st.session_state.last_model_r2:.4f}"
    ]
    table_blocks = [
        ("Historical", hist_df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"})),
        ("Factors", df_factors),
        ("Forecast", forecast_df)
    ]
    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECAST REPORT", summary_lines, table_blocks, image_bytes_list=images)
    if pdf_bytes:
        st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export requires reportlab; charts require kaleido.")

    # MySQL Save controls
    st.markdown("---")
    st.subheader("Optional: Save results to MySQL")
    if not MYSQL_AVAILABLE:
        st.info("MySQL connector not installed; enable 'mysql-connector-python' to use DB features.")
    else:
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Test DB connection"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    conn.close()
                    st.success("DB connection OK & tables ensured.")
                except Exception as e:
                    st.error(f"DB connection failed: {e}")
        with colB:
            if st.button("Save results to DB"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_results_to_db(conn, hist_df, df_factors, forecast_df)
                    conn.close()
                    st.success("Saved historical, factors & forecast to DB.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

# -------------------------
# MENU: Database (view)
# -------------------------
elif menu == "🗄️ Database":
    st.title("🗄️ Database — View / Query (MySQL)")
    if not MYSQL_AVAILABLE:
        st.info("MySQL support not available in this host (install mysql-connector-python).")
    else:
        if st.button("Test & Show table names"):
            try:
                conn = connect_db()
                cur = conn.cursor()
                cur.execute("SHOW TABLES;")
                tables = [r[0] for r in cur.fetchall()]
                st.write("Tables:", tables)
                # show few rows from energy_data if exists
                if "energy_data" in tables:
                    cur.execute("SELECT id, year, consumption, baseline_cost, fitted, adjusted, created_at FROM energy_data ORDER BY created_at DESC LIMIT 50;")
                    rows = cur.fetchall()
                    if rows:
                        dfdb = pd.DataFrame(rows, columns=["id","year","consumption","baseline_cost","fitted","adjusted","created_at"])
                        st.dataframe(dfdb)
                    else:
                        st.info("energy_data has no rows.")
                cur.close()
                conn.close()
            except Exception as e:
                st.error(f"Gagal sambung DB: {e}")

# -------------------------
# MENU: Reports
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("Use Energy Forecast to generate exports. This page can list saved reports in future (not persisted currently).")

# -------------------------
# MENU: Settings
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Database")
    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"])
    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        st.session_state.bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        [data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
        </style>
        """
        st.markdown(st.session_state.bg_css, unsafe_allow_html=True)
        st.success("Applied Dark theme.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        st.session_state.bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
        [data-testid="stSidebar"] {background-color: rgba(0,0,0,0.03);}
        </style>
        """
        st.markdown(st.session_state.bg_css, unsafe_allow_html=True)
        st.success("Applied Light theme.")
    else:
        img_url = st.text_input("Enter full image URL for background:")
        if img_url:
            st.session_state.bg_mode = "Custom"
            st.session_state.bg_css = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img_url}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """
            st.markdown(st.session_state.bg_css, unsafe_allow_html=True)
            st.success("Applied custom background.")
    st.markdown("---")
    st.subheader("Database configuration (optional)")
    st.markdown("Enter MySQL connection details (or set environment variables).")
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

# -------------------------
# MENU: Help & About
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    - Developed to forecast energy consumption, cost and CO₂ using historical data and simple models.
    - No hardware required (software-only).
    **Support / Report issues:**  
    📧 chikaenergyforecast@gmail.com
    """)
