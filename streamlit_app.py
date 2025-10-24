# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App with optional MySQL saving
Features included:
- Theme selector (Dark/Light/Custom image)
- Menu navigation: Dashboard, Energy Forecast, Device Management, Reports, Settings, Help & About
- Input: Upload CSV or Manual entry
- Adjustment factors, forecast (LinearRegression)
- Model accuracy (R^2)
- Graphs (5 total), Excel export, optional PDF export (reportlab)
- Optional MySQL connection: configure in Settings, test connection, Save results to DB
"""

import os
import io
from datetime import datetime
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional PDF
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

# Plotly image backend
PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except:
    pass

# Optional MySQL
MYSQL_AVAILABLE = True
try:
    import mysql.connector
except:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Page config and theme
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
    except:
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
        except:
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
        except:
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
            except:
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
    except:
        return None

# -------------------------
# Dashboard
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
# Energy Forecast
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
                st.error("CSV must contain 'year' and a consumption column.")
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
    df["baseline_cost"] = pd.to_numeric(df.get("baseline_cost", np.nan), errors="coerce")
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
            dev_name = device
            watt = WATT[device]
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

    # Site-level adjustment
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
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1)

    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    # Linear regression
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

    # -------------------------
    # Step 4: Graphs and metrics
    # -------------------------
    st.header("Step 4 — Visual comparisons & model accuracy")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("1. Baseline (historical)")
        fig_baseline = px.line(df, x="year", y="consumption", markers=True, title="Baseline kWh")
        st.plotly_chart(fig_baseline, use_container_width=True)

        st.subheader("2. Baseline vs Forecast (kWh)")
        plot_all = pd.concat([
            pd.DataFrame({"year": df["year"], "baseline": df["consumption"], "fitted": df["fitted"]}),
            pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"], "fitted": forecast_df["adjusted_consumption_kwh"]})
        ], ignore_index=True)
        fig_both = px.line(plot_all.sort_values("year"), x="year", y=["baseline","fitted"], markers=True, title="Baseline vs Forecast (kWh)")
        st.plotly_chart(fig_both, use_container_width=True)

        st.subheader("3. Baseline cost (RM)")
        fig_cost_base = px.bar(df, x="year", y="baseline_cost", title="Baseline Cost (RM)")
        st.plotly_chart(fig_cost_base, use_container_width=True)

        st.subheader("4. Baseline vs Forecast cost (RM)")
        fig_cost = px.bar(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], barmode="group", title="Baseline vs Forecast Cost (RM)")
        st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("5. CO₂ forecast (kg)")
        fig_co2 = px.bar(forecast_df, x="year", y=["baseline_co2_kg","adjusted_co2_kg"], barmode="group", title="CO₂ Forecast (kg)")
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
        total_cost_saving =
