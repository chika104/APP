# streamlit_app.py
"""
Smart Energy Forecasting — Monthly & Yearly
- Forecast monthly or yearly energy consumption and cost
- Device-level adjustment factors supported
- Excel & PDF export, optional MySQL save
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
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Session defaults
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "df_factors" not in st.session_state:
    st.session_state.df_factors = pd.DataFrame()
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "report_history" not in st.session_state:
    st.session_state.report_history = []

# -------------------------
# Utils
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

def make_pdf_bytes(title_text, summary_lines, table_blocks, image_bytes_list=None):
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
# App config
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management", "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# DASHBOARD
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("""
    Welcome — forecast energy and cost monthly or yearly.
    """)

# -------------------------
# ENERGY FORECAST
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV/Excel (needs 'year/month' & 'kWh' columns)", type=["csv","xlsx"])
        if uploaded:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)
            st.session_state.df = df_raw
    else:
        st.info("Manual entry not implemented yet for monthly data. Please upload CSV/Excel.")

    df = st.session_state.df.copy()
    if df.empty:
        st.warning("No data loaded yet.")
        st.stop()
    st.subheader("Loaded data")
    st.dataframe(df)

    # Step 2: Device factors
    st.header("Step 2 — Device Adjustment Factors")
    if st.session_state.df_factors.empty:
        st.session_state.df_factors = pd.DataFrame([{"device":"LED Lamp","units":0,"hours_per_year":0,"action":"Addition","kwh_per_year":0.0}])
    n_factors = st.number_input("Factor rows", min_value=1, max_value=10, value=len(st.session_state.df_factors))
    factors_edit = []
    WATT = {"LED":10, "CFL":15,"Fluorescent":40,"Computer":150,"Lab Equipment":500}
    for i in range(n_factors):
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        prev = st.session_state.df_factors.iloc[i].to_dict() if i < len(st.session_state.df_factors) else {}
        with c1:
            device = st.selectbox(f"Device {i+1}", ["Lamp - LED","Lamp - CFL","Lamp - Fluorescent","Computer","Lab Equipment"], index=0, key=f"dev_{i}")
        with c2:
            units = st.number_input("Units", min_value=0, value=int(prev.get("units",0)), key=f"units_{i}")
        with c3:
            hours = st.number_input("Hours/year", min_value=0, max_value=8760, value=int(prev.get("hours_per_year",0)), key=f"hours_{i}")
        with c4:
            action = st.selectbox("Action", ["Addition","Reduction"], index=0 if prev.get("action","Addition")=="Addition" else 1, key=f"action_{i}")
        subtype = device.split(" - ")[1] if " - " in device else device
        watt = WATT.get(subtype, 10)
        kwh_per_year = (watt*units*hours)/1000.0
        if action=="Reduction":
            kwh_per_year=-abs(kwh_per_year)
        factors_edit.append({"device":subtype+" Lamp" if "Lamp" in device else subtype, "units":units,"hours_per_year":hours,"action":action,"kwh_per_year":kwh_per_year})
    st.session_state.df_factors = pd.DataFrame(factors_edit)
    st.dataframe(st.session_state.df_factors)

    total_net_adjust_kwh = st.session_state.df_factors["kwh_per_year"].sum()
    st.info(f"Net adjustment: {total_net_adjust_kwh:.2f} kWh/year")

    # Step 3: Forecast
    st.header("Step 3 — Forecast Settings")
    forecast_type = st.radio("Forecast type:", ["Yearly","Monthly"])
    tariff = st.number_input("Electricity tariff RM/kWh", 0.0, 10.0, 0.52)
    
    if forecast_type=="Yearly":
        df_grouped = df.groupby("year")["kwh"].sum().reset_index()
        df_grouped["baseline_cost"] = df_grouped["kwh"]*tariff
        X = df_grouped[["year"]].values
        y = df_grouped["kwh"].values
        model = LinearRegression()
        model.fit(X,y)
        last_year = df_grouped["year"].max()
        future_years = [last_year+i for i in range(1,4)]
        future_X = np.array(future_years).reshape(-1,1)
        future_baseline = model.predict(future_X)
        future_adjusted = future_baseline + total_net_adjust_kwh
        forecast_df = pd.DataFrame({
            "year":future_years,
            "baseline_kwh":future_baseline,
            "adjusted_kwh":future_adjusted
        })
        forecast_df["baseline_cost"] = forecast_df["baseline_kwh"]*tariff
        forecast_df["adjusted_cost"] = forecast_df["adjusted_kwh"]*tariff
    else:
        df_grouped = df.groupby(["year","month"])["kwh"].sum().reset_index()
        df_grouped["baseline_cost"] = df_grouped["kwh"]*tariff
        X = np.arange(len(df_grouped)).reshape(-1,1)
        y = df_grouped["kwh"].values
        model = LinearRegression()
        model.fit(X,y)
        future_idx = np.arange(len(df_grouped), len(df_grouped)+12).reshape(-1,1)
        future_baseline = model.predict(future_idx)
        future_adjusted = future_baseline + total_net_adjust_kwh/12  # distribute monthly
        forecast_df = pd.DataFrame({
            "month_idx":np.arange(1,13),
            "baseline_kwh":future_baseline,
            "adjusted_kwh":future_adjusted
        })
        forecast_df["baseline_cost"] = forecast_df["baseline_kwh"]*tariff
        forecast_df["adjusted_cost"] = forecast_df["adjusted_kwh"]*tariff

    st.session_state.forecast_df = forecast_df

    st.header("Forecast Results")
    st.dataframe(forecast_df)

    # Plots
    fig1 = px.line(forecast_df, y=["baseline_kwh","adjusted_kwh"], markers=True, title="kWh Forecast")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(forecast_df, y=["baseline_cost","adjusted_cost"], barmode="group", title="Cost Forecast")
    st.plotly_chart(fig2, use_container_width=True)

    # Excel export
    excel_bytes = df_to_excel_bytes({"forecast":forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF export
    images = [try_get_plot_png(fig1), try_get_plot_png(fig2)]
    summary_lines = ["Forecast generated"]
    table_blocks = [("Forecast", forecast_df)]
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("Energy Forecast Report", summary_lines, table_blocks, image_bytes_list=images)
        if pdf_bytes:
            st.download_button("📄 Download PDF report", data=pdf_bytes, file_name="forecast.pdf", mime="application/pdf")
