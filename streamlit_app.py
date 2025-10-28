# streamlit_app.py (updated for monthly forecast)
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
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# ENERGY FORECAST (updated for monthly forecast)
# -------------------------
if menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Step 1: Input CSV
    uploaded = st.file_uploader("Upload CSV with 'MONTH', 'YEAR', 'kWh'", type=["csv", "xlsx"])
    if uploaded:
        if str(uploaded.name).lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        df_raw = normalize_cols(df_raw)

        # Pastikan ada kolum 'month', 'year', 'kwh'
        for c in ['month','year','kwh']:
            if c not in df_raw.columns:
                st.error(f"CSV mesti ada kolum '{c}'")
                st.stop()
        df_raw['year'] = pd.to_numeric(df_raw['year'], errors='coerce')
        df_raw['kwh'] = pd.to_numeric(df_raw['kwh'], errors='coerce')
        df_raw = df_raw.dropna(subset=['year','kwh'])
        df_raw['month'] = df_raw['month'].astype(str)
        st.session_state.df = df_raw

    df = st.session_state.df.copy()
    if df.empty:
        st.warning("Tiada data tersedia")
        st.stop()

    # Step 2: Tambah month_num & year_month
    month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                 "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    df['month_num'] = df['month'].map(month_map)
    df['year_month'] = df['year'] + (df['month_num']-1)/12

    st.subheader("Data Input")
    st.dataframe(df[['year','month','kwh']])

    # Step 3: Linear Regression (monthly)
    model = LinearRegression()
    X_hist = df[['year_month']].values
    y_hist = df['kwh'].values
    model.fit(X_hist, y_hist)
    df['fitted'] = model.predict(X_hist)
    r2 = r2_score(y_hist, df['fitted'])

    # Step 4: Forecast bulanan
    last_year = int(df['year'].max())
    last_month = int(df[df['year']==last_year]['month_num'].max())
    n_months_forecast = st.number_input("Forecast months ahead", min_value=1, max_value=36, value=12)

    future_years = []
    future_months = []
    for i in range(1, n_months_forecast+1):
        month = (last_month + i - 1) % 12 + 1
        year = last_year + (last_month + i - 1)//12
        future_years.append(year)
        future_months.append(month)

    future_year_month = [y + (m-1)/12 for y,m in zip(future_years, future_months)]
    future_X = np.array(future_year_month).reshape(-1,1)
    future_baseline_forecast = model.predict(future_X)

    forecast_df = pd.DataFrame({
        'year': future_years,
        'month_num': future_months,
        'baseline_consumption_kwh': future_baseline_forecast
    })

    # Tariff & CO2
    tariff = st.number_input("Electricity tariff (RM/kWh)", min_value=0.0, value=0.52)
    co2_factor = st.number_input("CO2 factor (kg/kWh)", min_value=0.0, value=0.75)

    forecast_df['baseline_cost_rm'] = forecast_df['baseline_consumption_kwh']*tariff
    forecast_df['baseline_co2_kg'] = forecast_df['baseline_consumption_kwh']*co2_factor

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    # Simpan di session
    st.session_state.forecast_df = forecast_df
