# streamlit_app.py — with monthly forecast
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
# Session defaults
# -------------------------
if "df_monthly_hist" not in st.session_state:
    st.session_state.df_monthly_hist = pd.DataFrame()
if "df_monthly_forecast" not in st.session_state:
    st.session_state.df_monthly_forecast = pd.DataFrame()

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

# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

st.title("⚡ Smart Energy Forecast")

# Step 1: Upload monthly CSV
uploaded = st.file_uploader("Upload CSV or Excel (month × year × kWh)", type=["csv", "xlsx"])
if uploaded:
    try:
        if str(uploaded.name).lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    df_raw = df_raw.rename(columns=lambda x: str(x).strip())
    months = df_raw.iloc[:,0].tolist()
    df_melt = df_raw.melt(id_vars=df_raw.columns[0], var_name="year_col", value_name="consumption")
    df_melt = df_melt.rename(columns={df_melt.columns[0]: "month"})
    df_melt = df_melt[df_melt["year_col"].str.isnumeric()]
    df_melt["year"] = df_melt["year_col"].astype(int)
    df_melt["consumption"] = pd.to_numeric(df_melt["consumption"], errors="coerce")
    df_melt = df_melt.dropna(subset=["consumption"])
    df_melt = df_melt[["year", "month", "consumption"]].sort_values(["year", "month"]).reset_index(drop=True)
    st.session_state.df_monthly_hist = df_melt.copy()
    st.success("Monthly historical data loaded!")

# Show historical monthly data
if not st.session_state.df_monthly_hist.empty:
    st.subheader("Historical monthly data")
    st.dataframe(st.session_state.df_monthly_hist)

# Step 2: Forecast settings
st.header("Step 2 — Forecast settings")
n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3)
tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)

# Step 3: Compute monthly forecast
if not st.session_state.df_monthly_hist.empty:
    df_hist = st.session_state.df_monthly_hist.copy()
    forecast_list = []
    for month in df_hist['month'].unique():
        df_m = df_hist[df_hist['month']==month]
        if len(df_m)>=2:
            model = LinearRegression()
            X = df_m[['year']].values
            y = df_m['consumption'].values
            model.fit(X, y)
            last_year = df_m['year'].max()
            future_years = np.array([last_year+i for i in range(1, n_years_forecast+1)]).reshape(-1,1)
            y_pred = model.predict(future_years)
            for i, yval in enumerate(y_pred):
                forecast_list.append({
                    'year': last_year+i+1,
                    'month': month,
                    'forecast_kwh': yval,
                    'forecast_cost_rm': yval*tariff,
                    'forecast_co2_kg': yval*co2_factor
                })

    df_forecast = pd.DataFrame(forecast_list)
    st.session_state.df_monthly_forecast = df_forecast.copy()

    st.subheader("Forecasted monthly data")
    st.dataframe(df_forecast)

    # Visualizations
    st.subheader("Historical vs Forecast")
    df_plot_hist = df_hist.rename(columns={'consumption':'kWh'})
    df_plot_fore = df_forecast.rename(columns={'forecast_kwh':'kWh'})
    df_plot_hist['type'] = 'Historical'
    df_plot_fore['type'] = 'Forecast'
    df_plot_all = pd.concat([df_plot_hist[['year','month','kWh','type']], df_plot_fore[['year','month','kWh','type']]])

    fig = px.line(df_plot_all, x='year', y='kWh', color='month', line_dash='type', markers=True,
                  title='Monthly Historical vs Forecast')
    st.plotly_chart(fig, use_container_width=True)

    # Excel download
    excel_bytes = df_to_excel_bytes({'historical': df_hist, 'forecast': df_forecast})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes,
                       file_name="monthly_energy_forecast.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")