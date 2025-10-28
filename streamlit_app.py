# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Session defaults
# -------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = pd.DataFrame()
if "df_monthly" not in st.session_state:
    st.session_state.df_monthly = pd.DataFrame()
if "df_yearly" not in st.session_state:
    st.session_state.df_yearly = pd.DataFrame()
if "forecast_monthly" not in st.session_state:
    st.session_state.forecast_monthly = pd.DataFrame()
if "forecast_yearly" not in st.session_state:
    st.session_state.forecast_yearly = pd.DataFrame()

# -------------------------
# Utility functions
# -------------------------
def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def preprocess_multiheader(df):
    # drop empty rows
    df = df.dropna(how="all")
    
    # flatten multi-header if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{a}_{b}" if b != "" else a for a,b in df.columns]
    
    # detect month column
    month_col = df.columns[0]  # assume first col is month
    
    # find kWh columns
    kwh_cols = [c for c in df.columns if "kWh" in c]
    if not kwh_cols:
        raise ValueError("Dataset must contain columns with 'kWh' in header")
    
    df_long = df.melt(id_vars=[month_col], value_vars=kwh_cols, var_name="year_type", value_name="kwh")
    df_long["year"] = df_long["year_type"].str.extract("(\d{4})").astype(int)
    df_long["month"] = df_long[month_col]
    df_long["kwh"] = pd.to_numeric(df_long["kwh"], errors="coerce")
    df_long = df_long.dropna(subset=["kwh"])
    
    return df_long[["year","month","kwh"]].sort_values(["year","month"]).reset_index(drop=True)

def forecast_linear(df, date_col="year", target_col="kwh", n_periods=12):
    model = LinearRegression()
    X_hist = np.arange(len(df)).reshape(-1,1)
    y_hist = df[target_col].values
    model.fit(X_hist, y_hist)
    future_idx = np.arange(len(df), len(df)+n_periods).reshape(-1,1)
    forecast = model.predict(future_idx)
    return forecast

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")
st.title("⚡ Smart Energy Forecasting — Monthly & Yearly Forecast")

# -------------------------
# Step 1: Upload dataset
# -------------------------
uploaded = st.file_uploader("Upload dataset (CSV/Excel)", type=["csv","xlsx"])
if uploaded:
    try:
        if str(uploaded.name).lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded, header=[0,1])  # allow multi-header
        else:
            df_raw = pd.read_excel(uploaded, header=[0,1])
        st.session_state.df_raw = df_raw
        st.success("Dataset loaded successfully.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# -------------------------
# Step 2: Preprocess monthly
# -------------------------
if not st.session_state.df_raw.empty:
    try:
        df_monthly = preprocess_multiheader(st.session_state.df_raw)
        st.session_state.df_monthly = df_monthly
        st.subheader("Monthly data")
        st.dataframe(df_monthly)
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        st.stop()

# -------------------------
# Step 3: Aggregate yearly
# -------------------------
if not st.session_state.df_monthly.empty:
    df_yearly = df_monthly.groupby("year")["kwh"].sum().reset_index()
    st.session_state.df_yearly = df_yearly
    st.subheader("Yearly aggregated data")
    st.dataframe(df_yearly)

# -------------------------
# Step 4: Forecast
# -------------------------
n_months_forecast = st.number_input("Months to forecast ahead", min_value=1, max_value=24, value=12)
n_years_forecast = st.number_input("Years to forecast ahead", min_value=1, max_value=10, value=3)

# Monthly forecast
if not st.session_state.df_monthly.empty:
    forecast_monthly = st.session_state.df_monthly.copy()
    forecast_values = forecast_linear(forecast_monthly, target_col="kwh", n_periods=int(n_months_forecast))
    future_months = pd.date_range(start="2024-01-01", periods=int(n_months_forecast), freq='MS').strftime("%B")
    future_years = pd.date_range(start="2024-01-01", periods=int(n_months_forecast), freq='MS').year
    df_forecast_monthly = pd.DataFrame({
        "year": future_years,
        "month": future_months,
        "forecast_kwh": forecast_values
    })
    st.session_state.forecast_monthly = df_forecast_monthly
    st.subheader("Monthly Forecast")
    st.dataframe(df_forecast_monthly)

    fig_monthly = px.line(pd.concat([forecast_monthly.rename(columns={"kwh":"historical_kwh"}), df_forecast_monthly.rename(columns={"forecast_kwh":"historical_kwh"})]), 
                          x="month", y="historical_kwh", color="year", markers=True, title="Monthly kWh Forecast")
    st.plotly_chart(fig_monthly, use_container_width=True)

# Yearly forecast
if not st.session_state.df_yearly.empty:
    forecast_values_y = forecast_linear(st.session_state.df_yearly, target_col="kwh", n_periods=int(n_years_forecast))
    last_year = st.session_state.df_yearly["year"].max()
    future_years_y = [last_year + i for i in range(1,int(n_years_forecast)+1)]
    df_forecast_yearly = pd.DataFrame({
        "year": future_years_y,
        "forecast_kwh": forecast_values_y
    })
    st.session_state.forecast_yearly = df_forecast_yearly
    st.subheader("Yearly Forecast")
    st.dataframe(df_forecast_yearly)

    fig_yearly = px.line(pd.concat([st.session_state.df_yearly.rename(columns={"kwh":"historical_kwh"}), df_forecast_yearly.rename(columns={"forecast_kwh":"historical_kwh"})]), 
                         x="year", y="historical_kwh", markers=True, title="Yearly kWh Forecast")
    st.plotly_chart(fig_yearly, use_container_width=True)

# -------------------------
# Step 5: Export Excel
# -------------------------
if not st.session_state.df_monthly.empty:
    excel_bytes = df_to_excel_bytes({
        "monthly_data": st.session_state.df_monthly,
        "monthly_forecast": st.session_state.forecast_monthly,
        "yearly_data": st.session_state.df_yearly,
        "yearly_forecast": st.session_state.forecast_yearly
    })
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
