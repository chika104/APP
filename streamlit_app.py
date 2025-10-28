# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io
from datetime import datetime

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Session defaults
# -------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = pd.DataFrame()
if "df_monthly" not in st.session_state:
    st.session_state.df_monthly = pd.DataFrame()
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
    # convert to long format
    kwh_cols = [c for c in df.columns if "kWh" in c]
    df_long = df.melt(id_vars=["MONTH"], value_vars=kwh_cols, var_name="year_type", value_name="kwh")
    df_long["year"] = df_long["year_type"].str.extract("(\d{4})").astype(int)
    df_long["month"] = df_long["MONTH"]
    df_long["kwh"] = pd.to_numeric(df_long["kwh"], errors="coerce")
    df_long = df_long.dropna(subset=["kwh"])
    return df_long[["year","month","kwh"]].sort_values(["year","month"]).reset_index(drop=True)

def forecast_linear(df, date_col="year", value_col="kwh", n_periods=3):
    model = LinearRegression()
    X = df[[date_col]].values
    y = df[value_col].values
    if len(X) < 2:
        df[f"fitted_{value_col}"] = y
        return df, pd.DataFrame()
    model.fit(X, y)
    df[f"fitted_{value_col}"] = model.predict(X)
    last_date = int(df[date_col].max())
    future_dates = np.array([last_date + i for i in range(1, n_periods+1)]).reshape(-1,1)
    forecast_vals = model.predict(future_dates)
    forecast_df = pd.DataFrame({
        date_col: future_dates.flatten(),
        value_col: forecast_vals
    })
    return df, forecast_df

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")
st.title("⚡ Smart Energy Forecasting — Monthly & Yearly")

# Sidebar
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "📈 Forecast", "⚙️ Settings"])

# -------------------------
# Dashboard
# -------------------------
if menu == "🏠 Dashboard":
    st.markdown("**Welcome!** Upload your multi-header Excel/CSV dataset and forecast energy consumption monthly or yearly.")
    if not st.session_state.df_raw.empty:
        st.subheader("Current loaded data")
        st.dataframe(st.session_state.df_raw)

# -------------------------
# Forecast
# -------------------------
elif menu == "📈 Forecast":
    st.header("Step 1 — Upload historical dataset")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx","csv"])
    if uploaded_file:
        try:
            if str(uploaded_file.name).lower().endswith(".csv"):
                df_input = pd.read_csv(uploaded_file, header=[0,1])
            else:
                df_input = pd.read_excel(uploaded_file, header=[0,1])
            st.session_state.df_raw = df_input.copy()
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    if not st.session_state.df_raw.empty:
        st.subheader("Step 2 — Preprocess dataset")
        df_monthly = preprocess_multiheader(st.session_state.df_raw)
        st.session_state.df_monthly = df_monthly
        st.dataframe(df_monthly)

        st.subheader("Step 3 — Forecast settings")
        forecast_years = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3)
        tariff = st.number_input("Electricity tariff RM/kWh", min_value=0.0, value=0.52, step=0.01)

        # Monthly forecast
        st.subheader("Monthly Forecast")
        df_monthly["baseline_cost_rm"] = df_monthly["kwh"] * tariff
        df_monthly, forecast_monthly = forecast_linear(df_monthly, date_col="year", value_col="kwh", n_periods=forecast_years)
        forecast_monthly["cost_rm"] = forecast_monthly["kwh"] * tariff
        st.session_state.forecast_monthly = forecast_monthly
        st.dataframe(forecast_monthly)

        # Yearly aggregation & forecast
        st.subheader("Yearly Forecast")
        df_yearly = df_monthly.groupby("year")["kwh"].sum().reset_index()
        df_yearly["baseline_cost_rm"] = df_yearly["kwh"] * tariff
        df_yearly, forecast_yearly = forecast_linear(df_yearly, date_col="year", value_col="kwh", n_periods=forecast_years)
        forecast_yearly["cost_rm"] = forecast_yearly["kwh"] * tariff
        st.session_state.forecast_yearly = forecast_yearly
        st.dataframe(forecast_yearly)

        # Visualizations
        st.subheader("Visual Comparison")
        fig_month = px.line(df_monthly, x="year", y="kwh", markers=True, title="Historical kWh (Monthly)")
        st.plotly_chart(fig_month, use_container_width=True)

        fig_forecast_month = px.line(forecast_monthly, x="year", y="kwh", markers=True, title="Forecast Monthly kWh")
        st.plotly_chart(fig_forecast_month, use_container_width=True)

        fig_year = px.line(df_yearly, x="year", y="kwh", markers=True, title="Historical kWh (Yearly)")
        st.plotly_chart(fig_year, use_container_width=True)

        fig_forecast_year = px.line(forecast_yearly, x="year", y="kwh", markers=True, title="Forecast Yearly kWh")
        st.plotly_chart(fig_forecast_year, use_container_width=True)

        # Export Excel
        excel_bytes = df_to_excel_bytes({
            "monthly_historical": df_monthly,
            "monthly_forecast": forecast_monthly,
            "yearly_historical": df_yearly,
            "yearly_forecast": forecast_yearly
        })
        st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes,
                           file_name="energy_forecast.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Settings
# -------------------------
elif menu == "⚙️ Settings":
    st.header("Settings")
    st.info("Currently no additional settings implemented. You can set tariff and forecast horizon in Forecast page.")
