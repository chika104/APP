import streamlit as st
import pandas as pd
import plotly.express as px
import mysql.connector
from datetime import datetime
import base64
import os

# -----------------------------
# DATABASE CONFIGURATION
# -----------------------------
DB_CONFIG = {
    "host": "switchback.proxy.rlwy.net",
    "port": 55398,
    "user": "root",
    "password": "polrwgDJZnGLaungxPtGkOTaduCuolEj",
    "database": "railway"
}

def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                year INT,
                total_kwh FLOAT,
                total_cost FLOAT,
                co2_saving FLOAT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS report_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                filename VARCHAR(255)
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.warning(f"⚠️ Database init error: {e}")

init_db()

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

# Keep dark background permanently
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1A1D23;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR MENU
# -----------------------------
menu = st.sidebar.radio("📋 Menu", ["Dashboard", "Energy Forecast", "Upload Data", "Report"])

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def parse_monthly_two_row_header_csv(file):
    df_raw = pd.read_csv(file, header=[0, 1])
    df_raw.columns = ['_'.join(map(str, col)).strip().replace(" ", "_") for col in df_raw.columns.values]

    st.write("🧩 Columns detected in file:", list(df_raw.columns))

    month_cols = [c for c in df_raw.columns if "MONTH" in c.upper()]
    if not month_cols:
        raise KeyError("No column containing 'MONTH' found in CSV header.")
    month_col = month_cols[0]

    df = pd.DataFrame()
    for year in [2019, 2020, 2021, 2022, 2023]:
        kwh_col = None
        cost_col = None

        for c in df_raw.columns:
            if str(year) in c and "kWh" in c and "RM(kWh)" not in c:
                kwh_col = c
            if str(year) in c and ("RM(Total)" in c or "RM" in c):
                cost_col = c

        if kwh_col and cost_col:
            temp = pd.DataFrame({
                "MONTH": df_raw[month_col],
                "YEAR": year,
                "kWh": pd.to_numeric(df_raw[kwh_col], errors="coerce"),
                "Cost_RM": pd.to_numeric(df_raw[cost_col], errors="coerce")
            })
            df = pd.concat([df, temp], ignore_index=True)

    df["MONTH"] = df["MONTH"].astype(str).str.strip()
    df = df.dropna(subset=["kWh"])
    return df

def save_forecast_to_db(year, total_kwh, total_cost, co2_saving):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO forecast_history (timestamp, year, total_kwh, total_cost, co2_saving)
            VALUES (%s, %s, %s, %s, %s)
        """, (datetime.now(), year, total_kwh, total_cost, co2_saving))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.warning(f"⚠️ DB save error: {e}")

def save_report_to_db(filename):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO report_history (timestamp, filename)
            VALUES (%s, %s)
        """, (datetime.now(), filename))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.warning(f"⚠️ DB save error: {e}")

# -----------------------------
# UPLOAD DATA PAGE
# -----------------------------
if menu == "Upload Data":
    st.title("📤 Upload Energy Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = parse_monthly_two_row_header_csv(uploaded_file)
        st.session_state.forecast_data = df
        st.success("✅ Dataset successfully loaded and stored in session.")
        st.dataframe(df.head())

# -----------------------------
# ENERGY FORECAST PAGE
# -----------------------------
elif menu == "Energy Forecast":
    st.title("⚡ Energy Forecast")

    if st.session_state.forecast_data is not None:
        df = st.session_state.forecast_data

        fig = px.line(df, x="MONTH", y="kWh", color="YEAR", title="Monthly Energy Usage (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        total_kwh = df.groupby("YEAR")["kWh"].sum().reset_index()
        total_cost = df.groupby("YEAR")["Cost_RM"].sum().reset_index()

        merged = pd.merge(total_kwh, total_cost, on="YEAR")
        merged["CO2_Saving_kg"] = merged["kWh"] * 0.527

        st.subheader("📊 Annual Summary")
        st.dataframe(merged)

        for _, row in merged.iterrows():
            save_forecast_to_db(int(row["YEAR"]), float(row["kWh"]), float(row["Cost_RM"]), float(row["CO2_Saving_kg"]))
    else:
        st.info("Please upload a dataset first in the 'Upload Data' menu.")

# -----------------------------
# DASHBOARD PAGE
# -----------------------------
elif menu == "Dashboard":
    st.title("📈 Energy Monitoring Dashboard")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql("SELECT * FROM forecast_history ORDER BY year", conn)
        conn.close()

        if not df.empty:
            fig1 = px.bar(df, x="year", y="total_kwh", title="Total Energy (kWh) by Year")
            fig2 = px.bar(df, x="year", y="co2_saving", title="CO₂ Saving (kg) by Year")

            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No forecast data found in database.")
    except Exception as e:
        st.warning(f"⚠️ DB load error: {e}")

# -----------------------------
# REPORT PAGE
# -----------------------------
elif menu == "Report":
    st.title("📑 Forecast Reports")

    st.write("Download last forecast result as PDF:")
    if st.button("📥 Download PDF"):
        filename = f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with open(filename, "w") as f:
            f.write("Energy Forecast Report (placeholder)")
        save_report_to_db(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Click here to download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.subheader("📚 Report History")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        df_report = pd.read_sql("SELECT * FROM report_history ORDER BY timestamp DESC", conn)
        conn.close()
        st.dataframe(df_report)
    except Exception as e:
        st.warning(f"⚠️ Report history load error: {e}")
