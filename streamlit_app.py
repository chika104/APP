# streamlit_app.py — Updated version with full 6 menus and expanded graphs

import os, io, base64
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
except:
    MYSQL_AVAILABLE = False

st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
     "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)

# --------------- UTILITY ----------------
def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def get_db_config():
    return {
        "host": st.session_state.get("db_host"),
        "user": st.session_state.get("db_user"),
        "password": st.session_state.get("db_password"),
        "database": st.session_state.get("db_database"),
        "port": int(st.session_state.get("db_port", 3306)),
    }

def connect_db():
    cfg = get_db_config()
    return mysql.connector.connect(**cfg)

def init_user_table(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100),
            password VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    c.close()

# --------------- DASHBOARD ----------------
if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard — Smart Energy Forecasting")
    st.markdown("Selamat datang ke sistem ramalan tenaga pintar anda ⚡")

# --------------- ENERGY FORECAST ----------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # INPUT DATA
    input_mode = st.radio("Input method:", ["Upload CSV", "Manual Entry"])
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (columns: year, consumption, cost optional)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = normalize_cols(df)
    else:
        rows = st.number_input("Number of rows", 1, 10, 5)
        data = []
        for i in range(rows):
            c1, c2, c3 = st.columns(3)
            with c1: y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i)
            with c2: cons = st.number_input(f"Consumption (kWh) {i+1}", 0.0, 9999999.0, 10000.0)
            with c3: cost = st.number_input(f"Cost (RM) {i+1}", 0.0, 9999999.0, 0.0)
            data.append({"year": int(y), "consumption": cons, "cost": cost})
        df = pd.DataFrame(data)

    if df is None or df.empty:
        st.warning("Please upload or input baseline data.")
        st.stop()

    df["baseline_cost"] = np.where(df["cost"] == 0, df["consumption"] * 0.52, df["cost"])
    st.subheader("Baseline Data")
    st.dataframe(df)

    # FORECAST MODEL
    model = LinearRegression()
    model.fit(df[["year"]], df["consumption"])
    years_future = [df["year"].max() + i for i in range(1, 6)]
    baseline_pred = model.predict(np.array(years_future).reshape(-1, 1))
    adjusted_pred = baseline_pred * 0.95  # example adjustment
    r2 = r2_score(df["consumption"], model.predict(df[["year"]]))

    forecast_df = pd.DataFrame({
        "year": years_future,
        "baseline_kwh": baseline_pred,
        "forecast_kwh": adjusted_pred,
        "baseline_cost": baseline_pred * 0.52,
        "forecast_cost": adjusted_pred * 0.52,
        "co2_forecast": adjusted_pred * 0.75,
    })

    # GRAPH 1: Baseline kWh
    st.subheader("📈 Baseline kWh")
    fig1 = px.line(df, x="year", y="consumption", markers=True, title="Baseline kWh")
    st.plotly_chart(fig1, use_container_width=True)

    # GRAPH 2: Baseline vs Forecast kWh
    st.subheader("📊 Baseline vs Forecast kWh")
    fig2 = px.line(forecast_df, x="year", y=["baseline_kwh", "forecast_kwh"],
                   markers=True, title="Baseline vs Forecast kWh")
    st.plotly_chart(fig2, use_container_width=True)

    # GRAPH 3: Baseline Cost
    st.subheader("💰 Baseline Cost")
    fig3 = px.bar(df, x="year", y="baseline_cost", title="Baseline Cost (RM)")
    st.plotly_chart(fig3, use_container_width=True)

    # GRAPH 4: Baseline vs Forecast Cost
    st.subheader("💸 Baseline vs Forecast Cost")
    fig4 = px.bar(forecast_df, x="year", y=["baseline_cost", "forecast_cost"],
                  barmode="group", title="Baseline vs Forecast Cost (RM)")
    st.plotly_chart(fig4, use_container_width=True)

    # GRAPH 5: CO2 Forecast
    st.subheader("🌱 CO₂ Forecast")
    fig5 = px.area(forecast_df, x="year", y="co2_forecast", title="CO₂ Forecast (kg)")
    st.plotly_chart(fig5, use_container_width=True)

    st.success(f"Model R² Score: {r2:.4f}")

    # AUTO SAVE TO DB
    if MYSQL_AVAILABLE and "db_host" in st.session_state:
        try:
            conn = connect_db()
            init_user_table(conn)
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("INSERT INTO user_records (username,password) VALUES (%s,%s)",
                               (f"user_{row['year']}", f"pass_{int(row['consumption'])}"))
            conn.commit()
            conn.close()
            st.info("✅ Data auto-saved into database successfully.")
        except Exception as e:
            st.error(f"DB Save Error: {e}")

# --------------- DEVICE MANAGEMENT ----------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.write("Add / Edit devices used in forecast.")

# --------------- REPORTS ----------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.write("View historical forecasts and download summaries here.")

# --------------- SETTINGS ----------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.subheader("Database Configuration")
    st.session_state.db_host = st.text_input("Host", st.session_state.get("db_host", ""))
    st.session_state.db_port = st.text_input("Port", st.session_state.get("db_port", "3306"))
    st.session_state.db_user = st.text_input("User", st.session_state.get("db_user", ""))
    st.session_state.db_password = st.text_input("Password", st.session_state.get("db_password", ""), type="password")
    st.session_state.db_database = st.text_input("Database", st.session_state.get("db_database", ""))

    if st.button("Save Settings"):
        st.success("Settings saved successfully.")

# --------------- HELP & ABOUT ----------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System (Chika Edition)**  
    Developed for automated baseline and forecast energy analysis.  
    - Author: Chika  
    - Assistant: Aiman  
    """)
