# streamlit_app.py
"""
Smart Energy Forecasting – Complete Streamlit App
With persistent dark background, solid black sidebar, and MySQL connectivity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import io
import base64
import mysql.connector

# ===============================
# 🎨 PAGE CONFIGURATION & THEME
# ===============================
st.set_page_config(page_title="Smart Energy Forecasting", page_icon="⚡", layout="wide")

# 🔒 Maintain background setting
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

# ===============================
# 💅 CUSTOM CSS
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FFFFFF;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}
@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        position: fixed !important;
        z-index: 9999 !important;
        background-color: #000000 !important;
        width: 80% !important;
        height: 100vh !important;
        overflow-y: auto !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.7);
    }
    .block-container {padding-top: 3rem !important;}
}
.main {background-color: transparent !important;}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 DATABASE CONNECTION (Railway)
# ===============================
def get_connection():
    try:
        conn = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            port=55398,
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway"
        )
        return conn
    except Exception as e:
        st.warning(f"⚠️ Database connection failed: {e}")
        return None

# ===============================
# 📊 SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio(
    "Navigate:",
    ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management", "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)

# ===============================
# 🧮 FORECAST FUNCTION
# ===============================
def energy_forecast(data):
    X = np.array(data['year']).reshape(-1, 1)
    y = np.array(data['consumption'])
    model = LinearRegression()
    model.fit(X, y)
    next_year = np.array([[data['year'].max() + 1]])
    forecast = model.predict(next_year)
    r2 = r2_score(y, model.predict(X))
    return forecast[0], r2

# ===============================
# 📂 MENU 1: DASHBOARD
# ===============================
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting Dashboard")
    st.write("""
    Welcome!  
    Use the left menu to navigate between forecasting, device management, reports, and settings.  
    You can forecast energy usage, track costs, and export your data easily.
    """)

# ===============================
# ⚡ MENU 2: ENERGY FORECAST
# ===============================
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecasting")

    if "forecast_data" not in st.session_state:
        st.session_state.forecast_data = pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024],
            "consumption": [1200, 1300, 1250, 1400, 1500],
            "baseline_cost": [240, 260, 250, 280, 300]
        })

    df = st.session_state.forecast_data

    st.subheader("📈 Historical Energy Consumption Data")
    st.dataframe(df)

    forecast, r2 = energy_forecast(df)
    next_year = df['year'].max() + 1
    forecast_cost = forecast * 0.2

    new_row = {"year": next_year, "consumption": round(forecast[0], 2),
               "baseline_cost": round(forecast_cost, 2), "forecast": "Yes"}
    st.session_state.forecast_data = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    st.success(f"Predicted consumption for {next_year}: **{round(forecast[0], 2)} kWh**")
    st.write(f"Model Accuracy (R²): {r2:.2f}")

    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS energy_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                year INT,
                consumption FLOAT,
                baseline_cost FLOAT,
                forecast VARCHAR(10)
            )
        """)
        conn.commit()

        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO energy_data (year, consumption, baseline_cost, forecast)
                VALUES (%s, %s, %s, %s)
            """, (int(row['year']), float(row['consumption']), float(row['baseline_cost']), 'No'))
        conn.commit()
        cursor.close()
        conn.close()
        st.info("✅ Data successfully saved to Railway MySQL.")

# ===============================
# 💡 MENU 3: DEVICE MANAGEMENT
# ===============================
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.write("Add, update, or remove devices from your energy tracking system.")
    device = st.text_input("Enter device name")
    usage = st.number_input("Energy usage (kWh/year)", min_value=0.0)
    if st.button("Add Device"):
        st.success(f"Device '{device}' added with usage {usage} kWh/year!")

# ===============================
# 📊 MENU 4: REPORTS
# ===============================
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.write("Export your data in Excel format.")

    if "forecast_data" in st.session_state:
        df = st.session_state.forecast_data
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast")
        excel_data = output.getvalue()

        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name="energy_forecast_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("⚠️ No data available to export.")

# ===============================
# ⚙️ MENU 5: SETTINGS
# ===============================
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    bg = st.selectbox("Background Mode", ["Dark", "Light", "Custom Image"], index=0)
    st.session_state.bg_mode = bg
    st.success(f"Background mode set to {bg} (will stay until user changes it).")

# ===============================
# ❓ MENU 6: HELP & ABOUT
# ===============================
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.write("""
    **Smart Energy Forecasting**  
    Version 2.0 (2025)  
    Developed for energy analysis and cost prediction.
    
    📘 Features:
    - Real-time forecasting using linear regression  
    - MySQL Cloud Database (Railway)  
    - Export to Excel & PDF  
    - Customizable UI  
    """)
