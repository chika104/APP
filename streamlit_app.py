import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mysql.connector
from mysql.connector import Error
import io
import base64

# -----------------------------
# PAGE CONFIG & STYLE
# -----------------------------
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_connection():
    try:
        conn = mysql.connector.connect(
            host=" containers-us-west-23.railway.app ",   # 🟢 Ganti ikut info Railway
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",             # 🟢 Ganti password kamu
            database="railway",
            port=3306
        )
        return conn
    except Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.radio(
    "📘 Main Menu",
    ["🏠 Dashboard", "⚡ Energy Forecast", "🔌 Device Management", "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)

# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Energy Dashboard Overview")
    st.write("Welcome to the Smart Energy Forecasting System Dashboard ⚡")
    st.write("View live energy data, cost trends, and performance insights.")

    conn = get_connection()
    if conn:
        df_sql = pd.read_sql("SELECT * FROM energy_data", conn)
        st.subheader("📊 Stored Forecast Data")
        st.dataframe(df_sql)

        fig = px.line(df_sql, x="year", y=["consumption", "forecast"],
                      title="Baseline vs Forecast Energy Consumption")
        st.plotly_chart(fig, use_container_width=True)
        conn.close()
    else:
        st.info("Database not connected or no data found yet.")

# -----------------------------
# ENERGY FORECAST
# -----------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast Module")

    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio("Choose Input Method", ("Upload CSV", "Manual Entry"))

    df = None
    if input_method == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    else:
        rows = st.number_input("Number of Records", 1, 20, 5)
        data = []
        for i in range(int(rows)):
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input(f"Year {i+1}", 2000, 2100, 2020 + i)
            with col2:
                consumption = st.number_input(f"Consumption (kWh) {i+1}", 0.0, 999999.0, 10000.0)
            data.append({"year": year, "consumption": consumption})
        df = pd.DataFrame(data)

    if df is not None and not df.empty:
        st.success("✅ Data successfully loaded!")
        st.dataframe(df)

        st.subheader("Scenario Settings")
        tariff = st.number_input("Electricity Tariff (RM/kWh)", 0.0, 5.0, 0.5, 0.01)
        factor_reduction = st.slider("Reduction Factor (%)", 0, 100, 10)

        # Baseline & forecast
        df["baseline_cost"] = df["consumption"] * tariff
        df["forecast"] = df["consumption"] * (1 - factor_reduction / 100)

        # Display results
        st.subheader("Forecast Table")
        st.dataframe(df)

        # Graphs
        st.subheader("Graphs")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(df, x="year", y=["consumption", "forecast"], markers=True,
                           title="Baseline vs Forecast Consumption")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(df, x="year", y=["baseline_cost"], title="Baseline Cost Trend (RM)")
            st.plotly_chart(fig2, use_container_width=True)

        # Save to DB
        if st.button("💾 Save Forecast Data to Database"):
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS energy_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        year INT,
                        consumption FLOAT,
                        baseline_cost FLOAT,
                        forecast FLOAT
                    )
                """)
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO energy_data (year, consumption, baseline_cost, forecast)
                        VALUES (%s, %s, %s, %s)
                    """, (int(row['year']), float(row['consumption']),
                          float(row['baseline_cost']), float(row['forecast'])))
                conn.commit()
                cursor.close()
                conn.close()
                st.success("✅ Data saved to MySQL database successfully!")

# -----------------------------
# DEVICE MANAGEMENT
# -----------------------------
elif menu == "🔌 Device Management":
    st.title("🔌 Device Management")
    st.write("Add, monitor, or remove connected devices.")
    st.info("Feature under development — future integration with IoT devices planned.")

# -----------------------------
# REPORTS
# -----------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports & Export")

    conn = get_connection()
    if conn:
        df_sql = pd.read_sql("SELECT * FROM energy_data", conn)
        st.dataframe(df_sql)

        # Export to Excel
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine="xlsxwriter") as writer:
            df_sql.to_excel(writer, index=False, sheet_name="ForecastData")

        st.download_button(
            "📥 Download Excel Report",
            data=excel.getvalue(),
            file_name="Energy_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        conn.close()
    else:
        st.warning("No data available to export yet.")

# -----------------------------
# SETTINGS
# -----------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Application Settings")

    st.subheader("🎨 Background Settings")
    bg_choice = st.selectbox("Choose background theme", ["Dark", "Light", "Blue", "Green"])
    if bg_choice == "Dark":
        st.markdown("<style>body { background-color: #0E1117; color: white; }</style>", unsafe_allow_html=True)
    elif bg_choice == "Light":
        st.markdown("<style>body { background-color: #FFFFFF; color: black; }</style>", unsafe_allow_html=True)
    elif bg_choice == "Blue":
        st.markdown("<style>body { background-color: #001F3F; color: white; }</style>", unsafe_allow_html=True)
    elif bg_choice == "Green":
        st.markdown("<style>body { background-color: #003300; color: white; }</style>", unsafe_allow_html=True)

    st.success(f"Theme updated to {bg_choice}")

# -----------------------------
# HELP & ABOUT
# -----------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.write("""
    This web application is developed for energy consumption analysis and forecasting.  
    Users can input data manually or via CSV, visualize trends, and export reports.
    
    📧 For support or to report a system issue, contact:  
    **nurshashiqah125@gmail.com**
    """)
    st.info("Developed with ❤️ by Chika using Streamlit and MySQL (Railway Cloud).")

