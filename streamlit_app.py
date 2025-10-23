import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mysql.connector

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# ---------------- DB CONNECTION ----------------
def get_connection():
    try:
        return mysql.connector.connect(
            host=st.session_state.get("db_host", "switchback.proxy.rlwy.net"),
            port=int(st.session_state.get("db_port", 55398)),
            user=st.session_state.get("db_user", "root"),
            password=st.session_state.get("db_password", "polrwgDJZnGLaungxPtGkOTaduCuolEj"),
            database=st.session_state.get("db_database", "railway")
        )
    except Exception as e:
        st.error(f"Gagal sambung DB: {e}")
        return None

# Create user table if not exists
def create_user_table():
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS login_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE,
                password VARCHAR(100)
            )
        """)
        conn.commit()
        conn.close()

def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO login_users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"DB error during registration: {e}")
            return False
    return False

def verify_login(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM login_users WHERE username=%s AND password=%s", (username, password))
            result = c.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            st.error(f"Login DB Error: {e}")
            return False
    return False

# ---------------- LOGIN SYSTEM ----------------
def login_page():
    st.title("🔐 User Login")

    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        uname = st.text_input("Username")
        pword = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_login(uname, pword):
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.success(f"Welcome back, {uname}!")
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")

    with tab2:
        new_uname = st.text_input("Create Username")
        new_pword = st.text_input("Create Password", type="password")
        if st.button("Register"):
            if register_user(new_uname, new_pword):
                st.success("Pendaftaran berjaya! Sila log masuk.")
            else:
                st.error("Gagal daftar pengguna baharu.")

# ---------------- MAIN APP ----------------
def main_dashboard():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #000000;
        }
        body {
            background: linear-gradient(120deg, #f4f4f4, #eaeaea);
        }
        </style>
    """, unsafe_allow_html=True)

    menu = st.sidebar.radio("📋 Menu",
        ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
         "📊 Reports", "⚙️ Settings", "❓ Help & About"])

    # ========== DASHBOARD ==========
    if menu == "🏠 Dashboard":
        st.title("🏠 Dashboard — Smart Energy Forecasting")
        st.write("Selamat datang, ", st.session_state.username)

    # ========== ENERGY FORECAST ==========
    elif menu == "⚡ Energy Forecast":
        st.title("⚡ Energy Forecast Analysis")

        input_mode = st.radio("Input Method", ["Upload CSV", "Manual Entry"])
        df = None

        if input_mode == "Upload CSV":
            file = st.file_uploader("Upload your baseline CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)
                df.columns = [c.lower().strip() for c in df.columns]
        else:
            rows = st.number_input("Number of rows", 1, 10, 5)
            data = []
            for i in range(rows):
                c1, c2, c3 = st.columns(3)
                with c1: year = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i)
                with c2: cons = st.number_input(f"Consumption (kWh) {i+1}", 0.0, 999999.0, 10000.0)
                with c3: cost = st.number_input(f"Cost (RM) {i+1}", 0.0, 999999.0, 5200.0)
                data.append({"year": year, "consumption": cons, "cost": cost})
            df = pd.DataFrame(data)

        if df is not None and not df.empty:
            df["baseline_cost"] = np.where(df["cost"] == 0, df["consumption"] * 0.52, df["cost"])
            st.subheader("Baseline Table")
            st.dataframe(df)

            # Model
            model = LinearRegression()
            model.fit(df[["year"]], df["consumption"])
            years_future = [df["year"].max() + i for i in range(1, 6)]
            forecast = model.predict(np.array(years_future).reshape(-1, 1))
            forecast_df = pd.DataFrame({
                "year": years_future,
                "baseline_kwh": model.predict(np.array(years_future).reshape(-1, 1)),
                "forecast_kwh": forecast * 0.95,
            })
            forecast_df["baseline_cost"] = forecast_df["baseline_kwh"] * 0.52
            forecast_df["forecast_cost"] = forecast_df["forecast_kwh"] * 0.52
            forecast_df["co2_forecast"] = forecast_df["forecast_kwh"] * 0.75

            # Graphs
            st.subheader("📊 Graphs")
            st.plotly_chart(px.line(df, x="year", y="consumption", markers=True, title="Baseline kWh", line_color="red"))
            st.plotly_chart(px.line(forecast_df, x="year", y=["baseline_kwh","forecast_kwh"],
                                    title="Baseline vs Forecast kWh", color_discrete_sequence=["red","darkblue"]))
            st.plotly_chart(px.bar(df, x="year", y="baseline_cost", title="Baseline Cost (RM)", color_discrete_sequence=["green"]))
            st.plotly_chart(px.bar(forecast_df, x="year", y=["baseline_cost","forecast_cost"],
                                   barmode="group", title="Baseline vs Forecast Cost", color_discrete_sequence=["orange","purple"]))
            st.plotly_chart(px.area(forecast_df, x="year", y="co2_forecast", title="CO₂ Forecast", color_discrete_sequence=["grey"]))

            # Auto save
            try:
                conn = get_connection()
                if conn:
                    c = conn.cursor()
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS energy_data (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            year INT,
                            consumption FLOAT,
                            cost FLOAT,
                            baseline_cost FLOAT
                        )
                    """)
                    for _, row in df.iterrows():
                        c.execute("INSERT INTO energy_data (year, consumption, cost, baseline_cost) VALUES (%s,%s,%s,%s)",
                                  (int(row["year"]), float(row["consumption"]), float(row["cost"]), float(row["baseline_cost"])))
                    conn.commit()
                    conn.close()
                    st.success("✅ Data saved to database.")
            except Exception as e:
                st.error(f"Database error: {e}")
        else:
            st.warning("Please upload or input your data.")

    # ========== DEVICE MANAGEMENT ==========
    elif menu == "💡 Device Management":
        st.title("💡 Device Management")
        st.info("Add or edit your energy devices.")

    # ========== REPORTS ==========
    elif menu == "📊 Reports":
        st.title("📊 Reports")
        st.write("View and export historical energy data here.")

    # ========== SETTINGS ==========
    elif menu == "⚙️ Settings":
        st.title("⚙️ Settings")
        st.session_state.db_host = st.text_input("DB Host", st.session_state.get("db_host", "switchback.proxy.rlwy.net"))
        st.session_state.db_port = st.text_input("DB Port", st.session_state.get("db_port", "55398"))
        st.session_state.db_user = st.text_input("DB User", st.session_state.get("db_user", "root"))
        st.session_state.db_password = st.text_input("DB Password", st.session_state.get("db_password", "polrwgDJZnGLaungxPtGkOTaduCuolEj"), type="password")
        st.session_state.db_database = st.text_input("DB Name", st.session_state.get("db_database", "railway"))
        if st.button("Save"):
            st.success("Settings saved!")

    # ========== HELP & ABOUT ==========
    elif menu == "❓ Help & About":
        st.title("❓ Help & About")
        st.markdown("""
        **Smart Energy Forecasting System (Chika Edition)**  
        Built for energy baseline, cost and CO₂ forecasting.  
        - Developer: Chika  
        - Assistant: Aiman  
        """)

# ---------------- MAIN EXEC ----------------
if "logged_in" not in st.session_state:
    create_user_table()
    login_page()
else:
    main_dashboard()
