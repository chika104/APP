# =========================================
# streamlit_app.py
# Final version by Aiman for Chika 💡
# =========================================

import streamlit as st
import mysql.connector
from mysql.connector import Error
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
DB_CONFIG = {
    "host": "containers-us-west-137.railway.app",
    "user": "root",
    "password": "passwordkau",  # Gantikan dengan password sebenar DB Railway Chika
    "database": "railway",
    "port": 3306
}

st.set_page_config(page_title="Energy Forecast System", layout="wide")


# ----------------------------
# DATABASE FUNCTIONS
# ----------------------------
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        st.error(f"❌ DB connection failed: {e}")
        return None


def create_user_table():
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_accounts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL
                )
            """)
            conn.commit()
            c.close()
        except Error as e:
            st.error(f"⚠️ Error creating user_accounts table: {e}")
        finally:
            conn.close()


def create_energy_table():
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS energy_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100),
                    baseline_kwh FLOAT,
                    forecast_kwh FLOAT,
                    baseline_cost FLOAT,
                    forecast_cost FLOAT,
                    co2 FLOAT
                )
            """)
            conn.commit()
            c.close()
        except Error as e:
            st.error(f"⚠️ Error creating energy_data table: {e}")
        finally:
            conn.close()


# ----------------------------
# SECURITY FUNCTIONS
# ----------------------------
def make_hash(password):
    return hashlib.sha256(str(password).encode()).hexdigest()


def add_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO user_accounts (username, password_hash) VALUES (%s, %s)",
                      (username, make_hash(password)))
            conn.commit()
            st.success("✅ Registration successful! You can now log in.")
        except Error as e:
            st.error(f"⚠️ DB error during registration: {e}")
        finally:
            conn.close()


def login_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor(dictionary=True)
            c.execute("SELECT * FROM user_accounts WHERE username=%s AND password_hash=%s",
                      (username, make_hash(password)))
            result = c.fetchone()
            c.close()
            return result is not None
        except Error as e:
            st.error(f"⚠️ DB error during login: {e}")
        finally:
            conn.close()
    return False


# ----------------------------
# ENERGY DATA HANDLING
# ----------------------------
def save_energy_data(username, df):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            for _, row in df.iterrows():
                c.execute("""
                    INSERT INTO energy_data (username, baseline_kwh, forecast_kwh, baseline_cost, forecast_cost, co2)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (username, row['Baseline_kWh'], row['Forecast_kWh'], row['Baseline_Cost'], row['Forecast_Cost'], row['CO2']))
            conn.commit()
            c.close()
            st.success("✅ Data saved successfully to database!")
        except Error as e:
            st.error(f"⚠️ Failed to save data: {e}")
        finally:
            conn.close()


# ----------------------------
# LOGIN / REGISTER PAGE
# ----------------------------
def login_page():
    st.title("🔐 Secure Login Portal")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            create_user_table()
            create_energy_table()
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password!")

    elif choice == "Register":
        st.subheader("Create New Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            if new_user and new_pass:
                create_user_table()
                add_user(new_user, new_pass)
            else:
                st.warning("⚠️ Please fill in both fields!")


# ----------------------------
# MAIN DASHBOARD SYSTEM
# ----------------------------
def main():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login_page()
    else:
        show_dashboard()


def show_dashboard():
    st.sidebar.title("⚡ Main Menu")
    menu = ["Dashboard", "Energy Forecast", "Device Management", "Reports", "Settings", "Help & About"]
    choice = st.sidebar.radio("Navigate", menu)

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    if choice == "Dashboard":
        st.title("📊 Dashboard")
        st.write("Welcome to your main dashboard.")

    elif choice == "Energy Forecast":
        show_energy_forecast(st.session_state["username"])

    elif choice == "Device Management":
        st.title("💡 Device Management")
        st.write("Manage connected IoT devices here.")

    elif choice == "Reports":
        st.title("📑 Reports")
        st.write("View and export your energy reports.")

    elif choice == "Settings":
        st.title("⚙️ Settings")
        st.write("Customize app preferences.")

    elif choice == "Help & About":
        st.title("❓ Help & About")
        st.write("Energy Forecast System by Chika 💡")


# ----------------------------
# ENERGY FORECAST PAGE
# ----------------------------
def show_energy_forecast(username):
    st.title("⚡ Energy Forecast Module")

    st.write("📂 You can upload CSV or input data manually below:")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data uploaded successfully!")
    else:
        st.subheader("Manual Input")
        num_rows = st.number_input("Number of entries:", min_value=1, max_value=50, value=5)
        data = {
            "Baseline_kWh": [st.number_input(f"Baseline kWh {i+1}", value=100.0) for i in range(num_rows)],
            "Forecast_kWh": [st.number_input(f"Forecast kWh {i+1}", value=110.0) for i in range(num_rows)],
            "Baseline_Cost": [st.number_input(f"Baseline Cost {i+1}", value=50.0) for i in range(num_rows)],
            "Forecast_Cost": [st.number_input(f"Forecast Cost {i+1}", value=55.0) for i in range(num_rows)],
            "CO2": [st.number_input(f"CO2 {i+1}", value=20.0) for i in range(num_rows)]
        }
        df = pd.DataFrame(data)

    if st.button("💾 Save to Database"):
        save_energy_data(username, df)

    st.subheader("📈 Forecast Graphs")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**1️⃣ Baseline kWh**")
        plt.figure()
        plt.plot(df["Baseline_kWh"], marker='o')
        plt.title("Baseline kWh")
        st.pyplot(plt)

    with col2:
        st.write("**2️⃣ Baseline vs Forecast kWh**")
        plt.figure()
        plt.plot(df["Baseline_kWh"], label="Baseline")
        plt.plot(df["Forecast_kWh"], label="Forecast")
        plt.legend()
        plt.title("Baseline vs Forecast kWh")
        st.pyplot(plt)

    col3, col4 = st.columns(2)

    with col3:
        st.write("**3️⃣ Baseline Cost**")
        plt.figure()
        plt.plot(df["Baseline_Cost"], color='green')
        plt.title("Baseline Cost")
        st.pyplot(plt)

    with col4:
        st.write("**4️⃣ Baseline vs Forecast Cost**")
        plt.figure()
        plt.plot(df["Baseline_Cost"], label="Baseline Cost", color='green')
        plt.plot(df["Forecast_Cost"], label="Forecast Cost", color='orange')
        plt.legend()
        plt.title("Baseline vs Forecast Cost")
        st.pyplot(plt)

    st.write("**5️⃣ CO₂ Forecast**")
    plt.figure()
    plt.bar(range(len(df["CO2"])), df["CO2"], color='gray')
    plt.title("CO₂ Forecast")
    st.pyplot(plt)


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    main()
