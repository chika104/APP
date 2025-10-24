import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import matplotlib.pyplot as plt
import numpy as np

# -------------------- DATABASE CONNECTION --------------------
def get_connection():
    return mysql.connector.connect(
        host="switchback.proxy.rlwy.net",
        user="root",
        password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
        database="railway",
        port=55398
    )

def create_user_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM user_accounts WHERE username=%s", (username,))
    if c.fetchone():
        st.warning("⚠️ Username already exists!")
    else:
        c.execute("INSERT INTO user_accounts (username, password_hash) VALUES (%s, %s)",
                  (username, make_hash(password)))
        conn.commit()
        st.success("✅ Account created successfully!")
    conn.close()

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM user_accounts WHERE username=%s AND password_hash=%s",
              (username, make_hash(password)))
    data = c.fetchone()
    conn.close()
    return data is not None

# -------------------- LOGIN PAGE --------------------
def login_page():
    st.title("🔐 Secure Login System")
    menu = ["Login", "Register"]
    choice = st.radio("Select action:", menu, horizontal=True)

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("❌ Incorrect username or password.")
    else:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            register_user(new_username, new_password)

# -------------------- DASHBOARD PAGE --------------------
def dashboard():
    st.header("📊 Dashboard Overview")
    st.write("Welcome to your main energy monitoring dashboard.")

# -------------------- ENERGY FORECAST PAGE --------------------
def energy_forecast():
    st.header("⚡ Energy Forecast Analysis")

    upload_choice = st.radio("Select input method:", ["Manual Entry", "Upload CSV"], horizontal=True)

    if upload_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload your energy data CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state["data"] = df
            st.success("✅ CSV uploaded successfully!")
    else:
        st.write("Enter data manually:")
        year = st.number_input("Year", 2000, 2100, 2025)
        consumption = st.number_input("Baseline Consumption (kWh)", 0.0)
        cost = st.number_input("Baseline Cost (RM)", 0.0)
        co2 = st.number_input("CO₂ Emission (kg)", 0.0)
        if st.button("Add Record"):
            new_row = pd.DataFrame({
                "Year": [year],
                "Baseline_kWh": [consumption],
                "Baseline_Cost": [cost],
                "CO2": [co2]
            })
            if "data" not in st.session_state:
                st.session_state["data"] = new_row
            else:
                st.session_state["data"] = pd.concat([st.session_state["data"], new_row], ignore_index=True)
            st.success("✅ Record added successfully!")

    if "data" in st.session_state:
        df = st.session_state["data"]
        st.subheader("📋 Current Dataset")
        st.dataframe(df)

        # Generate mock forecast data
        df["Forecast_kWh"] = df["Baseline_kWh"] * np.random.uniform(0.9, 1.1, len(df))
        df["Forecast_Cost"] = df["Baseline_Cost"] * np.random.uniform(0.9, 1.1, len(df))
        df["Forecast_CO2"] = df["CO2"] * np.random.uniform(0.85, 1.15, len(df))

        # -------------------- GRAPH 1 --------------------
        st.subheader("📈 Baseline kWh")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Year"], df["Baseline_kWh"], marker="o", label="Baseline kWh")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend()
        st.pyplot(fig1)

        # -------------------- GRAPH 2 --------------------
        st.subheader("📊 Baseline vs Forecast kWh")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Year"], df["Baseline_kWh"], marker="o", label="Baseline kWh")
        ax2.plot(df["Year"], df["Forecast_kWh"], marker="s", label="Forecast kWh")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Energy (kWh)")
        ax2.legend()
        st.pyplot(fig2)

        # -------------------- GRAPH 3 --------------------
        st.subheader("💰 Baseline Cost (RM)")
        fig3, ax3 = plt.subplots()
        ax3.bar(df["Year"], df["Baseline_Cost"], color="orange", label="Baseline Cost")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Cost (RM)")
        ax3.legend()
        st.pyplot(fig3)

        # -------------------- GRAPH 4 --------------------
        st.subheader("📉 Baseline Cost vs Forecast kWh")
        fig4, ax4 = plt.subplots()
        ax4.plot(df["Year"], df["Baseline_Cost"], label="Baseline Cost", color="orange")
        ax4.plot(df["Year"], df["Forecast_kWh"], label="Forecast kWh", color="green")
        ax4.set_xlabel("Year")
        ax4.legend()
        st.pyplot(fig4)

        # -------------------- GRAPH 5 --------------------
        st.subheader("🌿 CO₂ Forecast")
        fig5, ax5 = plt.subplots()
        ax5.bar(df["Year"], df["Forecast_CO2"], color="seagreen")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("CO₂ (kg)")
        st.pyplot(fig5)

# -------------------- DEVICE MANAGEMENT --------------------
def device_management():
    st.header("🔌 Device Management")
    st.write("Add, remove, or monitor IoT devices linked to your energy system.")

# -------------------- REPORT PAGE --------------------
def report_page():
    st.header("🧾 Energy Reports")
    st.write("Generate, view, and export your energy consumption reports.")

# -------------------- SETTINGS PAGE --------------------
def settings_page():
    st.header("⚙️ Settings")
    st.write("Adjust preferences and app configurations here.")

# -------------------- HELP & ABOUT --------------------
def help_about():
    st.header("❓ Help & About")
    st.write("""
    **Energy Forecast System v2.0**  
    Developed by Chika & Aiman 🌟  
    For Polytechnic Kota Kinabalu Project.  
    """)

# -------------------- MAIN APP --------------------
def main():
    create_user_table()
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
    else:
        with st.sidebar:
            st.title("🔧 Navigation")
            menu = [
                "Dashboard",
                "Energy Forecast",
                "Device Management",
                "Report",
                "Settings",
                "Help & About"
            ]
            choice = st.radio("Go to:", menu)

            if st.button("🚪 Logout"):
                st.session_state["logged_in"] = False
                st.session_state["user"] = None
                st.rerun()

        if choice == "Dashboard":
            dashboard()
        elif choice == "Energy Forecast":
            energy_forecast()
        elif choice == "Device Management":
            device_management()
        elif choice == "Report":
            report_page()
        elif choice == "Settings":
            settings_page()
        elif choice == "Help & About":
            help_about()

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()
