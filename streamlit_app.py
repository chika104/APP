import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import matplotlib.pyplot as plt

# ================================
# DATABASE CONNECTION
# ================================
def get_connection():
    return mysql.connector.connect(
        host="switchback.proxy.rlwy.net",
        user="root",
        password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
        database="railway",
        port=55398
    )

# ================================
# CREATE USER TABLE
# ================================
def create_user_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE,
            password_hash VARCHAR(255)
        )
    """)
    conn.commit()
    conn.close()

# ================================
# HASH FUNCTION
# ================================
def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# ================================
# ADD USER
# ================================
def add_user(username, password):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("INSERT INTO user_accounts (username, password_hash) VALUES (%s, %s)",
                  (username, make_hash(password)))
        conn.commit()
        conn.close()
        return True
    except mysql.connector.Error as err:
        st.error(f"DB error during registration: {err}")
        return False

# ================================
# LOGIN VALIDATION
# ================================
def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM user_accounts WHERE username=%s AND password_hash=%s",
              (username, make_hash(password)))
    result = c.fetchone()
    conn.close()
    return result

# ================================
# SETUP PAGE CONFIG
# ================================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# ================================
# CSS STYLING (MENU & BACKGROUND)
# ================================
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: black !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# INITIALIZE SESSION STATE
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================================
# LOGIN & REGISTER PAGE
# ================================
def login_page():
    st.title("🔒 Secure Login System")
    menu = ["Login", "Register"]
    choice = st.radio("Select an option", menu, horizontal=True)

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password!")

    elif choice == "Register":
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if add_user(new_user, new_pass):
                st.success("Account created successfully! You can now log in.")
            else:
                st.error("Registration failed!")

# ================================
# ENERGY FORECAST PAGE
# ================================
def energy_forecast_page():
    st.subheader("⚡ Energy Forecast Analysis")
    st.markdown("Upload your CSV file or input data manually below:")

    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
    else:
        st.info("Or manually enter data below:")
        df = pd.DataFrame({
            "Year": [2020, 2021, 2022],
            "Baseline (kWh)": [1200, 1300, 1400],
            "Forecast (kWh)": [1250, 1350, 1450]
        })

    st.dataframe(df)

    # Graph 1: Baseline kWh
    st.write("### Baseline Energy (kWh)")
    plt.figure(figsize=(6,3))
    plt.plot(df["Year"], df["Baseline (kWh)"], color="red", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("kWh")
    plt.grid(True)
    st.pyplot(plt)

    # Graph 2: Baseline vs Forecast
    st.write("### Baseline vs Forecast (kWh)")
    plt.figure(figsize=(6,3))
    plt.plot(df["Year"], df["Baseline (kWh)"], color="red", label="Baseline")
    plt.plot(df["Year"], df["Forecast (kWh)"], color="blue", label="Forecast")
    plt.legend()
    st.pyplot(plt)

    # Graph 3: Baseline Cost
    st.write("### Baseline Cost (RM)")
    df["Baseline Cost (RM)"] = df["Baseline (kWh)"] * 0.2
    plt.figure(figsize=(6,3))
    plt.bar(df["Year"], df["Baseline Cost (RM)"], color="orange")
    st.pyplot(plt)

    # Graph 4: Baseline vs Forecast Cost
    st.write("### Baseline vs Forecast Cost (RM)")
    df["Forecast Cost (RM)"] = df["Forecast (kWh)"] * 0.2
    plt.figure(figsize=(6,3))
    plt.plot(df["Year"], df["Baseline Cost (RM)"], color="orange", label="Baseline Cost")
    plt.plot(df["Year"], df["Forecast Cost (RM)"], color="purple", label="Forecast Cost")
    plt.legend()
    st.pyplot(plt)

    # Graph 5: CO₂ Forecast
    st.write("### CO₂ Emission Forecast (kg)")
    df["CO₂ Forecast (kg)"] = df["Forecast (kWh)"] * 0.233
    plt.figure(figsize=(6,3))
    plt.plot(df["Year"], df["CO₂ Forecast (kg)"], color="green", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("CO₂ (kg)")
    st.pyplot(plt)

# ================================
# OTHER MENU PAGES
# ================================
def dashboard_page():
    st.subheader("📊 Dashboard Overview")
    st.info("This is the main dashboard summary of the Energy Forecast system.")

def device_management_page():
    st.subheader("🖥️ Device Management")
    st.write("Manage connected IoT devices and sensors here.")

def report_page():
    st.subheader("📑 Reports & Data Export")
    st.write("Generate and download forecast reports or summaries.")

def settings_page():
    st.subheader("⚙️ Settings")
    st.write("Customize theme, background, and user preferences here.")

def help_about_page():
    st.subheader("❓ Help & About")
    st.write("This Energy Forecast System is developed as part of your project demonstration.")

# ================================
# MAIN DASHBOARD STRUCTURE
# ================================
def main_app():
    st.sidebar.title("Navigation Menu")
    menu = ["Dashboard", "Energy Forecast", "Device Management", "Report", "Settings", "Help & About"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Dashboard":
        dashboard_page()
    elif choice == "Energy Forecast":
        energy_forecast_page()
    elif choice == "Device Management":
        device_management_page()
    elif choice == "Report":
        report_page()
    elif choice == "Settings":
        settings_page()
    elif choice == "Help & About":
        help_about_page()

    st.sidebar.write("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# ================================
# RUN APP
# ================================
create_user_table()
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
