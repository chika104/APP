import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import numpy as np

# -------------------- CONFIGURATIONS --------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

DB_CONFIG = {
    "host": "switchback.proxy.rlwy.net",
    "port": 55398,
    "user": "root",
    "password": "polrwgDJZnGLaungxPtGkOTaduCuolEj",
    "database": "railway"
}

# -------------------- DB CONNECTION --------------------
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        st.error(f"❌ Gagal sambung DB: {e}")
        return None

# -------------------- USER AUTH --------------------
def get_user(username):
    conn = get_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()
        return user
    return None

def register_user(username, password):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password))
        conn.commit()
        conn.close()
        return True
    return False

# -------------------- BACKGROUND STYLE --------------------
if "bg_style" not in st.session_state:
    st.session_state.bg_style = "black"

def set_background(color):
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: #000000;
            color: white;
        }}
        [data-testid="stAppViewContainer"] {{
            background-color: {color};
            color: white;
        }}
        h1, h2, h3, h4, h5, h6, p, span, div {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(st.session_state.bg_style)

# -------------------- LOGIN & REGISTER --------------------
def login_register():
    st.title("🔐 Please login to access the dashboard")
    st.subheader("🔒 Secure Login")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = get_user(username)
            if user and user["password_hash"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")

    with tab_register:
        username_r = st.text_input("New Username", key="reg_user")
        password_r = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            user = get_user(username_r)
            if user:
                st.warning("Nama pengguna sudah wujud!")
            else:
                if register_user(username_r, password_r):
                    st.success("Akaun berjaya didaftarkan! Sila login semula.")

# -------------------- DASHBOARD MENU --------------------
def sidebar_menu():
    st.sidebar.title("💠 Smart Energy Forecasting")
    menu = st.sidebar.radio(
        "Navigate:",
        ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
         "📊 Reports", "⚙️ Settings", "❓ Help & About"]
    )
    return menu

# -------------------- GRAPH FUNCTION --------------------
def plot_graph(title, x, y, color, labelx, labely):
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linewidth=3)
    ax.set_title(title, color='white')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    st.pyplot(fig)

# -------------------- DASHBOARD CONTENT --------------------
def dashboard():
    st.header("📊 Dashboard Overview")
    st.info(f"Welcome, {st.session_state.username}!")

    x = np.arange(2020, 2027)
    baseline = np.random.randint(300, 600, len(x))
    forecast = baseline + np.random.randint(-100, 100, len(x))
    adjusted = baseline - np.random.randint(0, 80, len(x))
    cost_base = baseline * 0.25
    cost_forecast = forecast * 0.25
    co2_base = baseline * 0.3
    co2_forecast = forecast * 0.3

    plot_graph("Baseline Energy", x, baseline, "red", "Year", "kWh")
    plot_graph("Baseline vs Forecast", x, forecast, "blue", "Year", "kWh")
    plot_graph("Adjusted vs Forecast vs Baseline", x, adjusted, "orange", "Year", "kWh")
    plot_graph("Baseline Cost", x, cost_base, "purple", "Year", "RM")
    plot_graph("Forecast Cost vs Baseline Cost", x, cost_forecast, "green", "Year", "RM")
    plot_graph("CO₂ Baseline", x, co2_base, "grey", "Year", "kgCO₂")
    plot_graph("CO₂ Baseline vs Forecast", x, co2_forecast, "cyan", "Year", "kgCO₂")

# -------------------- SETTINGS --------------------
def settings():
    st.header("⚙️ Settings")
    color = st.color_picker("Pilih warna latar belakang:", st.session_state.bg_style)
    if st.button("Tukar Latar Belakang"):
        st.session_state.bg_style = color
        st.rerun()
    st.success("Warna latar belakang kekal walaupun tukar menu.")

# -------------------- APP --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_register()
    st.stop()
else:
    menu = sidebar_menu()

    if menu == "🏠 Dashboard":
        dashboard()
    elif menu == "⚡ Energy Forecast":
        st.header("⚡ Energy Forecast Module")
    elif menu == "💡 Device Management":
        st.header("💡 Device Management")
    elif menu == "📊 Reports":
        st.header("📊 Reports Section")
    elif menu == "⚙️ Settings":
        settings()
    elif menu == "❓ Help & About":
        st.header("❓ Help & About")
        st.info("Developed by Chika — Polytechnic Kota Kinabalu Project")

    st.sidebar.write("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
