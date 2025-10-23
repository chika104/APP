import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import plotly.express as px
import plotly.graph_objects as go

# ======================
# DB CONNECTION
# ======================
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
    except Error as e:
        st.error(f"Gagal sambung DB: {e}")
        return None

# ======================
# LOGIN SYSTEM
# ======================
def login(username, password):
    conn = get_connection()
    if conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        conn.close()
        if user and user["password_hash"] == password:
            return True
    return False

def register(username, password):
    conn = get_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing = cur.fetchone()
        if existing:
            st.error("Nama pengguna sudah wujud.")
        else:
            cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password))
            conn.commit()
            st.success("Pendaftaran berjaya! Sila log masuk.")
        conn.close()

# ======================
# APP STATE
# ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: black !important;
    color: white !important;
}
.menu-item {
    padding: 10px;
    color: white;
    font-weight: 600;
}
.menu-item:hover {
    background-color: #1a1a1a;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOGIN PAGE
# ======================
def login_page():
    st.title("🔐 Please login to access the dashboard")
    st.subheader("🔒 Secure Login")
    uname = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pass")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if login(uname, pwd):
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")
    with col2:
        if st.button("Register"):
            register(uname, pwd)

# ======================
# DASHBOARD MENU
# ======================
def navbar():
    st.markdown("""
        <div style='display:flex;justify-content:space-around;background-color:black;
        padding:10px;border-radius:10px;'>
            <a class='menu-item' href='#dashboard'>🏠 Dashboard</a>
            <a class='menu-item' href='#energy'>⚡ Energy Forecast</a>
            <a class='menu-item' href='#devices'>💡 Device Management</a>
            <a class='menu-item' href='#reports'>📊 Reports</a>
            <a class='menu-item' href='#settings'>⚙️ Settings</a>
            <a class='menu-item' href='#help'>❓ Help & About</a>
        </div>
    """, unsafe_allow_html=True)

# ======================
# GRAPH FUNCTION
# ======================
def show_graphs():
    st.header("📈 Energy Forecast Graphs")

    years = list(range(2018, 2026))
    baseline = [100, 120, 130, 140, 150, 160, 170, 180]
    forecast = [110, 125, 135, 145, 160, 175, 185, 195]
    adjusted = [90, 115, 125, 135, 140, 150, 160, 165]
    baseline_cost = [x * 0.22 for x in baseline]
    forecast_cost = [x * 0.22 for x in forecast]
    co2_baseline = [x * 0.8 for x in baseline]
    co2_forecast = [x * 0.7 for x in forecast]

    # 7 Graphs
    charts = {
        "Baseline": go.Figure(go.Scatter(x=years, y=baseline, name="Baseline", line=dict(color="red"))),
        "Baseline vs Forecast": go.Figure([
            go.Scatter(x=years, y=baseline, name="Baseline", line=dict(color="red")),
            go.Scatter(x=years, y=forecast, name="Forecast", line=dict(color="blue"))
        ]),
        "Adjusted vs Forecast vs Baseline": go.Figure([
            go.Scatter(x=years, y=baseline, name="Baseline", line=dict(color="red")),
            go.Scatter(x=years, y=forecast, name="Forecast", line=dict(color="blue")),
            go.Scatter(x=years, y=adjusted, name="Adjusted", line=dict(color="green"))
        ]),
        "Baseline Cost": go.Figure(go.Scatter(x=years, y=baseline_cost, name="Baseline Cost", line=dict(color="orange"))),
        "Forecast Cost vs Baseline Cost": go.Figure([
            go.Scatter(x=years, y=baseline_cost, name="Baseline Cost", line=dict(color="orange")),
            go.Scatter(x=years, y=forecast_cost, name="Forecast Cost", line=dict(color="purple"))
        ]),
        "CO2 Baseline": go.Figure(go.Scatter(x=years, y=co2_baseline, name="CO2 Baseline", line=dict(color="gray"))),
        "CO2 Baseline vs Forecast": go.Figure([
            go.Scatter(x=years, y=co2_baseline, name="CO2 Baseline", line=dict(color="gray")),
            go.Scatter(x=years, y=co2_forecast, name="CO2 Forecast", line=dict(color="lightblue"))
        ]),
    }

    for title, fig in charts.items():
        fig.update_layout(title=title, template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

# ======================
# MAIN APP
# ======================
if not st.session_state.logged_in:
    login_page()
    st.stop()

navbar()

st.title("🏠 Smart Energy Forecasting Dashboard")
st.markdown(f"Welcome, **{st.session_state.username}** 👋")

show_graphs()
