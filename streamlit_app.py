import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from io import StringIO

# ============================
# DATABASE CONNECTION
# ============================
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
        st.error(f"Gagal sambung DB: {e}")
        return None


# ============================
# AUTH SYSTEM
# ============================
def create_users_table():
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password VARCHAR(50)
            )
        """)
        conn.commit()
        conn.close()

def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            st.success("✅ Akaun berjaya didaftarkan! Sila log masuk.")
        except mysql.connector.IntegrityError:
            st.warning("⚠️ Nama pengguna sudah wujud.")
        finally:
            conn.close()

def login_user(username, password):
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        data = c.fetchone()
        conn.close()
        return data
    return None


# ============================
# PAGE SETTINGS
# ============================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# Background kekal
if "bg" not in st.session_state:
    st.session_state.bg = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {st.session_state.bg};
    background-attachment: fixed;
}}
.sidebar .sidebar-content {{
    background-color: black !important;
}}
[data-testid="stSidebar"] {{
    background-color: black !important;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ============================
# LOGIN PAGE
# ============================
create_users_table()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])

    with tab1:
        st.subheader("Login ke Dashboard")
        uname = st.text_input("Username")
        pword = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(uname, pword)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")

    with tab2:
        st.subheader("Daftar Akaun Baharu")
        new_uname = st.text_input("Username Baru")
        new_pword = st.text_input("Kata Laluan Baru", type="password")
        if st.button("Daftar"):
            if new_uname and new_pword:
                register_user(new_uname, new_pword)
            else:
                st.warning("Isi semua ruangan sebelum daftar.")
    st.stop()

# ============================
# SIDEBAR MENU
# ============================
menu = st.sidebar.radio(
    "📊 Menu Pilihan",
    ["Dashboard", "Manual Data Entry", "Upload CSV", "Baseline Table", "Comparison Table", "Visualization", "Settings"]
)

# ============================
# DATA MANAGEMENT
# ============================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Year", "Consumption", "Forecast", "Baseline", "Adjusted", "Cost", "CO2"])

# Manual entry
if menu == "Manual Data Entry":
    st.subheader("➕ Masukkan Data Manual")
    year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
    consumption = st.number_input("Consumption (kWh)", min_value=0.0)
    forecast = st.number_input("Forecast (kWh)", min_value=0.0)
    baseline = st.number_input("Baseline (kWh)", min_value=0.0)
    adjusted = st.number_input("Adjusted (kWh)", min_value=0.0)
    cost = st.number_input("Cost (RM)", min_value=0.0)
    co2 = st.number_input("CO2 (kg)", min_value=0.0)
    if st.button("Simpan Data"):
        new_data = pd.DataFrame([{
            "Year": year,
            "Consumption": consumption,
            "Forecast": forecast,
            "Baseline": baseline,
            "Adjusted": adjusted,
            "Cost": cost,
            "CO2": co2
        }])
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
        st.success("Data berjaya ditambah!")

# Upload CSV
elif menu == "Upload CSV":
    st.subheader("📤 Muat Naik Fail CSV")
    uploaded_file = st.file_uploader("Pilih fail CSV", type="csv")
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
        st.success("Fail CSV berjaya dimuat naik!")

# Baseline Table
elif menu == "Baseline Table":
    st.subheader("📘 Baseline Data Table")
    st.dataframe(st.session_state.data[["Year", "Baseline", "Consumption", "Cost", "CO2"]])

# Comparison Table
elif menu == "Comparison Table":
    st.subheader("📊 Comparison Table (Baseline vs Forecast vs Adjusted)")
    compare_cols = ["Year", "Baseline", "Forecast", "Adjusted", "Cost", "CO2"]
    st.dataframe(st.session_state.data[compare_cols])

# Visualization
elif menu == "Visualization":
    st.subheader("📈 Energy Forecast Graphs")
    df = st.session_state.data
    if df.empty:
        st.warning("Tiada data untuk dipaparkan.")
    else:
        graphs = [
            ("Baseline", ["Year", "Baseline"], ["#ff3333"]),
            ("Baseline vs Forecast", ["Baseline", "Forecast"], ["#ff3333", "#0047ab"]),
            ("Adjusted vs Forecast vs Baseline", ["Adjusted", "Forecast", "Baseline"], ["#00b300", "#0047ab", "#ff3333"]),
            ("Baseline Cost", ["Cost"], ["#ff6600"]),
            ("Forecast Cost vs Baseline Cost", ["Forecast", "Cost"], ["#0047ab", "#ff6600"]),
            ("CO2 Baseline", ["CO2"], ["#888888"]),
            ("CO2 Baseline vs Forecast", ["CO2", "Forecast"], ["#888888", "#0047ab"])
        ]

        for title, cols, colors in graphs:
            plt.figure()
            for col, color in zip(cols, colors):
                plt.plot(df["Year"], df[col], label=col, color=color, linewidth=2)
            plt.title(title, fontsize=14)
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.legend()
            st.pyplot(plt)

# Settings
elif menu == "Settings":
    st.subheader("⚙️ Change Background")
    color = st.color_picker("Pilih warna latar belakang")
    if st.button("Tukar Background"):
        st.session_state.bg = color
        st.success("Warna latar belakang ditukar!")
        st.rerun()
