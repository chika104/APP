# streamlit_app.py
"""
Smart Energy Forecasting — with Secure Login (MySQL-based)
Chika's Project Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, io, base64
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px

# Try MySQL
MYSQL_AVAILABLE = True
try:
    import mysql.connector
except Exception:
    MYSQL_AVAILABLE = False

# Page setup
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# ---------------------------
# Default style (black sidebar background fix)
# ---------------------------
DEFAULT_STYLE = """
<style>
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #F5F5F5;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
</style>
"""
st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)

# ---------------------------
# Database helpers
# ---------------------------
def get_db_conn():
    if not MYSQL_AVAILABLE:
        st.error("MySQL tidak disokong dalam environment ini.")
        return None
    try:
        conn = mysql.connector.connect(
            host=os.environ.get("DB_HOST", st.session_state.get("db_host", "localhost")),
            user=os.environ.get("DB_USER", st.session_state.get("db_user", "root")),
            password=os.environ.get("DB_PASSWORD", st.session_state.get("db_password", "")),
            database=os.environ.get("DB_DATABASE", st.session_state.get("db_database", "energydb")),
            port=int(os.environ.get("DB_PORT", st.session_state.get("db_port", 3306)))
        )
        return conn
    except Exception as e:
        st.error(f"Gagal sambung DB: {e}")
        return None

def init_user_table():
    conn = get_db_conn()
    if not conn: return
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        role VARCHAR(20) DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def check_user(username, password):
    conn = get_db_conn()
    if not conn: return False, None
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cur.fetchone()
    conn.close()
    if user:
        return True, user
    return False, None

def register_user(username, password):
    conn = get_db_conn()
    if not conn: return False, "DB connection failed"
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        conn.close()
        return True, "Pendaftaran berjaya! Anda boleh login sekarang."
    except mysql.connector.IntegrityError:
        return False, "Username sudah wujud!"
    except Exception as e:
        return False, str(e)

# ---------------------------
# Authentication UI
# ---------------------------
def login_ui():
    st.title("🔐 Smart Energy Forecasting Login")
    choice = st.radio("Pilih tindakan:", ["Login", "Daftar Akaun Baru"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, user = check_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = user["username"]
                st.success(f"Selamat datang, {user['username']}!")
                st.rerun()
            else:
                st.error("Username atau password salah.")

    else:
        st.subheader("🆕 Daftar Akaun Baru")
        new_user = st.text_input("Pilih Username")
        new_pass = st.text_input("Pilih Password", type="password")
        if st.button("Daftar"):
            ok, msg = register_user(new_user, new_pass)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# ---------------------------
# Logout button
# ---------------------------
def logout_button():
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Anda telah log keluar.")
        st.rerun()

# ---------------------------
# Initialize
# ---------------------------
init_user_table()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------
# If not logged in → show login page
# ---------------------------
if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ---------------------------
# Main app menu (only visible after login)
# ---------------------------
menu = st.sidebar.radio(
    f"Selamat datang, {st.session_state.username}",
    ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management", "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)
logout_button()

# ---------------------------
# DASHBOARD
# ---------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting Dashboard")
    st.markdown("""
    Selamat datang ke **Smart Energy Forecasting System**!
    
    Dari sini, anda boleh:
    - Menjana ramalan penggunaan tenaga.
    - Menambah faktor peranti.
    - Memuat turun laporan dalam format PDF atau Excel.
    """)
    st.info("Gunakan menu di kiri untuk navigasi modul lain.")

# ---------------------------
# ENERGY FORECAST (ringkas untuk contoh)
# ---------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")
    st.write("Bahagian ini memaparkan fungsi ramalan tenaga (versi penuh seperti sebelum ini).")

# ---------------------------
# DEVICE MANAGEMENT
# ---------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.write("Tambah atau urus peranti yang digunakan dalam ramalan.")

# ---------------------------
# REPORTS
# ---------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.write("Gunakan laporan ramalan untuk analisis prestasi tenaga anda.")

# ---------------------------
# SETTINGS
# ---------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("Masukkan konfigurasi MySQL (simpan ke session).")
    db_host = st.text_input("DB host", value=st.session_state.get("db_host","localhost"))
    db_port = st.text_input("DB port", value=str(st.session_state.get("db_port","3306")))
    db_user = st.text_input("DB user", value=st.session_state.get("db_user","root"))
    db_password = st.text_input("DB password", value=st.session_state.get("db_password",""), type="password")
    db_database = st.text_input("DB database", value=st.session_state.get("db_database","energydb"))
    if st.button("Simpan DB Settings"):
        st.session_state.db_host = db_host
        st.session_state.db_port = db_port
        st.session_state.db_user = db_user
        st.session_state.db_password = db_password
        st.session_state.db_database = db_database
        st.success("Tetapan DB disimpan dalam session.")

# ---------------------------
# HELP
# ---------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System (Chika Edition)**  
    Sistem ini direka untuk menjana ramalan tenaga, kos dan pelepasan CO₂.

    🔸 **Dibangunkan oleh:** Chika  
    🔸 **Dibantu oleh:** Aiman  
    🔸 **Fungsi:** Login selamat + MySQL penyimpanan pengguna & data ramalan.
    """)
