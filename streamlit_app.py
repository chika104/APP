import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from mysql.connector import Error

# ================================
# 🎯 DATABASE CONNECTION SETTINGS
# ================================
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


# ================================
# 🔐 CREATE USERS TABLE (Auto Fix)
# ================================
def create_users_table():
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE
                )
            """)
            # Tambah kolum password jika belum ada
            c.execute("SHOW COLUMNS FROM users LIKE 'password'")
            result = c.fetchone()
            if not result:
                c.execute("ALTER TABLE users ADD COLUMN password VARCHAR(100)")
                conn.commit()
        except mysql.connector.Error as e:
            st.error(f"Gagal cipta/ubah table users: {e}")
        finally:
            conn.close()


# ================================
# 👥 REGISTER USER
# ================================
def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            st.success("Pendaftaran berjaya! Sila log masuk.")
        except mysql.connector.Error as e:
            st.error(f"Gagal register ke DB: {e}")
        finally:
            conn.close()


# ================================
# 🔑 LOGIN CHECK
# ================================
def login_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor(dictionary=True)
            c.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = c.fetchone()
            return user
        except mysql.connector.Error as e:
            st.error(f"Gagal semak DB: {e}")
        finally:
            conn.close()
    return None


# ================================
# 🎨 APP SETTINGS
# ================================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#101010"  # default dark
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()


# ================================
# 🌈 BACKGROUND STYLING
# ================================
def apply_bg():
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {st.session_state.bg_color};
            color: white;
        }}
        [data-testid="stSidebar"] {{
            background-color: #000000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ================================
# 🔐 LOGIN FORM
# ================================
def login_form():
    st.title("🔐 Log Masuk Dashboard")
    tab1, tab2 = st.tabs(["Masuk", "Daftar"])

    with tab1:
        uname = st.text_input("Nama Pengguna")
        pword = st.text_input("Kata Laluan", type="password")
        if st.button("Log Masuk"):
            user = login_user(uname, pword)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = uname
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")

    with tab2:
        new_uname = st.text_input("Daftar Nama Pengguna")
        new_pword = st.text_input("Daftar Kata Laluan", type="password")
        if st.button("Daftar"):
            register_user(new_uname, new_pword)


# ================================
# 📈 DASHBOARD PAGE
# ================================
def dashboard():
    st.title("⚡ Energy Forecast Dashboard")

    # Background changer
    with st.sidebar:
        new_color = st.color_picker("🎨 Tukar warna latar belakang", st.session_state.bg_color)
        if new_color != st.session_state.bg_color:
            st.session_state.bg_color = new_color
            st.rerun()

    st.markdown("### 📊 Data Input")

    data_mode = st.radio("Pilih cara masukkan data:", ["Manual", "Muat naik CSV"])
    if data_mode == "Manual":
        baseline = st.number_input("Baseline Energy (kWh)", min_value=0.0)
        forecast = st.number_input("Forecast Energy (kWh)", min_value=0.0)
        adjusted = st.number_input("Adjusted Energy (kWh)", min_value=0.0)
        if st.button("Tambah Data"):
            new_row = pd.DataFrame({
                "Baseline": [baseline],
                "Forecast": [forecast],
                "Adjusted": [adjusted]
            })
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success("Data berjaya ditambah!")
    else:
        file = st.file_uploader("Muat naik fail CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.session_state.data = df
            st.success("Data CSV dimuat naik!")

    if not st.session_state.data.empty:
        st.markdown("### 📋 Jadual Baseline")
        st.dataframe(st.session_state.data)

        st.markdown("### 🔁 Jadual Perbandingan")
        comparison = st.session_state.data.copy()
        if "Baseline" in comparison and "Forecast" in comparison:
            comparison["Perbezaan (%)"] = (
                (comparison["Forecast"] - comparison["Baseline"]) / comparison["Baseline"]
            ) * 100
        st.dataframe(comparison)

        # ===== Graph Section =====
        st.markdown("## 📈 Visualisasi Graf")
        df = st.session_state.data

        fig1, ax1 = plt.subplots()
        ax1.plot(df["Baseline"], color="red", label="Baseline")
        ax1.set_title("Baseline")
        st.pyplot(fig1)

        if "Forecast" in df:
            fig2, ax2 = plt.subplots()
            ax2.plot(df["Baseline"], color="red", label="Baseline")
            ax2.plot(df["Forecast"], color="blue", label="Forecast")
            ax2.legend()
            ax2.set_title("Baseline vs Forecast")
            st.pyplot(fig2)

        if "Adjusted" in df:
            fig3, ax3 = plt.subplots()
            ax3.plot(df["Baseline"], color="red", label="Baseline")
            ax3.plot(df["Forecast"], color="blue", label="Forecast")
            ax3.plot(df["Adjusted"], color="green", label="Adjusted")
            ax3.legend()
            ax3.set_title("Adjusted vs Forecast vs Baseline")
            st.pyplot(fig3)

        # Baseline Cost
        fig4, ax4 = plt.subplots()
        cost_baseline = df["Baseline"] * 0.5
        ax4.plot(cost_baseline, color="orange", label="Baseline Cost")
        ax4.legend()
        ax4.set_title("Baseline Cost")
        st.pyplot(fig4)

        # Forecast vs Baseline Cost
        if "Forecast" in df:
            fig5, ax5 = plt.subplots()
            cost_forecast = df["Forecast"] * 0.5
            ax5.plot(cost_baseline, color="orange", label="Baseline Cost")
            ax5.plot(cost_forecast, color="purple", label="Forecast Cost")
            ax5.legend()
            ax5.set_title("Forecast Cost vs Baseline Cost")
            st.pyplot(fig5)

        # CO2 Baseline
        fig6, ax6 = plt.subplots()
        co2_baseline = df["Baseline"] * 0.8
        ax6.plot(co2_baseline, color="gray", label="CO2 Baseline")
        ax6.legend()
        ax6.set_title("CO2 Baseline")
        st.pyplot(fig6)

        # CO2 Baseline vs Forecast
        if "Forecast" in df:
            fig7, ax7 = plt.subplots()
            co2_forecast = df["Forecast"] * 0.8
            ax7.plot(co2_baseline, color="gray", label="CO2 Baseline")
            ax7.plot(co2_forecast, color="cyan", label="CO2 Forecast")
            ax7.legend()
            ax7.set_title("CO2 Baseline vs CO2 Forecast")
            st.pyplot(fig7)


# ================================
# 🚀 MAIN APP
# ================================
create_users_table()
apply_bg()

if not st.session_state.logged_in:
    login_form()
    st.stop()
else:
    dashboard()
