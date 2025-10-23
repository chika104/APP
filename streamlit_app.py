# streamlit_app.py
"""
Smart Energy Forecasting — with Login System (Railway DB)
"""

import os, io, base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Database (Railway)
import mysql.connector

# ============= CONFIG =============
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# ---------- Custom CSS ----------
MAIN_STYLE = """
<style>
/* Background persistence */
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #000000;
    color: white;
}
[data-testid="stHeader"] {
    background: #000000;
}

/* Responsive tweaks */
@media (max-width: 768px) {
    h1, h2, h3, h4 { font-size: 95%; }
    button, input, select { font-size: 90%; }
    [data-testid="stSidebar"] { width: 100% !important; }
}
</style>
"""
st.markdown(MAIN_STYLE, unsafe_allow_html=True)

# ---------- DB Connection ----------
def get_conn():
    try:
        return mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            port=55398,
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway"
        )
    except Exception as e:
        st.error(f"Gagal sambung DB: {e}")
        return None


# ---------- User Auth ----------
def register_user(username, password):
    conn = get_conn()
    if not conn:
        return
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=%s", (username,))
    if c.fetchone():
        st.warning("❌ Nama pengguna sudah wujud.")
    else:
        c.execute("INSERT INTO users (username, password_hash) VALUES (%s,%s)", (username, password))
        conn.commit()
        st.success("✅ Akaun berjaya didaftarkan. Sila log masuk.")
    conn.close()


def login_user(username, password):
    conn = get_conn()
    if not conn:
        return
    c = conn.cursor(dictionary=True)
    c.execute("SELECT * FROM users WHERE username=%s", (username,))
    u = c.fetchone()
    conn.close()
    if u and u["password_hash"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.rerun()
    else:
        st.error("❌ Nama pengguna atau kata laluan salah!")


# ---------- Login Form ----------
def login_screen():
    st.title("🔐 Smart Energy Forecasting Login")
    st.markdown("Masukkan akaun Railway anda untuk log masuk.")

    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            login_user(u, p)

    with tab2:
        u2 = st.text_input("Username (baru)")
        p2 = st.text_input("Password (baru)", type="password")
        if st.button("Register"):
            register_user(u2, p2)

# ---------- If Not Logged In ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ---------- Menu ----------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=60)
st.sidebar.markdown(f"### 👋 Hai, {st.session_state.username}")
menu = st.sidebar.radio("📂 Navigasi", [
    "🏠 Dashboard",
    "⚡ Energy Forecast",
    "💡 Device Management",
    "📊 Reports",
    "⚙️ Settings",
    "📘 Help & About",
    "🚪 Logout"
])

if menu == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ---------- Persistent Background ----------
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

# ---------- DASHBOARD ----------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting Dashboard")
    st.markdown("Selamat datang ke sistem peramalan tenaga pintar 🔋")
    st.markdown("- Forecast penggunaan tenaga dan kos masa depan")
    st.markdown("- Simpan data ke pangkalan Railway DB")

# ---------- ENERGY FORECAST ----------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Data Input
    uploaded = st.file_uploader("Upload CSV (year, consumption)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = [c.lower() for c in df.columns]
        if "year" in df.columns and "consumption" in df.columns:
            model = LinearRegression()
            X = df[["year"]]
            y = df["consumption"]
            model.fit(X, y)
            future_years = np.arange(df["year"].max() + 1, df["year"].max() + 8)
            y_pred = model.predict(future_years.reshape(-1, 1))
            df_future = pd.DataFrame({"year": future_years, "forecast": y_pred})

            # Gabung
            df_all = pd.concat([df, df_future.rename(columns={"forecast": "consumption"})])

            # 7 graf warna terang vs gelap
            colors = ["#FF4C4C", "#0047AB", "#00CC66", "#FFA500", "#9932CC", "#CCCC00", "#A0A0A0"]

            st.plotly_chart(px.line(df_all, x="year", y="consumption", title="Energy Consumption Forecast", color_discrete_sequence=[colors[0]]))
            st.plotly_chart(px.bar(df_all, x="year", y="consumption", title="Consumption Bar", color_discrete_sequence=[colors[1]]))
            st.plotly_chart(px.area(df_all, x="year", y="consumption", title="Energy Area Graph", color_discrete_sequence=[colors[2]]))
            st.plotly_chart(px.scatter(df_all, x="year", y="consumption", title="Scatter Plot", color_discrete_sequence=[colors[3]]))
            st.plotly_chart(px.line(df_all, x="year", y="consumption", title="Line Graph 2", color_discrete_sequence=[colors[4]]))
            st.plotly_chart(px.bar(df_all, x="year", y="consumption", title="Bar Graph 2", color_discrete_sequence=[colors[5]]))
            st.plotly_chart(px.line(df_all, x="year", y="consumption", title="Line Graph 3", color_discrete_sequence=[colors[6]]))
        else:
            st.error("CSV mesti ada lajur 'year' dan 'consumption'.")

# ---------- DEVICE MANAGEMENT ----------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    if "devices" not in st.session_state:
        st.session_state.devices = []
    d_name = st.text_input("Nama peranti")
    d_watt = st.number_input("Kuasa (W)", 0, 10000, 10)
    if st.button("Tambah peranti"):
        st.session_state.devices.append({"Device": d_name, "Power (W)": d_watt})
        st.success("Peranti ditambah!")
    st.dataframe(pd.DataFrame(st.session_state.devices))

# ---------- REPORTS ----------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("Gunakan halaman Energy Forecast untuk muat turun laporan Excel/PDF.")

# ---------- SETTINGS ----------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    mode = st.radio("Pilih tema:", ["Dark", "Light", "Custom Image"])
    if mode == "Dark":
        st.session_state.bg_mode = "Dark"
        st.success("Tema gelap digunakan.")
    elif mode == "Light":
        st.session_state.bg_mode = "Light"
        st.markdown("<style>[data-testid='stAppViewContainer']{background:#FFFFFF;color:#000}</style>", unsafe_allow_html=True)
        st.success("Tema cerah digunakan.")
    else:
        img_url = st.text_input("Masukkan URL gambar latar:")
        if img_url:
            custom_style = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url('{img_url}');
                background-size: cover;
                background-position: center;
            }}
            </style>
            """
            st.markdown(custom_style, unsafe_allow_html=True)
            st.session_state.bg_mode = "Custom"
            st.success("Latar belakang disimpan dalam session_state.")

# ---------- HELP ----------
elif menu == "📘 Help & About":
    st.title("📘 Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Versi Streamlit dengan Railway Database Login  

    Dibangunkan oleh: **Chika (Politeknik Kota Kinabalu)**  
    🔧 Sokongan: [chikaenergyforecast@gmail.com](mailto:chikaenergyforecast@gmail.com)
    """)

# End of file
