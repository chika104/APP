import streamlit as st
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# ===================== DATABASE CONNECTION =====================
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            port=55398,
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway"
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"Gagal sambung DB: {err}")
        return None


# ===================== REGISTER FUNCTION =====================
def register_user(username, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            st.warning("Nama pengguna sudah wujud!")
        else:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password)
            )
            conn.commit()
            st.success("Pendaftaran berjaya! Anda boleh log masuk sekarang.")
        cursor.close()
        conn.close()


# ===================== LOGIN FUNCTION =====================
def login_user(username, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and user["password_hash"] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success(f"Selamat datang, {username}!")
            st.rerun()
        else:
            st.error("Nama pengguna atau kata laluan salah!")


# ===================== LOGIN FORM =====================
def login_form():
    st.markdown("<h2 style='text-align:center;'>🔒 Secure Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            login_user(username, password)
    with col2:
        if st.button("Register"):
            register_user(username, password)


# ===================== PAGE SETUP =====================
st.set_page_config(page_title="Energy Dashboard", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#0d1117"  # default dark background

# ===================== LOGIN CHECK =====================
if not st.session_state.logged_in:
    st.markdown(
        f"<h1 style='text-align:center;color:white;'>🔐 Please login to access the dashboard</h1>",
        unsafe_allow_html=True
    )
    login_form()
    st.stop()

# ===================== DASHBOARD =====================
st.markdown(
    f"""
    <style>
        body {{
            background-color: {st.session_state.bg_color};
            color: white;
        }}
        .main {{
            background-color: {st.session_state.bg_color};
        }}
        .menu-bar {{
            background-color: #000;
            padding: 12px;
            text-align: center;
            border-radius: 12px;
        }}
        .menu-item {{
            color: white;
            margin: 0 15px;
            text-decoration: none;
            font-weight: bold;
        }}
        .menu-item:hover {{
            color: #00b4d8;
        }}
    </style>
    <div class='menu-bar'>
        <a class='menu-item' href='#'>Dashboard</a>
        <a class='menu-item' href='#'>Usage</a>
        <a class='menu-item' href='#'>Forecast</a>
        <a class='menu-item' href='#'>Reports</a>
        <a class='menu-item' href='#'>Settings</a>
        <a class='menu-item' href='#'>Profile</a>
        <a class='menu-item' href='#'>Logout</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 Energy Usage Dashboard")
st.markdown("Warna graf kontras — terang vs gelap (merah, biru tua, hijau, oren, ungu, kuning, kelabu).")

# Contoh dataset rawak
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    "Energy_Use": [120, 135, 150, 145, 160, 155, 170],
    "Cost": [300, 340, 360, 355, 380, 370, 400],
}
df = pd.DataFrame(data)

# 7 graf
colors = ["red", "darkblue", "green", "orange", "purple", "gold", "gray"]
titles = [
    "Energy Usage Trend",
    "Cost Comparison",
    "Monthly Growth",
    "Energy vs Cost",
    "Forecast Accuracy",
    "Daily Average",
    "Yearly Projection",
]

for i, title in enumerate(titles):
    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Energy_Use"], color=colors[i % len(colors)], linewidth=3)
    ax.set_title(title, color="white")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    st.pyplot(fig)

# Tukar warna background
st.sidebar.header("⚙️ Settings")
bg_color = st.sidebar.color_picker("Pilih warna latar belakang", st.session_state.bg_color)
if st.sidebar.button("Simpan warna latar"):
    st.session_state.bg_color = bg_color
    st.sidebar.success("Warna latar disimpan! 🌈")
    st.rerun()
