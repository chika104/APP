import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt

# -------------------------------
# 🔧 DATABASE CONNECTION
# -------------------------------
def connect_db():
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
        st.error(f"DB connection failed: {e}")
        return None

# -------------------------------
# 🔐 USER MANAGEMENT
# -------------------------------
def create_user_table():
    conn = connect_db()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_accounts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            """)
            conn.commit()
        except Error as e:
            st.error(f"Error creating user_accounts table: {e}")
        finally:
            conn.close()

def register_user(username, password):
    conn = connect_db()
    if conn:
        try:
            c = conn.cursor()
            create_user_table()
            c.execute("INSERT INTO user_accounts (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            st.success("✅ Registration successful!")
        except mysql.connector.Error as err:
            st.error(f"DB error during registration: {err}")
        finally:
            conn.close()
    else:
        st.error("DB connection failed.")

def login_user(username, password):
    conn = connect_db()
    if conn:
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM user_accounts WHERE username=%s AND password=%s", (username, password))
            user = c.fetchone()
            return user
        except mysql.connector.Error as err:
            st.error(f"DB error during login: {err}")
        finally:
            conn.close()
    return None

# -------------------------------
# 🎨 APP STYLE (BLACK MENU BAR)
# -------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #000000;
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
        [data-testid="stAppViewContainer"] {
            background-size: cover;
            background-attachment: fixed;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🌈 BACKGROUND & SESSION
# -------------------------------
if "bg_url" not in st.session_state:
    st.session_state.bg_url = "https://images.unsplash.com/photo-1504384308090-c894fdcc538d"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{st.session_state.bg_url}");
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔑 LOGIN PAGE
# -------------------------------
def login_page():
    st.title("🔐 Login to Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Nama pengguna atau kata laluan salah!")
    with col2:
        if st.button("Register"):
            register_user(username, password)

# -------------------------------
# 📊 DASHBOARD CONTENT
# -------------------------------
def dashboard():
    st.title(f"📈 Welcome, {st.session_state.username}")
    st.write("Energy Forecast Dashboard")

def data_input():
    st.title("📥 Data Input")
    st.write("Masukkan data manual atau muat naik CSV.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["data"] = df
        st.success("✅ Data berjaya dimuat naik dan disimpan!")
    else:
        data = st.text_area("Masukkan data secara manual (contoh: tarikh, penggunaan)")
        if st.button("Simpan Data Manual") and data:
            df = pd.DataFrame({"Data": [data]})
            st.session_state["data"] = df
            st.success("✅ Data manual berjaya disimpan!")

def comparison_table():
    st.title("📊 Comparison Table")
    if "data" in st.session_state:
        st.dataframe(st.session_state["data"])
    else:
        st.info("Tiada data untuk dibandingkan.")

def baseline_table():
    st.title("📘 Baseline Table")
    if "data" in st.session_state:
        st.dataframe(st.session_state["data"])
    else:
        st.info("Tiada data baseline tersedia.")

def energy_forecast():
    st.title("⚡ Energy Forecast Graphs")

    graphs = [
        "Baseline Only",
        "Baseline vs Forecast",
        "Adjusted vs Forecast vs Baseline",
        "Baseline Cost",
        "Forecast Cost vs Baseline Cost",
        "CO2 Baseline",
        "CO2 Baseline vs CO2 Forecast"
    ]

    colors = ["red", "blue", "green", "orange", "purple", "grey", "darkblue"]

    for i, title in enumerate(graphs):
        st.subheader(title)
        x = range(1, 11)
        y = [v * (i + 1) for v in x]
        plt.figure()
        plt.plot(x, y, color=colors[i], label=title)
        plt.legend()
        st.pyplot(plt)

def settings_page():
    st.title("⚙️ Settings")
    st.write("Tukar latar belakang dashboard.")
    new_bg = st.text_input("Masukkan URL imej latar belakang:")
    if st.button("Tukar Background"):
        st.session_state.bg_url = new_bg
        st.rerun()

# -------------------------------
# 🚪 MAIN LOGIC
# -------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    menu = st.sidebar.radio(
        "Menu",
        ["Dashboard", "Data Input", "Comparison Table", "Baseline Table", "Energy Forecast", "Settings"]
    )

    if menu == "Dashboard":
        dashboard()
    elif menu == "Data Input":
        data_input()
    elif menu == "Comparison Table":
        comparison_table()
    elif menu == "Baseline Table":
        baseline_table()
    elif menu == "Energy Forecast":
        energy_forecast()
    elif menu == "Settings":
        settings_page()
