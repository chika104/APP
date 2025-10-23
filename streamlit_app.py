import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px

# ==============================
# ⚙️ Database Connection Setup
# ==============================
DB_CONFIG = {
    "host": "switchback.proxy.rlwy.net",
    "port": 55398,
    "user": "root",
    "password": "polrwgDJZnGLaungxPtGkOTaduCuolEj",
    "database": "railway"
}

def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Gagal sambung DB: {e}")
        return None

def create_users_table():
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE,
                    password VARCHAR(100)
                )
            """)
            conn.commit()
        except mysql.connector.Error as e:
            st.error(f"Gagal cipta table users: {e}")
        finally:
            conn.close()

def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            st.success("✅ Pendaftaran berjaya! Sila log masuk.")
        except mysql.connector.Error as e:
            st.error(f"Gagal register ke DB: {e}")
        finally:
            conn.close()

def login_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            return c.fetchone()
        except mysql.connector.Error as e:
            st.error(f"Gagal log masuk DB: {e}")
        finally:
            conn.close()
    return None


# ============================================
# 🎨 UI - Background, Theme & Persistent Style
# ============================================
st.markdown("""
    <style>
        body {
            background-color: #0a0a0a;
            color: white;
        }
        [data-testid="stSidebar"] {
            background-color: #111 !important;
        }
        [data-testid="stSidebarNav"]::before {
            color: white;
            font-weight: bold;
            font-size: 18px;
            content: "⚡ Energy Forecast Dashboard";
            margin-left: 10px;
            margin-bottom: 20px;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 🧭 Login & Register System
# ===========================
create_users_table()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_form():
    st.subheader("🔐 User Login")
    uname = st.text_input("Username")
    pword = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            user = login_user(uname, pword)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = uname
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")
    with col2:
        if st.button("Register"):
            register_user(uname, pword)

if not st.session_state.logged_in:
    st.title("🔒 Welcome to Energy Forecast System")
    login_form()
    st.stop()

# ===========================
# 🧭 Sidebar Menu
# ===========================
menu = st.sidebar.radio("📂 Menu", [
    "🏠 Dashboard",
    "📊 Upload Data",
    "📈 Forecast Graphs",
    "📋 Table Comparison",
    "⚙️ Settings",
    "🚪 Logout"
])

# ===========================
# 📊 Data Section
# ===========================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

# ==================================
# MENU 1: DASHBOARD
# ==================================
if menu == "🏠 Dashboard":
    st.title("⚡ Energy Forecast Dashboard")
    st.write("Welcome,", st.session_state.user)
    st.success("This dashboard shows your energy forecasting data and insights.")

# ==================================
# MENU 2: UPLOAD DATA
# ==================================
elif menu == "📊 Upload Data":
    st.header("📊 Upload or Input Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("✅ Data successfully uploaded!")

    st.write("Or add manually:")
    with st.form("manual_input"):
        year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
        consumption = st.number_input("Consumption (kWh)")
        baseline_cost = st.number_input("Baseline Cost (RM)")
        forecast = st.number_input("Forecast (kWh)")
        submit_btn = st.form_submit_button("Add Data")

        if submit_btn:
            new_row = pd.DataFrame({
                "year": [year],
                "consumption": [consumption],
                "baseline_cost": [baseline_cost],
                "forecast": [forecast]
            })
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success("✅ Data added manually!")

    if not st.session_state.data.empty:
        st.write("### Current Data Table")
        st.dataframe(st.session_state.data)

# ==================================
# MENU 3: FORECAST GRAPHS
# ==================================
elif menu == "📈 Forecast Graphs":
    st.header("📈 Energy Forecast Visualizations")
    df = st.session_state.data

    if df.empty:
        st.warning("Please upload or input data first.")
    else:
        graph_options = {
            "Baseline Only": ("year", "consumption", "blue"),
            "Baseline vs Forecast": ("year", ["consumption", "forecast"], ["red", "darkblue"]),
            "Adjusted vs Forecast vs Baseline": ("year", ["consumption", "forecast", "baseline_cost"], ["orange", "green", "purple"]),
            "Baseline Cost": ("year", "baseline_cost", "cyan"),
            "Forecast vs Baseline Cost": ("year", ["forecast", "baseline_cost"], ["red", "blue"]),
            "CO2 Baseline": ("year", "consumption", "gray"),
            "CO2 Baseline vs CO2 Forecast": ("year", ["consumption", "forecast"], ["green", "purple"])
        }

        selected_graph = st.selectbox("Choose Graph", list(graph_options.keys()))
        x_col, y_col, colors = graph_options[selected_graph]

        if isinstance(y_col, list):
            fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=colors, markers=True)
        else:
            fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=[colors], markers=True)

        fig.update_layout(
            plot_bgcolor="#0a0a0a",
            paper_bgcolor="#0a0a0a",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================
# MENU 4: TABLE COMPARISON
# ==================================
elif menu == "📋 Table Comparison":
    st.header("📋 Data Comparison Table")
    df = st.session_state.data
    if not df.empty:
        df["Forecast Error (%)"] = ((df["forecast"] - df["consumption"]) / df["consumption"]) * 100
        st.dataframe(df)
    else:
        st.warning("Please upload or input data first.")

# ==================================
# MENU 5: SETTINGS
# ==================================
elif menu == "⚙️ Settings":
    st.header("⚙️ User Settings")
    bg_color = st.color_picker("Choose Background Color", "#0a0a0a")
    st.session_state.bg_color = bg_color
    st.markdown(f"<style>body{{background-color:{bg_color};}}</style>", unsafe_allow_html=True)

# ==================================
# MENU 6: LOGOUT
# ==================================
elif menu == "🚪 Logout":
    st.session_state.logged_in = False
    st.success("You have logged out.")
    st.rerun()
