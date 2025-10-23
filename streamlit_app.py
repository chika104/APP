import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import hashlib

# ==============================
# DATABASE CONNECTION
# ==============================
def create_connection():
    try:
        conn = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway",
            port=55398
        )
        return conn
    except Error as e:
        st.error(f"Gagal sambung DB: {e}")
        return None


# ==============================
# USER AUTH FUNCTIONS
# ==============================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed):
    if make_hashes(password) == hashed:
        return True
    return False

def create_usertable():
    conn = create_connection()
    if conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password_hash VARCHAR(255)
        )''')
        conn.commit()
        conn.close()

def add_userdata(username, password):
    conn = create_connection()
    if conn:
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', (username, make_hashes(password)))
        conn.commit()
        conn.close()

def login_user(username, password):
    conn = create_connection()
    if conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=%s', (username,))
        data = c.fetchone()
        conn.close()
        if data and check_hashes(password, data[2]):
            return True
    return False


# ==============================
# SETUP PAGE & BACKGROUND
# ==============================
st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

if "background_color" not in st.session_state:
    st.session_state.background_color = "#0e1117"

page_bg = f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-color: {st.session_state.background_color};
        color: white;
    }}
    [data-testid="stSidebar"] {{
        background-color: black;
    }}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# ==============================
# LOGIN / SIGNUP SYSTEM
# ==============================
menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

create_usertable()

if choice == "Sign Up":
    st.title("🔐 User Registration")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Create Account"):
        if new_user and new_password:
            try:
                add_userdata(new_user, new_password)
                st.success("✅ Account created successfully! Please login.")
            except:
                st.error("⚠️ Username already exists!")
        else:
            st.warning("Please fill in all fields.")

elif choice == "Login":
    st.title("🔑 Login to Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success(f"Welcome, {username}! Redirecting...")
        else:
            st.error("Incorrect Username/Password")

# ==============================
# MAIN DASHBOARD
# ==============================
if "authenticated" in st.session_state and st.session_state["authenticated"]:

    st.sidebar.title("📊 Dashboard Menu")
    menu = ["Home", "Energy Forecast", "Database"]
    selected = st.sidebar.radio("Go to", menu)

    if selected == "Home":
        st.title("🏠 Welcome to the Energy Forecasting Dashboard")
        st.write("Manage and visualize your energy consumption efficiently.")
        color = st.color_picker("🎨 Choose Background Color", st.session_state.background_color)
        if color != st.session_state.background_color:
            st.session_state.background_color = color
            st.experimental_rerun()

    elif selected == "Energy Forecast":
        st.title("⚡ Energy Forecasting System")

        if "forecast_data" not in st.session_state:
            st.session_state.forecast_data = pd.DataFrame(columns=["Year", "Consumption (kWh)", "Cost (RM)", "Forecast (kWh)"])

        year = st.number_input("Enter Year", min_value=2000, max_value=2100, step=1)
        consumption = st.number_input("Energy Consumption (kWh)", min_value=0.0)
        cost = st.number_input("Estimated Cost (RM)", min_value=0.0)
        forecast = st.number_input("Forecast (kWh)", min_value=0.0)

        if st.button("Save Record"):
            new_row = {"Year": year, "Consumption (kWh)": consumption, "Cost (RM)": cost, "Forecast (kWh)": forecast}
            st.session_state.forecast_data = pd.concat([st.session_state.forecast_data, pd.DataFrame([new_row])], ignore_index=True)
            st.success("✅ Record added successfully!")

        st.dataframe(st.session_state.forecast_data)

    elif selected == "Database":
        st.title("🗄️ Database Viewer")
        conn = create_connection()
        if conn:
            c = conn.cursor()
            c.execute("SHOW TABLES")
            tables = [x[0] for x in c.fetchall()]
            st.write("Available tables:", tables)
            if tables:
                table_choice = st.selectbox("Select a table to view", tables)
                if st.button("Load Table"):
                    df = pd.read_sql(f"SELECT * FROM {table_choice}", conn)
                    st.dataframe(df)
            conn.close()

else:
    st.info("👈 Please login from the sidebar to access the dashboard.")
