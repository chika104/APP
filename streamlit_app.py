import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import matplotlib.pyplot as plt

# ==============================
# DATABASE CONNECTION
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
    except Exception as e:
        st.error(f"Gagal sambung DB: {e}")
        return None

# ==============================
# USER AUTHENTICATION
# ==============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(input_password, stored_password):
    hashed_input = hash_password(input_password)
    # auto-detect hashed or plain
    return input_password == stored_password or hashed_input == stored_password

def login_user(username, password):
    conn = get_connection()
    if not conn:
        return False
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and verify_password(password, result[0]):
        return True
    return False

def register_user(username, password):
    conn = get_connection()
    if not conn:
        return False
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
    conn.commit()
    conn.close()
    return True

# ==============================
# PAGE CONFIG & BACKGROUND
# ==============================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#F8F9FA"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {st.session_state.bg_color};
}}
[data-testid="stSidebar"] {{
    background-color: black;
}}
[data-testid="stSidebarNav"] {{
    background-color: black;
}}
[data-testid="stSidebarNav"] a {{
    color: white !important;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# LOGIN / REGISTER PAGE
# ==============================
def login_register():
    st.title("🔐 Login to Energy Dashboard")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        uname = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(uname, pwd):
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Nama pengguna atau kata laluan salah!")

    with tab2:
        new_uname = st.text_input("New Username")
        new_pwd = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(new_uname, new_pwd):
                st.success("✅ Registered successfully! You can now login.")
            else:
                st.error("❌ Registration failed.")

# ==============================
# MAIN DASHBOARD
# ==============================
def dashboard():
    st.sidebar.title("📊 Menu")
    menu = st.sidebar.radio("Navigate", [
        "🏠 Home",
        "📥 Data Input",
        "📈 Forecast Graphs",
        "📊 Tables Comparison",
        "⚙️ Settings",
        "🚪 Logout"
    ])

    # ============ HOME ============
    if menu == "🏠 Home":
        st.title("Energy Forecasting Dashboard")
        st.write(f"Welcome, **{st.session_state.username}** 👋")
        st.write("This dashboard provides visualization and forecasting for energy consumption trends.")

    # ============ DATA INPUT ============
    elif menu == "📥 Data Input":
        st.subheader("📊 Upload or Enter Data")
        option = st.radio("Choose input method", ["Manual Entry", "Upload CSV"])

        if option == "Manual Entry":
            year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
            consumption = st.number_input("Consumption (kWh)", min_value=0.0)
            cost = st.number_input("Cost (RM)", min_value=0.0)

            if st.button("Save Data"):
                conn = get_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO energy_data (year, consumption, baseline_cost)
                        VALUES (%s, %s, %s)
                    """, (year, consumption, cost))
                    conn.commit()
                    conn.close()
                    st.success("✅ Data saved successfully!")

        elif option == "Upload CSV":
            file = st.file_uploader("Upload CSV file", type="csv")
            if file:
                df = pd.read_csv(file)
                st.dataframe(df)
                if st.button("Save to Database"):
                    conn = get_connection()
                    if conn:
                        cur = conn.cursor()
                        for _, row in df.iterrows():
                            cur.execute("""
                                INSERT INTO energy_data (year, consumption, baseline_cost)
                                VALUES (%s, %s, %s)
                            """, (row['year'], row['consumption'], row['baseline_cost']))
                        conn.commit()
                        conn.close()
                        st.success("✅ CSV Data saved!")

    # ============ FORECAST GRAPHS ============
    elif menu == "📈 Forecast Graphs":
        st.subheader("📉 Forecast Visualization")
        conn = get_connection()
        if conn:
            df = pd.read_sql("SELECT * FROM energy_data", conn)
            conn.close()

            if not df.empty:
                df = df.sort_values("year")
                df["forecast"] = df["consumption"] * 1.1
                df["adjusted"] = df["consumption"] * 1.05
                df["co2_baseline"] = df["consumption"] * 0.8
                df["co2_forecast"] = df["forecast"] * 0.8
                df["forecast_cost"] = df["baseline_cost"] * 1.15

                colors = ["red", "darkblue", "green", "orange", "purple", "grey"]

                graphs = [
                    ("Baseline Consumption", ["consumption"]),
                    ("Baseline vs Forecast", ["consumption", "forecast"]),
                    ("Adjusted vs Forecast vs Baseline", ["adjusted", "forecast", "consumption"]),
                    ("Baseline Cost", ["baseline_cost"]),
                    ("Forecast vs Baseline Cost", ["forecast_cost", "baseline_cost"]),
                    ("CO2 Baseline", ["co2_baseline"]),
                    ("CO2 Baseline vs Forecast", ["co2_baseline", "co2_forecast"])
                ]

                for title, cols in graphs:
                    fig, ax = plt.subplots()
                    for i, col in enumerate(cols):
                        ax.plot(df["year"], df[col], label=col, color=colors[i % len(colors)], linewidth=2)
                    ax.set_title(title)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.warning("⚠️ No data available.")

    # ============ TABLES COMPARISON ============
    elif menu == "📊 Tables Comparison":
        st.subheader("📋 Data Comparison Table")
        conn = get_connection()
        if conn:
            df = pd.read_sql("SELECT * FROM energy_data", conn)
            conn.close()
            df["forecast"] = df["consumption"] * 1.1
            df["adjusted"] = df["consumption"] * 1.05
            df["forecast_cost"] = df["baseline_cost"] * 1.15
            df["co2_baseline"] = df["consumption"] * 0.8
            df["co2_forecast"] = df["forecast"] * 0.8
            st.dataframe(df)

    # ============ SETTINGS ============
    elif menu == "⚙️ Settings":
        st.subheader("🎨 Change Background Color")
        color = st.color_picker("Pick Background Color", value=st.session_state.bg_color)
        if st.button("Apply Background"):
            st.session_state.bg_color = color
            st.rerun()

    # ============ LOGOUT ============
    elif menu == "🚪 Logout":
        st.session_state.logged_in = False
        st.success("You have logged out.")
        st.rerun()

# ==============================
# APP FLOW
# ==============================
if not st.session_state.logged_in:
    login_register()
else:
    dashboard()
