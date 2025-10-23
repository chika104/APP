import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import plotly.express as px

# -------------------- DATABASE CONNECTION --------------------
def get_connection():
    return mysql.connector.connect(
        host="switchback.proxy.rlwy.net",
        port=55398,
        user="root",
        password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
        database="railway"
    )

def create_user_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE,
            password VARCHAR(100)
        )
    """)
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO user_accounts (username, password) VALUES (%s, %s)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM user_accounts WHERE username=%s AND password=%s", (username, password))
    data = c.fetchone()
    conn.close()
    return data

create_user_table()

# -------------------- STYLING --------------------
st.set_page_config(page_title="Smart Energy Forecast", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0b0b0b;
        color: white;
        background-size: cover;
    }
    div[data-testid="stSidebar"] {
        background-color: black;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOGIN SECTION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_section():
    st.title("🔐 Login to Smart Energy Forecast Dashboard")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        uname = st.text_input("Username")
        pword = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(uname, pword)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = uname
                st.success("Login successful ✅")
            else:
                st.error("Nama pengguna atau kata laluan salah!")
    
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            try:
                add_user(new_user, new_pass)
                st.success("Pendaftaran berjaya! Sila login.")
            except:
                st.warning("Nama pengguna sudah wujud.")

# -------------------- DASHBOARD --------------------
def dashboard():
    st.sidebar.title("⚙️ Menu Utama")
    menu = st.sidebar.radio("Pilih menu:", [
        "🏠 Dashboard",
        "⚡ Energy Forecast",
        "🔌 Device Management",
        "📊 Report",
        "⚙️ Settings",
        "💬 Help & About"
    ])

    # -------------------- HOME --------------------
    if menu == "🏠 Dashboard":
        st.title("⚡ Smart Energy Forecast Dashboard")
        st.write("Selamat datang ke sistem ramalan tenaga pintar 💡")

        if "df" in st.session_state:
            st.dataframe(st.session_state.df)
        else:
            st.info("Tiada data. Pergi ke menu Settings atau Upload CSV di Energy Forecast.")

    # -------------------- ENERGY FORECAST --------------------
    elif menu == "⚡ Energy Forecast":
        st.title("⚡ Energy Forecast Analysis")

        # Upload Data
        upload_option = st.radio("Pilih kaedah data:", ["Manual", "Upload CSV"])
        if upload_option == "Manual":
            year = st.number_input("Tahun", min_value=2000, max_value=2100)
            consumption = st.number_input("Penggunaan (kWh)")
            if st.button("Tambah Data"):
                new_data = pd.DataFrame({"year": [year], "consumption": [consumption]})
                if "df" not in st.session_state:
                    st.session_state.df = new_data
                else:
                    st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                st.success("Data berjaya ditambah!")
        else:
            uploaded_file = st.file_uploader("Muat naik fail CSV", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("CSV dimuat naik!")

        if "df" in st.session_state:
            df = st.session_state.df
            df = df.sort_values("year")
            st.subheader("📋 Data Asal")
            st.dataframe(df)

            # 1️⃣ Baseline KWh
            fig1 = px.line(df, x="year", y="consumption", title="Baseline Energy Consumption (kWh)", color_discrete_sequence=["#FF0000"])
            st.plotly_chart(fig1, use_container_width=True)

            # Linear Forecast
            m, b = np.polyfit(df["year"], df["consumption"], 1)
            future_years = np.arange(df["year"].max() + 1, df["year"].max() + 6)
            forecast = m * future_years + b
            forecast_df = pd.DataFrame({"year": future_years, "forecast": forecast})
            st.session_state.forecast_df = forecast_df

            # 2️⃣ Baseline vs Forecast (kWh)
            df_combined = pd.concat([df, forecast_df], ignore_index=True)
            fig2 = px.line(df_combined, x="year", y=["consumption", "forecast"], title="Baseline vs Forecast (kWh)", color_discrete_sequence=["#FF4500", "#0066CC"])
            st.plotly_chart(fig2, use_container_width=True)

            # 3️⃣ Baseline Cost
            df["baseline_cost"] = df["consumption"] * 0.2
            fig3 = px.bar(df, x="year", y="baseline_cost", title="Baseline Cost (RM)", color_discrete_sequence=["#008B8B"])
            st.plotly_chart(fig3, use_container_width=True)

            # 4️⃣ Baseline vs Forecast Cost
            forecast_df["forecast_cost"] = forecast_df["forecast"] * 0.2
            cost_compare = pd.concat([df[["year", "baseline_cost"]], forecast_df[["year", "forecast_cost"]]], ignore_index=True)
            fig4 = px.line(cost_compare, x="year", y=["baseline_cost", "forecast_cost"], title="Baseline vs Forecast Cost (RM)", color_discrete_sequence=["#FFA500", "#0000CD"])
            st.plotly_chart(fig4, use_container_width=True)

            # 5️⃣ CO₂ Forecast
            forecast_df["co2_forecast"] = forecast_df["forecast"] * 0.000233
            fig5 = px.bar(forecast_df, x="year", y="co2_forecast", title="CO₂ Forecast (kg)", color_discrete_sequence=["#32CD32"])
            st.plotly_chart(fig5, use_container_width=True)

    # -------------------- DEVICE MANAGEMENT --------------------
    elif menu == "🔌 Device Management":
        st.title("🔌 Device Management")
        st.write("Tambah dan semak peranti IoT yang digunakan.")
        if "devices" not in st.session_state:
            st.session_state.devices = []
        device_name = st.text_input("Nama Peranti")
        status = st.selectbox("Status", ["Active", "Inactive"])
        if st.button("Tambah Peranti"):
            st.session_state.devices.append({"Device": device_name, "Status": status})
        st.table(st.session_state.devices)

    # -------------------- REPORT --------------------
    elif menu == "📊 Report":
        st.title("📊 Full Report Summary")
        if "forecast_df" in st.session_state:
            st.dataframe(st.session_state.forecast_df)
        else:
            st.warning("Tiada data forecast.")

    # -------------------- SETTINGS --------------------
    elif menu == "⚙️ Settings":
        st.title("⚙️ Settings")
        color = st.color_picker("Tukar warna latar belakang:", "#0b0b0b")
        st.session_state.bg_color = color
        st.write(f"Background diset kepada {color}")

    # -------------------- HELP & ABOUT --------------------
    elif menu == "💬 Help & About":
        st.title("💬 Help & About")
        st.write("""
        **Smart Energy Forecast System**
        Dibangunkan oleh Chika di Politeknik Kota Kinabalu 💡  
        Projek ini bertujuan membantu pengguna menganalisis dan meramal penggunaan tenaga menggunakan pembelajaran mesin.
        """)

# -------------------- MAIN APP --------------------
if not st.session_state.logged_in:
    login_section()
else:
    dashboard()
