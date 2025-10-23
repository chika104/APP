import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import plotly.express as px
from datetime import datetime
from io import BytesIO

# --------------------- DB CONNECTION ---------------------
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

# --------------------- STYLING ---------------------
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0b0b0b;
        background-attachment: fixed;
        background-size: cover;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: black;
    }
    div[data-testid="stSidebarNav"] ul {
        background-color: black;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------- LOGIN PAGE ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_section():
    st.title("🔐 Login to Dashboard")
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

# --------------------- DASHBOARD ---------------------
def dashboard():
    st.sidebar.title("⚙️ Dashboard Menu")
    menu = st.sidebar.radio("Select a Menu:", [
        "🏠 Home", 
        "📊 Baseline Analysis", 
        "📈 Forecasting", 
        "💰 Cost Analysis", 
        "🌿 CO₂ Emission", 
        "📁 Data Upload", 
        "📜 Report"
    ])

    if menu == "🏠 Home":
        st.title("⚡ Energy Forecasting Dashboard")
        st.write("Selamat datang ke sistem ramalan tenaga pintar 💡")

    elif menu == "📁 Data Upload":
        st.title("📂 Upload Data")
        upload_method = st.radio("Pilih kaedah input:", ["Manual", "Upload CSV"])
        if upload_method == "Manual":
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
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("CSV dimuat naik!")

        if "df" in st.session_state:
            st.subheader("📋 Data Semasa")
            st.dataframe(st.session_state.df)

    elif menu == "📊 Baseline Analysis":
        st.title("📊 Baseline Analysis")
        if "df" in st.session_state:
            df = st.session_state.df
            st.dataframe(df)
            fig = px.line(df, x="year", y="consumption", title="Baseline Energy Consumption (kWh)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tiada data. Sila muat naik atau masukkan data dahulu.")

    elif menu == "📈 Forecasting":
        st.title("🔮 Forecasting Results")
        if "df" in st.session_state:
            df = st.session_state.df
            X = df["year"].values.reshape(-1, 1)
            y = df["consumption"].values
            m, b = np.polyfit(df["year"], df["consumption"], 1)
            future_years = np.arange(df["year"].max()+1, df["year"].max()+6)
            forecast = m * future_years + b
            forecast_df = pd.DataFrame({"year": future_years, "forecast": forecast})
            st.session_state.forecast_df = forecast_df
            st.dataframe(forecast_df)

            fig = px.line(forecast_df, x="year", y="forecast", title="Forecast Energy Consumption (kWh)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sila muat naik data dahulu.")

    elif menu == "💰 Cost Analysis":
        st.title("💰 Cost Comparison")
        if "forecast_df" in st.session_state:
            df = st.session_state.forecast_df
            df["baseline_cost"] = df["forecast"] * 0.2
            fig = px.bar(df, x="year", y="baseline_cost", title="Baseline Cost (RM)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tiada forecast data!")

    elif menu == "🌿 CO₂ Emission":
        st.title("🌿 CO₂ Emission Forecast")
        if "forecast_df" in st.session_state:
            df = st.session_state.forecast_df
            df["co2_forecast"] = df["forecast"] * 0.000233
            fig = px.bar(df, x="year", y="co2_forecast", title="CO₂ Forecast (kg)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tiada data forecast!")

    elif menu == "📜 Report":
        st.title("📜 Report Summary")
        if "df" in st.session_state and "forecast_df" in st.session_state:
            st.dataframe(st.session_state.forecast_df)
            st.success("All analysis complete ✅")
        else:
            st.warning("Tiada data untuk laporan.")

# --------------------- MAIN ---------------------
if not st.session_state.logged_in:
    login_section()
else:
    dashboard()
