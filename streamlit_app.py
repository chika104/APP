import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import bcrypt
import io

# -----------------------------------------------------
# DATABASE CONNECTION (RAILWAY)
# -----------------------------------------------------
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway",
            port=55398
        )
        return connection
    except Error as e:
        st.error(f"❌ Gagal sambung ke DB: {e}")
        return None

# -----------------------------------------------------
# AUTHENTICATION SYSTEM
# -----------------------------------------------------
def create_users_table():
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password_hash VARCHAR(255)
            )
        """)
        connection.commit()
        cursor.close()
        connection.close()

def register_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password_hash))
            connection.commit()
            st.success("✅ Akaun berjaya didaftarkan!")
        except mysql.connector.IntegrityError:
            st.error("❌ Nama pengguna telah wujud.")
        finally:
            cursor.close()
            connection.close()

def login_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user and bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            return user
        else:
            return None

# -----------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------
def calculate_forecast(data):
    df = data.copy()
    df['Forecast'] = df['Baseline (kWh)'] * np.random.uniform(0.9, 1.1, len(df))
    df['Adjusted'] = df['Forecast'] * np.random.uniform(0.95, 1.05, len(df))
    df['Baseline Cost (RM)'] = df['Baseline (kWh)'] * 0.5
    df['Forecast Cost (RM)'] = df['Forecast'] * 0.5
    df['Adjusted Cost (RM)'] = df['Adjusted'] * 0.5
    df['CO2 Baseline'] = df['Baseline (kWh)'] * 0.0007
    df['CO2 Forecast'] = df['Forecast'] * 0.0007
    return df

# -----------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------
def plot_graph(df, graph_type):
    fig = go.Figure()
    if graph_type == "Baseline Only":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Baseline (kWh)'], name="Baseline", line=dict(color="#FF4C4C")))
    elif graph_type == "Baseline vs Forecast":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Baseline (kWh)'], name="Baseline", line=dict(color="#FF4C4C")))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast'], name="Forecast", line=dict(color="#0050A0")))
    elif graph_type == "Adjusted vs Forecast vs Baseline":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Baseline (kWh)'], name="Baseline", line=dict(color="#FF4C4C")))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast'], name="Forecast", line=dict(color="#0050A0")))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Adjusted'], name="Adjusted", line=dict(color="#00B050")))
    elif graph_type == "Baseline Cost":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Baseline Cost (RM)'], name="Baseline Cost", line=dict(color="#FFA500")))
    elif graph_type == "Forecast Cost vs Baseline Cost":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast Cost (RM)'], name="Forecast Cost", line=dict(color="#0050A0")))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Baseline Cost (RM)'], name="Baseline Cost", line=dict(color="#FF4C4C")))
    elif graph_type == "CO2 Baseline":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CO2 Baseline'], name="CO₂ Baseline", line=dict(color="#8000FF")))
    elif graph_type == "CO2 Baseline vs CO2 Forecast":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CO2 Baseline'], name="CO₂ Baseline", line=dict(color="#8000FF")))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CO2 Forecast'], name="CO₂ Forecast", line=dict(color="#FFD700")))

    fig.update_layout(
        title=graph_type,
        xaxis_title="Date",
        yaxis_title="Values",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------
def main_dashboard():
    st.title("⚡ Energy Forecasting Dashboard")

    menu = ["Upload Data", "Manual Input", "Graphs", "Table Comparison", "Settings", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload Data":
        st.subheader("📂 Upload CSV Data")
        file = st.file_uploader("Upload your CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            st.dataframe(df)
            st.session_state['data'] = calculate_forecast(df)

    elif choice == "Manual Input":
        st.subheader("✏️ Manual Data Entry")
        date = st.date_input("Date")
        baseline = st.number_input("Baseline (kWh)", min_value=0.0)
        if st.button("Add Data"):
            new_df = pd.DataFrame({"Date": [date], "Baseline (kWh)": [baseline]})
            if 'data' not in st.session_state:
                st.session_state['data'] = new_df
            else:
                st.session_state['data'] = pd.concat([st.session_state['data'], new_df], ignore_index=True)
            st.success("✅ Data added!")

    elif choice == "Graphs":
        st.subheader("📊 Energy Graphs")
        if 'data' in st.session_state:
            df = st.session_state['data']
            for g in [
                "Baseline Only", "Baseline vs Forecast", "Adjusted vs Forecast vs Baseline",
                "Baseline Cost", "Forecast Cost vs Baseline Cost", "CO2 Baseline", "CO2 Baseline vs CO2 Forecast"
            ]:
                plot_graph(df, g)
        else:
            st.warning("⚠️ Tiada data dimuat naik lagi!")

    elif choice == "Table Comparison":
        if 'data' in st.session_state:
            st.subheader("📋 Comparison Table")
            df = st.session_state['data']
            comparison = df[['Date', 'Baseline (kWh)', 'Forecast', 'Adjusted',
                             'Baseline Cost (RM)', 'Forecast Cost (RM)', 'Adjusted Cost (RM)',
                             'CO2 Baseline', 'CO2 Forecast']]
            st.dataframe(comparison)
        else:
            st.warning("⚠️ Tiada data untuk dipaparkan.")

    elif choice == "Settings":
        st.subheader("⚙️ Settings")
        theme = st.selectbox("Choose Background Theme", ["Dark", "Light", "Blue"])
        st.session_state['theme'] = theme
        st.success(f"Tema ditukar kepada {theme}")

    elif choice == "Logout":
        st.session_state.clear()
        st.success("✅ Logout berjaya. Sila refresh halaman.")

# -----------------------------------------------------
# LOGIN & REGISTER PAGE
# -----------------------------------------------------
def login_register_page():
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state['user'] = user['username']
                st.experimental_rerun()
            else:
                st.error("❌ Nama pengguna atau kata laluan salah!")

    with tab2:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            register_user(new_username, new_password)

# -----------------------------------------------------
# APP ENTRY POINT
# -----------------------------------------------------
create_users_table()

if 'user' not in st.session_state:
    login_register_page()
else:
    st.sidebar.markdown(f"👋 Welcome, **{st.session_state['user']}**")
    main_dashboard()
