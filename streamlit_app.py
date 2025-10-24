# =========================
# streamlit_app.py (Bahagian 1)
# Security + DB Connection + Auth
# =========================

import streamlit as st
import mysql.connector
from mysql.connector import Error
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Forecast System", layout="wide")

# =========================
# DATABASE CONNECTION
# =========================
def get_connection():
    try:
        return mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"],
            port=st.secrets["DB_PORT"]
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# =========================
# SECURITY & AUTH SYSTEM
# =========================
def make_hash(password):
    return hashlib.sha256(str(password).encode()).hexdigest()


def create_user_table():
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_accounts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL
                )
            """)
            conn.commit()
            c.close()
        except Error as e:
            st.error(f"Error creating user_accounts table: {e}")
        finally:
            conn.close()


def add_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO user_accounts (username, password_hash) VALUES (%s, %s)",
                      (username, make_hash(password)))
            conn.commit()
            st.success("✅ Registration successful! You can now log in.")
        except Error as e:
            st.error(f"DB error during registration: {e}")
        finally:
            conn.close()


def login_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor(dictionary=True)
            c.execute("SELECT * FROM user_accounts WHERE username=%s AND password_hash=%s",
                      (username, make_hash(password)))
            result = c.fetchone()
            c.close()
            if result:
                return True
        except Error as e:
            st.error(f"DB error during login: {e}")
        finally:
            conn.close()
    return False


# =========================
# LOGIN & REGISTER PAGE
# =========================
def login_page():
    st.title("🔐 Secure Login Portal")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login to Access Dashboard")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            create_user_table()
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password!")

    elif choice == "Register":
        st.subheader("Create New Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            if new_user and new_pass:
                create_user_table()
                add_user(new_user, new_pass)
            else:
                st.warning("⚠️ Please fill in both fields!")


# =========================
# MAIN ROUTING
# =========================
def main():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login_page()
    else:
        show_dashboard()


def show_dashboard():
    st.sidebar.title("⚡ Main Menu")
    menu = ["Dashboard", "Energy Forecast", "Device Management", "Reports", "Settings", "Help & About"]
    choice = st.sidebar.radio("Navigate", menu)

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

    if choice == "Dashboard":
        st.title("📊 Dashboard")
        st.write("Welcome to your energy forecasting dashboard!")

    elif choice == "Energy Forecast":
        show_energy_forecast()

    elif choice == "Device Management":
        st.title("💡 Device Management")
        st.write("Manage connected devices and sensors here.")

    elif choice == "Reports":
        st.title("📑 Reports")
        st.write("Generate and view performance reports here.")

    elif choice == "Settings":
        st.title("⚙️ Settings")
        st.write("Customize your application preferences.")

    elif choice == "Help & About":
        st.title("❓ Help & About")
        st.write("Energy Forecast System — Developed by Chika 💡")


def show_energy_forecast():
    st.title("⚡ Energy Forecast Module")
    st.write("Upload or input your data below to generate forecast graphs.")
    st.info("Graphs will appear here once data is provided.")

# Run app
if __name__ == "__main__":
    main()
