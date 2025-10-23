import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# DATABASE CONNECTION
# ---------------------------
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway",
            port=55398
        )
        return conn
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return None

# ---------------------------
# USER AUTH FUNCTIONS
# ---------------------------
def login_user(username, password):
    conn = get_db_connection()
    if conn:
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM user_accounts WHERE username=%s AND password=%s", (username, password))
        user = c.fetchone()
        conn.close()
        return user
    return None

def register_user(username, password):
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO user_accounts (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"DB error during registration: {e}")
            return False

# ---------------------------
# LOGIN PAGE
# ---------------------------
def login_page():
    st.markdown("<h2 style='text-align:center;color:white;'>🔐 Login to Energy Forecast Dashboard</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password!")

    st.markdown("---")
    st.markdown("<h4 style='text-align:center;color:white;'>Don't have an account?</h4>", unsafe_allow_html=True)
    with st.form("register_form"):
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        reg_submit = st.form_submit_button("Register")
        if reg_submit:
            if register_user(new_user, new_pass):
                st.success("✅ Registration successful! You can now login.")
            else:
                st.error("❌ Registration failed.")

# ---------------------------
# DASHBOARD
# ---------------------------
def dashboard():
    st.sidebar.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: black;
            }
            [data-testid="stSidebar"] * {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("📊 Dashboard Menu")
    menu = st.sidebar.radio("Navigate", ["Home", "Upload Data", "Baseline Table", "Comparison Table", 
                                         "Graphs", "Settings", "Logout"])

    if "bg_color" not in st.session_state:
        st.session_state.bg_color = "#001f3f"

    st.markdown(
        f"""
        <style>
            body {{
                background-color: {st.session_state.bg_color};
            }}
            .stApp {{
                background-color: {st.session_state.bg_color};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if menu == "Home":
        st.title("🏠 Welcome to Energy Forecast Dashboard")
        st.write("This dashboard helps you analyze and forecast your energy consumption data interactively.")
    elif menu == "Upload Data":
        st.subheader("📁 Upload or Enter Data")
        method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

        if method == "Manual Input":
            year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
            consumption = st.number_input("Energy Consumption (kWh)", min_value=0.0)
            baseline_cost = st.number_input("Baseline Cost (RM)", min_value=0.0)
            forecast = st.number_input("Forecast Value", min_value=0.0)

            if st.button("Save Data"):
                new_row = pd.DataFrame({
                    "year": [year],
                    "consumption": [consumption],
                    "baseline_cost": [baseline_cost],
                    "forecast": [forecast]
                })
                if "data" not in st.session_state:
                    st.session_state.data = new_row
                else:
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.success("✅ Data saved successfully!")
        else:
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success("✅ CSV uploaded successfully!")

    elif menu == "Baseline Table":
        st.subheader("📄 Baseline Data Table")
        if "data" in st.session_state:
            st.dataframe(st.session_state.data)
        else:
            st.warning("No data available. Please upload or input data first.")

    elif menu == "Comparison Table":
        st.subheader("📊 Comparison Table (Baseline vs Forecast)")
        if "data" in st.session_state:
            df = st.session_state.data.copy()
            df["difference"] = df["forecast"] - df["baseline_cost"]
            st.dataframe(df)
        else:
            st.warning("No data available.")

    elif menu == "Graphs":
        st.subheader("📈 Data Visualization")

        if "data" in st.session_state:
            df = st.session_state.data

            def line_chart(x, y1, y2=None, label1="Series 1", label2="Series 2", color1="red", color2="blue"):
                plt.figure()
                plt.plot(x, y1, marker='o', color=color1, label=label1)
                if y2 is not None:
                    plt.plot(x, y2, marker='x', color=color2, label=label2)
                plt.legend()
                st.pyplot(plt)

            line_chart(df["year"], df["baseline_cost"], label1="Baseline", color1="red")
            line_chart(df["year"], df["baseline_cost"], df["forecast"], label1="Baseline", label2="Forecast", color1="red", color2="darkblue")
            line_chart(df["year"], df["consumption"], df["forecast"], label1="Adjusted", label2="Forecast", color1="green", color2="orange")
            line_chart(df["year"], df["baseline_cost"], label1="Baseline Cost", color1="purple")
            line_chart(df["year"], df["forecast"], df["baseline_cost"], label1="Forecast Cost", label2="Baseline Cost", color1="orange", color2="gray")
            line_chart(df["year"], df["consumption"] * 0.0007, label1="CO2 Baseline", color1="brown")
            line_chart(df["year"], df["consumption"] * 0.0007, df["forecast"] * 0.0007, label1="CO2 Baseline", label2="CO2 Forecast", color1="brown", color2="black")
        else:
            st.warning("Please upload or input data first.")

    elif menu == "Settings":
        st.subheader("⚙️ Settings")
        new_color = st.color_picker("Choose background color", value=st.session_state.bg_color)
        if st.button("Apply"):
            st.session_state.bg_color = new_color
            st.success("✅ Background color updated!")

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully!")
        st.rerun()

# ---------------------------
# MAIN
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login_page()
