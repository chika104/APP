import streamlit as st
import pandas as pd
import mysql.connector
import bcrypt
import matplotlib.pyplot as plt

# ---------------------------
# DATABASE CONNECTION
# ---------------------------
def create_connection():
    try:
        return mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            port=55398,
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway"
        )
    except mysql.connector.Error as err:
        st.error(f"Gagal sambung DB: {err}")
        return None


# ---------------------------
# USER AUTH FUNCTIONS
# ---------------------------
def create_user(username, password):
    conn = create_connection()
    if not conn:
        return False
    cursor = conn.cursor(dictionary=True)
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
    conn.commit()
    conn.close()
    return True


def login_user(username, password):
    conn = create_connection()
    if not conn:
        return None
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    conn.close()

    if user:
        stored_pw = user['password_hash']
        try:
            if bcrypt.checkpw(password.encode(), stored_pw.encode()):
                return user
        except ValueError:
            if password == stored_pw:
                return user
    return None


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#121212"  # default background
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = pd.DataFrame()


# ---------------------------
# CSS STYLE
# ---------------------------
st.markdown(
    f"""
    <style>
    body {{
        background-color: {st.session_state.bg_color};
        color: white;
    }}
    .block-container {{
        background-color: {st.session_state.bg_color};
    }}
    div[data-testid="stSidebar"] {{
        background-color: black !important;
    }}
    div[data-testid="stSidebarNav"] > div {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# LOGIN & REGISTER PAGE
# ---------------------------
def login_register_page():
    st.title("🔐 Energy Forecast Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login berjaya!")
                st.rerun()
            else:
                st.error("Nama pengguna atau kata laluan salah!")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if new_user and new_pass:
                if create_user(new_user, new_pass):
                    st.success("Pendaftaran berjaya! Sila login.")
                else:
                    st.error("Ralat semasa pendaftaran.")
            else:
                st.warning("Sila isi semua maklumat.")


# ---------------------------
# DASHBOARD PAGE
# ---------------------------
def dashboard_page():
    st.title("⚡ Energy Forecast Dashboard")
    st.sidebar.header(f"Welcome, {st.session_state.user}")
    st.sidebar.write("📅 Manage your energy data and forecasts")

    menu = st.sidebar.radio("Navigation Menu", [
        "📊 Dashboard Overview",
        "📥 Upload or Input Data",
        "📈 Forecast Charts",
        "📘 Baseline Table",
        "📗 Comparison Table",
        "⚙️ Settings"
    ])

    # ---------------- DATA UPLOAD / INPUT ----------------
    if menu == "📥 Upload or Input Data":
        st.subheader("Upload CSV or Enter Data Manually")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.forecast_data = df
            st.success("CSV file uploaded successfully!")

        with st.expander("Manual Data Entry"):
            year = st.number_input("Year", step=1)
            consumption = st.number_input("Consumption (kWh)", step=1.0)
            baseline_cost = st.number_input("Baseline Cost (RM)", step=0.01)
            forecast = st.number_input("Forecast (kWh)", step=1.0)
            if st.button("Add Data"):
                new_row = {"year": year, "consumption": consumption, "baseline_cost": baseline_cost, "forecast": forecast}
                st.session_state.forecast_data = pd.concat(
                    [st.session_state.forecast_data, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Data added successfully!")

        st.dataframe(st.session_state.forecast_data)

    # ---------------- BASELINE TABLE ----------------
    elif menu == "📘 Baseline Table":
        if not st.session_state.forecast_data.empty:
            st.subheader("Baseline Energy Data")
            st.dataframe(st.session_state.forecast_data[["year", "consumption", "baseline_cost"]])
        else:
            st.info("No data available.")

    # ---------------- COMPARISON TABLE ----------------
    elif menu == "📗 Comparison Table":
        if not st.session_state.forecast_data.empty:
            df = st.session_state.forecast_data.copy()
            df["Forecast Cost"] = df["forecast"] * (df["baseline_cost"] / df["consumption"])
            df["CO2 Baseline"] = df["consumption"] * 0.233
            df["CO2 Forecast"] = df["forecast"] * 0.233
            st.subheader("Comparison Table")
            st.dataframe(df)
        else:
            st.info("No data to compare yet.")

    # ---------------- FORECAST CHARTS ----------------
    elif menu == "📈 Forecast Charts":
        if st.session_state.forecast_data.empty:
            st.warning("No data yet — please upload or enter data first.")
        else:
            df = st.session_state.forecast_data.copy()
            colors = {
                "baseline": "red",
                "forecast": "blue",
                "adjusted": "green",
                "cost": "orange",
                "co2": "purple"
            }

            st.subheader("1️⃣ Baseline Only")
            plt.figure()
            plt.plot(df["year"], df["consumption"], color=colors["baseline"], label="Baseline")
            plt.legend(); st.pyplot(plt)

            st.subheader("2️⃣ Baseline vs Forecast")
            plt.figure()
            plt.plot(df["year"], df["consumption"], color=colors["baseline"])
            plt.plot(df["year"], df["forecast"], color=colors["forecast"])
            plt.legend(["Baseline", "Forecast"]); st.pyplot(plt)

            st.subheader("3️⃣ Adjusted vs Forecast vs Baseline")
            plt.figure()
            adjusted = df["forecast"] * 0.9
            plt.plot(df["year"], df["consumption"], color=colors["baseline"])
            plt.plot(df["year"], df["forecast"], color=colors["forecast"])
            plt.plot(df["year"], adjusted, color=colors["adjusted"])
            plt.legend(["Baseline", "Forecast", "Adjusted"]); st.pyplot(plt)

            st.subheader("4️⃣ Baseline Cost")
            plt.figure()
            plt.plot(df["year"], df["baseline_cost"], color=colors["cost"])
            plt.legend(["Baseline Cost"]); st.pyplot(plt)

            st.subheader("5️⃣ Forecast Cost vs Baseline Cost")
            forecast_cost = df["forecast"] * (df["baseline_cost"] / df["consumption"])
            plt.figure()
            plt.plot(df["year"], df["baseline_cost"], color=colors["cost"])
            plt.plot(df["year"], forecast_cost, color="grey")
            plt.legend(["Baseline Cost", "Forecast Cost"]); st.pyplot(plt)

            st.subheader("6️⃣ CO2 Baseline")
            co2_base = df["consumption"] * 0.233
            plt.figure()
            plt.plot(df["year"], co2_base, color=colors["co2"])
            plt.legend(["CO2 Baseline"]); st.pyplot(plt)

            st.subheader("7️⃣ CO2 Baseline vs Forecast")
            co2_fore = df["forecast"] * 0.233
            plt.figure()
            plt.plot(df["year"], co2_base, color=colors["co2"])
            plt.plot(df["year"], co2_fore, color="cyan")
            plt.legend(["CO2 Baseline", "CO2 Forecast"]); st.pyplot(plt)

    # ---------------- SETTINGS ----------------
    elif menu == "⚙️ Settings":
        st.subheader("Theme & Background Settings")
        color = st.color_picker("Choose Background Color", st.session_state.bg_color)
        if st.button("Save Background"):
            st.session_state.bg_color = color
            st.success("Background updated successfully!")


# ---------------------------
# MAIN EXECUTION
# ---------------------------
if not st.session_state.logged_in:
    login_register_page()
else:
    dashboard_page()
