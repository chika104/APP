import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt

# ======================
# DATABASE CONNECTION
# ======================
def connect_db():
    try:
        conn = mysql.connector.connect(
            host="switchback.proxy.rlwy.net",
            user="root",
            password="polrwgDJZnGLaungxPtGkOTaduCuolEj",
            database="railway",
            port=55398
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Gagal sambung DB: {e}")
        return None


# ======================
# LOGIN SYSTEM
# ======================
def login_form():
    st.markdown("<h2 style='text-align:center;'>🔒 Secure Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "12345":
            st.session_state.logged_in = True
            st.success("Login berjaya!")
            st.rerun()  # ✅ FIXED — ganti daripada experimental_rerun()
        else:
            st.error("Nama pengguna atau kata laluan salah!")


# ======================
# BACKGROUND SETTING
# ======================
def set_background(color="#0a0a0a"):
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {color};
        background-size: cover;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0,0,0,0.9);
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0.8);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


# ======================
# LOAD DATA
# ======================
def load_data():
    conn = connect_db()
    if conn:
        df = pd.read_sql("SELECT * FROM energy_data", conn)
        conn.close()
        return df
    return pd.DataFrame(columns=["year", "consumption", "baseline_cost", "forecast"])


# ======================
# PLOT GRAPHS
# ======================
def plot_graphs(df):
    st.subheader("📊 Energy Forecast Graphs")

    graph_options = [
        "Baseline Only",
        "Baseline vs Forecast",
        "Adjusted vs Forecast vs Baseline",
        "Baseline Cost",
        "Forecast Cost vs Baseline Cost",
        "CO₂ Baseline",
        "CO₂ Baseline vs CO₂ Forecast"
    ]

    choice = st.selectbox("Select Graph Type", graph_options)

    plt.figure(figsize=(8, 4))
    if choice == "Baseline Only":
        plt.plot(df["year"], df["consumption"], color="red", label="Baseline")

    elif choice == "Baseline vs Forecast":
        plt.plot(df["year"], df["consumption"], color="red", label="Baseline")
        plt.plot(df["year"], df["forecast"], color="blue", label="Forecast")

    elif choice == "Adjusted vs Forecast vs Baseline":
        df["adjusted"] = df["consumption"] * 0.9
        plt.plot(df["year"], df["consumption"], color="red", label="Baseline")
        plt.plot(df["year"], df["forecast"], color="blue", label="Forecast")
        plt.plot(df["year"], df["adjusted"], color="green", label="Adjusted")

    elif choice == "Baseline Cost":
        plt.plot(df["year"], df["baseline_cost"], color="red", label="Baseline Cost")

    elif choice == "Forecast Cost vs Baseline Cost":
        plt.plot(df["year"], df["baseline_cost"], color="red", label="Baseline Cost")
        plt.plot(df["year"], df["forecast"] * 0.2, color="blue", label="Forecast Cost")

    elif choice == "CO₂ Baseline":
        plt.plot(df["year"], df["consumption"] * 0.6, color="red", label="CO₂ Baseline")

    elif choice == "CO₂ Baseline vs CO₂ Forecast":
        plt.plot(df["year"], df["consumption"] * 0.6, color="red", label="CO₂ Baseline")
        plt.plot(df["year"], df["forecast"] * 0.6, color="blue", label="CO₂ Forecast")

    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


# ======================
# DASHBOARD MENU
# ======================
def main_dashboard():
    menu = ["Dashboard", "Data Table", "Upload Data", "Forecast Results", "Visualization", "Settings"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Dashboard":
        st.title("⚡ Energy Forecast Dashboard")
        st.write("Welcome to your smart energy forecasting system.")
        df = load_data()
        st.dataframe(df)

    elif choice == "Data Table":
        st.subheader("📋 View Stored Data")
        df = load_data()
        st.dataframe(df)

    elif choice == "Upload Data":
        st.subheader("📂 Upload CSV File")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df_new = pd.read_csv(uploaded_file)
            conn = connect_db()
            if conn:
                cursor = conn.cursor()
                for _, row in df_new.iterrows():
                    cursor.execute("""
                        INSERT INTO energy_data (year, consumption, baseline_cost, forecast)
                        VALUES (%s, %s, %s, %s)
                    """, (int(row['year']), float(row['consumption']),
                          float(row['baseline_cost']), float(row['forecast'])))
                conn.commit()
                conn.close()
                st.success("Data uploaded successfully!")

    elif choice == "Forecast Results":
        st.subheader("📈 Forecasted Energy Data")
        df = load_data()
        if not df.empty:
            next_year = df['year'].max() + 1
            forecast_value = df['forecast'].iloc[-1] * 1.05
            forecast_cost = forecast_value * 0.2

            new_row = {
                "year": next_year,
                "consumption": round(forecast_value, 2),
                "baseline_cost": round(forecast_cost, 2),
                "forecast": round(forecast_value, 2)
            }

            st.write(pd.DataFrame([new_row]))
        else:
            st.warning("No data available for forecasting.")

    elif choice == "Visualization":
        df = load_data()
        if not df.empty:
            plot_graphs(df)
        else:
            st.warning("No data to visualize.")

    elif choice == "Settings":
        st.subheader("⚙️ Settings")
        bg_color = st.color_picker("Choose background color", "#0a0a0a")
        if st.button("Apply Background"):
            set_background(bg_color)
            st.success("Background color updated!")


# ======================
# MAIN APP
# ======================
set_background("#0a0a0a")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Please login to access the dashboard")
    login_form()
    st.stop()

# After login → main dashboard
main_dashboard()
