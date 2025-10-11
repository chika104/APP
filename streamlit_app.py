# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# ---------------------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------------------
st.set_page_config(page_title="Smart Energy Forecasting System", layout="wide")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1509395176047-4a66953fd231");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.8);
}
h1, h2, h3 {
    color: #023047;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("⚡ Smart Energy Forecasting System")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Energy Forecast", "💡 Device Management",
     "📈 Reports", "⚙️ Settings", "❓ Help & About"]
)

# ---------------------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.markdown("""
    Selamat datang ke **Smart Energy Forecasting System** 💡  
    Sistem ini membantu anda:
    - Menganalisis dan meramal penggunaan tenaga.
    - Menilai potensi penjimatan kos & pengurangan karbon.
    - Mengurus peranti dan menjana laporan secara interaktif.
    """)
    st.success("Pilih menu di sebelah kiri untuk mula menggunakan aplikasi ini 🚀")

# ---------------------------------------------------------------------
# ENERGY FORECAST PAGE
# ---------------------------------------------------------------------
elif menu == "📊 Energy Forecast":
    st.title("📊 Energy Forecast Module")
    st.markdown("Ramalkan penggunaan tenaga & kos berdasarkan data sejarah 🔍")

    # ---------------------------
    # Utility functions
    # ---------------------------
    def normalize_cols(df):
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def safe_to_numeric_series(s):
        return pd.to_numeric(s, errors="coerce")

    # ---------------------------
    # Step 1: Input data
    # ---------------------------
    st.header("Step 1 — Input Baseline Data")
    input_mode = st.radio("Input Method", ("Upload CSV", "Manual Entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (columns: year, consumption, [optional baseline cost])", type=["csv", "xlsx"])
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)

            year_col = [c for c in df_raw.columns if "year" in c][0]
            cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum", "kwh", "energy"])][0]

            df = df_raw[[year_col, cons_col]].copy()
            df.columns = ["year", "consumption"]

            if any("cost" in c for c in df_raw.columns):
                cost_col = [c for c in df_raw.columns if "cost" in c][0]
                df["baseline_cost"] = pd.to_numeric(df_raw[cost_col], errors="coerce")
            else:
                df["baseline_cost"] = np.nan

    elif input_mode == "Manual Entry":
        rows = st.number_input("How many historical rows?", min_value=1, max_value=20, value=5)
        years, consumptions, baseline_costs = [], [], []
        for i in range(int(rows)):
            c1, c2 = st.columns(2)
            with c1:
                y = st.number_input(f"Year {i+1}", min_value=2000, max_value=2100, value=2020+i, key=f"y{i}")
                years.append(y)
            with c2:
                c = st.number_input(f"Consumption kWh ({y})", min_value=0.0, value=10000.0, key=f"c{i}")
                consumptions.append(c)
            b = st.number_input(f"Baseline cost RM ({y}) (optional)", min_value=0.0, value=0.0, key=f"b{i}")
            baseline_costs.append(b if b > 0 else np.nan)
        df = pd.DataFrame({"year": years, "consumption": consumptions, "baseline_cost": baseline_costs})

    if df is None or df.empty:
        st.warning("⚠️ Please upload or enter data to continue.")
        st.stop()

    df["year"] = df["year"].astype(int)
    df["consumption"] = safe_to_numeric_series(df["consumption"])
    df["baseline_cost"] = safe_to_numeric_series(df["baseline_cost"])
    df = df.sort_values("year").reset_index(drop=True)

    st.dataframe(df)

    # ---------------------------
    # Step 2: Baseline
    # ---------------------------
    st.header("Step 2 — Baseline Calculations")
    tariff = st.number_input("Enter tariff (RM/kWh)", min_value=0.0, value=0.5, step=0.01)

    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_energy_saving"] = 0.0
    df["baseline_cost_saving"] = 0.0
    df["baseline_co2_reduction"] = 0.0
    st.success("Baseline calculated ✅")

    # ---------------------------
    # Step 3: Adjusted Scenario
    # ---------------------------
    st.header("Step 3 — Adjusted Scenario")
    reduction_pct = st.slider("Reduction percentage (%)", 0, 100, 10)

    df["adjusted_consumption"] = df["consumption"] * (1 - reduction_pct/100)
    df["adjusted_cost"] = df["adjusted_consumption"] * tariff
    df["adjusted_energy_saving"] = df["consumption"] - df["adjusted_consumption"]
    df["adjusted_cost_saving"] = df["baseline_cost"] - df["adjusted_cost"]
    df["adjusted_co2_reduction"] = df["adjusted_energy_saving"] * 0.00069

    st.balloons()

    # ---------------------------
    # Step 4: Visualization
    # ---------------------------
    st.header("Step 4 — Visualization")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Consumption", "Cost", "Energy Saving", "Cost Saving", "CO₂ Reduction"
    ])

    with tab1:
        st.subheader("Baseline vs Adjusted Consumption")
        fig = px.line(df, x="year", y=["consumption", "adjusted_consumption"], markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Baseline vs Adjusted Cost")
        fig = px.line(df, x="year", y=["baseline_cost", "adjusted_cost"], markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Energy Saving (kWh)")
        fig = px.bar(df, x="year", y="adjusted_energy_saving")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Cost Saving (RM)")
        fig = px.bar(df, x="year", y="adjusted_cost_saving")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("CO₂ Reduction")
        fig = px.bar(df, x="year", y="adjusted_co2_reduction")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# DEVICE MANAGEMENT PAGE
# ---------------------------------------------------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.info("Feature coming soon: Manage your connected IoT devices, status, and energy usage.")

# ---------------------------------------------------------------------
# REPORTS PAGE
# ---------------------------------------------------------------------
elif menu == "📈 Reports":
    st.title("📈 Reports")
    st.info("Generate and download customized energy reports in PDF or Excel format (coming soon).")

# ---------------------------------------------------------------------
# SETTINGS PAGE
# ---------------------------------------------------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.info("Adjust system configurations and preferences.")

# ---------------------------------------------------------------------
# HELP & ABOUT PAGE
# ---------------------------------------------------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Version 1.0 — Developed by Chika 💻  
    This system utilizes data-driven analysis to predict and visualize energy consumption trends.
    """)
