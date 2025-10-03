import streamlit as st
st.title("Hello Chika 👋")
st.write("App sudah jalan dengan betul 🎉")

import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import io

# =========================
# 🔹 Load Lottie Animation
# =========================
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_energy = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")

# =========================
# 🔹 Function to create Excel bytes
# =========================
def excel_bytes_from_dfs(dfs: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()

# =========================
# 🔹 App Layout
# =========================
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# 🎬 Show animation ONCE
if "anim_shown" not in st.session_state:
    st.session_state.anim_shown = False

if not st.session_state.anim_shown:
    st_lottie(lottie_energy, speed=1, height=400, key="energy")
    st.session_state.anim_shown = True
    st.stop()

st.title("⚡ Energy Forecast & Savings Dashboard")

# =========================
# 🔹 Step 1: Data Input
# =========================
st.header("📥 Step 1: Input Data")

option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if option == "Manual Input":
    start_year = st.number_input("Start Year", value=2020)
    end_year = st.number_input("End Year", value=2024)
    years = list(range(start_year, end_year + 1))

    data = []
    for y in years:
        consumption = st.number_input(f"Energy consumption for {y} (kWh)", min_value=0.0, value=1000.0, step=100.0)
        cost = st.number_input(f"Total cost for {y} (RM)", min_value=0.0, value=500.0, step=50.0)
        co2 = st.number_input(f"CO₂ emissions for {y} (kg)", min_value=0.0, value=200.0, step=10.0)
        data.append({"year": y, "consumption": consumption, "cost": cost, "co2": co2})
    df = pd.DataFrame(data)

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

st.dataframe(df)

# =========================
# 🔹 Step 2: Apply Factors
# =========================
st.header("⚙️ Step 2: Apply Factors (Additions / Reductions)")

factors = {
    "LED Lamp (kWh/year)": st.number_input("LED Lamp", value=0.0, step=10.0),
    "CFL Lamp (kWh/year)": st.number_input("CFL Lamp", value=0.0, step=10.0),
    "Fluorescent Lamp (kWh/year)": st.number_input("Fluorescent Lamp", value=0.0, step=10.0),
    "Computers (kWh/year)": st.number_input("Computers", value=0.0, step=10.0),
    "Lab Equipment (kWh/year)": st.number_input("Lab Equipment", value=0.0, step=10.0),
    "Operating Hours (hours/year)": st.number_input("Operating Hours (hours/year)", value=0.0, step=10.0),
}

# Adjusted consumption
df["adjusted_consumption"] = df["consumption"] + sum(factors.values())

# Adjusted cost (RM)
cost_per_kwh = (df["cost"] / df["consumption"]).mean()
df["adjusted_cost"] = df["adjusted_consumption"] * cost_per_kwh

# Adjusted CO2
co2_per_kwh = (df["co2"] / df["consumption"]).mean()
df["adjusted_co2"] = df["adjusted_consumption"] * co2_per_kwh

# =========================
# 🔹 Step 3: Forecast Graphs
# =========================
st.header("📊 Step 3: Forecast & Comparison Graphs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Graph 1: Baseline vs Adjusted Consumption")
    fig1 = px.line(df, x="year", y=["consumption", "adjusted_consumption"], markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Graph 2: Baseline vs Adjusted Cost")
    fig2 = px.line(df, x="year", y=["cost", "adjusted_cost"], markers=True)
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Graph 3: Baseline vs Adjusted Energy Saving (kWh)")
    df["saving_kwh"] = df["consumption"] - df["adjusted_consumption"]
    fig3 = px.bar(df, x="year", y="saving_kwh", color="saving_kwh")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Graph 4: Baseline vs Adjusted CO₂ Reduction (kg)")
    df["saving_co2"] = df["co2"] - df["adjusted_co2"]
    fig4 = px.bar(df, x="year", y="saving_co2", color="saving_co2")
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# 🔹 Step 4: Download Results
# =========================
st.header("📥 Step 4: Download Results")

all_dfs = {"Forecast Results": df}
excel_bytes = excel_bytes_from_dfs(all_dfs)
st.download_button("Download Excel", data=excel_bytes, file_name="forecast_results.xlsx")

st.success("✅ Analysis Complete! You can now download results.")
