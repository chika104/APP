import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import base64
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Smart Energy Forecasting Dashboard", layout="wide")

# Default background (dark)
default_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background: rgba(30, 30, 30, 0.8);
}
</style>
"""
st.markdown(default_bg, unsafe_allow_html=True)

# ------------------------------------------------------------
# APP NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("🔹 Navigation Menu")
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
     "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)

# ------------------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting System")
    st.markdown("""
    **Welcome to the Smart Energy Forecasting Dashboard**  
    This platform allows you to:
    - Forecast future energy usage and cost  
    - Analyze CO₂ emission reduction  
    - Compare baseline and adjusted scenarios  
    - Export data into formal reports (PDF & Excel)
    """)

# ------------------------------------------------------------
# ENERGY FORECAST PAGE
# ------------------------------------------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecasting Module")

    # Step 1 — Input Data
    st.header("Step 1 — Input Baseline Data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (columns: year, consumption, [optional cost])", type=["csv", "xlsx"])
        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "year" not in df.columns or "consumption" not in df.columns:
                st.error("CSV must contain 'year' and 'consumption' columns.")
                st.stop()
    else:
        rows = st.number_input("How many historical records?", 1, 20, 5)
        data = []
        for i in range(int(rows)):
            c1, c2, c3 = st.columns(3)
            with c1:
                y = st.number_input(f"Year {i+1}", min_value=2000, max_value=2100, value=2020+i, key=f"year{i}")
            with c2:
                c = st.number_input(f"Consumption (kWh) {i+1}", min_value=0.0, value=10000.0, key=f"cons{i}")
            with c3:
                cost = st.number_input(f"Baseline Cost (RM) {i+1}", min_value=0.0, value=0.0, key=f"cost{i}")
            data.append({"year": y, "consumption": c, "baseline_cost": cost if cost > 0 else np.nan})
        df = pd.DataFrame(data)

    if df is not None and not df.empty:
        st.dataframe(df)

        # Step 2 — Baseline Calculation
        st.header("Step 2 — Baseline Calculation")
        tariff = st.number_input("Tariff rate (RM/kWh)", min_value=0.0, value=0.50, step=0.01)
        df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)

        # Step 3 — Adjusted Scenario
        st.header("Step 3 — Adjustment Factors")
        st.write("Adjust the factors to simulate energy reduction:")
        lamp = st.slider("Lighting reduction (%) 💡", 0, 100, 10)
        pc = st.slider("Computer usage reduction (%) 💻", 0, 100, 5)
        lab = st.slider("Lab equipment reduction (%) ⚗️", 0, 100, 8)
        hours = st.slider("Operating hours reduction (%) ⏱️", 0, 100, 10)

        total_reduction = (lamp + pc + lab + hours) / 400
        df["adjusted_consumption"] = df["consumption"] * (1 - total_reduction)
        df["adjusted_cost"] = df["adjusted_consumption"] * tariff
        df["energy_saving"] = df["consumption"] - df["adjusted_consumption"]
        df["cost_saving"] = df["baseline_cost"] - df["adjusted_cost"]
        df["co2_reduction"] = df["energy_saving"] * 0.00069

        # Step 4 — Visualization
        st.header("Step 4 — Visualization")
        tab1, tab2, tab3, tab4 = st.tabs(["Baseline Forecast", "Adjusted Forecast", "Cost Trend", "CO₂ Trend"])

        with tab1:
            fig = px.line(df, x="year", y="consumption", title="Baseline Energy Forecast (kWh)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.line(df, x="year", y="adjusted_consumption", title="Adjusted Forecast (kWh)", markers=True, color_discrete_sequence=["#00CC96"])
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            fig = px.bar(df, x="year", y=["baseline_cost", "adjusted_cost"], barmode="group", title="Cost Trend (RM)")
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            fig = px.bar(df, x="year", y="co2_reduction", title="CO₂ Reduction (tons)", color_discrete_sequence=["#FFA15A"])
            st.plotly_chart(fig, use_container_width=True)

        # Step 5 — Export Data
        st.header("Step 5 — Export Report")

        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Energy Forecast")
        b64_excel = base64.b64encode(excel_buffer.getvalue()).decode()
        st.download_button("⬇️ Download Excel", data=excel_buffer, file_name="Energy_Forecast.xlsx", mime="application/vnd.ms-excel")

        # PDF export
        def create_pdf(df):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            title = Paragraph("<b>SMART ENERGY FORECASTING REPORT</b>", styles["Title"])
            subtitle = Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')}", styles["Normal"])
            elements += [title, subtitle, Spacer(1, 20)]

            summary = Paragraph(
                "This report presents baseline and adjusted energy forecasts, including cost and CO₂ reduction analysis.",
                styles["Normal"]
            )
            elements.append(summary)
            elements.append(Spacer(1, 20))

            table_data = [list(df.columns)] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
            ]))
            elements.append(table)

            doc.build(elements)
            pdf = buffer.getvalue()
            buffer.close()
            return pdf

        pdf_data = create_pdf(df)
        st.download_button("📄 Download PDF Report", data=pdf_data, file_name="Energy_Report.pdf", mime="application/pdf")

# ------------------------------------------------------------
# DEVICE MANAGEMENT PAGE
# ------------------------------------------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.info("Feature for managing connected devices and schedules (coming soon).")

# ------------------------------------------------------------
# REPORTS PAGE
# ------------------------------------------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.info("Historical and comparative reports can be viewed or downloaded here.")

# ------------------------------------------------------------
# SETTINGS PAGE
# ------------------------------------------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    bg_choice = st.radio("Select Background Theme", ["Dark Mode (Default)", "Light Mode", "Custom Image"])
    if bg_choice == "Light Mode":
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f5f5f5;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    elif bg_choice == "Custom Image":
        img_url = st.text_input("Enter background image URL:")
        if img_url:
            st.markdown(f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img_url}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------
# HELP & ABOUT PAGE
# ------------------------------------------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed by: *Chika (Politeknik Kota Kinabalu)*  
    - Interactive data visualization  
    - Forecasting based on machine learning-ready dataset  
    - Exportable reports for academic & practical use
    """)
