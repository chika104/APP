import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mysql.connector
import io
from datetime import datetime
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Smart Energy Forecasting Dashboard", layout="wide")

# ------------------------------------------------------
# DATABASE CONNECTION (XAMPP MySQL)
# ------------------------------------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="energy_forecast_db"
    )

# ------------------------------------------------------
# STYLING
# ------------------------------------------------------
theme = st.sidebar.radio("🎨 Choose Theme", ["Dark", "Light"])
if theme == "Dark":
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #1e1e1e; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f4f4f4; color: black; }
    [data-testid="stSidebar"] { background-color: #ffffff; color: black; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# NAVIGATION MENU
# ------------------------------------------------------
menu = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Dashboard", "🔋 Energy Forecast", "🧠 Device Management", "📈 Reports", "⚙️ Settings", "❓ Help & About"]
)

# ------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------
if menu == "🏠 Dashboard":
    st.title("⚡ Smart Energy Forecasting Dashboard")
    st.write("Selamat datang ke sistem ramalan tenaga pintar!")
    st.info("Gunakan menu di sebelah kiri untuk mengakses fungsi seperti ramalan tenaga, laporan, dan pengurusan peranti.")

# ------------------------------------------------------
# ENERGY FORECAST MENU
# ------------------------------------------------------
elif menu == "🔋 Energy Forecast":
    st.title("🔋 Energy Forecast Module")

    # Step 1 — Data Input
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Choose input method", ("Upload CSV", "Manual entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (columns: year, consumption, [optional cost])", type=["csv", "xlsx"])
        if uploaded:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    else:
        rows = st.number_input("Number of data rows", 1, 20, 5)
        data = {"year": [], "consumption": [], "baseline_cost": []}
        for i in range(int(rows)):
            c1, c2, c3 = st.columns(3)
            with c1:
                y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i)
            with c2:
                e = st.number_input(f"Consumption (kWh) {i+1}", 0.0, 1e9, 10000.0)
            with c3:
                b = st.number_input(f"Cost (RM) {i+1}", 0.0, 1e9, 0.0)
            data["year"].append(y)
            data["consumption"].append(e)
            data["baseline_cost"].append(b)
        df = pd.DataFrame(data)

    if df is not None and not df.empty:
        st.dataframe(df)

        # Step 2 — Tariff & Factors
        st.header("Step 2 — Adjust Factors")
        tariff = st.number_input("Enter tariff (RM/kWh)", 0.0, 10.0, 0.5)
        lamp_factor = st.slider("Lamp usage reduction (%) 💡", 0, 50, 10)
        pc_factor = st.slider("Computer efficiency increase (%) 💻", 0, 50, 5)
        lab_factor = st.slider("Lab equipment optimization (%) ⚗️", 0, 50, 8)
        time_factor = st.slider("Operating hours adjustment (%) ⏱️", 0, 50, 5)

        total_factor = (lamp_factor + pc_factor + lab_factor + time_factor) / 400  # Combined effect
        df["baseline_cost"] = df["baseline_cost"].replace(0, np.nan).fillna(df["consumption"] * tariff)
        df["forecast_consumption"] = df["consumption"] * (1 - total_factor)
        df["forecast_cost"] = df["forecast_consumption"] * tariff

        # Step 3 — Forecast table
        st.subheader("📊 Forecast Results")
        df["energy_saving"] = df["consumption"] - df["forecast_consumption"]
        df["cost_saving"] = df["baseline_cost"] - df["forecast_cost"]
        df["co2_reduction"] = df["energy_saving"] * 0.00069
        st.dataframe(df)

        # Step 4 — Graphs
        st.subheader("📈 Visualization")
        tab1, tab2 = st.tabs(["Baseline vs Forecast", "Cost vs Forecast"])
        with tab1:
            fig1 = px.line(df, x="year", y=["consumption", "forecast_consumption"], markers=True, title="Baseline vs Forecast (kWh)")
            st.plotly_chart(fig1, use_container_width=True)
        with tab2:
            fig2 = px.line(df, x="year", y=["baseline_cost", "forecast_cost"], markers=True, title="Cost Comparison (RM)")
            st.plotly_chart(fig2, use_container_width=True)

        # Step 5 — Save to MySQL
        if st.button("💾 Save Forecast to Database"):
            conn = get_db_connection()
            cur = conn.cursor()
            for _, r in df.iterrows():
                cur.execute("""
                    INSERT INTO forecast_data (year, consumption, baseline_cost, adjusted_consumption, adjusted_cost,
                    adjusted_energy_saving, adjusted_cost_saving, adjusted_co2_reduction, uploaded_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                """, (
                    r["year"], r["consumption"], r["baseline_cost"],
                    r["forecast_consumption"], r["forecast_cost"],
                    r["energy_saving"], r["cost_saving"], r["co2_reduction"]
                ))
            conn.commit()
            conn.close()
            st.success("✅ Data successfully saved to MySQL database!")

# ------------------------------------------------------
# DEVICE MANAGEMENT
# ------------------------------------------------------
elif menu == "🧠 Device Management":
    st.title("🧠 Device Management")
    st.info("Feature to monitor connected IoT devices (future integration).")

# ------------------------------------------------------
# REPORTS
# ------------------------------------------------------
elif menu == "📈 Reports":
    st.title("📈 Reports & Data Export")

    conn = get_db_connection()
    df_db = pd.read_sql("SELECT * FROM forecast_data ORDER BY year", conn)
    conn.close()

    st.dataframe(df_db)

    # Graph
    fig = px.line(df_db, x="year", y=["consumption", "adjusted_consumption"], markers=True, title="Database: Baseline vs Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.subheader("📂 Export Options")
    if st.button("📄 Download as Excel"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            df_db.to_excel(writer, index=False, sheet_name="Forecast_Report")
        st.download_button("⬇️ Download Excel File", data=buffer.getvalue(), file_name="forecast_report.xlsx")

    if st.button("🧾 Download as PDF"):
        pdf_path = "forecast_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Energy Forecast Report", styles['Heading1']), Spacer(1, 12)]
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        data = [df_db.columns.tolist()] + df_db.values.tolist()
        story.append(Table(data))
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF File", f, file_name="forecast_report.pdf")

# ------------------------------------------------------
# SETTINGS
# ------------------------------------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.info("This section allows theme and system preferences configuration.")

# ------------------------------------------------------
# HELP & ABOUT
# ------------------------------------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    ### 📬 Contact Support
    If you encounter any issues or system errors, please contact:
    **Email:** [chikaprojectsupport@gmail.com](mailto:chikaprojectsupport@gmail.com)
    
    ### 🧾 Version
    - App Version: 2.0  
    - Developer: Chika @ Polytechnic Kota Kinabalu
    """)
