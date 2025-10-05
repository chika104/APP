import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io
import xlsxwriter
from reportlab.pdfgen import canvas

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="⚡ Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("⚡ Energy Forecast Dashboard")
st.markdown("Selamat datang ke **Energy Forecast App**! 🎉")
st.markdown("Upload dataset anda untuk mula membuat analisis, forecast tenaga, kos & penjimatan.")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Controls")
    n_days = st.sidebar.slider("Number of forecast days", 7, 30, 14)
    energy_rate = st.sidebar.number_input("Energy cost per kWh (RM)", 0.1, 5.0, 0.50)
    saving_rate = st.sidebar.slider("Expected saving rate (%)", 1, 50, 10)

    # -----------------------------
    # Forecasting (simple regression)
    # -----------------------------
    if "Energy" in df.columns:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["Energy"].values
        model = LinearRegression()
        model.fit(X, y)

        future_x = np.arange(len(df), len(df) + n_days).reshape(-1, 1)
        forecast = model.predict(future_x)

        forecast_dates = pd.date_range(start=pd.to_datetime(df.index[-1]), periods=n_days + 1, freq="D")[1:]
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast_Energy": forecast})

        # Combine actual + forecast
        full_data = pd.concat([df, forecast_df.set_index("Date")], axis=0)

        # -----------------------------
        # Plot Charts
        # -----------------------------
        st.subheader("📈 Energy Usage Forecast")
        fig = px.line(full_data, x=full_data.index, y=full_data.columns[0], title="Energy Forecast (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Key Metrics
        # -----------------------------
        st.subheader("📌 Key Metrics")
        total_actual_energy = df["Energy"].sum()
        forecast_energy = forecast.sum()
        total_cost = total_actual_energy * energy_rate
        forecast_cost = forecast_energy * energy_rate
        saving_estimate = forecast_cost * (saving_rate / 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Total Energy (kWh)", f"{total_actual_energy:,.2f}")
        col2.metric("💰 Total Cost (RM)", f"{total_cost:,.2f}")
        col3.metric("✅ Potential Saving (RM)", f"{saving_estimate:,.2f}")

        # -----------------------------
        # Download Reports
        # -----------------------------
        st.subheader("📥 Download Reports")

        # Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Actual Data")
            forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
        excel_data = output.getvalue()

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name="energy_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF
        pdf_output = io.BytesIO()
        c = canvas.Canvas(pdf_output)
        c.setFont("Helvetica", 12)
        c.drawString(100, 800, "Energy Forecast Report")
        c.drawString(100, 780, f"Total Actual Energy: {total_actual_energy:,.2f} kWh")
        c.drawString(100, 760, f"Forecasted Next {n_days} Days: {forecast_energy:,.2f} kWh")
        c.drawString(100, 740, f"Total Cost: RM {total_cost:,.2f}")
        c.drawString(100, 720, f"Forecast Cost (Next {n_days} Days): RM {forecast_cost:,.2f}")
        c.drawString(100, 700, f"Estimated Saving ({saving_rate}%): RM {saving_estimate:,.2f}")
        c.showPage()
        c.save()
        pdf_output.seek(0)

        st.download_button(
            label="📑 Download PDF Report",
            data=pdf_output,
            file_name="energy_forecast.pdf",
            mime="application/pdf"
        )

    else:
        st.error("⚠️ Column `Energy` tidak dijumpai dalam dataset. Pastikan dataset ada column bernama `Energy`.")
else:
    st.info("⬆️ Upload dataset anda untuk mula.")
