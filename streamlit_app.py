"""
Smart Energy Forecasting — Full Streamlit App
"""
import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Optional dependencies
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

PLOTLY_IMG_OK = False
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

MYSQL_AVAILABLE = True
try:
    import mysql.connector
except Exception:
    MYSQL_AVAILABLE = False

EXCEL_ENGINE = "xlsxwriter"

# -------------------------
# Utility functions
# -------------------------
def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def try_get_plot_png(fig):
    if PLOTLY_IMG_OK:
        try:
            return fig.to_image(format="png", width=900, height=540, scale=2)
        except Exception:
            return None
    return None

def make_pdf_bytes(title_text, summary_lines, table_blocks, image_bytes_list=None, logo_bytes=None):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title_text, styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    for line in summary_lines:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 12))
    if image_bytes_list:
        for im_bytes in image_bytes_list:
            try:
                imgbuf = io.BytesIO(im_bytes)
                img = RLImage(imgbuf, width=450, height=280)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
    for title, df in table_blocks:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")
st.sidebar.title("🔹 Smart Energy Forecasting")

menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# -------------------------
# Dashboard
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("Use sidebar to navigate to Forecast, Reports, or Settings.")

# -------------------------
# Forecast
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # Step 1: Input
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (must contain 'year' and 'consumption')", type=["csv","xlsx"])
        if uploaded:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)
            df = df_raw[["year","consumption"]]
    else:
        rows = st.number_input("Number of years:", 1, 20, 5)
        data = []
        for i in range(rows):
            y = st.number_input(f"Year {i+1}", 2000, 2100, 2020+i, key=f"y_{i}")
            c = st.number_input(f"Consumption (kWh) {y}", 0.0, 1e7, 10000.0, key=f"c_{i}")
            data.append({"year": y, "consumption": c})
        df = pd.DataFrame(data)

    if df is None or df.empty:
        st.stop()

    st.dataframe(df)

    # Step 2: Forecast
    st.header("Step 2 — Forecast model")
    forecast_years = st.number_input("Forecast next N years:", 1, 10, 3)
    tariff = st.number_input("Electricity tariff (RM/kWh)", 0.0, 10.0, 0.5)
    model = LinearRegression()
    X = df[["year"]]
    y = df["consumption"]
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))

    future_years = np.arange(df["year"].max()+1, df["year"].max()+1+forecast_years)
    forecast_df = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": model.predict(future_years.reshape(-1,1)),
    })
    forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
    forecast_df["adjusted_consumption_kwh"] = forecast_df["baseline_consumption_kwh"] * 0.95
    forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
    forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
    forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
    forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * 0.584
    forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * 0.584
    forecast_df["saving_co2_kg"] = forecast_df["saving_kwh"] * 0.584

    st.dataframe(forecast_df)

    # Step 3: Charts
    st.header("Step 3 — Visualizations")

    fig_baseline = px.line(df, x="year", y="consumption", title="Baseline kWh (Historical)")
    st.plotly_chart(fig_baseline, use_container_width=True)

    fig_kwh = px.line(forecast_df, x="year", y=["baseline_consumption_kwh","adjusted_consumption_kwh"], title="Baseline vs Forecast kWh")
    st.plotly_chart(fig_kwh, use_container_width=True)

    fig_cost = px.line(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], title="Baseline vs Forecast Cost")
    st.plotly_chart(fig_cost, use_container_width=True)

    fig_co2 = px.bar(forecast_df, x="year", y=["baseline_co2_kg","adjusted_co2_kg"], barmode="group", title="CO₂ Forecast (kg)")
    st.plotly_chart(fig_co2, use_container_width=True)

    # Step 4: Summary metrics
    st.header("Step 4 — Summary")
    total_baseline_kwh = forecast_df["baseline_consumption_kwh"].sum()
    total_adjusted_kwh = forecast_df["adjusted_consumption_kwh"].sum()
    total_kwh_saving = forecast_df["saving_kwh"].sum()
    total_cost_saving = forecast_df["saving_cost_rm"].sum()
    total_co2_saving = forecast_df["saving_co2_kg"].sum()

    st.metric("Total energy saving (kWh)", f"{total_kwh_saving:,.0f}")
    st.metric("Total cost saving (RM)", f"RM {total_cost_saving:,.2f}")
    st.metric("Total CO₂ reduction (kg)", f"{total_co2_saving:,.0f}")

    # Step 6: Export & (optional) Save to DB
    st.header("Step 6 — Export results & Save")
    excel_bytes = df_to_excel_bytes({"historical": df, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    images = []
    for fig in (fig_baseline, fig_kwh, fig_cost, fig_co2):
        img = try_get_plot_png(fig)
        if img:
            images.append(img)

    summary_lines = [
        f"Forecast period: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
        f"Total energy saving (kWh): {total_kwh_saving:,.2f}",
        f"Total cost saving (RM): RM {total_cost_saving:,.2f}",
        f"Total CO₂ reduction (kg): {total_co2_saving:,.2f}",
        f"Model R²: {r2:.4f}"
    ]

    table_blocks = [
        ("Historical (baseline)", df),
        ("Forecast results", forecast_df)
    ]

    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
    if pdf_bytes:
        st.download_button("📄 Download formal PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export not available (reportlab not installed).")

    st.markdown("---")
    st.subheader("Optional: Save results to MySQL database")
    if not MYSQL_AVAILABLE:
        st.info("MySQL not available. Install 'mysql-connector-python'.")
    else:
        st.markdown("Configure DB in Settings, then test or save here.")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Test DB connection"):
                st.success("DB connection successful (placeholder).")
        with colB:
            if st.button("Save results to DB"):
                st.success("Results saved successfully (placeholder).")

# -------------------------
# Device Management
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.info("Add and manage device types.")

# -------------------------
# Reports
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("Use export options in Energy Forecast page to generate Excel/PDF reports.")

# -------------------------
# Settings
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Database")
    st.markdown("Configure background and database connection here.")

# -------------------------
# Help
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("Smart Energy Forecasting System by Chika.")
