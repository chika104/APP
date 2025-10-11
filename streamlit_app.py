# streamlit_app.py
"""
Smart Energy Forecasting — Full Streamlit App
Features:
- Dark mode default (user can switch to Light / Custom image)
- Menu navigation: Dashboard, Energy Forecast, Device Management, Reports, Settings, Help & About
- Input: Upload CSV or Manual entry
- Adjustment factors: lamp types (LED, CFL, Fluorescent), computer, lab equipment, operating hours (hours/year)
- Forecasting: Linear Regression, forecast for configurable years ahead
- Model accuracy (R^2)
- Graphs: baseline vs forecast, future forecast, cost trend, CO2 trend
- Exports: Excel (.xlsx) and formal PDF (with summary, tables, attempt to include graphs if environment supports)
- Help email included
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime

# plotting
import plotly.express as px

# model & metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# try to import PDF libraries
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# helper: try to enable plotly image export (kaleido)
PLOTLY_IMG_OK = False
try:
    # use fig.to_image with kaleido backend
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    PLOTLY_IMG_OK = True
except Exception:
    PLOTLY_IMG_OK = False

# helper: excel writer engine fallback
EXCEL_ENGINE = "xlsxwriter"  # typically available

# -------------------------
# PAGE CONFIG + DEFAULT THEME
# -------------------------
st.set_page_config(page_title="Smart Energy Forecasting", layout="wide")

# default background: dark
DEFAULT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {background-color: #0E1117; color: #F5F5F5;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.04);}
</style>
"""
st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)

# -------------------------
# SIDEBAR: Global UI
# -------------------------
st.sidebar.title("🔹 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "⚡ Energy Forecast", "💡 Device Management",
                                     "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# background / theme selector (stored in session_state)
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"

# apply theme selection UI in Settings page; but show quick toggle here
if st.sidebar.button("Reset to default dark theme"):
    st.session_state.bg_mode = "Dark"
    st.experimental_rerun()

# -------------------------
# Utility functions
# -------------------------
def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def safe_num(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def df_to_excel_bytes(dfs: dict):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine=EXCEL_ENGINE) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def make_pdf_bytes(title_text, summary_lines, table_blocks, image_bytes_list=None, logo_bytes=None):
    """
    Create a formal PDF using reportlab if available.
    - title_text: main title
    - summary_lines: list of strings
    - table_blocks: list of tuples (title, pandas.DataFrame)
    - image_bytes_list: list of PNG bytes to embed (optional)
    - logo_bytes: optional PNG bytes for logo
    """
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # header: logo + title
    if logo_bytes:
        try:
            logo_buf = io.BytesIO(logo_bytes)
            img = RLImage(logo_buf, width=80, height=80)
            elements.append(img)
        except Exception:
            pass

    elements.append(Paragraph(title_text, styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # summary text
    for line in summary_lines:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # images (plots)
    if image_bytes_list:
        for im_bytes in image_bytes_list:
            try:
                imgbuf = io.BytesIO(im_bytes)
                img = RLImage(imgbuf, width=450, height=280)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass

    # tables
    for title, df in table_blocks:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        elements.append(Spacer(1, 6))
        data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        elements.append(tbl)
    try:
        doc.build(elements)
        return buf.getvalue()
    except Exception:
        return None

# -------------------------
# DASHBOARD PAGE
# -------------------------
if menu == "🏠 Dashboard":
    st.title("🏠 Smart Energy Forecasting")
    st.markdown("""
    **Welcome** — use the left menu to go to the Energy Forecast module, manage devices, or download reports.
    """)
    st.markdown("- Clean, professional dashboard.\n- Forecast energy and cost, compare baseline vs adjusted scenarios.\n- Export formal PDF & Excel reports.")
    st.info("Tip: Use Settings to change background (Dark / Light / Custom image).")

# -------------------------
# ENERGY FORECAST PAGE
# -------------------------
elif menu == "⚡ Energy Forecast":
    st.title("⚡ Energy Forecast")

    # --- Step 1: Input data (CSV or Manual)
    st.header("Step 1 — Input baseline data")
    input_mode = st.radio("Input method:", ("Upload CSV", "Manual Entry"))

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel (needs 'year' & 'consumption' columns)", type=["csv","xlsx"])
        if uploaded:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_raw = normalize_cols(df_raw)
            # find candidates
            if "year" not in df_raw.columns or not any(c for c in df_raw.columns if "consum" in c or "kwh" in c or "energy" in c):
                st.error("CSV must contain 'year' and a consumption column (e.g. 'consumption', 'kwh').")
                st.stop()
            # choose columns
            year_col = "year"
            cons_col = [c for c in df_raw.columns if any(k in c for k in ["consum","kwh","energy"])][0]
            df = pd.DataFrame({
                "year": df_raw[year_col].astype(int),
                "consumption": pd.to_numeric(df_raw[cons_col], errors="coerce")
            })
            # optional baseline cost
            cost_cols = [c for c in df_raw.columns if "cost" in c]
            if cost_cols:
                df["baseline_cost"] = pd.to_numeric(df_raw[cost_cols[0]], errors="coerce")
            else:
                df["baseline_cost"] = np.nan
    else:
        rows = st.number_input("Number of historical rows:", min_value=1, max_value=20, value=5)
        data = []
        for i in range(int(rows)):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                y = st.number_input(f"Year {i+1}", 2000, 2100, 2020 + i, key=f"year_{i}")
            with c2:
                cons = st.number_input(f"Consumption kWh ({y})", 0.0, 10_000_000.0, 10000.0, key=f"cons_{i}")
            with c3:
                cost = st.number_input(f"Baseline cost RM ({y}) (optional, 0 = compute from tariff)", 0.0, 10_000_000.0, 0.0, key=f"cost_{i}")
            data.append({"year": int(y), "consumption": float(cons), "baseline_cost": float(cost) if cost>0 else np.nan})
        df = pd.DataFrame(data)

    if df is None or df.empty:
        st.warning("Please upload a valid file or enter manual data to continue.")
        st.stop()

    # ensure types and sort
    df["year"] = df["year"].astype(int)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0.0)
    if "baseline_cost" not in df.columns:
        df["baseline_cost"] = np.nan
    df["baseline_cost"] = pd.to_numeric(df["baseline_cost"], errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)

    st.subheader("Loaded baseline data")
    st.dataframe(df)

    # --- Step 2: Factors (detailed)
    st.header("Step 2 — Adjustment factors (additions or reductions)")

    st.markdown("Enter device-level adjustments. You may add multiple factors. Hours are per YEAR.")

    # default wattages
    WATT = {"LED": 10, "CFL": 15, "Fluorescent": 40, "Computer": 150, "Lab Equipment": 500}

    # allow multiple factors via number input
    n_factors = st.number_input("How many factor rows do you want to add?", min_value=1, max_value=10, value=1, key="n_factors")
    factor_rows = []
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            device = st.selectbox(f"Device type (factor {i+1})", options=["Lamp - LED", "Lamp - CFL", "Lamp - Fluorescent", "Computer", "Lab Equipment"], key=f"dev_{i}")
        with c2:
            units = st.number_input(f"Units (factor {i+1})", min_value=0, value=0, step=1, key=f"units_{i}")
        with c3:
            hours = st.number_input(f"Hours per YEAR (factor {i+1})", min_value=0, max_value=8760, value=0, step=1, key=f"hours_{i}")
        with c4:
            action = st.selectbox(f"Action (factor {i+1})", options=["Addition", "Reduction"], key=f"action_{i}")
        # compute watt
        if device.startswith("Lamp"):
            subtype = device.split(" - ")[1]
            watt = WATT[subtype]
            dev_name = f"{subtype} Lamp"
        else:
            dev_key = device
            watt = WATT[dev_key]
            dev_name = dev_key
        kwh_per_year = (watt * units * hours) / 1000.0
        if action == "Reduction":
            kwh_per_year = -abs(kwh_per_year)
        else:
            kwh_per_year = abs(kwh_per_year)
        factor_rows.append({
            "device": dev_name,
            "units": int(units),
            "hours_per_year": int(hours),
            "action": action,
            "kwh_per_year": kwh_per_year
        })

    df_factors = pd.DataFrame(factor_rows)
    st.subheader("Factors summary (kWh per year)")
    st.dataframe(df_factors)

    # optional site-level operating hours change (not device-specific) — specify hours/year and average kW
    st.markdown("General site-level operating hours change (positive = add load, negative = reduce load)")
    general_hours = st.number_input("General extra/reduced hours per year", min_value=-8760, max_value=8760, value=0)
    general_avg_load_kw = st.number_input("Assumed average site load for general hours (kW)", min_value=0.0, value=2.0, step=0.1)
    general_kwh = float(general_avg_load_kw) * float(general_hours) if general_hours != 0 else 0.0

    total_net_adjust_kwh = df_factors["kwh_per_year"].sum() + general_kwh
    if total_net_adjust_kwh > 0:
        st.info(f"Net adjustment (additional consumption): {total_net_adjust_kwh:,.2f} kWh/year")
    elif total_net_adjust_kwh < 0:
        st.info(f"Net adjustment (reduction): {abs(total_net_adjust_kwh):,.2f} kWh/year")
    else:
        st.info("Net adjustment: 0 kWh/year")

    # --- Step 3: Forecast settings & compute
    st.header("Step 3 — Forecast settings & compute")

    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)
    n_years_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1)

    # Historical baseline cost fill
    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    # Build linear regression model using year -> consumption
    model = LinearRegression()
    X_hist = df[["year"]].values
    y_hist = df["consumption"].values
    if len(X_hist) >= 2:
        model.fit(X_hist, y_hist)
        df["fitted"] = model.predict(X_hist)
        r2 = r2_score(y_hist, df["fitted"])
    else:
        # not enough history - use last value as flat forecast
        df["fitted"] = df["consumption"]
        r2 = 1.0

    last_year = int(df["year"].max())
    future_years = [last_year + i for i in range(1, int(n_years_forecast)+1)]
    future_X = np.array(future_years).reshape(-1,1)
    future_baseline_forecast = model.predict(future_X) if len(X_hist) >= 2 else np.array([df["consumption"].iloc[-1]]*len(future_years))

    # apply net adjustment to future years (adjustment applied equally to each forecast year)
    adjusted_forecast = future_baseline_forecast + total_net_adjust_kwh

    forecast_df = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": future_baseline_forecast,
        "adjusted_consumption_kwh": adjusted_forecast
    })
    forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
    forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
    forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * co2_factor
    forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * co2_factor
    forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
    forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
    forecast_df["saving_co2_kg"] = forecast_df["baseline_co2_kg"] - forecast_df["adjusted_co2_kg"]

    # --- Step 4: Visualizations (graphs)
    st.header("Step 4 — Visual comparisons & model accuracy")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Baseline (historical) vs Forecast")
        # combine historical + forecast for plot
        hist_plot = pd.DataFrame({
            "year": df["year"],
            "baseline": df["consumption"],
            "fitted": df["fitted"]
        })
        plot_all = pd.concat([hist_plot, pd.DataFrame({"year": forecast_df["year"], "baseline": forecast_df["baseline_consumption_kwh"], "fitted": forecast_df["adjusted_consumption_kwh"]})], ignore_index=True)
        fig_both = px.line(plot_all.sort_values("year"), x="year", y=["baseline","fitted"], markers=True,
                           labels={"value":"kWh","variable":"Series"},
                           title="Baseline vs Forecast (kWh)")
        st.plotly_chart(fig_both, use_container_width=True)

        st.subheader("Future forecast (baseline vs adjusted)")
        fig_future = px.line(forecast_df, x="year", y=["baseline_consumption_kwh","adjusted_consumption_kwh"], markers=True,
                             labels={"value":"kWh","variable":"Series"}, title="Future Forecast (kWh)")
        st.plotly_chart(fig_future, use_container_width=True)

        st.subheader("Cost trend (RM) — forecast period")
        fig_cost = px.bar(forecast_df, x="year", y=["baseline_cost_rm","adjusted_cost_rm"], barmode="group", title="Cost Trend (RM)")
        st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("CO₂ trend (kg) — forecast period")
        fig_co2 = px.bar(forecast_df, x="year", y=["baseline_co2_kg","adjusted_co2_kg"], barmode="group", title="CO₂ Trend (kg)")
        st.plotly_chart(fig_co2, use_container_width=True)

    with col2:
        st.subheader("Model performance")
        st.markdown(f"**R² (coefficient of determination):** `{r2:.4f}`")
        if r2 >= 0.8:
            st.success("Model accuracy: High")
        elif r2 >= 0.6:
            st.warning("Model accuracy: Moderate")
        else:
            st.error("Model accuracy: Low — consider more historical data or more features")

        st.markdown("**Totals over forecast period**")
        total_baseline_kwh = forecast_df["baseline_consumption_kwh"].sum()
        total_adjusted_kwh = forecast_df["adjusted_consumption_kwh"].sum()
        total_kwh_saving = total_baseline_kwh - total_adjusted_kwh
        total_cost_saving = forecast_df["saving_cost_rm"].sum()
        total_co2_saving = forecast_df["saving_co2_kg"].sum()

        st.metric("Baseline kWh (forecast period)", f"{total_baseline_kwh:,.0f} kWh")
        st.metric("Adjusted kWh (forecast period)", f"{total_adjusted_kwh:,.0f} kWh")
        st.metric("Total energy saving (kWh)", f"{total_kwh_saving:,.0f} kWh")
        st.metric("Total cost saving (RM)", f"RM {total_cost_saving:,.2f}")
        st.metric("Total CO₂ reduction (kg)", f"{total_co2_saving:,.0f} kg")

    # --- Step 5: Forecast tables
    st.header("Step 5 — Forecast tables")
    st.subheader("Historical (baseline)")
    st.dataframe(df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"}))
    st.subheader("Forecast results")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh":"{:.0f}",
        "adjusted_consumption_kwh":"{:.0f}",
        "baseline_cost_rm":"{:.2f}",
        "adjusted_cost_rm":"{:.2f}",
        "saving_kwh":"{:.0f}",
        "saving_cost_rm":"{:.2f}",
        "saving_co2_kg":"{:.0f}"
    }))

    # --- Step 6: Export (Excel + PDF)
    st.header("Step 6 — Export results")

    # Excel data
    excel_bytes = df_to_excel_bytes({"historical": df, "factors": df_factors, "forecast": forecast_df})
    st.download_button("⬇️ Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF creation: summary + tables + try to capture plots
    def try_get_plot_png(fig):
        if PLOTLY_IMG_OK:
            try:
                png = fig.to_image(format="png", width=900, height=540, scale=2)
                return png
            except Exception:
                return None
        return None

    images = []
    # attempt to capture main charts
    for fig in (fig_both, fig_future, fig_cost, fig_co2):
        try:
            img = try_get_plot_png(fig)
            if img:
                images.append(img)
        except Exception:
            pass

    summary_lines = [
        f"Forecast period: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
        f"Net adjustment applied (kWh/year): {total_net_adjust_kwh:.2f}",
        f"Total energy saving (kWh): {total_kwh_saving:,.2f}",
        f"Total cost saving (RM): RM {total_cost_saving:,.2f}",
        f"Total CO₂ reduction (kg): {total_co2_saving:,.2f}",
        f"Model R²: {r2:.4f}"
    ]

    # Build tables for PDF
    table_blocks = [
        ("Historical (baseline)", df[["year","consumption","baseline_cost"]].rename(columns={"consumption":"consumption_kwh","baseline_cost":"baseline_cost_rm"})),
        ("Factors (kWh/year)", df_factors[["device","units","hours_per_year","action","kwh_per_year"]]),
        ("Forecast results", forecast_df)
    ]

    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        pdf_bytes = make_pdf_bytes("SMART ENERGY FORECASTING REPORT", summary_lines, table_blocks, image_bytes_list=images)
    else:
        st.info("PDF export not available on this system (reportlab not installed). Excel export is available.")

    if pdf_bytes:
        st.download_button("📄 Download formal PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
    else:
        st.info("PDF report generation skipped — install 'reportlab' (and 'kaleido' for embedding graphs) to enable full PDF export with graphs.")

# -------------------------
# DEVICE MANAGEMENT
# -------------------------
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.markdown("Add and manage common device types used in forecasts. (This module is a simple registry — further IoT integration is optional.)")
    # Simple register UI (local only)
    if "devices" not in st.session_state:
        st.session_state.devices = []
    with st.form("add_device"):
        d_name = st.text_input("Device name (e.g. 'LED 10W')", value="")
        d_watt = st.number_input("Power (W)", min_value=0.0, value=10.0)
        d_note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Add device")
        if submitted and d_name:
            st.session_state.devices.append({"name":d_name,"watt":d_watt,"note":d_note})
            st.success("Device added.")
    if st.session_state.devices:
        st.table(pd.DataFrame(st.session_state.devices))

# -------------------------
# REPORTS
# -------------------------
elif menu == "📊 Reports":
    st.title("📊 Reports")
    st.markdown("Saved exports will appear in Downloads (the app does not persist files server-side). Use Excel/PDF export on Energy Forecast screen.")

# -------------------------
# SETTINGS
# -------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Appearance & Preferences")
    choice = st.radio("Background / Theme:", ["Dark (default)", "Light", "Custom image URL"])
    if choice == "Dark (default)":
        st.session_state.bg_mode = "Dark"
        st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
        st.success("Applied Dark theme.")
    elif choice == "Light":
        st.session_state.bg_mode = "Light"
        light_style = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
        [data-testid="stSidebar"] {background-color: rgba(0,0,0,0.03);}
        </style>
        """
        st.markdown(light_style, unsafe_allow_html=True)
        st.success("Applied Light theme.")
    else:
        img_url = st.text_input("Enter a full image URL to use as background:")
        if img_url:
            custom_style = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img_url}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """
            st.markdown(custom_style, unsafe_allow_html=True)
            st.success("Applied custom background image.")

    st.markdown("---")
    st.markdown("**PDF Export:** A formal PDF (with tables) is included if `reportlab` is installed on the host. Embedding charts into PDF requires `kaleido` for Plotly image export.")

# -------------------------
# HELP & ABOUT
# -------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed for forecasting and scenario comparison of energy consumption, cost and CO₂.

    **Support / Report issues:**  
    📧 **Email:** chikaenergyforecast@gmail.com

    Note: This app uses offline historical data you upload or enter manually — no hardware (IoT) is required to use the forecasting features.
    """)

