# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import base64
from datetime import datetime

# optional libs
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Smart Energy Forecasting System", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
WATT_DEFAULTS_W = {
    "LED": 10,
    "CFL": 15,
    "Fluorescent": 40,
    "Computer": 150,
    "Lab Equipment": 500
}


def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def safe_to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def calc_kwh_from_watts(watt_per_unit, units, hours_per_year):
    return (watt_per_unit * units * hours_per_year) / 1000.0


def excel_bytes_from_dfs(dfs_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in dfs_dict.items():
            try:
                df.to_excel(writer, sheet_name=name[:31], index=False)
            except Exception:
                # fallback: convert everything to str
                df.astype(str).to_excel(writer, sheet_name=name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def pdf_bytes_report(title, logo_image, summary_lines, df_tables, plot_images=None):
    """
    Create simple PDF report. plot_images is a list of (png_bytes, caption) tuples (optional).
    """
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin

    # logo centered top if provided
    if logo_image is not None:
        try:
            logo = ImageReader(io.BytesIO(logo_image))
            logo_w = 160
            logo_h = 60
            c.drawImage(logo, (w - logo_w) / 2, y - logo_h, width=logo_w, height=logo_h, preserveAspectRatio=True)
            y -= (logo_h + 10)
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, y, title)
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 18

    # summary
    for line in summary_lines:
        if y < 120:
            c.showPage()
            y = h - margin
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, str(line))
        y -= 14

    # add plot images if any
    if plot_images:
        for img_bytes, caption in plot_images:
            if y < 260:
                c.showPage()
                y = h - margin
            try:
                img = ImageReader(io.BytesIO(img_bytes))
                img_w = w - 2 * margin
                img_h = 200
                c.drawImage(img, margin, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True)
                y -= (img_h + 8)
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(margin, y, caption)
                y -= 18
            except Exception:
                # skip if cannot render
                pass

    # add short tables (only first few rows)
    for name, df in df_tables.items():
        if y < 120:
            c.showPage()
            y = h - margin
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, name)
        y -= 14
        c.setFont("Helvetica", 8)
        txt = df.head(8).to_string(index=False).splitlines()
        for line in txt:
            if y < 60:
                c.showPage()
                y = h - margin
            c.drawString(margin, y, line[:120])
            y -= 11
        y -= 8

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ---------------------------
# Sidebar: navigation & settings
# ---------------------------
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Energy Forecast", "Device Management", "Reports", "Settings", "Help & About"])

# Theme / background
st.sidebar.markdown("---")
bg_choice = st.sidebar.selectbox("Background / Theme", ["Dark (default)", "Light", "Upload image"])
bg_image_bytes = None
if bg_choice == "Upload image":
    uploaded_bg = st.sidebar.file_uploader("Upload background image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_bg is not None:
        bg_image_bytes = uploaded_bg.read()

# Header: logo upload and title centered
st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
logo_file = st.file_uploader("Upload logo (will show in header and PDF) — optional", type=["png", "jpg", "jpeg"])
logo_bytes = None
if logo_file is not None:
    logo_bytes = logo_file.read()
    st.image(logo_bytes, width=250)
else:
    # show text title (center)
    st.markdown("<h1 style='text-align:center'>Smart Energy Forecasting System</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# apply basic background via markdown style if uploaded
if bg_choice == "Upload image" and bg_image_bytes is not None:
    b64 = base64.b64encode(bg_image_bytes).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)
elif bg_choice == "Light":
    st.markdown(
        """<style>[data-testid="stAppViewContainer"]{background-color: #ffffff; color: #111}</style>""",
        unsafe_allow_html=True)
else:
    # dark default
    st.markdown(
        """<style>[data-testid="stAppViewContainer"]{background-color: #0f1720; color: #fff}</style>""",
        unsafe_allow_html=True)

# ---------------------------
# Dashboard page
# ---------------------------
if menu == "Dashboard":
    st.header("Dashboard — Overview")
    st.markdown("""
    **Welcome to the Smart Energy Forecasting System.**  
    Use the _Energy Forecast_ module to upload historical data or enter manually, add adjustment factors (lamp, computer, lab equipment, operating hours), and compare baseline vs adjusted scenarios.
    """)
    st.info("Navigate to **Energy Forecast** to start analysis. Use **Reports** to export results.")

# ---------------------------
# Energy Forecast page
# ---------------------------
if menu == "Energy Forecast":
    st.header("Energy Forecast")

    st.markdown("**Step 1 — Input historical baseline data**")
    input_mode = st.radio("Choose input method", ["Upload CSV", "Manual entry"], horizontal=True)

    df = None
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV/XLSX (must contain year and consumption columns)", type=["csv", "xlsx"])
        if uploaded is not None:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
                df_raw = normalize_cols(df_raw)
                # find year & consumption
                year_cols = [c for c in df_raw.columns if "year" in c]
                cons_cols = [c for c in df_raw.columns if any(k in c for k in ["consum", "kwh", "energy"])]
                if not year_cols or not cons_cols:
                    st.error("CSV must include 'year' and a consumption column (e.g., consumption, kwh).")
                    st.stop()
                year_col = year_cols[0]
                cons_col = cons_cols[0]
                df = df_raw[[year_col, cons_col]].copy()
                df.columns = ["year", "consumption"]
                # optional baseline cost column
                cost_cols = [c for c in df_raw.columns if "cost" in c]
                if cost_cols:
                    df["baseline_cost"] = safe_to_numeric_series(df_raw[cost_cols[0]])
                else:
                    df["baseline_cost"] = np.nan
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                st.stop()

    elif input_mode == "Manual entry":
        st.markdown("Enter historical rows (year, consumption kWh, optional baseline cost RM)")
        n_rows = st.number_input("How many historical rows?", min_value=1, max_value=20, value=5)
        years = []
        consumptions = []
        baseline_costs = []
        for i in range(int(n_rows)):
            c1, c2, c3 = st.columns([1.2, 1.8, 1.6])
            with c1:
                y = st.number_input(f"Year {i+1}", min_value=1900, max_value=2100, value=2020 + i, key=f"y_{i}")
            with c2:
                cons = st.number_input(f"Consumption kWh ({y})", min_value=0.0, value=10000.0, key=f"cons_{i}")
            with c3:
                cost = st.number_input(f"Baseline cost RM ({y}) (optional)", min_value=0.0, value=0.0, key=f"cost_{i}")
            years.append(int(y))
            consumptions.append(float(cons))
            baseline_costs.append(float(cost) if cost > 0 else np.nan)
        df = pd.DataFrame({"year": years, "consumption": consumptions, "baseline_cost": baseline_costs})

    # validate input
    if df is None or df.empty:
        st.warning("Please provide historical data (upload CSV or manual entry).")
        st.stop()

    # normalize & sort
    df["year"] = df["year"].astype(int)
    df["consumption"] = safe_to_numeric_series(df["consumption"]).fillna(0).astype(float)
    if "baseline_cost" not in df.columns:
        df["baseline_cost"] = np.nan
    df["baseline_cost"] = safe_to_numeric_series(df["baseline_cost"])
    df = df.sort_values("year").reset_index(drop=True)

    st.subheader("Historical Data")
    st.dataframe(df, use_container_width=True)

    # Step 2 Baseline calculations
    st.markdown("**Step 2 — Baseline & assumptions**")
    tariff = st.number_input("Electricity tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
    co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)

    # fill baseline cost historical
    df["baseline_cost"] = df["baseline_cost"].fillna(df["consumption"] * tariff)
    df["baseline_co2_kg"] = df["consumption"] * co2_factor

    # Step 3 Factors
    st.markdown("**Step 3 — Define adjustment factors**")
    st.markdown("Add factors. Each factor converts (units, hours/year, watt) -> kWh/year. Select Action: Addition or Reduction.")

    n_factors = st.number_input("Number of factors to enter", min_value=0, max_value=10, value=0)
    factors = []
    for i in range(int(n_factors)):
        st.markdown(f"**Factor {i+1}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            dev = st.selectbox(f"Device (factor {i+1})", options=["Lamp", "Computer", "Lab Equipment"], key=f"dev_{i}")
            subtype = None
            if dev == "Lamp":
                subtype = st.selectbox(f"Lamp type (factor {i+1})", options=["LED", "CFL", "Fluorescent"], key=f"lamp_{i}")
        with c2:
            units = st.number_input(f"Units (factor {i+1})", min_value=0, value=0, step=1, key=f"units_{i}")
        with c3:
            hours_year = st.number_input(f"Hours per YEAR (factor {i+1})", min_value=0, max_value=8760, value=0, step=1, key=f"hours_{i}")
        c4, c5 = st.columns([2, 1])
        with c4:
            action = st.selectbox(f"Action (Add/Reduce) (factor {i+1})", options=["Addition", "Reduction"], key=f"act_{i}")
        with c5:
            start_year = st.selectbox(f"Apply from year (factor {i+1})", options=list(range(df["year"].max()+1, df["year"].max()+6)), index=0, key=f"start_{i}")
        # assign watt
        if dev == "Lamp":
            watt = WATT_DEFAULTS_W[subtype]
            dev_name = f"{subtype} Lamp"
        else:
            watt = WATT_DEFAULTS_W[dev]
            dev_name = dev
        kwh_year = calc_kwh_from_watts(watt, units, hours_year)
        if action == "Reduction":
            kwh_year = -kwh_year
        factors.append({
            "device": dev_name,
            "units": int(units),
            "hours_per_year": int(hours_year),
            "action": action,
            "kwh_per_year": float(kwh_year),
            "start_year": int(start_year)
        })

    df_factors = pd.DataFrame(factors) if factors else pd.DataFrame(columns=["device", "units", "hours_per_year", "action", "kwh_per_year", "start_year"])
    st.subheader("Factors summary")
    st.dataframe(df_factors, use_container_width=True)

    # General site-level operating hours adjustment (hours per year) - treated as average site load (kW)
    st.markdown("Optional site-level operating hours change (not device-specific).")
    general_hours = st.number_input("General extra/reduced hours per year (positive=add, negative=reduce)", min_value=-8760, max_value=8760, value=0, step=1)
    general_avg_load_kw = st.number_input("Assumed average site load for general hours (kW)", min_value=0.0, value=0.0, step=0.1)
    general_kwh = 0.0
    if general_hours != 0 and general_avg_load_kw > 0:
        general_kwh = general_avg_load_kw * general_hours
        st.info(f"Site-level adjustment = {general_kwh:,.2f} kWh per year")

    # Step 4 Forecast settings
    st.markdown("**Step 4 — Forecast settings**")
    n_forecast_years = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=3, step=1)
    apply_factors_start = st.selectbox("If multiple start years selected for factors, factors apply from their defined start years. (This control defines default if none)", options=[df["year"].max()+i for i in range(1, n_forecast_years+1)], index=0)

    # Build baseline forecast using linear trend on historical points
    hist_x = df["year"].values.astype(float)
    hist_y = df["consumption"].values.astype(float)
    if len(hist_x) >= 2:
        # use polyfit on year values to avoid index mismatch
        coeffs = np.polyfit(hist_x, hist_y, 1)
        slope, intercept = coeffs[0], coeffs[1]
    else:
        slope, intercept = 0.0, float(hist_y[-1] if len(hist_y) > 0 else 0.0)

    last_year = int(df["year"].max())
    future_years = [last_year + i for i in range(1, int(n_forecast_years)+1)]

    baseline_forecast = [slope * y + intercept for y in future_years]

    # compute adjusted forecast: for each forecast year, sum contributions from factors that start <= year
    adjusted_forecast = []
    for fy, base_kwh in zip(future_years, baseline_forecast):
        # sum factor kwh for those factors whose start_year <= fy
        factor_sum = 0.0
        for f in factors:
            if fy >= int(f["start_year"]):
                factor_sum += f["kwh_per_year"]
        # include site-level general_kwh (if general_hours applied) - assume applies to all forecast years
        total_adj = base_kwh + factor_sum + general_kwh
        adjusted_forecast.append(total_adj)

    forecast_df = pd.DataFrame({
        "year": future_years,
        "baseline_consumption_kwh": baseline_forecast,
        "adjusted_consumption_kwh": adjusted_forecast
    })
    forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
    forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
    forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * co2_factor
    forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * co2_factor
    forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
    forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
    forecast_df["saving_co2_kg"] = forecast_df["baseline_co2_kg"] - forecast_df["adjusted_co2_kg"]

    # Step 5 Visuals
    st.markdown("**Step 5 — Visual comparisons**")
    # Graph 1: Baseline vs Adjusted Consumption (historic + forecast)
    fig1 = px.line(title="Baseline vs Adjusted Consumption (Historic + Forecast)")
    fig1.add_scatter(x=df["year"], y=df["consumption"], mode="markers+lines", name="Historical (baseline)")
    fig1.add_scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"], mode="markers+lines", name="Baseline forecast")
    fig1.add_scatter(x=forecast_df["year"], y=forecast_df["adjusted_consumption_kwh"], mode="markers+lines", name="Adjusted forecast")
    fig1.update_layout(xaxis_title="Year", yaxis_title="kWh")

    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Baseline cost vs adjusted cost (historic + forecast)
    fig2 = px.line(title="Baseline cost vs Adjusted cost (RM)")
    fig2.add_scatter(x=df["year"], y=df["baseline_cost"], mode="markers+lines", name="Historical baseline cost (RM)")
    fig2.add_scatter(x=forecast_df["year"], y=forecast_df["baseline_cost_rm"], mode="markers+lines", name="Baseline forecast cost (RM)")
    fig2.add_scatter(x=forecast_df["year"], y=forecast_df["adjusted_cost_rm"], mode="markers+lines", name="Adjusted forecast cost (RM)")
    fig2.update_layout(xaxis_title="Year", yaxis_title="RM")
    st.plotly_chart(fig2, use_container_width=True)

    # Graph 3: CO2 trend
    fig3 = px.line(title="CO₂ Emissions: Baseline vs Adjusted (kg)")
    fig3.add_scatter(x=forecast_df["year"], y=forecast_df["baseline_co2_kg"], mode="markers+lines", name="Baseline CO₂ (kg)")
    fig3.add_scatter(x=forecast_df["year"], y=forecast_df["adjusted_co2_kg"], mode="markers+lines", name="Adjusted CO₂ (kg)")
    fig3.update_layout(xaxis_title="Year", yaxis_title="kg CO₂")
    st.plotly_chart(fig3, use_container_width=True)

    # Graph 4: Savings (kWh & RM)
    fig4 = px.bar(forecast_df, x="year", y=["saving_kwh", "saving_cost_rm"], barmode="group", title="Yearly Savings: kWh & RM")
    fig4.update_yaxes(title_text="Value (kWh / RM)")
    st.plotly_chart(fig4, use_container_width=True)

    # Forecast table
    st.subheader("Forecast table (baseline vs adjusted)")
    st.dataframe(forecast_df.style.format({
        "baseline_consumption_kwh": "{:,.0f}",
        "adjusted_consumption_kwh": "{:,.0f}",
        "baseline_cost_rm": "{:,.2f}",
        "adjusted_cost_rm": "{:,.2f}",
        "saving_kwh": "{:,.0f}",
        "saving_cost_rm": "{:,.2f}",
        "saving_co2_kg": "{:,.0f}"
    }), use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total baseline kWh (forecast)", f"{forecast_df['baseline_consumption_kwh'].sum():,.0f}")
    col2.metric("Total adjusted kWh (forecast)", f"{forecast_df['adjusted_consumption_kwh'].sum():,.0f}")
    col3.metric("Total kWh saving", f"{forecast_df['saving_kwh'].sum():,.0f}")
    col4.metric("Total cost saving (RM)", f"RM {forecast_df['saving_cost_rm'].sum():,.2f}")

    # Step 6: Exports
    st.markdown("**Step 6 — Export results**")
    all_dfs = {"historical": df, "factors": df_factors, "forecast": forecast_df}
    excel_bytes = excel_bytes_from_dfs(all_dfs)
    st.download_button("Download Excel (.xlsx)", data=excel_bytes, file_name="energy_forecast_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF export - try to include plot images if plotly -> png export available (kaleido)
    pdf_ready = True if REPORTLAB_AVAILABLE else False
    if pdf_ready:
        show_pdf = st.button("Generate PDF report (may include graphs if available)")
        if show_pdf:
            summary_lines = [
                f"Forecast years: {forecast_df['year'].min()} - {forecast_df['year'].max()}",
                f"Net site-level adjustment (kWh/year): {general_kwh:.2f}",
                f"Total kWh saved (forecast period): {forecast_df['saving_kwh'].sum():.2f}",
                f"Total cost saved (RM): {forecast_df['saving_cost_rm'].sum():.2f}",
                f"Total CO₂ reduction (kg): {forecast_df['saving_co2_kg'].sum():.2f}"
            ]
            plot_images = []
            # try to export fig1..fig4 as png using kaleido
            for fig in (fig1, fig2, fig3, fig4):
                try:
                    img_bytes = fig.to_image(format="png", width=1000, height=450)
                    plot_images.append((img_bytes, fig.layout.title.text if fig.layout.title else ""))
                except Exception:
                    # skip if not possible
                    pass
            pdf_bytes = pdf_bytes_report("Smart Energy Forecasting System", logo_bytes, summary_lines, {"Historical": df, "Factors": df_factors, "Forecast": forecast_df}, plot_images=plot_images)
            if pdf_bytes:
                st.download_button("Download PDF report", data=pdf_bytes, file_name="energy_forecast_report.pdf", mime="application/pdf")
            else:
                st.error("PDF generation failed (reportlab missing or error).")
    else:
        st.info("PDF export not available — install 'reportlab' in environment to enable.")

# ---------------------------
# Device Management page (simple CRUD-like placeholder)
# ---------------------------
if menu == "Device Management":
    st.header("Device Management")
    st.markdown("Manage common device wattage defaults and add custom device types for factoring.")
    st.markdown("Defaults (editable):")
    colA, colB = st.columns(2)
    with colA:
        st.write("LED (Watt per unit)")
        w_led = st.number_input("LED watt", value=WATT_DEFAULTS_W["LED"], key="w_led")
    with colB:
        st.write("CFL (Watt per unit)")
        w_cfl = st.number_input("CFL watt", value=WATT_DEFAULTS_W["CFL"], key="w_cfl")
    WATT_DEFAULTS_W["LED"] = w_led
    WATT_DEFAULTS_W["CFL"] = w_cfl
    st.write("Computer and Lab defaults:")
    c1, c2 = st.columns(2)
    with c1:
        w_comp = st.number_input("Computer watt", value=WATT_DEFAULTS_W["Computer"], key="w_comp")
    with c2:
        w_lab = st.number_input("Lab Equipment watt", value=WATT_DEFAULTS_W["Lab Equipment"], key="w_lab")
    WATT_DEFAULTS_W["Computer"] = w_comp
    WATT_DEFAULTS_W["Lab Equipment"] = w_lab
    st.success("Device defaults updated (session only).")

# ---------------------------
# Reports page - quick access to last generated excel/pdf
# ---------------------------
if menu == "Reports":
    st.header("Reports")
    st.markdown("Use Energy Forecast page to generate Excel/PDF exports. This page can be extended to show saved reports (requires persistent storage).")
    st.info("If you want persistent DB and saved reports, we can add MySQL / SQLite integration in next iteration.")

# ---------------------------
# Settings page
# ---------------------------
if menu == "Settings":
    st.header("Settings")
    st.markdown("App settings & preferences.")
    st.markdown("- Contact & branding can be changed in Help & About.")
    st.markdown("- For production deploy, ensure requirements.txt includes required libs (plotly, reportlab, kaleido if needed).")

# ---------------------------
# Help & About
# ---------------------------
if menu == "Help & About":
    st.header("Help & About")
    st.markdown("**Smart Energy Forecasting System** — Desktop / Web forecasting tool.\n\nIf you find issues, please contact:")
    st.write("📧 Email: **nurshashiqah125@gmail.com**")
    st.markdown("**Project**: Smart Energy Forecasting System — generates baseline and adjusted forecasts, cost & CO₂ comparisions, and exports reports.")
    st.markdown("Logo uploaded will appear in header and inside generated PDF (if provided).")

# Footer
st.markdown("---")
st.caption("Built for Polytechnic Kota Kinabalu — Smart Energy Forecasting System")

