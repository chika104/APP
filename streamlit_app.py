# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO

# ----------------------------
# Helper utilities
# ----------------------------
WATT_DEFAULTS_W = {
    "LED": 10,
    "CFL": 15,
    "Fluorescent": 40,
    "Computer": 150,
    "Lab Equipment": 500
}

def clean_string_numbers(s):
    """Remove common thousands separators and whitespace, return string safe for numeric conversion."""
    if pd.isna(s):
        return s
    return str(s).replace(",", "").strip()

def to_numeric_series(s):
    s2 = s.astype(str).apply(clean_string_numbers)
    return pd.to_numeric(s2, errors="coerce")

def detect_year_and_consumption(df):
    """Try to find year & consumption-like columns (returns normalized df with 'year' and 'consumption')."""
    cols_low = [c.lower().strip() for c in df.columns]
    year_idx = None
    cons_idx = None

    # find year column
    for i, c in enumerate(cols_low):
        if "year" in c or c in ("yr", "y"):
            year_idx = i
            break
    # fallback: look for integer-like column (4-digit)
    if year_idx is None:
        for i, c in enumerate(df.columns):
            sample = df[c].astype(str).head(10).str.replace(",", "").str.strip()
            if sample.str.match(r"^\d{4}$").sum() >= 1:
                year_idx = i
                break

    # find consumption-like column
    for i, c in enumerate(cols_low):
        if any(k in c for k in ["consum", "kwh", "energy", "usage", "total"]):
            cons_idx = i
            break

    if year_idx is None or cons_idx is None:
        return None

    # build normalized df
    out = pd.DataFrame()
    # clean year and consumption
    out["year"] = to_numeric_series(df.iloc[:, year_idx])
    out["consumption"] = to_numeric_series(df.iloc[:, cons_idx])

    # optional baseline cost column
    cost_idx = None
    for i, c in enumerate(cols_low):
        if any(k in c for k in ["cost", "rm", "price", "total_cost", "totalcost"]):
            cost_idx = i
            break
    if cost_idx is not None:
        out["baseline_cost"] = to_numeric_series(df.iloc[:, cost_idx])
    else:
        out["baseline_cost"] = np.nan

    # drop rows where year or consumption not parseable
    out = out.dropna(subset=["year", "consumption"]).reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    out["consumption"] = out["consumption"].astype(float)
    return out

def linear_forecast(years, values, n_future):
    """
    Forecast using linear regression on year (actual calendar year).
    years: array-like of ints (calendar year)
    values: corresponding consumption values
    returns: (years_all, baseline_forecast_full, future_years)
    """
    # ensure numeric arrays
    X = np.array(years).reshape(-1, 1)
    y = np.array(values).reshape(-1, 1)

    if len(X) == 0:
        return None

    if len(X) == 1:
        # cannot fit model — flat forecast equal to the single value
        last_year = int(X.flatten()[-1])
        years_hist = list(X.flatten())
        years_future = [last_year + i for i in range(1, n_future + 1)]
        hist_vals = list(y.flatten())
        future_vals = [float(hist_vals[-1])] * n_future
        all_years = years_hist + years_future
        all_vals = hist_vals + future_vals
        return all_years, all_vals, years_future

    model = LinearRegression()
    model.fit(X, y)
    # historical predicted (for smooth plotting)
    y_fit = model.predict(X).flatten().tolist()
    last_year = int(X.flatten()[-1])
    future_years = [last_year + i for i in range(1, n_future + 1)]
    X_future = np.array(future_years).reshape(-1, 1)
    y_future = model.predict(X_future).flatten().tolist()

    all_years = list(X.flatten()) + future_years
    all_vals = list(y_fit) + y_future
    return all_years, all_vals, future_years

def kwh_from_watts(watt_per_unit, units, hours_per_year):
    return (watt_per_unit * units * hours_per_year) / 1000.0

def build_excel_bytes(dfs_dict):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for name, df in dfs_dict.items():
                # safe sheet name
                sheet = name[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
            writer.close()
        return output.getvalue()
    except Exception as e:
        return None

# ----------------------------
# Streamlit layout & logic
# ----------------------------
st.set_page_config(page_title="Energy Forecast — Baseline vs Adjusted", layout="wide")
st.title("⚡ Energy Forecast — Baseline vs Adjusted Scenarios")

st.markdown(
    "Pilih input (Manual / CSV), tentukan faktor (penambahan/pengurangan) — dan lihat perbandingan **Baseline vs Adjusted** secara interaktif."
)

# Step 1: Input mode
st.header("Step 1 — Input data")
input_mode = st.radio("Pilih cara input:", ("Manual entry (default 2020-2024)", "Upload CSV"))

df_hist = None
if input_mode.startswith("Upload"):
    uploaded = st.file_uploader("Upload CSV (kolum: year & consumption) atau Excel (.xlsx)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            df_detect = detect_year_and_consumption(df_raw)
            if df_detect is None or df_detect.empty:
                st.error("CSV / Excel tidak mengandungi kolum 'year' & 'consumption' yang boleh dikesan. Sila semak fail.")
                st.stop()
            df_hist = df_detect.sort_values("year").reset_index(drop=True)
        except Exception as e:
            st.error("Gagal baca fail: " + str(e))
            st.stop()
else:
    st.info("Masukkan nilai historical untuk 5 tahun (contoh 2020–2024). Anda boleh ubah bilangan baris jika perlu.")
    n_rows = st.number_input("Berapa tahun historical?", min_value=1, max_value=20, value=5, step=1)
    years = []
    consumptions = []
    costs = []
    default_start = 2020
    for i in range(int(n_rows)):
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            y = st.number_input(f"Year #{i+1}", min_value=1900, max_value=2100, value=default_start + i, key=f"yr_{i}")
            years.append(int(y))
        with col2:
            c = st.number_input(f"Consumption kWh (Year {int(y)})", min_value=0.0, value=float((i+1)*1000000), step=1000.0, key=f"cons_{i}")
            consumptions.append(float(c))
        with col3:
            cost = st.number_input(f"Baseline cost RM (optional)", min_value=0.0, value=0.0, step=1.0, key=f"cost_{i}")
            costs.append(float(cost) if cost > 0 else np.nan)
    df_hist = pd.DataFrame({"year": years, "consumption": consumptions, "baseline_cost": costs})
    df_hist = df_hist.sort_values("year").reset_index(drop=True)

st.subheader("Historical data loaded")
st.dataframe(df_hist)

# Step 2: Factors
st.header("Step 2 — Define factors (addition / reduction)")
st.markdown("Masukkan nilai logik (unit, jam operasi setahun, dan pilih jenis lampu bila berkenaan). Perubahan dikira sebagai kWh/year.")

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Lampu 💡")
    lamp_action = st.selectbox("Tindakan", ("Addition", "Reduction"), key="lamp_action")
    lamp_type = st.selectbox("Jenis lampu", ("LED", "CFL", "Fluorescent"), key="lamp_type")
    lamp_units = st.slider("Bilangan unit (lamp)", min_value=0, max_value=1000, value=0, step=1, key="lamp_units")
    lamp_hours = st.slider("Jam operasi per tahun (jam)", min_value=0, max_value=8760, value=0, step=10, key="lamp_hours")

with col_r:
    st.subheader("Komputer 💻")
    comp_action = st.selectbox("Tindakan", ("Addition", "Reduction"), key="comp_action")
    comp_units = st.slider("Bilangan unit (computer)", min_value=0, max_value=1000, value=0, step=1, key="comp_units")
    comp_hours = st.slider("Jam operasi per tahun (jam)", min_value=0, max_value=8760, value=0, step=10, key="comp_hours")

st.markdown("---")
col_l2, col_r2 = st.columns(2)
with col_l2:
    st.subheader("Peralatan makmal ⚗️")
    lab_action = st.selectbox("Tindakan", ("Addition", "Reduction"), key="lab_action")
    lab_units = st.slider("Bilangan unit (lab eq)", min_value=0, max_value=200, value=0, step=1, key="lab_units")
    lab_hours = st.slider("Jam operasi per tahun (jam)", min_value=0, max_value=8760, value=0, step=10, key="lab_hours")

with col_r2:
    st.subheader("Operating hours (site-level) ⏱️")
    # user asked hours in hours (not in kWh) — so we ask hours and an average kW (site-level)
    site_hours = st.number_input("Extra/reduced hours per year (positive=add, negative=reduce)", min_value=-8760, max_value=8760, value=0, step=1, key="site_hours")
    site_avg_load_kw = st.number_input("Average site load for those hours (kW)", min_value=0.0, value=1.0, step=0.1, key="site_avg_kw")

# Compute kWh impact for each factor
lamp_w = WATT_DEFAULTS_W.get(lamp_type, 10)
lamp_kwh = kwh_from_watts(lamp_w, lamp_units, lamp_hours)
if lamp_action == "Reduction":
    lamp_kwh = -lamp_kwh

comp_kwh = kwh_from_watts(WATT_DEFAULTS_W["Computer"], comp_units, comp_hours)
if comp_action == "Reduction":
    comp_kwh = -comp_kwh

lab_kwh = kwh_from_watts(WATT_DEFAULTS_W["Lab Equipment"], lab_units, lab_hours)
if lab_action == "Reduction":
    lab_kwh = -lab_kwh

site_kwh = site_avg_load_kw * site_hours  # positive/negative as user entered

total_adjust_kwh = lamp_kwh + comp_kwh + lab_kwh + site_kwh

st.markdown("**Ringkasan faktor (kWh/year)**")
st.write(f"Lamp ({lamp_type}): {lamp_kwh:,.2f} kWh/year")
st.write(f"Computer: {comp_kwh:,.2f} kWh/year")
st.write(f"Lab equipment: {lab_kwh:,.2f} kWh/year")
st.write(f"Site hours effect: {site_kwh:,.2f} kWh/year")
if total_adjust_kwh >= 0:
    st.success(f"Net adjustment (additional): {total_adjust_kwh:,.2f} kWh/year")
else:
    st.success(f"Net adjustment (reduction): {abs(total_adjust_kwh):,.2f} kWh/year")

# Step 3: Forecast settings
st.header("Step 3 — Forecast settings")
n_forecast = st.number_input("Forecast years ahead", min_value=1, max_value=10, value=5, step=1)
tariff = st.number_input("Tariff (RM per kWh)", min_value=0.0, value=0.52, step=0.01)
co2_factor = st.number_input("CO₂ factor (kg CO₂ per kWh)", min_value=0.0, value=0.75, step=0.01)

# Prepare baseline historical cost if missing
if "baseline_cost" not in df_hist.columns:
    df_hist["baseline_cost"] = np.nan
# if baseline_cost is nan -> compute from tariff
df_hist["baseline_cost"] = df_hist["baseline_cost"].fillna(df_hist["consumption"] * tariff)
df_hist["baseline_co2_kg"] = df_hist["consumption"] * co2_factor

# Forecast baseline (linear on calendar year)
years_hist = df_hist["year"].tolist()
cons_hist = df_hist["consumption"].tolist()
res = linear_forecast(years_hist, cons_hist, int(n_forecast))
if res is None:
    st.error("Tiada data historical untuk forecast.")
    st.stop()
all_years, baseline_vals, future_years = res

# Build forecast dataframe
# historical predicted values are in baseline_vals for historical years too (we use observed historical consumption for plot for clarity)
future_mask = [y in future_years for y in all_years]
forecast_only_vals = [v for y, v in zip(all_years, baseline_vals) if y in future_years]

forecast_df = pd.DataFrame({
    "year": future_years,
    "baseline_consumption_kwh": forecast_only_vals
})
# adjusted forecast = baseline + total_adjust_kwh (applied equally to each future year)
forecast_df["adjusted_consumption_kwh"] = forecast_df["baseline_consumption_kwh"] + total_adjust_kwh

# costs & co2
forecast_df["baseline_cost_rm"] = forecast_df["baseline_consumption_kwh"] * tariff
forecast_df["adjusted_cost_rm"] = forecast_df["adjusted_consumption_kwh"] * tariff
forecast_df["baseline_co2_kg"] = forecast_df["baseline_consumption_kwh"] * co2_factor
forecast_df["adjusted_co2_kg"] = forecast_df["adjusted_consumption_kwh"] * co2_factor

forecast_df["saving_kwh"] = forecast_df["baseline_consumption_kwh"] - forecast_df["adjusted_consumption_kwh"]
forecast_df["saving_cost_rm"] = forecast_df["baseline_cost_rm"] - forecast_df["adjusted_cost_rm"]
forecast_df["saving_co2_kg"] = forecast_df["baseline_co2_kg"] - forecast_df["adjusted_co2_kg"]

# Step 4: Visualization — 4 graphs requested
st.header("Step 4 — Visual comparisons")

# Graph A: Baseline forecast (historical + future baseline)
fig_a = go.Figure()
# historical actual
fig_a.add_trace(go.Scatter(x=df_hist["year"], y=df_hist["consumption"],
                           mode="lines+markers", name="Historical (actual)", line=dict(color="#00BFFF")))
# baseline forecast (future)
fig_a.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"],
                           mode="lines+markers", name="Baseline forecast", line=dict(dash="dash", color="#FFA500")))
fig_a.update_layout(title="Baseline consumption (historical + baseline forecast)",
                    xaxis_title="Year", yaxis_title="kWh",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="white"))
st.plotly_chart(fig_a, use_container_width=True)

# Graph B: Forecast with factors (Baseline vs Adjusted)
fig_b = go.Figure()
fig_b.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_consumption_kwh"],
                           mode="lines+markers", name="Baseline forecast", line=dict(color="#FFA500")))
fig_b.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["adjusted_consumption_kwh"],
                           mode="lines+markers", name="Adjusted forecast", line=dict(color="#00FF7F")))
fig_b.update_layout(title="Forecast: Baseline vs Adjusted (factors applied)",
                    xaxis_title="Year", yaxis_title="kWh",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="white"))
st.plotly_chart(fig_b, use_container_width=True)

# Graph C: Cost trend (Baseline vs Adjusted)
fig_c = go.Figure()
# historical cost (computed earlier)
fig_c.add_trace(go.Bar(x=df_hist["year"], y=df_hist["baseline_cost"], name="Historical cost (RM)", marker_color="#3a7bd5"))
fig_c.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_cost_rm"], mode="lines+markers",
                           name="Baseline forecast cost (RM)", line=dict(color="#FFA500")))
fig_c.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["adjusted_cost_rm"], mode="lines+markers",
                           name="Adjusted forecast cost (RM)", line=dict(color="#00FF7F")))
fig_c.update_layout(title="Cost trend (RM): Baseline vs Adjusted",
                    xaxis_title="Year", yaxis_title="RM",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="white"))
st.plotly_chart(fig_c, use_container_width=True)

# Graph D: CO2 trend (Baseline vs Adjusted)
fig_d = go.Figure()
# historical co2
fig_d.add_trace(go.Bar(x=df_hist["year"], y=df_hist["baseline_co2_kg"], name="Historical CO₂ (kg)", marker_color="#6a11cb"))
fig_d.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["baseline_co2_kg"], mode="lines+markers",
                           name="Baseline forecast CO₂ (kg)", line=dict(color="#FFA500")))
fig_d.add_trace(go.Scatter(x=forecast_df["year"], y=forecast_df["adjusted_co2_kg"], mode="lines+markers",
                           name="Adjusted forecast CO₂ (kg)", line=dict(color="#00FF7F")))
fig_d.update_layout(title="CO₂ trend (kg): Baseline vs Adjusted",
                    xaxis_title="Year", yaxis_title="kg CO₂",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="white"))
st.plotly_chart(fig_d, use_container_width=True)

# Step 5: Summary metrics & download
st.header("Step 5 — Summary & Export")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total baseline kWh (forecast period)", f"{forecast_df['baseline_consumption_kwh'].sum():,.0f}")
col2.metric("Total adjusted kWh (forecast period)", f"{forecast_df['adjusted_consumption_kwh'].sum():,.0f}")
col3.metric("Total kWh saved (forecast period)", f"{forecast_df['saving_kwh'].sum():,.0f}")
col4.metric("Total cost saved (RM)", f"RM {forecast_df['saving_cost_rm'].sum():,.2f}")

st.write("Total CO₂ reduction (kg) (forecast period):", f"{forecast_df['saving_co2_kg'].sum():,.0f} kg")

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

# Downloads: excel with historical, factors summary, forecast
factors_summary = pd.DataFrame([
    {"factor": f"{lamp_type} lamp", "kwh_per_year": lamp_kwh, "action": lamp_action, "units": lamp_units, "hours": lamp_hours},
    {"factor": "Computer", "kwh_per_year": comp_kwh, "action": comp_action, "units": comp_units, "hours": comp_hours},
    {"factor": "Lab equipment", "kwh_per_year": lab_kwh, "action": lab_action, "units": lab_units, "hours": lab_hours},
    {"factor": "Site hours", "kwh_per_year": site_kwh, "action": "Site-level", "units": "", "hours": site_hours}
])

all_dfs = {
    "historical": df_hist,
    "factors": factors_summary,
    "forecast": forecast_df
}
excel_bytes = build_excel_bytes(all_dfs)
if excel_bytes:
    st.download_button("Download results (.xlsx)", data=excel_bytes,
                       file_name="energy_forecast_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Excel export tidak tersedia (pakej xlsxwriter mungkin tiada).")

st.success("Selesai — ubah nilai faktor (slider) untuk lihat perubahan secara langsung pada graf & ringkasan.")
