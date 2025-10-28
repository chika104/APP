import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Data Dashboard", layout="wide")

st.title("⚡ Monthly Energy Usage Dashboard")
st.markdown("Upload your monthly energy usage CSV (with multi-row headers).")

# -------------------------------
# CSV PARSER FUNCTION (FIXED)
# -------------------------------
def parse_monthly_two_row_header_csv(file):
    try:
        df_raw = pd.read_csv(file, header=[0, 1])
    except Exception:
        # fallback if only single header
        df_raw = pd.read_csv(file, header=0)
    df_raw.columns = [
        "_".join(map(str, col)).strip().replace(" ", "_") for col in df_raw.columns.values
    ]

    st.write("🧩 Columns detected:\n", list(df_raw.columns))

    # --- Identify the MONTH column more flexibly ---
    month_col_candidates = [
        c for c in df_raw.columns if "MONTH" in c.upper() or "BULAN" in c.upper()
    ]
    if not month_col_candidates:
        st.warning(f"⚠️ No 'MONTH' column found — using first column: {df_raw.columns[0]}")
        month_col_candidates = [df_raw.columns[0]]

    month_col = month_col_candidates[0]
    df_raw = df_raw.rename(columns={month_col: "MONTH"})

    # --- Extract data for each year ---
    df = pd.DataFrame()
    for year in [2019, 2020, 2021, 2022, 2023]:
        kwh_col = None
        cost_col = None

        for c in df_raw.columns:
            if str(year) in c and "KWH" in c.upper() and "RM" not in c.upper():
                kwh_col = c
            if str(year) in c and ("RM(TOTAL)" in c.upper() or "RMTOTAL" in c.upper()):
                cost_col = c

        if kwh_col and cost_col:
            temp = pd.DataFrame({
                "MONTH": df_raw["MONTH"],
                "YEAR": year,
                "kWh": pd.to_numeric(df_raw[kwh_col], errors="coerce"),
                "Cost_RM": pd.to_numeric(df_raw[cost_col], errors="coerce")
            })
            df = pd.concat([df, temp], ignore_index=True)

    df["MONTH"] = df["MONTH"].astype(str).str.strip()
    df = df.dropna(subset=["kWh"])
    return df


# -------------------------------
# MAIN APP
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = parse_monthly_two_row_header_csv(uploaded_file)

        st.success("✅ File successfully processed!")
        st.dataframe(df)

        # --- Plotting ---
        st.subheader("📊 Monthly Energy Usage (kWh)")
        fig, ax = plt.subplots(figsize=(10, 5))
        for year in df["YEAR"].unique():
            subset = df[df["YEAR"] == year]
            ax.plot(subset["MONTH"], subset["kWh"], marker="o", label=str(year))
        ax.set_xlabel("Month")
        ax.set_ylabel("Energy Usage (kWh)")
        ax.set_title("Monthly Energy Usage by Year")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # --- Cost Chart ---
        st.subheader("💰 Monthly Energy Cost (RM)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for year in df["YEAR"].unique():
            subset = df[df["YEAR"] == year]
            ax2.plot(subset["MONTH"], subset["Cost_RM"], marker="o", label=str(year))
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Cost (RM)")
        ax2.set_title("Monthly Energy Cost by Year")
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # --- Summary ---
        st.subheader("📈 Yearly Summary")
        summary = df.groupby("YEAR")[["kWh", "Cost_RM"]].sum().reset_index()
        st.dataframe(summary)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
