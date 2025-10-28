# streamlit_app.py — fixed monthly dataset structure
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime

# optional
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except:
    MYSQL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

# --------------------
# DATABASE CONFIG
# --------------------
DB_CONFIG = {
    "host": "switchback.proxy.rlwy.net",
    "port": 55398,
    "user": "root",
    "password": "<YOUR_RAILWAY_PASSWORD>",
    "database": "railway"
}

# --------------------
# STYLE PERSISTENCE
# --------------------
DEFAULT_STYLE = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.8);
}
</style>
"""

if "bg_style" not in st.session_state:
    st.session_state.bg_style = DEFAULT_STYLE
st.markdown(st.session_state.bg_style, unsafe_allow_html=True)

# --------------------
# DATABASE FUNCTIONS
# --------------------
def connect_db():
    if not MYSQL_AVAILABLE:
        raise Exception("MySQL connector not installed.")
    return mysql.connector.connect(**DB_CONFIG)

def init_db_tables(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS energy_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        month VARCHAR(20),
        year INT,
        kwh FLOAT,
        cost FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forecast_reports (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        pdf LONGBLOB
    )""")
    conn.commit()

def save_data_to_db(conn, df):
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute(
            "INSERT INTO energy_data (month, year, kwh, cost) VALUES (%s,%s,%s,%s)",
            (r["MONTH"], int(r["YEAR"]), float(r["kWh"]), float(r["Cost_RM"]))
        )
    conn.commit()

def save_pdf_to_db(conn, filename, pdf_bytes):
    cur = conn.cursor()
    cur.execute("INSERT INTO forecast_reports (filename, pdf) VALUES (%s,%s)", (filename, pdf_bytes))
    conn.commit()

def get_pdf_history(conn):
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, filename, created_at FROM forecast_reports ORDER BY created_at DESC")
    return cur.fetchall()

# --------------------
# PARSE DATASET
# --------------------
def parse_monthly_two_row_header_csv(file):
    df_raw = pd.read_csv(file, header=[0, 1])
    # flatten header
    df_raw.columns = ['_'.join(map(str, col)).strip().replace(" ", "_") for col in df_raw.columns.values]
    df = pd.DataFrame()

    for year in [2019, 2020, 2021, 2022, 2023]:
        kwh_col = f'Baseline_{year}_kWh'
        rm_col = f'Baseline_{year}_RM(kWh)'
        # handle flexible naming
        matches = [c for c in df_raw.columns if str(year) in c and "kWh" in c and "RM(kWh)" not in c]
        if matches:
            kwh_col = matches[0]
        if rm_col not in df_raw.columns:
            candidates = [c for c in df_raw.columns if str(year) in c and "RM(kWh)" in c]
            if candidates:
                rm_col = candidates[0]
        if kwh_col in df_raw.columns and rm_col in df_raw.columns:
            temp = pd.DataFrame({
                "MONTH": df_raw.filter(like="MONTH").iloc[:, 0],
                "YEAR": year,
                "kWh": pd.to_numeric(df_raw[kwh_col], errors="coerce"),
                "Cost_RM": pd.to_numeric(df_raw[rm_col], errors="coerce")
            })
            df = pd.concat([df, temp], ignore_index=True)

    df["MONTH"] = df["MONTH"].astype(str).str.strip()
    return df.dropna(subset=["kWh"])

# --------------------
# PDF MAKER
# --------------------
def make_pdf_bytes(title, summary, df):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.2 * inch)]
    for s in summary:
        story.append(Paragraph(s, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    tbl = Table([df.columns.tolist()] + df.values.tolist())
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(tbl)
    doc.build(story)
    buf.seek(0)
    return buf.read()

# --------------------
# SIDEBAR NAV
# --------------------
menu = st.sidebar.radio("📍 Menu", ["📈 Energy Forecast", "📊 Reports", "⚙️ Settings", "❓ Help"])

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = pd.DataFrame()

# --------------------
# ENERGY FORECAST
# --------------------
if menu == "📈 Energy Forecast":
    st.title("📈 Energy Forecast Dashboard")
    file = st.file_uploader("Upload CSV (two-row header)", type=["csv"])

    if file:
        df = parse_monthly_two_row_header_csv(file)
        st.session_state.uploaded_data = df
        st.dataframe(df.head())

        fig = px.line(df, x="MONTH", y="kWh", color="YEAR", title="Monthly Energy Usage (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        # forecast
        growth = df.groupby("YEAR")["kWh"].sum().pct_change().mean()
        next_year = df["YEAR"].max() + 1
        last_year_df = df[df["YEAR"] == df["YEAR"].max()].copy()
        forecast_df = last_year_df.copy()
        forecast_df["YEAR"] = next_year
        forecast_df["kWh"] = forecast_df["kWh"] * (1 + growth)
        forecast_df["Cost_RM"] = forecast_df["Cost_RM"] * (1 + growth)
        st.session_state.forecast_df = forecast_df

        st.subheader("Forecast Preview")
        st.dataframe(forecast_df)

        if st.button("💾 Save Dataset to Database"):
            try:
                conn = connect_db()
                init_db_tables(conn)
                save_data_to_db(conn, df)
                conn.close()
                st.success("Data saved to Railway DB!")
            except Exception as e:
                st.error(f"DB Error: {e}")

        if REPORTLAB_AVAILABLE:
            summary = [f"Forecast for {next_year}", f"Avg growth rate: {growth*100:.2f}%"]
            pdf = make_pdf_bytes("Energy Forecast Report", summary, forecast_df)
            fname = f"forecast_{next_year}.pdf"
            st.download_button("📄 Download PDF", data=pdf, file_name=fname, mime="application/pdf")
            if st.button("🧾 Save PDF to DB"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_pdf_to_db(conn, fname, pdf)
                    conn.close()
                    st.success("PDF saved in database.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Install reportlab for PDF export.")

# --------------------
# REPORTS
# --------------------
elif menu == "📊 Reports":
    st.title("📊 Reports History")
    try:
        conn = connect_db()
        init_db_tables(conn)
        rows = get_pdf_history(conn)
        conn.close()
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No reports found in DB.")
    except Exception as e:
        st.error(f"DB Error: {e}")

# --------------------
# SETTINGS
# --------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Theme & Database")
    theme = st.radio("Theme:", ["Dark", "Light", "Custom Image URL"])
    if theme == "Dark":
        st.session_state.bg_style = DEFAULT_STYLE
        st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
    elif theme == "Light":
        style = """<style>
        [data-testid="stAppViewContainer"] {background-color: #fff; color: #000;}
        [data-testid="stSidebar"] {background-color: rgba(0,0,0,0.05);}
        </style>"""
        st.session_state.bg_style = style
        st.markdown(style, unsafe_allow_html=True)
    else:
        img_url = st.text_input("Enter Image URL:")
        if img_url:
            custom = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url('{img_url}');
                background-size: cover;
                color: white;
            }}
            </style>"""
            st.session_state.bg_style = custom
            st.markdown(custom, unsafe_allow_html=True)
    st.success("Theme saved & persistent across menus.")

    st.markdown("**Railway DB Config (default):**")
    st.write(DB_CONFIG)

# --------------------
# HELP
# --------------------
elif menu == "❓ Help":
    st.title("❓ Help & About")
    st.markdown("""
    Developed by Aiman for Chika — Politeknik Kota Kinabalu  
    Forecast, visualize & export energy reports.  
    All data saved on Railway MySQL.
    """)
