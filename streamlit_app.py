# streamlit_app.py
# Smart Energy Forecasting System (Enhanced Persistent Version)
# Author: Aiman for Chika (Politeknik Kota Kinabalu)
# Last updated: Oct 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import base64
import os

# --- Optional libraries ---
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


# -------------------------------
# CONSTANT SETTINGS
# -------------------------------
DB_CONFIG = {
    "host": "switchback.proxy.rlwy.net",
    "port": 55398,
    "user": "root",
    "password": "<YOUR_RAILWAY_PASSWORD>",
    "database": "railway"
}

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

# Apply persistent theme
if "bg_mode" not in st.session_state:
    st.session_state.bg_mode = "Dark"
if "bg_style" not in st.session_state:
    st.session_state.bg_style = DEFAULT_STYLE
st.markdown(st.session_state.bg_style, unsafe_allow_html=True)


# -------------------------------
# DATABASE FUNCTIONS
# -------------------------------
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
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pdf LONGBLOB
        )
    """)
    conn.commit()


def save_data_to_db(conn, df):
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO energy_data (month, year, kwh, cost)
            VALUES (%s, %s, %s, %s)
        """, (row['MONTH'], row['YEAR'], row['kWh'], row['Cost_RM']))
    conn.commit()


def save_pdf_to_db(conn, filename, pdf_bytes):
    cur = conn.cursor()
    cur.execute("INSERT INTO forecast_reports (filename, pdf) VALUES (%s, %s)", (filename, pdf_bytes))
    conn.commit()


def get_pdf_history(conn):
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, filename, created_at FROM forecast_reports ORDER BY created_at DESC")
    return cur.fetchall()


# -------------------------------
# DATASET PARSER
# -------------------------------
def parse_monthly_two_row_header_csv(uploaded_file):
    df_raw = pd.read_csv(uploaded_file, header=[0, 1])
    df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
    df = pd.DataFrame()

    for year in [2019, 2020, 2021, 2022, 2023]:
        kwh_col = f'Baseline {year}_kWh'
        rm_col = f'Baseline {year}_RM(kWh)'
        if kwh_col in df_raw.columns and rm_col in df_raw.columns:
            tmp = pd.DataFrame({
                'MONTH': df_raw['MONTH_'],
                'YEAR': year,
                'kWh': df_raw[kwh_col],
                'Cost_RM': df_raw[rm_col]
            })
            df = pd.concat([df, tmp], ignore_index=True)
    return df


# -------------------------------
# PDF REPORT GENERATION
# -------------------------------
def make_pdf_bytes(title, summary_lines, df, images=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles['Title']), Spacer(1, 0.2 * inch)]

    for line in summary_lines:
        story.append(Paragraph(line, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    table_data = [df.columns.tolist()] + df.values.tolist()
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(tbl)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# -------------------------------
# STREAMLIT MAIN APP
# -------------------------------
st.sidebar.title("🔋 Smart Energy Forecasting")
menu = st.sidebar.radio("Navigation", ["📈 Energy Forecast", "📊 Reports", "⚙️ Settings", "❓ Help & About"])

# Persistent DataFrames
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = pd.DataFrame()

# -------------------------------
# ENERGY FORECAST PAGE
# -------------------------------
if menu == "📈 Energy Forecast":
    st.title("📈 Energy Forecast Dashboard")
    uploaded = st.file_uploader("Upload monthly dataset (two-row header CSV)", type=["csv"])
    if uploaded:
        df = parse_monthly_two_row_header_csv(uploaded)
        st.session_state.uploaded_data = df
        st.success("Dataset loaded successfully!")

        # Visualization
        fig = px.line(df, x="MONTH", y="kWh", color="YEAR", title="Monthly Energy Usage (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        # Forecast simulation (simple trend)
        avg_growth = df.groupby("YEAR")["kWh"].sum().pct_change().mean()
        future_year = df["YEAR"].max() + 1
        forecast_df = df[df["YEAR"] == df["YEAR"].max()].copy()
        forecast_df["YEAR"] = future_year
        forecast_df["kWh"] = forecast_df["kWh"] * (1 + avg_growth)
        forecast_df["Cost_RM"] = forecast_df["Cost_RM"] * (1 + avg_growth)

        st.session_state.forecast_df = forecast_df

        st.subheader("Forecast Preview")
        st.dataframe(forecast_df)

        # Save to DB
        if st.button("💾 Save dataset to Database"):
            try:
                conn = connect_db()
                init_db_tables(conn)
                save_data_to_db(conn, df)
                st.success("Data successfully saved to Railway database!")
                conn.close()
            except Exception as e:
                st.error(f"Database error: {e}")

        # Export PDF
        if REPORTLAB_AVAILABLE:
            summary = [
                f"Forecast generated for year {future_year}",
                f"Average growth rate: {avg_growth*100:.2f}%"
            ]
            pdf_bytes = make_pdf_bytes("Smart Energy Forecast Report", summary, forecast_df)
            filename = f"energy_forecast_{future_year}.pdf"
            st.download_button("📄 Download Forecast PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")

            # Save PDF to DB
            if st.button("🧾 Save Report to Database"):
                try:
                    conn = connect_db()
                    init_db_tables(conn)
                    save_pdf_to_db(conn, filename, pdf_bytes)
                    conn.close()
                    st.success("PDF report saved to database!")
                except Exception as e:
                    st.error(f"Error saving PDF: {e}")
        else:
            st.warning("Install 'reportlab' for PDF export support.")


# -------------------------------
# REPORT HISTORY PAGE
# -------------------------------
elif menu == "📊 Reports":
    st.title("📊 Saved Reports History")
    try:
        conn = connect_db()
        init_db_tables(conn)
        rows = get_pdf_history(conn)
        conn.close()
        if rows:
            df_reports = pd.DataFrame(rows)
            st.dataframe(df_reports)
        else:
            st.info("No reports found in the database yet.")
    except Exception as e:
        st.error(f"Database error: {e}")


# -------------------------------
# SETTINGS PAGE
# -------------------------------
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings — Theme & Database")

    choice = st.radio("Theme:", ["Dark (default)", "Light", "Custom Image URL"])
    if choice == "Dark (default)":
        st.session_state.bg_style = DEFAULT_STYLE
        st.session_state.bg_mode = "Dark"
        st.markdown(DEFAULT_STYLE, unsafe_allow_html=True)
    elif choice == "Light":
        light_style = """
        <style>
        [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #000000;}
        [data-testid="stSidebar"] {background-color: rgba(0,0,0,0.05);}
        </style>
        """
        st.session_state.bg_style = light_style
        st.session_state.bg_mode = "Light"
        st.markdown(light_style, unsafe_allow_html=True)
    else:
        img_url = st.text_input("Enter background image URL:")
        if img_url:
            custom_style = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{img_url}");
                background-size: cover;
                background-position: center;
                color: white;
            }}
            [data-testid="stSidebar"] {{background-color: rgba(0,0,0,0.6);}}
            </style>
            """
            st.session_state.bg_style = custom_style
            st.session_state.bg_mode = "Custom"
            st.markdown(custom_style, unsafe_allow_html=True)

    st.success("Theme will persist even after switching menus.")

    st.markdown("---")
    st.subheader("Database Configuration (Default: Railway)")
    st.write(f"**Host:** {DB_CONFIG['host']}")
    st.write(f"**Port:** {DB_CONFIG['port']}")
    st.write(f"**User:** {DB_CONFIG['user']}")
    st.write("**Database:** railway")


# -------------------------------
# HELP PAGE
# -------------------------------
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.markdown("""
    **Smart Energy Forecasting System**  
    Developed by Aiman for Chika (Politeknik Kota Kinabalu).  

    💡 Upload your monthly dataset to generate forecasts, visualize energy trends, and export PDF reports.  
    💾 All data and reports are saved in the Railway database for persistence.  

    📧 Support: chikaenergyforecast@gmail.com
    """)
