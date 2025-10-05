import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from io import BytesIO
import xlsxwriter

# ==============================================
# ⚙️ PAGE CONFIG
# ==============================================
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ==============================================
# 🌈 CUSTOM CSS (Professional Look)
# ==============================================
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
        font-family: "Poppins", sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #fafafa !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# 🧭 SIDEBAR NAVIGATION
# ==============================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4727/4727487.png", width=100)
st.sidebar.title("⚡ Dashboard Navigation")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ["🏠 Utama", "📂 Upload Dataset", "✍️ Input Manual", "📊 Hasil Ramalan"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Dibangunkan oleh Chika & Aiman")
st.sidebar.markdown("© 2025 Energy Forecast Project")

# ==============================================
# 📈 FUNGSI RAMALAN
# ==============================================
def run_forecast(df):
    df.columns = ['Time', 'Energy']
    df = df.dropna()

    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Energy'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # Ramalan 5 langkah ke depan
    future_steps = 5
    future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
    future_pred = model.predict(future_X)

    future_df = pd.DataFrame({
        'Time': [f'Future {i+1}' for i in range(future_steps)],
        'Predicted Energy': future_pred
    })

    # ====================
    # GRAF ANIMATED
    # ====================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['Energy'],
        mode='lines+markers',
        name='Data Sebenar',
        line=dict(color='deepskyblue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'], y=y_pred,
        mode='lines',
        name='Ramalan (Training)',
        line=dict(color='lime', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=future_df['Time'], y=future_df['Predicted Energy'],
        mode='lines+markers',
        name='Forecast (Next 5)',
        line=dict(color='orange', width=3)
    ))

    fig.update_layout(
        title="📊 Ramalan Penggunaan Tenaga (Interaktif)",
        template="plotly_dark",
        hovermode="x unified",
        transition_duration=800,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Model siap dilatih — MSE: **{mse:.4f}**")

    st.subheader("📘 Hasil Ramalan:")
    st.dataframe(future_df)

    # Export Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data Asal', index=False)
        future_df.to_excel(writer, sheet_name='Ramalan', index=False)

    st.download_button(
        label="💾 Muat Turun (Excel)",
        data=output.getvalue(),
        file_name="energy_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    return future_df

# ==============================================
# 🏠 PAGE 1: UTAMA
# ==============================================
if menu == "🏠 Utama":
    st.title("⚡ Energy Forecast Dashboard")
    st.markdown("""
        Selamat datang ke **Energy Forecast Dashboard** 💡  
        Aplikasi ini membantu anda menganalisis dan meramal penggunaan tenaga dengan model **Machine Learning (Linear Regression)**.
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972397.png", width=250)
    st.markdown("""
        ### 🌟 Fungsi utama:
        - 📂 Upload dataset CSV sebenar
        - ✍️ Masukkan data secara manual
        - 📊 Lihat graf interaktif & hasil ramalan
        - 💾 Muat turun laporan dalam Excel
    """)

# ==============================================
# 📂 PAGE 2: UPLOAD CSV
# ==============================================
elif menu == "📂 Upload Dataset":
    st.title("📂 Muat Naik Dataset CSV")
    uploaded_file = st.file_uploader("Pilih fail CSV untuk dianalisis", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Pratonton Data")
            st.dataframe(df.head())
            st.session_state["uploaded_df"] = df
        except Exception as e:
            st.error(f"⚠️ Ralat semasa membaca fail: {e}")

# ==============================================
# ✍️ PAGE 3: INPUT MANUAL
# ==============================================
elif menu == "✍️ Input Manual":
    st.title("✍️ Masukkan Data Manual")
    n = st.number_input("Berapa banyak rekod yang ingin dimasukkan?", 3, 50, 5, step=1)

    manual_data = []
    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            time = st.text_input(f"Masa {i+1}", value=f"T{i+1}")
        with col2:
            energy = st.number_input(f"Tenaga {i+1} (kWh)", value=float(i+1)*10)
        manual_data.append({"Time": time, "Energy": energy})

    if st.button("🚀 Jalankan Ramalan"):
        df_manual = pd.DataFrame(manual_data)
        st.session_state["manual_df"] = df_manual
        st.success("✅ Data manual berjaya disimpan! Pergi ke halaman '📊 Hasil Ramalan' untuk lihat hasil.")

# ==============================================
# 📊 PAGE 4: HASIL RAMALAN
# ==============================================
elif menu == "📊 Hasil Ramalan":
    st.title("📊 Hasil Ramalan")
    if "uploaded_df" in st.session_state:
        st.info("Dataset digunakan: **Fail CSV yang dimuat naik**")
        run_forecast(st.session_state["uploaded_df"])
    elif "manual_df" in st.session_state:
        st.info("Dataset digunakan: **Data manual**")
        run_forecast(st.session_state["manual_df"])
    else:
        st.warning("⚠️ Tiada data dimasukkan lagi. Sila pergi ke halaman 'Upload Dataset' atau 'Input Manual'.")
