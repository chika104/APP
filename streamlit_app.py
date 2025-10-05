import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from io import BytesIO
import xlsxwriter

# ==============================
# 🧭 APP CONFIGURATION
# ==============================
st.set_page_config(page_title="Energy Forecast Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ Energy Forecast Dashboard")
st.write("Selamat datang ke aplikasi ramalan tenaga. Anda boleh pilih untuk **upload dataset CSV** atau **masukkan data manual** untuk meramal penggunaan tenaga.")

# ==============================
# 🔧 PILIHAN INPUT
# ==============================
option = st.radio(
    "Pilih cara untuk masukkan data:",
    ["Upload CSV", "Masukkan Data Manual"]
)

# ==============================
# 🧮 FUNGSI RAMALAN
# ==============================
def run_forecast(df):
    # Pastikan dataset ada dua kolum minimum
    if df.shape[1] < 2:
        st.error("Dataset mesti ada sekurang-kurangnya dua kolum (contoh: 'Masa' dan 'Tenaga').")
        return None

    df.columns = ['Time', 'Energy']  # standardize columns
    df = df.dropna()

    # Feature & target
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Energy'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # Ramalan masa depan (5 step)
    future_steps = 5
    future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
    future_pred = model.predict(future_X)

    # Gabungkan hasil ramalan
    future_df = pd.DataFrame({
        'Time': [f'Future {i+1}' for i in range(future_steps)],
        'Predicted Energy': future_pred
    })

    # Paparan graf
    fig = px.line(df, x='Time', y='Energy', title="📊 Data Sebenar vs Ramalan")
    fig.add_scatter(x=df['Time'], y=y_pred, mode='lines', name='Predicted (Training)', line=dict(dash='dot'))
    fig.add_scatter(x=future_df['Time'], y=future_df['Predicted Energy'], mode='lines+markers', name='Forecast (Next 5)', line=dict(color='orange'))

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Model Linear Regression siap dilatih — MSE: {mse:.4f}")

    # Paparan hasil
    st.subheader("📈 Hasil Ramalan:")
    st.dataframe(future_df)

    # Fungsi muat turun hasil ke Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data Asal', index=False)
        future_df.to_excel(writer, sheet_name='Ramalan', index=False)
    st.download_button(
        label="💾 Muat Turun Hasil Ramalan (Excel)",
        data=output.getvalue(),
        file_name="energy_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================
# 📂 UPLOAD CSV
# ==============================
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Muat naik fail CSV anda", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Pratonton Data")
            st.dataframe(df.head())
            run_forecast(df)
        except Exception as e:
            st.error(f"Ralat semasa membaca fail CSV: {e}")

# ==============================
# ✍️ INPUT MANUAL
# ==============================
elif option == "Masukkan Data Manual":
    st.subheader("Masukkan Data Penggunaan Tenaga")
    n = st.number_input("Berapa banyak rekod yang ingin dimasukkan?", min_value=3, max_value=50, value=5, step=1)

    manual_data = []
    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            time = st.text_input(f"Masa {i+1}", value=f"T{i+1}")
        with col2:
            energy = st.number_input(f"Tenaga {i+1} (kWh)", value=float(i+1) * 10.0)
        manual_data.append({"Time": time, "Energy": energy})

    if st.button("Jalankan Ramalan 🔮"):
        df_manual = pd.DataFrame(manual_data)
        st.subheader("📊 Data Manual Dimasukkan")
        st.dataframe(df_manual)
        run_forecast(df_manual)
