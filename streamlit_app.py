import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from io import BytesIO
import xlsxwriter
import time

# ==============================
# ⚙️ PAGE CONFIG
# ==============================
st.set_page_config(page_title="Energy Management Dashboard", page_icon="⚡", layout="wide")

# ==============================
# 🎨 CUSTOM CSS FOR PROFESSIONAL UI
# ==============================
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #00BFFF;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #A9A9A9;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #00BFFF;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🌐 SIDEBAR MENU
# ==============================
menu = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Dashboard", "📈 Energy Forecast", "💡 Device Management", "📊 Reports", "⚙️ Settings", "❓ Help & About"]
)

# ==============================
# 🧠 FUNCTION: FORECAST MODEL
# ==============================
def run_forecast(df):
    df.columns = ['Time', 'Energy']
    df = df.dropna()

    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Energy'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # future predictions
    future_steps = 5
    future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
    future_pred = model.predict(future_X)
    future_df = pd.DataFrame({
        'Time': [f'Future {i+1}' for i in range(future_steps)],
        'Predicted Energy': future_pred
    })

    fig = px.line(df, x='Time', y='Energy', title="Energy Usage and Forecast",
                  markers=True, line_shape='spline', color_discrete_sequence=['#00BFFF'])
    fig.add_scatter(x=df['Time'], y=y_pred, mode='lines', name='Predicted (Train)', line=dict(dash='dot', color='orange'))
    fig.add_scatter(x=future_df['Time'], y=future_df['Predicted Energy'], mode='lines+markers',
                    name='Forecast (Next 5)', line=dict(color='#FFD700'))

    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        title_font=dict(size=22),
        transition_duration=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Model siap dilatih — MSE: {mse:.4f}")

    with st.expander("📊 Hasil Ramalan"):
        st.dataframe(future_df)

    # Export option
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
# 🏠 DASHBOARD
# ==============================
if menu == "🏠 Dashboard":
    st.markdown('<p class="main-title">⚡ Energy Management Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pantau, ramal dan kawal penggunaan tenaga dengan mudah.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Devices", "12", "+2")
    with col2:
        st.metric("Average Daily Usage (kWh)", "34.5", "-1.2")
    with col3:
        st.metric("Efficiency", "89%", "+4.1%")

    st.markdown("---")

    # Simulated energy trend chart
    time.sleep(0.3)
    df = pd.DataFrame({
        "Time": pd.date_range(start="2025-09-01", periods=30),
        "Usage": np.random.randint(20, 50, 30)
    })
    fig = px.area(df, x="Time", y="Usage", title="Energy Consumption Trend (Last 30 Days)",
                  color_discrete_sequence=['#00BFFF'])
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        transition_duration=800
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 📈 ENERGY FORECAST
# ==============================
elif menu == "📈 Energy Forecast":
    st.markdown('<p class="main-title">📈 Energy Forecast Module</p>', unsafe_allow_html=True)
    option = st.radio("Pilih sumber data:", ["Upload CSV", "Masukkan Data Manual"])

    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Muat naik dataset CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Pratonton Data")
            st.dataframe(df.head())
            run_forecast(df)

    else:
        n = st.number_input("Bilangan rekod:", min_value=3, max_value=50, value=5, step=1)
        manual_data = []
        for i in range(n):
            col1, col2 = st.columns(2)
            with col1:
                t = st.text_input(f"Masa {i+1}", value=f"T{i+1}")
            with col2:
                e = st.number_input(f"Tenaga {i+1} (kWh)", value=float(i+1)*10)
            manual_data.append({"Time": t, "Energy": e})

        if st.button("🔮 Jalankan Ramalan"):
            df_manual = pd.DataFrame(manual_data)
            run_forecast(df_manual)

# ==============================
# 💡 DEVICE MANAGEMENT
# ==============================
elif menu == "💡 Device Management":
    st.markdown('<p class="main-title">💡 Device Management</p>', unsafe_allow_html=True)
    st.info("Modul ini digunakan untuk memantau dan mengawal peranti tenaga.")
    device_list = ["Air Conditioner", "Water Heater", "Lights", "Fridge"]
    selected_device = st.selectbox("Pilih peranti untuk dipantau:", device_list)
    st.write(f"Status semasa {selected_device}: ✅ Aktif")
    st.progress(np.random.randint(40, 100))

# ==============================
# 📊 REPORTS
# ==============================
elif menu == "📊 Reports":
    st.markdown('<p class="main-title">📊 Energy Reports</p>', unsafe_allow_html=True)
    st.info("Bahagian ini memaparkan laporan prestasi penggunaan tenaga.")
    chart_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Usage": np.random.randint(100, 300, 5)
    })
    fig = px.bar(chart_data, x="Month", y="Usage", text="Usage", color="Usage", color_continuous_scale="Blues")
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        transition_duration=800
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# ⚙️ SETTINGS
# ==============================
elif menu == "⚙️ Settings":
    st.markdown('<p class="main-title">⚙️ Application Settings</p>', unsafe_allow_html=True)
    theme = st.selectbox("Pilih tema warna:", ["Dark Mode", "Light Mode", "Auto"])
    st.toggle("Notifikasi Harian", value=True)
    st.toggle("Auto Update", value=False)
    st.success("✅ Tetapan disimpan secara automatik.")

# ==============================
# ❓ HELP & ABOUT
# ==============================
elif menu == "❓ Help & About":
    st.markdown('<p class="main-title">❓ Help & About</p>', unsafe_allow_html=True)
    st.write("""
        **Energy Forecast Dashboard v2.0**  
        Dibangunkan oleh **Chika @ Polytechnic Kota Kinabalu**  
        Projek ini memanfaatkan **Machine Learning (Linear Regression)** untuk menjana ramalan tenaga.
    """)
    st.markdown("💬 *Untuk bantuan lanjut, hubungi penyelaras projek atau lihat dokumentasi penuh di GitHub.*")
