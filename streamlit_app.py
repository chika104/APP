# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background: #0f1116;
        color: #fff;
    }
    h1, h2, h3 {
        color: #00c8ff;
    }
    .stButton>button {
        background-color: #00c8ff;
        color: white;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0093cc;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("⚙️ App Navigation")
menu = st.sidebar.radio("Select Page:", [
    "🏠 Dashboard",
    "🔋 Energy Forecast",
    "💡 Device Management",
    "📊 Reports",
    "⚙️ Settings",
    "❓ Help & About"
])

# --- DASHBOARD PAGE ---
if menu == "🏠 Dashboard":
    st.title("⚡ Smart Energy Dashboard")
    st.markdown("Welcome **Chika!** 👋 This is your live energy monitoring and forecasting platform.")
    
    # Sample summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Usage (kWh)", "245", "+3%")
    col2.metric("Predicted Tomorrow (kWh)", "252", "↑ 7")
    col3.metric("Monthly Cost (RM)", "128.50", "↓ RM2.30")
    col4.metric("CO₂ Emissions (kg)", "56.3", "+0.5%")
    
    st.markdown("---")
    st.subheader("📈 Energy Usage Over Time")
    
    date_rng = pd.date_range(datetime.now() - timedelta(days=30), periods=30)
    usage = np.random.randint(200, 280, 30)
    fig = px.line(x=date_rng, y=usage, markers=True, title="Energy Usage (Last 30 Days)", 
                  labels={'x': 'Date', 'y': 'Energy (kWh)'})
    fig.update_traces(line_color='#00c8ff')
    st.plotly_chart(fig, use_container_width=True)


# --- ENERGY FORECAST PAGE ---
elif menu == "🔋 Energy Forecast":
    st.title("🔋 Energy Forecasting System")

    # Step 1: Choose Input Method
    st.subheader("Step 1: Choose Input Method")
    input_method = st.radio("Select Input Type:", ["Manual Entry", "Upload CSV"])
    
    if input_method == "Manual Entry":
        days = st.slider("Number of Days to Forecast:", 7, 60, 30)
        base_energy = st.number_input("Baseline Daily Energy (kWh)", 100, 1000, 250)
        data = pd.DataFrame({
            "Date": pd.date_range(datetime.now(), periods=days),
            "Baseline": np.linspace(base_energy, base_energy + 20, days)
        })
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("✅ File successfully uploaded!")
        else:
            st.warning("Please upload a CSV to proceed.")
            st.stop()

    # Step 2: Adjust Factors
    st.subheader("Step 2: Adjust Energy Factors ⚙️")
    col1, col2, col3, col4 = st.columns(4)
    light_factor = col1.slider("💡 Lights", -50, 50, 0, step=5)
    comp_factor = col2.slider("💻 Computers", -50, 50, 0, step=5)
    lab_factor = col3.slider("⚗️ Lab Equipment", -50, 50, 0, step=5)
    hour_factor = col4.slider("⏱️ Operating Hours", -50, 50, 0, step=5)

    # Calculate Adjusted Forecast
    adj_factor = 1 + (light_factor + comp_factor + lab_factor + hour_factor) / 400
    data["Adjusted"] = data["Baseline"] * adj_factor
    data["Cost"] = data["Adjusted"] * 0.52  # RM0.52 per kWh
    data["CO2"] = data["Adjusted"] * 0.233  # kg CO₂ per kWh

    st.markdown("---")

    # Step 3: Graphs
    st.subheader("📊 Scenario Comparison")
    tab1, tab2, tab3, tab4 = st.tabs(["Baseline Forecast", "Adjusted Forecast", "Cost Trend", "CO₂ Trend"])

    with tab1:
        fig1 = px.line(data, x="Date", y="Baseline", title="Baseline Forecast (kWh)",
                       line_shape="spline", markers=True)
        fig1.update_traces(line_color="#0077ff")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = px.line(data, x="Date", y="Adjusted", title="Forecast with Adjusted Factors",
                       line_shape="spline", markers=True, color_discrete_sequence=["#00ff88"])
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.bar(data, x="Date", y="Cost", title="Energy Cost Trend (RM)", color="Cost",
                      color_continuous_scale="viridis")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=data["Date"], y=data["CO2"], mode="lines+markers", line=dict(color="orange")))
        fig4.update_layout(title="CO₂ Emission Trend (kg)", xaxis_title="Date", yaxis_title="CO₂ (kg)")
        st.plotly_chart(fig4, use_container_width=True)


# --- DEVICE MANAGEMENT PAGE ---
elif menu == "💡 Device Management":
    st.title("💡 Device Management")
    st.info("This section allows you to add, monitor, and control your connected devices.")
    st.write("Coming soon: device monitoring API integration.")


# --- REPORTS PAGE ---
elif menu == "📊 Reports":
    st.title("📊 Reports & Analytics")
    st.write("Generate detailed monthly or custom period energy usage reports.")
    st.write("Feature in development — CSV and PDF exports will be available here.")


# --- SETTINGS PAGE ---
elif menu == "⚙️ Settings":
    st.title("⚙️ Application Settings")
    st.write("Adjust preferences, themes, and configurations here.")
    theme = st.selectbox("Choose Theme:", ["Dark", "Light", "Auto"])
    st.success(f"Theme set to {theme} mode ✅")


# --- HELP & ABOUT PAGE ---
elif menu == "❓ Help & About":
    st.title("❓ Help & About")
    st.write("""
        **Smart Energy Forecast App v2.0**
        - Developed by Chika 💻  
        - Built with Streamlit + Plotly  
        - Interactive forecasting with scenario comparison  
        - Version: October 2025
    """)
    st.markdown("If you encounter issues, please contact [support@example.com](mailto:support@example.com)")
