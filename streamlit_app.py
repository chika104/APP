import streamlit as st
import mysql.connector
from mysql.connector import Error
import bcrypt

# ========================
# DATABASE CONNECTION
# ========================
def get_connection():
    try:
        return mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            database=st.secrets["DB_NAME"]
        )
    except Error as e:
        st.error(f"DB connection failed: {e}")
        return None


# ========================
# CREATE USER TABLE (AUTO)
# ========================
def create_user_table():
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_accounts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL
            )
        """)
        conn.commit()
        c.close()
        conn.close()


# ========================
# REGISTER USER
# ========================
def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor()
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            c.execute("INSERT INTO user_accounts (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
            conn.commit()
            st.success("Pendaftaran berjaya! Anda boleh log masuk sekarang.")
        except Error as e:
            st.error(f"DB error during registration: {e}")
        finally:
            c.close()
            conn.close()


# ========================
# LOGIN USER
# ========================
def login_user(username, password):
    conn = get_connection()
    if conn:
        try:
            c = conn.cursor(dictionary=True)
            c.execute("SELECT * FROM user_accounts WHERE username = %s", (username,))
            user = c.fetchone()
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                return True
            else:
                return False
        except Error as e:
            st.error(f"Login failed: {e}")
        finally:
            c.close()
            conn.close()
    return False


# ========================
# MAIN APP
# ========================
def main():
    st.set_page_config(page_title="Secure Dashboard", layout="wide")

    # Auto create table
    create_user_table()

    # Background style
    st.markdown("""
        <style>
            body {
                background-color: #0a0a0a;
                color: white;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            input, textarea {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            .stButton>button {
                background-color: #007bff;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)

    # Session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    # Login / Register
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if not st.session_state.logged_in:
        if choice == "Login":
            st.title("🔐 Secure Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Selamat datang, {username}!")
                else:
                    st.error("Nama pengguna atau kata laluan salah!")

        elif choice == "Register":
            st.title("📝 Pendaftaran Akaun Baru")
            new_username = st.text_input("Username Baharu")
            new_password = st.text_input("Kata Laluan Baharu", type="password")
            if st.button("Daftar"):
                if new_username and new_password:
                    register_user(new_username, new_password)
                else:
                    st.warning("Sila isi semua ruangan!")

    # ========================
    # DASHBOARD
    # ========================
    if st.session_state.logged_in:
        st.sidebar.success(f"Log masuk sebagai: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

        st.title("📊 Dashboard Utama")
        st.markdown("Selamat datang ke sistem ramalan tenaga ⚡")

        # Contoh: pengguna masukkan data CSV/manual
        upload = st.file_uploader("Muat naik fail CSV (opsyenal)")
        if upload:
            import pandas as pd
            df = pd.read_csv(upload)
            st.dataframe(df)

            # Simpan automatik ke DB
            conn = get_connection()
            if conn:
                try:
                    c = conn.cursor()
                    for _, row in df.iterrows():
                        c.execute(
                            "INSERT INTO user_data (col1, col2, col3) VALUES (%s, %s, %s)",
                            tuple(row)
                        )
                    conn.commit()
                    st.success("✅ Data berjaya disimpan ke DB.")
                except Error as e:
                    st.error(f"Gagal simpan data ke DB: {e}")
                finally:
                    c.close()
                    conn.close()

        # Table comparison (contoh baseline)
        st.subheader("📈 Table Comparison")
        st.table({
            "Model": ["Baseline", "Forecast"],
            "Accuracy": ["85%", "93%"],
            "Loss": ["0.15", "0.07"]
        })


if __name__ == "__main__":
    main()
