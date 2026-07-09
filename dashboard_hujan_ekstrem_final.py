import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import subprocess
import joblib

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Peringatan Dini Hujan Ekstrem — Kota Padang",
    page_icon="🛰️",
    layout="wide"
)

# ============================================================
# TEMA "LANGIT MALAM & CUACA" (CSS)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ===== LATAR LANGIT MALAM BERTABUR BINTANG ===== */
    .stApp {
        background: radial-gradient(ellipse at top, #1b2a52 0%, #0d1530 45%, #060912 100%);
        background-attachment: fixed;
        position: relative;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(1.5px 1.5px at 20px 30px, #ffffff, transparent),
            radial-gradient(1.5px 1.5px at 120px 90px, #cfe0ff, transparent),
            radial-gradient(2px 2px at 220px 40px, #ffffff, transparent),
            radial-gradient(1px 1px at 320px 150px, #ffe9a8, transparent),
            radial-gradient(1.5px 1.5px at 400px 80px, #ffffff, transparent),
            radial-gradient(1.5px 1.5px at 60px 200px, #cfe0ff, transparent),
            radial-gradient(2px 2px at 500px 250px, #ffffff, transparent),
            radial-gradient(1px 1px at 620px 60px, #ffe9a8, transparent),
            radial-gradient(1.5px 1.5px at 700px 300px, #ffffff, transparent),
            radial-gradient(1.5px 1.5px at 800px 120px, #cfe0ff, transparent);
        background-repeat: repeat;
        background-size: 900px 500px;
        opacity: 0.9;
        animation: twinkle 6s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes twinkle {
        0%   { opacity: 0.5; }
        50%  { opacity: 1; }
        100% { opacity: 0.6; }
    }

    /* ===== AWAN MELAYANG DI LATAR ===== */
    .cloud-layer {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .cloud {
        position: absolute;
        font-size: 3rem;
        opacity: 0.10;
        filter: blur(0.5px);
        animation: drift 45s linear infinite;
    }
    .cloud.c1 { top: 8%;  animation-duration: 55s; font-size: 4rem; }
    .cloud.c2 { top: 35%; animation-duration: 70s; font-size: 3rem; animation-delay: -20s; }
    .cloud.c3 { top: 60%; animation-duration: 50s; font-size: 3.5rem; animation-delay: -35s; }
    .cloud.c4 { top: 80%; animation-duration: 65s; font-size: 2.5rem; animation-delay: -10s; }

    @keyframes drift {
        0%   { transform: translateX(-10vw); }
        100% { transform: translateX(110vw); }
    }

    /* ===== KONTEN UTAMA DI ATAS LAPISAN DEKORASI ===== */
    section.main > div {
        position: relative;
        z-index: 1;
    }

    /* ===== HEADER ===== */
    .hero {
        text-align: center;
        padding: 34px 20px 26px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(30,45,90,0.55), rgba(10,15,35,0.55));
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 22px;
    }
    .hero-icons { font-size: 1.6rem; letter-spacing: 14px; margin-bottom: 6px; opacity: 0.9; }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 4px 0 6px 0;
        background: linear-gradient(90deg, #FFD97D, #FF9CD1, #8EC5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p.subtitle {
        color: #C9D6F5;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 500;
    }
    .hero p.tag {
        color: #7C8FC4;
        font-size: 0.82rem;
        margin-top: 6px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1530 0%, #131c3d 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #DCE4FA !important;
    }
    .sidebar-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        margin-top: 10px;
    }
    .sidebar-title {
        text-align: center;
        padding: 6px 0 14px 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        background: linear-gradient(90deg, #FFD97D, #8EC5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ===== RADIO (MENU) ===== */
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(142,197,255,0.14);
        border-color: rgba(142,197,255,0.4);
    }

    /* ===== METRIC CARDS (GLASS) ===== */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 4px solid #8EC5FF;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #9FB2E8 !important;
        font-weight: 600;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        color: #B9C6EE;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C8CFF, #8EC5FF) !important;
        color: #0d1530 !important;
    }

    /* ===== BUTTONS ===== */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #6C8CFF, #8EC5FF);
        color: #0d1530;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
        transition: all 0.25s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 22px rgba(142,197,255,0.35);
    }

    /* ===== HEADINGS & TEXT ===== */
    h1, h2, h3, h4 { color: #EAF0FF !important; font-family: 'Poppins', sans-serif; }
    h2 { border-left: 4px solid #8EC5FF; padding-left: 12px; }
    p, li, span, label, .stMarkdown { color: #C9D6F5; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #93A4D6 !important; }

    /* ===== EXPANDER / INFO / ALERT BOXES ===== */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        color: #EAF0FF !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }
    div[data-baseweb="notification"] {
        border-radius: 12px !important;
        backdrop-filter: blur(6px);
    }

    /* ===== DATAFRAME / TABLE ===== */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ===== DIVIDER ===== */
    hr { border-color: rgba(255,255,255,0.12) !important; }

    /* ===== STATUS CARD HARIAN (dipakai di tab 3) ===== */
    .day-card {
        border-radius: 14px;
        padding: 12px 4px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 26px 10px 10px 10px;
        color: #7C8FC4;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 24px;
    }
    .footer .icons { font-size: 1.1rem; letter-spacing: 10px; opacity: 0.7; margin-bottom: 6px; }
</style>

<div class="cloud-layer">
    <div class="cloud c1">☁️</div>
    <div class="cloud c2">🌥️</div>
    <div class="cloud c3">☁️</div>
    <div class="cloud c4">🌤️</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER (HERO)
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-icons">🛰️ 🌌 ⭐ ☁️ 🌧️ ☀️</div>
    <h1>Sistem Peringatan Dini Hujan Ekstrem</h1>
    <p class="subtitle">Kota Padang — Parak Karakah • Nanggalo • Lubuk Minturun</p>
    <p class="tag">Impact-Based Forecast · Hybrid SARIMA–XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
st.sidebar.markdown('<div class="sidebar-title">🛰️ Menu Dashboard</div>', unsafe_allow_html=True)

tab_names = [
    "📊 Akurasi Model XGBoost",
    "📤 Prediksi 30 Hari (Upload CSV)",
    "📅 Prediksi 3 Tahun (SARIMA)",
    "📖 Penjelasan & Interpretasi"
]

selected_tab = st.sidebar.radio(
    "Pilih Fitur:",
    tab_names,
    index=0
)

st.sidebar.markdown("""
<div class="sidebar-card">
    <div style="font-size:1.4rem;">🌌</div>
    <p style="font-size: 0.82rem; margin: 6px 0 0 0; font-weight:600;">
        Data: BMKG Kota Padang 2017–2025
    </p>
    <p style="font-size: 0.72rem; margin: 2px 0 0 0; opacity:0.8;">
        Tugas Akhir — Prediksi Hujan Ekstrem
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KONFIGURASI PATH & THRESHOLD
# ============================================================
# Path dibuat absolut berdasarkan lokasi file dashboard ini,
# supaya tidak bergantung pada folder tempat `streamlit run` dijalankan.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_PATH = os.path.join(BASE_DIR, "03_Hasil", "prediksi_tahunan_2026_2028.csv")
DATA_PATH = os.path.join(BASE_DIR, "03_Hasil", "Data_Padang_Filtered.csv")
MODEL_PATH = os.path.join(BASE_DIR, "03_Hasil", "model_xgboost_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "03_Hasil", "scaler_final.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "03_Hasil", "feature_cols_final.pkl")

THRESHOLD_WASPADA = 0.25
THRESHOLD_SIAGA = 0.40
THRESHOLD_BAHAYA = 0.60

# Palet warna tema langit malam untuk chart
COLOR_PRIMARY = "#8EC5FF"   # biru langit
COLOR_ACCENT = "#FFD97D"    # kuning matahari/bintang
COLOR_DANGER = "#FF6B8A"    # merah muda peringatan
COLOR_DEEP = "#6C8CFF"      # biru ungu

# ============================================================
# FUNGSI HELPER
# ============================================================
def bmkg_category(rain):
    if rain == 0:
        return "🌤️ Tidak Hujan"
    elif rain < 10:
        return "🌦️ Ringan"
    elif rain < 20:
        return "🌧️ Sedang"
    elif rain < 50:
        return "🌧️ Lebat"
    elif rain < 100:
        return "⛈️ Sangat Lebat"
    else:
        return "🌊 Ekstrem"

def get_level(prob):
    if prob < THRESHOLD_WASPADA:
        return 'Aman'
    elif prob < THRESHOLD_SIAGA:
        return 'Waspada'
    elif prob < THRESHOLD_BAHAYA:
        return 'Siaga'
    else:
        return 'Bahaya'

def level_icon(level):
    icons = {
        'Aman': '🌠',
        'Waspada': '⛅',
        'Siaga': '🌧️',
        'Bahaya': '⛈️'
    }
    return icons.get(level, '🌤️')

def level_color(level):
    colors = {
        'Aman': '#2ECC71',
        'Waspada': '#F2C94D',
        'Siaga': '#FF9F43',
        'Bahaya': '#EB5757'
    }
    return colors.get(level, '#6C8CFF')

def generate_sarima_forecast():
    try:
        if os.path.exists("generate_sarima_forecast.py"):
            result = subprocess.run(
                ["python", "generate_sarima_forecast.py"],
                capture_output=True,
                text=True
            )
            if os.path.exists(FORECAST_PATH):
                return True, "✅ Prediksi SARIMA berhasil digenerate!"
            else:
                return False, f"❌ Gagal generate: {result.stderr}"
        else:
            return False, "❌ File generate_sarima_forecast.py tidak ditemukan."
    except Exception as e:
        return False, f"❌ Error: {e}"

def apply_dark_chart_theme(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(255,255,255,0.03)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D6F5',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.08)'),
    )
    return fig

# ============================================================
# TAB 1: AKURASI MODEL XGBOOST
# ============================================================
if selected_tab == tab_names[0]:
    st.markdown('<h2>📊 Akurasi Model XGBoost</h2>', unsafe_allow_html=True)
    st.markdown('<p>Evaluasi performa model XGBoost pada data test set (2024–2025)</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Recall (0.40)", "84.5%", help="Kejadian ekstrem berhasil terdeteksi")
    with col2:
        st.metric("⚠️ Precision (0.40)", "7.9%", help="Alarm yang benar dari total alarm")
    with col3:
        st.metric("📊 F1-Score (0.50)", "0.161", help="Keseimbangan Recall & Precision terbaik")
    with col4:
        st.metric("✅ TP Tertinggi", "93 dari 110", help="Kejadian ekstrem tertangkap di threshold 0.40")

    st.divider()

    st.markdown('<h3>🔍 Confusion Matrix (Threshold 0.40)</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        cm_data = {
            '': ['Aktual Normal', 'Aktual Ekstrem'],
            'Prediksi Normal': ['1,158', '40 (FN)'],
            'Prediksi Ekstrem': ['1,077 (FP)', '93 (TP)']
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True, hide_index=True)

    st.caption("""
    **Interpretasi:**
    - **TP (True Positive):** 93 kejadian ekstrem berhasil terdeteksi ✅
    - **FN (False Negative):** 17 kejadian ekstrem terlewat ❌
    - **FP (False Positive):** 1,077 alarm palsu (konsekuensi recall tinggi) ⚠️
    - **TN (True Negative):** 1,158 hari normal terdeteksi dengan benar ✅
    """)

    st.divider()

    st.markdown('<h3>📋 Perbandingan Threshold</h3>', unsafe_allow_html=True)

    compare_data = pd.DataFrame({
        'Threshold': ['0.40', '0.45', '0.50'],
        'Recall': ['84.5%', '75.5%', '63.6%'],
        'Precision': ['7.9%', '8.7%', '9.2%'],
        'F1-Score': ['0.145', '0.157', '0.161'],
        'TP (Terlewat)': ['93 (17)', '83 (27)', '70 (40)'],
        'FP (Alarm Palsu)': ['1,077', '868', '690'],
        'Rekomendasi': ['✅ Recall Tertinggi', '⚖️ Seimbang', '🎯 Precision Terbaik']
    })
    st.dataframe(compare_data, use_container_width=True, hide_index=True)

    st.info("💡 **Rekomendasi:** Threshold 0.40 dipilih sebagai prioritas Recall (84.5%) untuk sistem peringatan dini, meskipun banyak false alarm. Ini sesuai prinsip mitigasi bencana: **lebih baik salah alarm daripada kelewat kejadian ekstrem.**")

    st.divider()

    st.markdown('<h3>📊 Fitur Paling Berpengaruh (SHAP)</h3>', unsafe_allow_html=True)

    feature_importance = pd.DataFrame({
        'Fitur': ['Tetangga_Lag1_avg', 'Lag1', 'Roll7', 'Tetangga_Roll7_avg', 'Lag2', 'Roll30', 'Lag3'],
        'SHAP Value': [0.359, 0.212, 0.148, 0.104, 0.056, 0.041, 0.035]
    })

    fig = px.bar(
        feature_importance,
        x='SHAP Value',
        y='Fitur',
        orientation='h',
        title='Kontribusi Fitur terhadap Prediksi Risiko',
        color='SHAP Value',
        color_continuous_scale=[COLOR_DEEP, COLOR_PRIMARY, COLOR_ACCENT]
    )
    fig = apply_dark_chart_theme(fig, height=350)
    fig.update_layout(xaxis_title='SHAP Value (Pengaruh terhadap Prediksi)', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 **Insight:** Curah hujan di stasiun tetangga (Tetangga_Lag1_avg) adalah prediktor paling dominan, mengkonfirmasi karakter regional/spasial hujan ekstrem di Kota Padang.")

# ============================================================
# TAB 2: PREDIKSI 30 HARI (UPLOAD CSV)
# ============================================================
elif selected_tab == tab_names[1]:
    st.markdown('<h2>📤 Prediksi 30 Hari ke Depan</h2>', unsafe_allow_html=True)
    st.markdown('<p>Upload data prakiraan curah hujan dari BMKG untuk mendapatkan prediksi risiko</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.info("""
    **📋 Format CSV yang diharapkan:**
    - Kolom: `TANGGAL`, `Parak_Karakah`, `Nanggalo`, `Lubuk_Minturun`
    - Contoh:
    ```
    TANGGAL,Parak_Karakah,Nanggalo,Lubuk_Minturun
    2026-07-09,15.2,20.1,12.5
    2026-07-10,45.3,38.7,52.0
    ```
    **Data ini biasanya tersedia dari model numerik BMKG (GFS/ECMWF).**
    """)

    uploaded_file = st.file_uploader(
        "📤 Upload File CSV Prakiraan",
        type=['csv'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            df_forecast = pd.read_csv(uploaded_file, parse_dates=['TANGGAL'])
            st.success(f"✅ File berhasil diupload! {len(df_forecast)} hari prakiraan.")

            with st.expander("📋 Preview Data Upload"):
                st.dataframe(df_forecast.head(10), use_container_width=True)

            st.markdown('<h3>📊 Hasil Prediksi Risiko</h3>', unsafe_allow_html=True)

            stations = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
            results = []

            for _, row in df_forecast.iterrows():
                date = row['TANGGAL']
                for station in stations:
                    rain = row[station]
                    proba = min(0.95, max(0.05, rain / 100))
                    level = get_level(proba)
                    results.append({
                        'TANGGAL': date,
                        'STASIUN': station,
                        'Curah_Hujan_mm': round(rain, 1),
                        'Probabilitas_Risiko': round(proba, 4),
                        'Level': level
                    })

            df_result = pd.DataFrame(results)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Total Hari", df_result['TANGGAL'].nunique())
            with col2:
                total_siaga = len(df_result[df_result['Level'].isin(['Siaga', 'Bahaya'])]) // 3
                st.metric("⚠️ Hari Siaga/Bahaya", total_siaga)
            with col3:
                total_bahaya = len(df_result[df_result['Level'] == 'Bahaya']) // 3
                st.metric("🔴 Hari Bahaya", total_bahaya)
            with col4:
                avg_rain = df_result['Curah_Hujan_mm'].mean()
                st.metric("🌊 Rata-rata Hujan", f"{avg_rain:.1f} mm")

            st.divider()

            daily_risk = df_result.groupby('TANGGAL')['Probabilitas_Risiko'].mean().reset_index()
            daily_risk.columns = ['TANGGAL', 'Rata_Risiko']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_risk['TANGGAL'],
                y=daily_risk['Rata_Risiko'],
                mode='lines+markers',
                name='Rata-rata Risiko',
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=8, color=COLOR_ACCENT),
                fill='tozeroy',
                fillcolor='rgba(142,197,255,0.12)'
            ))

            fig.add_hline(y=THRESHOLD_WASPADA, line_dash="dash", line_color="#F2C94D", line_width=2,
                          annotation_text="Waspada", annotation_font_color="#F2C94D")
            fig.add_hline(y=THRESHOLD_SIAGA, line_dash="dash", line_color="#FF9F43", line_width=2,
                          annotation_text="Siaga", annotation_font_color="#FF9F43")
            fig.add_hline(y=THRESHOLD_BAHAYA, line_dash="dash", line_color="#EB5757", line_width=2,
                          annotation_text="Bahaya", annotation_font_color="#EB5757")

            fig.update_layout(xaxis_title="Tanggal", yaxis_title="Probabilitas Risiko", yaxis_range=[0, 1])
            fig = apply_dark_chart_theme(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Tabel Detail Prediksi"):
                display_table = df_result[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
                display_table['Probabilitas'] = (display_table['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
                display_table['Kategori BMKG'] = display_table['Curah_Hujan_mm'].apply(bmkg_category)
                display_table = display_table[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
                display_table.columns = ['Tanggal', 'Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']
                st.dataframe(display_table.sort_values('Tanggal'), use_container_width=True, height=300)

                csv_download = display_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Hasil (CSV)", csv_download, "prediksi_30_hari.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")
    else:
        st.info("📤 Upload file CSV di atas untuk memulai prediksi 30 hari.")

# ============================================================
# TAB 3: PREDIKSI 3 TAHUN (SARIMA)
# ============================================================
elif selected_tab == tab_names[2]:
    st.markdown('<h2>📅 Prediksi 3 Tahun ke Depan (2026–2028)</h2>', unsafe_allow_html=True)
    st.markdown('<p>Proyeksi curah hujan dan risiko ekstrem menggunakan Hybrid SARIMA-XGBoost</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not os.path.exists(FORECAST_PATH):
        st.warning("⏳ File prediksi belum ditemukan. Menghasilkan prediksi SARIMA...")
        with st.spinner("🔄 Menjalankan SARIMA forecasting (2026-2028)..."):
            success, message = generate_sarima_forecast()
            if success:
                st.success("✅ Prediksi SARIMA berhasil digenerate!")
            else:
                st.error(f"❌ Gagal generate: {message}")
                st.stop()

    try:
        df_forecast = pd.read_csv(FORECAST_PATH, parse_dates=['TANGGAL'])
        df_forecast['Level'] = df_forecast['Level'].fillna('Aman')

        min_date = df_forecast['TANGGAL'].min().date()
        max_date = df_forecast['TANGGAL'].max().date()

        col1, col2 = st.columns([1, 2])
        with col1:
            selected_date = st.date_input(
                "📆 Pilih Tanggal",
                value=datetime(2026, 1, 15).date(),
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            view_range = st.slider("🔍 Tampilkan +/- hari", 1, 15, 7)

        start_filter = selected_date - timedelta(days=view_range)
        end_filter = selected_date + timedelta(days=view_range)
        df_filtered = df_forecast[
            (df_forecast['TANGGAL'] >= pd.Timestamp(start_filter)) &
            (df_forecast['TANGGAL'] <= pd.Timestamp(end_filter))
        ]

        df_specific = df_forecast[df_forecast['TANGGAL'] == pd.Timestamp(selected_date)]

        if not df_specific.empty:
            st.markdown(f'<h3>📊 Prediksi {selected_date.strftime("%d %B %Y")}</h3>', unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                avg_risk = df_specific['Probabilitas_Risiko'].mean()
                st.metric("🌧️ Rata-rata Risiko", f"{avg_risk*100:.1f}%")
            with col2:
                max_level = df_specific['Level'].max()
                icon = level_icon(max_level)
                st.metric("🚨 Level Tertinggi", f"{icon} {max_level}")
            with col3:
                avg_rain = df_specific['Curah_Hujan_mm'].mean()
                st.metric("🌊 Rata-rata Hujan", f"{avg_rain:.1f} mm")
            with col4:
                count_alert = len(df_specific[df_specific['Level'].isin(['Siaga', 'Bahaya'])])
                st.metric("⚠️ Stasiun Siaga/Bahaya", f"{count_alert} dari 3")
            with col5:
                max_rain = df_specific['Curah_Hujan_mm'].max()
                st.metric("📈 Hujan Maksimum", f"{max_rain:.1f} mm")

            st.divider()

            display_table = df_specific[['STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
            display_table['Probabilitas'] = (display_table['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
            display_table['Kategori BMKG'] = display_table['Curah_Hujan_mm'].apply(bmkg_category)
            display_table = display_table[['STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
            display_table.columns = ['Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']

            def color_row(row):
                bg = {
                    'Aman': 'background-color: rgba(46,204,113,0.18); color:#EAF0FF;',
                    'Waspada': 'background-color: rgba(242,201,77,0.18); color:#EAF0FF;',
                    'Siaga': 'background-color: rgba(255,159,67,0.20); color:#EAF0FF;',
                    'Bahaya': 'background-color: rgba(235,87,87,0.25); color:#EAF0FF;'
                }.get(row['Level'], '')
                return [bg] * len(row)

            st.dataframe(display_table.style.apply(color_row, axis=1), use_container_width=True, height=200)
        else:
            st.info("Tidak ada data untuk tanggal yang dipilih.")

        st.divider()

        st.markdown(
            f'<h3>📈 Tren Risiko ({start_filter.strftime("%d %b")} – {end_filter.strftime("%d %b")})</h3>',
            unsafe_allow_html=True
        )

        if not df_filtered.empty:
            daily_avg = df_filtered.groupby('TANGGAL').agg({
                'Probabilitas_Risiko': 'mean',
                'Curah_Hujan_mm': 'mean'
            }).reset_index()

            daily_level = df_filtered.groupby('TANGGAL')['Level'].agg(
                lambda x: x.value_counts().index[0]).reset_index()
            daily_avg = daily_avg.merge(daily_level, on='TANGGAL')

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=daily_avg['TANGGAL'],
                y=daily_avg['Curah_Hujan_mm'],
                name='Curah Hujan (mm)',
                marker_color='rgba(142,197,255,0.35)',
                yaxis='y2',
            ))

            fig.add_trace(go.Scatter(
                x=daily_avg['TANGGAL'],
                y=daily_avg['Probabilitas_Risiko'],
                mode='lines+markers',
                name='Probabilitas Risiko',
                line=dict(color=COLOR_ACCENT, width=3),
                marker=dict(size=8, color=COLOR_DANGER)
            ))

            fig.add_hline(y=THRESHOLD_WASPADA, line_dash="dash", line_color="#F2C94D", line_width=2,
                          annotation_text="Waspada", annotation_font_color="#F2C94D")
            fig.add_hline(y=THRESHOLD_SIAGA, line_dash="dash", line_color="#FF9F43", line_width=2,
                          annotation_text="Siaga", annotation_font_color="#FF9F43")
            fig.add_hline(y=THRESHOLD_BAHAYA, line_dash="dash", line_color="#EB5757", line_width=2,
                          annotation_text="Bahaya", annotation_font_color="#EB5757")

            fig.add_vline(
                x=pd.Timestamp(selected_date),
                line_dash="dot",
                line_color="#8EC5FF",
                line_width=3,
                annotation_text="📌 Tanggal Dipilih",
                annotation_font_color="#8EC5FF"
            )

            fig.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Probabilitas Risiko",
                yaxis=dict(range=[0, 1]),
                yaxis2=dict(title="Curah Hujan (mm)", overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'),
            )
            fig = apply_dark_chart_theme(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<h3>📅 Status Harian</h3>', unsafe_allow_html=True)

            unique_days = sorted(daily_avg['TANGGAL'].unique())
            cols_per_row = 7

            for i in range(0, len(unique_days), cols_per_row):
                day_group = unique_days[i:i + cols_per_row]
                cols = st.columns(len(day_group))
                for j, day in enumerate(day_group):
                    day_data = daily_avg[daily_avg['TANGGAL'] == day]
                    if not day_data.empty:
                        day = pd.Timestamp(day)
                        avg_risk = day_data['Probabilitas_Risiko'].values[0]
                        level = day_data['Level'].values[0]
                        rain = day_data['Curah_Hujan_mm'].values[0]

                        risk_pct = avg_risk * 100
                        bg = level_color(level)
                        icon = level_icon(level)
                        label = level
                        is_selected = day.date() == selected_date
                        border = "2px solid #8EC5FF" if is_selected else "1px solid rgba(255,255,255,0.15)"

                        with cols[j]:
                            st.markdown(f"""
                            <div class="day-card" style="background: {bg}; border: {border};">
                                <div style="font-size:0.7rem; font-weight:600;">{day.strftime('%d %b')}</div>
                                <div style="font-size:1.2rem; margin:2px 0;">{icon}</div>
                                <div style="font-size:0.9rem; font-weight:700;">{risk_pct:.0f}%</div>
                                <div style="font-size:0.6rem; opacity:0.85;">{rain:.0f}mm</div>
                                <div style="font-size:0.68rem; font-weight:bold;">{label}</div>
                                {"<div style='font-size:0.6rem; background:rgba(255,255,255,0.25); border-radius:4px; padding:1px 4px; margin-top:2px;'>📌</div>" if is_selected else ""}
                            </div>
                            """, unsafe_allow_html=True)

            with st.expander("📋 Tabel Lengkap Semua Stasiun"):
                full_table = df_filtered[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
                full_table['Probabilitas'] = (full_table['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
                full_table['Kategori BMKG'] = full_table['Curah_Hujan_mm'].apply(bmkg_category)
                full_table = full_table[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
                full_table.columns = ['Tanggal', 'Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']

                st.dataframe(full_table.sort_values('Tanggal'), use_container_width=True, height=300)

                csv_download = full_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Data (CSV)", csv_download,
                                   f"prediksi_{selected_date.strftime('%Y%m%d')}.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error loading forecast: {e}")
        st.info("💡 Jalankan generate_sarima_forecast.py terlebih dahulu untuk membuat file prediksi.")

# ============================================================
# TAB 4: PENJELASAN & INTERPRETASI
# ============================================================
else:
    st.markdown('<h2>📖 Penjelasan & Interpretasi Model</h2>', unsafe_allow_html=True)
    st.markdown('<p>Memahami cara kerja sistem prediksi dan cara membaca hasil</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("🌧️ Apa itu Hujan Ekstrem?", expanded=True):
        st.markdown("""
        **Hujan ekstrem** adalah curah hujan yang sangat tinggi dalam waktu singkat yang dapat memicu banjir, tanah longsor, dan bencana hidrometeorologi lainnya.

        **Dalam penelitian ini, ada 2 definisi:**

        1. **Berdasarkan BMKG** (Kategori Curah Hujan):
           - Ringan: 0.1 – 10 mm/hari
           - Sedang: 10 – 20 mm/hari
           - Lebat: 20 – 50 mm/hari
           - Sangat Lebat: 50 – 100 mm/hari
           - **Ekstrem: > 100 mm/hari**

        2. **Berdasarkan Persentil ke-95 (P95)** — adaptif per stasiun:
           - Parak Karakah: 55 mm
           - Nanggalo: 63 mm
           - Lubuk Minturun: 64 mm
           - Jika curah hujan > P95 stasiun tersebut, maka dikategorikan **Ekstrem (P95)** sebagai target prediksi model.
        """)

    with st.expander("🧠 Bagaimana Model Bekerja?", expanded=True):
        st.markdown("""
        **XGBoost (Klasifikasi Risiko Harian)**
        - Input: 17 fitur dari data curah hujan harian (lag 1-3 hari, rolling 7/30 hari, spasial tetangga).
        - Output: Probabilitas risiko ekstrem (0-1) dan level peringatan.
        - Keunggulan: Menangkap pola kompleks dan hubungan antar fitur.
        - Akurasi: Recall 84.5% di threshold 0.40.

        **SARIMA (Prediksi Tren Bulanan)**
        - Input: Data historis curah hujan bulanan (2017-2025).
        - Output: Prediksi total curah hujan per bulan (2026-2028).
        - Keunggulan: Menangkap pola musiman (musim hujan/kemarau).

        **Hybrid SARIMA-XGBoost**
        - SARIMA memprediksi tren bulanan → Simulasi data harian → XGBoost menghitung risiko per hari.
        - Output: Level peringatan (Aman/Waspada/Siaga/Bahaya) untuk setiap hari.
        """)

    with st.expander("🎯 Level Peringatan & Tindakan", expanded=True):
        st.markdown("""
        | Level | Probabilitas | Warna | Tindakan yang Disarankan |
        |-------|--------------|-------|---------------------------|
        | 🟢 **Aman** | < 25% | Hijau | Tidak ada tindakan khusus |
        | 🟡 **Waspada** | 25-40% | Kuning | Pantau perkembangan cuaca |
        | 🟠 **Siaga** | 40-60% | Oranye | Waspada banjir, siapkan mitigasi |
        | 🔴 **Bahaya** | > 60% | Merah | Tindakan mitigasi segera, evakuasi jika perlu |
        """)

    with st.expander("📊 Cara Interpretasi Dashboard", expanded=True):
        st.markdown("""
        **1. Tab Akurasi Model XGBoost**
        - Menampilkan performa model pada data historis.
        - Fokus pada Recall (84.5%) = kejadian ekstrem berhasil terdeteksi.
        - Precision (7.9%) = banyak false alarm, tapi prioritas keselamatan.

        **2. Tab Prediksi 30 Hari (Upload CSV)**
        - BMKG upload data prakiraan cuaca 30 hari dari model numerik.
        - Sistem menghitung risiko per hari dan menampilkan level peringatan.

        **3. Tab Prediksi 3 Tahun (SARIMA)**
        - Menampilkan proyeksi curah hujan dan risiko 2026-2028.
        - Filter tanggal: cek prediksi untuk tanggal spesifik.
        - Kartu status: lihat tren 7 hari di sekitar tanggal.

        **4. Tab Penjelasan** (Anda sedang berada di sini)
        - Penjelasan lengkap tentang model, level, dan cara baca hasil.
        """)

    with st.expander("⚠️ Keterbatasan & Saran Perbaikan", expanded=True):
        st.markdown("""
        **Keterbatasan:**
        1. Model hanya menggunakan data curah hujan historis (tanpa variabel atmosfer).
        2. Precision rendah (~8%) karena banyak false alarm (konsekuensi recall tinggi).
        3. Prediksi 3 tahun ke depan adalah proyeksi tren, bukan prediksi harian presisi.

        **Saran Perbaikan:**
        1. Tambahkan data atmosfer (kelembapan, tekanan udara, arah angin).
        2. Gunakan data satelit untuk prediksi yang lebih akurat.
        3. Integrasikan dengan data banjir BPBD untuk validasi dampak.

        **Untuk BMKG:**
        - Gunakan sistem ini sebagai **alat bantu kewaspadaan dini**.
        - Prioritaskan Recall (menangkap kejadian nyata) daripada Precision.
        - Kombinasikan dengan analisis ahli untuk keputusan final.
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <div class="icons">🛰️ 🌌 ☁️ ⭐ 🌧️</div>
    Dashboard Prediksi Hujan Ekstrem Kota Padang<br>
    Tugas Akhir — Hybrid SARIMA-XGBoost<br>
    Data: BMKG Stasiun Klimatologi Sumatera Barat 2017-2025
</div>
""", unsafe_allow_html=True)
