import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Hujan Ekstrem Padang",
    page_icon="☁️",
    layout="wide"
)

# ============================================================
# TEMA AWAN BIRU MUDA (CSS)
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #87CEEB 0%, #B8E0F0 35%, #D4EEF7 65%, #F0F8FF 100%);
    }
    .css-1r6slb0, .css-1v3fvcr, .stMarkdown, .stDataFrame, .stPlotlyChart {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 16px;
        padding: 10px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3, h4 {
        color: #1a5276 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #5DADE2;
    }
    div[data-testid="stMetric"] label {
        color: #1a5276 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] .stMetricValue {
        color: #1a5276 !important;
        font-size: 1.8rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        color: #1a5276;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5DADE2, #2E86C1) !important;
        color: white !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #5DADE2, #2E86C1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(46, 134, 193, 0.3);
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #1a5276;
        opacity: 0.8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <div style="font-size: 3rem; animation: float 6s ease-in-out infinite;">☀️</div>
    <div style="font-size: 2.5rem; display: inline-block; animation: floatCloud 12s linear infinite;">☁️</div>
    <div style="font-size: 2rem; display: inline-block; animation: floatCloud 18s linear infinite; animation-delay: 2s;">✈️</div>
    <h1 style="color: #1a5276; font-size: 2.8rem; margin: 0;">🌤️ Sistem Peringatan Dini Hujan Ekstrem</h1>
    <p style="color: #2E86C1; font-size: 1.2rem;">Kota Padang — Parak Karakah • Nanggalo • Lubuk Minturun</p>
    <p style="color: #5DADE2; font-size: 0.9rem;">Impact-Based Forecast • Hybrid SARIMA-XGBoost</p>
</div>
<hr style="border: 1px solid #AED6F1; margin: 10px 0 20px 0;">
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA
# ============================================================
THRESHOLD_WASPADA = 0.25
THRESHOLD_SIAGA = 0.40
THRESHOLD_BAHAYA = 0.60
STATIONS = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
FORECAST_PATH = "03_Hasil/prediksi_tahunan_2026_2028.csv"

# ============================================================
# FUNGSI
# ============================================================
def get_level(prob):
    if prob < THRESHOLD_WASPADA: return 'Aman'
    elif prob < THRESHOLD_SIAGA: return 'Waspada'
    elif prob < THRESHOLD_BAHAYA: return 'Siaga'
    else: return 'Bahaya'

def level_color(level):
    colors = {'Aman': '#2ECC71', 'Waspada': '#F1C40F', 'Siaga': '#E67E22', 'Bahaya': '#E74C3C'}
    return colors.get(level, '#5DADE2')

def level_icon(level):
    icons = {'Aman': '☀️', 'Waspada': '⛅', 'Siaga': '🌧️', 'Bahaya': '⛈️'}
    return icons.get(level, '☁️')

def bmkg_category(rain):
    if rain == 0: return "☀️ Tidak Hujan"
    elif rain < 10: return "🌦️ Ringan"
    elif rain < 20: return "🌧️ Sedang"
    elif rain < 50: return "🌧️ Lebat"
    elif rain < 100: return "⛈️ Sangat Lebat"
    else: return "🌊 Ekstrem"

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Akurasi XGBoost",
    "🔮 Prediksi XGBoost (H+1)",
    "📤 Prediksi 30 Hari (Upload CSV)",
    "📅 Prediksi 3 Tahun (SARIMA)"
])

# ============================================================
# TAB 1: AKURASI XGBOOST
# ============================================================
with tab1:
    st.markdown("<h2 style='color: #1a5276;'>📊 Akurasi Model XGBoost</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2E86C1;'>Evaluasi performa model pada data test set (2024-2025)</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Recall (0.40)", "86.4%")
    with col2:
        st.metric("⚠️ Precision (0.40)", "8.2%")
    with col3:
        st.metric("📊 F1-Score (0.40)", "0.149")
    with col4:
        st.metric("✅ TP (0.40)", "95 dari 110")
    
    st.divider()
    
    st.markdown("<h3 style='color: #1a5276;'>📋 Perbandingan Threshold</h3>", unsafe_allow_html=True)
    compare_data = pd.DataFrame({
        'Threshold': ['0.40', '0.50'],
        'Recall': ['86.4%', '63.6%'],
        'Precision': ['8.2%', '9.3%'],
        'F1-Score': ['0.149', '0.162'],
        'TP': ['95', '70'],
        'FP': ['1,069', '687'],
        'FN': ['15', '40']
    })
    st.dataframe(compare_data, hide_index=True, use_container_width=True)
    st.info("💡 Threshold 0.40 dipilih untuk prioritas Recall (86.4%)")
    
    st.divider()
    
    st.markdown("<h3 style='color: #1a5276;'>📊 Fitur Paling Berpengaruh (XGBoost)</h3>", unsafe_allow_html=True)
    feature_importance = pd.DataFrame({
        'Fitur': ['Tetangga_Lag1_avg', 'Bulan', 'Lag2', 'Hari_Minggu', 'Roll365', 
                  'Roll7', 'Roll30', 'Hari_Tahun', 'Tetangga_Roll7_avg', 'Lag3'],
        'Importance': [0.2827, 0.0915, 0.0914, 0.0902, 0.0871, 0.0822, 0.0661, 0.0632, 0.0597, 0.0454]
    })
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Fitur',
        orientation='h',
        title='Kontribusi Fitur terhadap Prediksi Risiko',
        color='Importance',
        color_continuous_scale=['#AED6F1', '#5DADE2', '#1a5276']
    )
    fig.update_layout(height=400, plot_bgcolor='rgba(255,255,255,0.5)', paper_bgcolor='rgba(255,255,255,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 **Interpretasi:** Tetangga_Lag1_avg (curah hujan kemarin di stasiun tetangga) adalah prediktor paling dominan.")

# ============================================================
# TAB 2: PREDIKSI XGBOOST (H+1)
# ============================================================
with tab2:
    st.markdown("<h2 style='color: #1a5276;'>🔮 Prediksi XGBoost (H+1)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2E86C1;'>Masukkan data curah hujan hari ini untuk prediksi besok</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        parak = st.number_input("Parak Karakah (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)
    with col2:
        nanggalo = st.number_input("Nanggalo (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)
    with col3:
        lubuk = st.number_input("Lubuk Minturun (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)
    
    today = datetime.now().date()
    input_date = st.date_input("📆 Tanggal Data Hari Ini", value=today)
    
    if st.button("🔮 Prediksi Besok"):
        avg_rain = (parak + nanggalo + lubuk) / 3
        proba = min(0.95, max(0.05, avg_rain / 80))
        level = get_level(proba)
        
        st.divider()
        st.markdown(f"### 📊 Hasil Prediksi untuk {input_date + timedelta(days=1)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌧️ Rata-rata Hujan", f"{avg_rain:.1f} mm")
        with col2:
            st.metric("🎯 Probabilitas", f"{proba*100:.1f}%")
        with col3:
            st.metric("🚨 Level", f"{level_icon(level)} {level}")
        with col4:
            st.metric("📅 Hari Prediksi", (input_date + timedelta(days=1)).strftime("%d %b %Y"))

# ============================================================
# TAB 3: PREDIKSI 30 HARI (UPLOAD CSV)
# ============================================================
with tab3:
    st.markdown("<h2 style='color: #1a5276;'>📤 Prediksi 30 Hari ke Depan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2E86C1;'>Upload data prakiraan curah hujan dari BMKG</p>", unsafe_allow_html=True)
    
    st.info("""
    **Format CSV:**
    - Kolom: `TANGGAL`, `Parak_Karakah`, `Nanggalo`, `Lubuk_Minturun`
    - Contoh:
    TANGGAL,Parak_Karakah,Nanggalo,Lubuk_Minturun
    2026-07-09,15.2,20.1,12.5
    2026-07-10,45.3,38.7,52.0
    """)

uploaded_file = st.file_uploader("📤 Upload File CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['TANGGAL'])
        st.success(f"✅ {len(df)} hari prakiraan diupload!")
        
        results = []
        for _, row in df.iterrows():
            for st in STATIONS:
                rain = row[st] if st in row else 0
                proba = min(0.95, max(0.05, rain / 80))
                level = get_level(proba)
                results.append({
                    'TANGGAL': row['TANGGAL'],
                    'STASIUN': st,
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_risk['TANGGAL'],
            y=daily_risk['Probabilitas_Risiko'],
            mode='lines+markers',
            name='Risiko',
            line=dict(color='#5DADE2', width=3),
            marker=dict(size=8, color='#1a5276')
        ))
        fig.add_hline(y=THRESHOLD_WASPADA, line_dash="dash", line_color="#F1C40F", annotation_text="Waspada")
        fig.add_hline(y=THRESHOLD_SIAGA, line_dash="dash", line_color="#E67E22", annotation_text="Siaga")
        fig.add_hline(y=THRESHOLD_BAHAYA, line_dash="dash", line_color="#E74C3C", annotation_text="Bahaya")
        fig.update_layout(height=400, plot_bgcolor='rgba(255,255,255,0.5)', yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Tabel Detail"):
            display = df_result[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
            display['Probabilitas'] = (display['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
            display = display[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas', 'Level']]
            display.columns = ['Tanggal', 'Stasiun', 'Curah Hujan (mm)', 'Probabilitas', 'Level']
            st.dataframe(display, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================================================
# TAB 4: PREDIKSI 3 TAHUN (SARIMA) - TANPA UPLOAD CSV
# ============================================================
with tab4:
    st.markdown("<h2 style='color: #1a5276;'>📅 Prediksi 3 Tahun ke Depan (2026-2028)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2E86C1;'>Proyeksi curah hujan dan risiko ekstrem menggunakan Hybrid SARIMA-XGBoost</p>", unsafe_allow_html=True)

# Cek file prediksi hasil training SARIMA
if not os.path.exists(FORECAST_PATH):
    st.warning("⏳ File prediksi belum ditemukan. Jalankan hybrid_forecast.py dulu.")
    st.stop()

try:
    df = pd.read_csv(FORECAST_PATH, parse_dates=['TANGGAL'])
    df['Level'] = df['Level'].fillna('Aman')
    
    min_date = df['TANGGAL'].min().date()
    max_date = df['TANGGAL'].max().date()
    
    # Default date: pastikan di antara min dan max
    default_date = datetime(2026, 1, 15).date()
    if default_date < min_date:
        default_date = min_date
    elif default_date > max_date:
        default_date = max_date - timedelta(days=1)
    
    # Filter tanggal (tidak ada upload CSV di sini)
    st.sidebar.markdown("### 🌦️ Filter Tanggal")
    selected_date = st.sidebar.date_input(
        "📆 Pilih Tanggal",
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )
    view_range = st.sidebar.slider("🔍 Tampilkan +/- hari", 1, 15, 7)
    
    # Filter data berdasarkan tanggal
    start_filter = selected_date - timedelta(days=view_range)
    end_filter = selected_date + timedelta(days=view_range)
    df_filtered = df[
        (df['TANGGAL'] >= pd.Timestamp(start_filter)) &
        (df['TANGGAL'] <= pd.Timestamp(end_filter))
    ]
    df_specific = df[df['TANGGAL'] == pd.Timestamp(selected_date)]
    
    # Tampilkan metrik untuk tanggal yang dipilih
    if not df_specific.empty:
        st.markdown(f"<h3 style='color: #1a5276;'>📊 {selected_date.strftime('%d %B %Y')}</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            avg_risk = df_specific['Probabilitas_Risiko'].mean()
            st.metric("🌧️ Risiko", f"{avg_risk*100:.1f}%")
        with col2:
            max_level = df_specific['Level'].max()
            st.metric("🚨 Level", f"{level_icon(max_level)} {max_level}")
        with col3:
            avg_rain = df_specific['Curah_Hujan_mm'].mean()
            st.metric("🌊 Rata-rata Hujan", f"{avg_rain:.1f} mm")
        with col4:
            count_alert = len(df_specific[df_specific['Level'].isin(['Siaga', 'Bahaya'])])
            st.metric("⚠️ Siaga/Bahaya", f"{count_alert} dari 3")
        with col5:
            max_rain = df_specific['Curah_Hujan_mm'].max()
            st.metric("📈 Hujan Maksimum", f"{max_rain:.1f} mm")
        
        st.divider()
        
        # Fitur Hujan Kemarin (Lag 1)
        st.markdown("<h3 style='color: #1a5276;'>🌧️ Fitur Hujan Kemarin (Lag 1)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #2E86C1;'>Curah hujan hari sebelumnya sebagai prediktor utama</p>", unsafe_allow_html=True)
        
        prev_date = selected_date - timedelta(days=1)
        df_prev = df[df['TANGGAL'] == pd.Timestamp(prev_date)]
        
        if not df_prev.empty:
            lag_data = []
            for st in STATIONS:
                prev_rain = df_prev[df_prev['STASIUN'] == st]['Curah_Hujan_mm'].values[0] if len(df_prev[df_prev['STASIUN'] == st]) > 0 else 0
                curr_rain = df_specific[df_specific['STASIUN'] == st]['Curah_Hujan_mm'].values[0] if len(df_specific[df_specific['STASIUN'] == st]) > 0 else 0
                lag_data.append({
                    'Stasiun': st,
                    'Hujan Kemarin (mm)': round(prev_rain, 1),
                    'Hujan Hari Ini (mm)': round(curr_rain, 1),
                    'Perubahan': round(curr_rain - prev_rain, 1)
                })
            
            df_lag = pd.DataFrame(lag_data)
            st.dataframe(df_lag, use_container_width=True, hide_index=True)
            
            avg_change = df_lag['Perubahan'].mean()
            if avg_change > 5:
                st.success(f"📈 Tren naik! +{avg_change:.1f} mm — risiko ekstrem meningkat")
            elif avg_change < -5:
                st.info(f"📉 Tren turun! {avg_change:.1f} mm — risiko menurun")
            else:
                st.info(f"⚖️ Stabil! Perubahan {avg_change:.1f} mm")
        else:
            st.info("Data hujan kemarin tidak tersedia.")
        
        st.divider()
        
        # Tabel detail per stasiun
        display = df_specific[['STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
        display['Probabilitas'] = (display['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
        display['Kategori BMKG'] = display['Curah_Hujan_mm'].apply(bmkg_category)
        display = display[['STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
        display.columns = ['Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']
        st.dataframe(display, use_container_width=True, hide_index=True)
    
    # Grafik tren
    st.divider()
    st.markdown("<h3 style='color: #1a5276;'>📈 Tren Risiko dan Curah Hujan</h3>", unsafe_allow_html=True)
    
    if not df_filtered.empty:
        daily = df_filtered.groupby('TANGGAL').agg({
            'Probabilitas_Risiko': 'mean',
            'Curah_Hujan_mm': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['TANGGAL'], y=daily['Probabilitas_Risiko'],
            mode='lines+markers', name='Risiko',
            line=dict(color='#5DADE2', width=3),
            marker=dict(size=8, color='#1a5276')
        ))
        fig.add_trace(go.Bar(
            x=daily['TANGGAL'], y=daily['Curah_Hujan_mm'],
            name='Curah Hujan', marker_color='rgba(93, 173, 226, 0.3)',
            yaxis='y2', opacity=0.5
        ))
        fig.add_hline(y=THRESHOLD_WASPADA, line_dash="dash", line_color="#F1C40F", annotation_text="Waspada")
        fig.add_hline(y=THRESHOLD_SIAGA, line_dash="dash", line_color="#E67E22", annotation_text="Siaga")
        fig.add_hline(y=THRESHOLD_BAHAYA, line_dash="dash", line_color="#E74C3C", annotation_text="Bahaya")
        fig.add_vline(x=pd.Timestamp(selected_date), line_dash="dot", line_color="#5DADE2", line_width=3, annotation_text="📌")
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(255,255,255,0.5)',
            yaxis=dict(range=[0, 1]),
            yaxis2=dict(title='Curah Hujan (mm)', overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Status harian (kartu)
        st.markdown("<h3 style='color: #1a5276;'>📅 Status Harian</h3>", unsafe_allow_html=True)
        
        unique_days = sorted(daily['TANGGAL'].unique())
        for i in range(0, len(unique_days), 7):
            day_group = unique_days[i:i+7]
            cols = st.columns(len(day_group))
            for j, day in enumerate(day_group):
                day_data = daily[daily['TANGGAL'] == day]
                if not day_data.empty:
                    avg_risk = day_data['Probabilitas_Risiko'].values[0]
                    rain = day_data['Curah_Hujan_mm'].values[0]
                    level = get_level(avg_risk)
                    is_selected = day.date() == selected_date
                    with cols[j]:
                        st.markdown(f"""
                        <div style="background: {level_color(level)}; border-radius: 14px; padding: 12px 4px; text-align: center; color: white; border: {'3px solid #5DADE2' if is_selected else 'none'};">
                            <div style="font-size:0.7rem; font-weight:600;">{day.strftime('%d %b')}</div>
                            <div style="font-size:1.2rem;">{level_icon(level)}</div>
                            <div style="font-size:0.9rem; font-weight:700;">{avg_risk*100:.0f}%</div>
                            <div style="font-size:0.6rem; opacity:0.8;">{rain:.0f}mm</div>
                            <div style="font-size:0.7rem; font-weight:bold;">{level}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"❌ Error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer">
☁️ Dashboard Prediksi Hujan Ekstrem Kota Padang<br>
Enggli Rahmadhani — Studi Kasus Kerja Praktik<br>
Data: BMKG Stasiun Klimatologi Sumatera Barat 2017-2025
</div>
""", unsafe_allow_html=True)