import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ============================================================
# KONFIGURASI - PAKAI PATH RELATIF
# ============================================================
st.set_page_config(page_title="Peringatan Dini Hujan Ekstrem - Padang", layout="wide", page_icon="🌧️")

THRESHOLD_WASPADA = 0.25
THRESHOLD_SIAGA = 0.40
THRESHOLD_BAHAYA = 0.60

# PATH RELATIF (sesuai dengan struktur di GitHub)
DATA_PATH = "03_Hasil/Data_Padang_Filtered.csv"
FORECAST_PATH = "03_Hasil/prediksi_tahunan_2026_2028.csv"

# ============================================================
# AUTO-GENERATE FILE PREDIKSI JIKA BELUM ADA
# ============================================================
if not os.path.exists(FORECAST_PATH):
    st.warning("⏳ File prediksi belum ditemukan. Menghasilkan data prediksi 2026-2028 otomatis...")
    try:
        # Coba jalankan generate_forecast_manual.py
        if os.path.exists("generate_forecast_manual.py"):
            result = subprocess.run(["python", "generate_forecast_manual.py"], capture_output=True, text=True)
            if os.path.exists(FORECAST_PATH):
                st.success("✅ Data prediksi berhasil digenerate!")
            else:
                st.error(f"❌ Gagal generate: {result.stderr}")
        else:
            st.error("❌ File generate_forecast_manual.py tidak ditemukan.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================================================
# CSS HEADER
# ============================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #4A90D9 0%, #87CEEB 35%, #B8E0F0 70%, #E8F4F8 100%); }
.weather-header { position: relative; height: 180px; overflow: hidden; border-radius: 20px; background: linear-gradient(135deg, #2C5F8A 0%, #4A90D9 50%, #6FB1E8 100%); margin-bottom: 25px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); }
.cloud { position: absolute; background: white; border-radius: 50px; opacity: 0.85; }
.cloud::before, .cloud::after { content: ''; position: absolute; background: white; border-radius: 50%; }
.cloud1 { width: 100px; height: 35px; top: 25px; left: -120px; animation: floatCloud 18s linear infinite; }
.cloud1::before { width: 55px; height: 55px; top: -25px; left: 10px; }
.cloud1::after { width: 40px; height: 40px; top: -18px; left: 55px; }
.cloud2 { width: 130px; height: 40px; top: 70px; left: -160px; animation: floatCloud 25s linear infinite; animation-delay: -8s; opacity: 0.7; }
.cloud2::before { width: 65px; height: 65px; top: -30px; left: 15px; }
.cloud2::after { width: 45px; height: 45px; top: -20px; left: 70px; }
.cloud3 { width: 80px; height: 28px; top: 115px; left: -100px; animation: floatCloud 15s linear infinite; animation-delay: -4s; opacity: 0.6; }
.cloud3::before { width: 40px; height: 40px; top: -18px; left: 8px; }
.cloud3::after { width: 30px; height: 30px; top: -12px; left: 42px; }
@keyframes floatCloud { from { transform: translateX(0); } to { transform: translateX(calc(100vw + 200px)); } }
.rain-drop { position: absolute; bottom: 100%; width: 2px; height: 18px; background: linear-gradient(180deg, rgba(255,255,255,0), rgba(200,230,255,0.9)); animation: fallRain linear infinite; }
@keyframes fallRain { to { transform: translateY(220px); opacity: 0; } }
.header-title { position: relative; z-index: 10; color: white; text-align: center; padding-top: 55px; text-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.header-title h1 { font-size: 2.2rem; margin-bottom: 5px; }
.header-title p { font-size: 1rem; opacity: 0.95; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.85); border-radius: 16px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #4A90D9; }
div[data-testid="stExpander"] { background: rgba(255,255,255,0.7); border-radius: 14px; }
.pulse-warning { animation: pulseWarning 1.5s ease-in-out infinite; }
@keyframes pulseWarning { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,75,75,0.5); } 50% { box-shadow: 0 0 0 12px rgba(255,75,75,0); } }
</style>
""", unsafe_allow_html=True)

def render_weather_header(avg_risk):
    n_drops = int(15 + avg_risk * 45)
    rain_html = ""
    for i in range(n_drops):
        left = np.random.randint(0, 100)
        delay = round(np.random.uniform(0, 2), 2)
        duration = round(np.random.uniform(0.6, 1.3), 2)
        rain_html += f'<div class="rain-drop" style="left:{left}%; animation-delay:{delay}s; animation-duration:{duration}s;"></div>'
    header_html = f"""
    <div class="weather-header">
        <div class="cloud cloud1"></div>
        <div class="cloud cloud2"></div>
        <div class="cloud cloud3"></div>
        {rain_html}
        <div class="header-title">
            <h1>🌧️ Sistem Peringatan Dini Hujan Ekstrem</h1>
            <p>Kota Padang — Parak Karakah • Nanggalo • Lubuk Minturun</p>
            <p style="font-size:0.8rem; opacity:0.7;">BMKG Kota Padang — Prakiraan Berbasis Dampak (Impact-Based Forecast)</p>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# ============================================================
# FUNGSI KATEGORI BMKG
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
    if prob < THRESHOLD_WASPADA: return 'Aman'
    elif prob < THRESHOLD_SIAGA: return 'Waspada'
    elif prob < THRESHOLD_BAHAYA: return 'Siaga'
    else: return 'Bahaya'

# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    render_weather_header(0.3)
    
    # SIDEBAR: Mode Operasi
    st.sidebar.markdown("## 🛠️ Mode Operasi")
    mode = st.sidebar.radio(
        "Pilih Mode:",
        [
            "📊 Evaluasi Historis",
            "🔮 Prakiraan 30 Hari (Upload CSV)",
            "📅 Prediksi Tahunan 2026-2028 (Hybrid)"
        ],
        index=0
    )
    
    # ============================================================
    # MODE 1: EVALUASI HISTORIS (Menampilkan Performa Model)
    # ============================================================
    if mode == "📊 Evaluasi Historis":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 Filter Data Historis")
        
        st.markdown("""
        ### 📊 Performa Model Final XGBoost (Spasial 9 Stasiun)
        
        Berikut adalah hasil evaluasi model pada data test set (2024-2025) dengan berbagai threshold:
        """)
        
        # Tabel Performa Model
        data_performa = {
            'Threshold': ['0.40 (Siaga)', '0.45', '0.50', '0.55', '0.60 (Bahaya)'],
            'Recall': ['84.5%', '75.5%', '63.6%', '53.6%', '33.6%'],
            'Precision': ['7.9%', '8.7%', '9.2%', '10.4%', '10.7%'],
            'F1-Score': ['0.145', '0.156', '0.161', '0.175', '0.151'],
            'TP (tertangkap)': ['93', '83', '70', '59', '37'],
            'FP (alarm palsu)': ['1077', '868', '690', '506', '308'],
            'FN (terlewat)': ['17', '27', '40', '51', '73']
        }
        df_performa = pd.DataFrame(data_performa)
        st.dataframe(df_performa, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### 📌 Kesimpulan Evaluasi
        
        - **Threshold 0.40** dipilih sebagai **prioritas Recall** (84.5%) untuk memaksimalkan deteksi kejadian ekstrem.
        - Konsekuensi: **banyak false alarm (1077)** — ini wajar untuk sistem peringatan dini (early warning) yang mengutamakan keselamatan.
        - **Precision rendah (7.9%)** menunjukkan keterbatasan data curah hujan historis tanpa variabel atmosfer pendukung.
        
        ### 🎯 Rekomendasi untuk BMKG
        
        - Gunakan **Threshold 0.40** untuk peringatan **Siaga** (tindakan kesiapsiagaan).
        - Gunakan **Threshold 0.60** untuk peringatan **Bahaya** (tindakan evakuasi).
        """)
        
        st.divider()
        st.caption("Mode Evaluasi Historis — Data: BMKG Kota Padang 2017-2025")
    
    # ============================================================
    # MODE 2: PRAKIRAAN 30 HARI (UPLOAD CSV)
    # ============================================================
    elif mode == "🔮 Prakiraan 30 Hari (Upload CSV)":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📤 Upload Data Prakiraan BMKG")
        st.sidebar.markdown("""
        Format CSV:
        `TANGGAL`, `Parak_Karakah`, `Nanggalo`, `Lubuk_Minturun`
        """)
        
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['TANGGAL'])
            st.success(f"✅ {len(df)} hari prakiraan diupload!")
            
            st.subheader("📋 Preview Data Prakiraan")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.info("""
            **Proses Prediksi:**
            1. Data prakiraan dari BMKG akan diproses dengan fitur lag dan rolling.
            2. XGBoost akan menghitung probabilitas risiko ekstrem per hari.
            3. Hasil: Level peringatan (Aman, Waspada, Siaga, Bahaya).
            
            📌 **Fitur ini sedang dalam pengembangan untuk integrasi penuh.**
            """)
        else:
            st.info("📤 Upload file CSV prakiraan di sidebar untuk memulai.")
    
    # ============================================================
    # MODE 3: PREDIKSI TAHUNAN 2026-2028 (HYBRID)
    # ============================================================
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📅 Filter Tanggal (2026-2028)")
        st.sidebar.markdown("Cari tahu prediksi risiko untuk tanggal tertentu!")
        
        # Cek apakah file prediksi ada
        if not os.path.exists(FORECAST_PATH):
            st.error(f"❌ File prediksi tidak ditemukan di: `{FORECAST_PATH}`")
            st.warning("⚠️ Jalankan `generate_forecast_manual.py` terlebih dahulu, atau pastikan file CSV sudah di-generate.")
            st.stop()
        
        # Load data prediksi
        df_forecast = pd.read_csv(FORECAST_PATH, parse_dates=['TANGGAL'])
        df_forecast['Level'] = df_forecast['Level'].fillna('Aman')
        
        # Pastikan kolom yang diperlukan ada
        required_cols = ['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']
        if not all(col in df_forecast.columns for col in required_cols):
            # Coba mapping jika nama kolom beda
            if 'Curah_Hujan_mm' not in df_forecast.columns and 'Curah_Hujan' in df_forecast.columns:
                df_forecast.rename(columns={'Curah_Hujan': 'Curah_Hujan_mm'}, inplace=True)
            if 'Probabilitas_Risiko' not in df_forecast.columns and 'Probabilitas' in df_forecast.columns:
                df_forecast.rename(columns={'Probabilitas': 'Probabilitas_Risiko'}, inplace=True)
        
        # Filter data berdasarkan rentang tanggal
        min_date = df_forecast['TANGGAL'].min().date()
        max_date = df_forecast['TANGGAL'].max().date()
        
        # Default ke tanggal tengah atau awal
        default_date = datetime(2026, 1, 25).date() if datetime(2026, 1, 25).date() >= min_date else min_date
        
        selected_date = st.sidebar.date_input(
            "📆 Pilih Tanggal",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )
        
        view_range = st.sidebar.slider("🔍 Tampilkan +/- hari", 1, 15, 3)
        
        # Filter data untuk tanggal yang dipilih + range
        start_filter = selected_date - timedelta(days=view_range)
        end_filter = selected_date + timedelta(days=view_range)
        
        df_filtered = df_forecast[
            (df_forecast['TANGGAL'] >= pd.Timestamp(start_filter)) &
            (df_forecast['TANGGAL'] <= pd.Timestamp(end_filter))
        ]
        
        df_specific = df_forecast[df_forecast['TANGGAL'] == pd.Timestamp(selected_date)]
        
        # ============================================================
        # HEADER & METRIK
        # ============================================================
        st.subheader(f"📊 Prediksi untuk Tanggal: **{selected_date.strftime('%d %B %Y')}**")
        st.caption(f"Menampilkan data dari {start_filter.strftime('%d %b %Y')} hingga {end_filter.strftime('%d %b %Y')}")
        
        if not df_specific.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            avg_risk = df_specific['Probabilitas_Risiko'].mean()
            max_level = df_specific['Level'].max()
            avg_rain = df_specific['Curah_Hujan_mm'].mean()
            count_alert = len(df_specific[df_specific['Level'].isin(['Siaga', 'Bahaya'])])
            max_rain = df_specific['Curah_Hujan_mm'].max()
            
            with col1:
                st.metric("🌧️ Rata-rata Risiko", f"{avg_risk*100:.1f}%")
            with col2:
                icon_level = "🟢" if max_level == "Aman" else "🟡" if max_level == "Waspada" else "🟠" if max_level == "Siaga" else "🔴"
                st.metric("🚨 Level Tertinggi", f"{icon_level} {max_level}")
            with col3:
                st.metric("🌊 Rata-rata Hujan", f"{avg_rain:.1f} mm")
            with col4:
                st.metric("⚠️ Stasiun Siaga/Bahaya", f"{count_alert} dari 3")
            with col5:
                st.metric("📈 Hujan Maksimum", f"{max_rain:.1f} mm")
        else:
            st.warning("Tidak ada data untuk tanggal yang dipilih.")
        
        st.divider()
        
        # ============================================================
        # TABEL DETAIL TANGGAL DIPILIH
        # ============================================================
        st.subheader(f"📋 Detail Prediksi {selected_date.strftime('%d %B %Y')} (3 Stasiun)")
        
        if not df_specific.empty:
            display_table = df_specific[['STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
            display_table['Probabilitas'] = (display_table['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
            display_table['Kategori BMKG'] = display_table['Curah_Hujan_mm'].apply(bmkg_category)
            display_table = display_table[['STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
            display_table.columns = ['Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas Risiko', 'Level Peringatan']
            
            st.dataframe(display_table, use_container_width=True, height=200)
        else:
            st.info("Tidak ada data untuk tanggal ini.")
        
        st.divider()
        
        # ============================================================
        # GRAFIK TREND
        # ============================================================
        st.subheader(f"📈 Tren Risiko {view_range*2 + 1} Hari Sekitar {selected_date.strftime('%d %b %Y')}")
        
        if not df_filtered.empty:
            daily_avg = df_filtered.groupby('TANGGAL').agg({
                'Probabilitas_Risiko': 'mean',
                'Curah_Hujan_mm': 'mean'
            }).reset_index()
            
            daily_level = df_filtered.groupby('TANGGAL')['Level'].agg(lambda x: x.value_counts().index[0]).reset_index()
            daily_avg = daily_avg.merge(daily_level, on='TANGGAL')
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=daily_avg['TANGGAL'],
                y=daily_avg['Curah_Hujan_mm'],
                name='Curah Hujan (mm)',
                marker_color='lightblue',
                yaxis='y2',
                opacity=0.6
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_avg['TANGGAL'],
                y=daily_avg['Probabilitas_Risiko'],
                mode='lines+markers',
                name='Probabilitas Risiko',
                line=dict(color='#4A90D9', width=2.5),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=THRESHOLD_WASPADA, line_dash="dash", line_color="orange", line_width=1.5, annotation_text="Waspada")
            fig.add_hline(y=THRESHOLD_SIAGA, line_dash="dash", line_color="red", line_width=2, annotation_text="Siaga")
            fig.add_hline(y=THRESHOLD_BAHAYA, line_dash="dash", line_color="darkred", line_width=2, annotation_text="Bahaya")
            
            fig.add_vline(x=pd.Timestamp(selected_date), line_dash="dot", line_color="green", line_width=3, annotation_text="📌 Tanggal Dipilih")
            
            fig.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Probabilitas Risiko",
                yaxis=dict(range=[0, 1]),
                yaxis2=dict(title="Curah Hujan (mm)", overlaying='y', side='right'),
                height=450,
                plot_bgcolor='rgba(255,255,255,0.6)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ============================================================
            # KARTU STATUS PER HARI
            # ============================================================
            st.subheader(f"📅 Status Harian ({start_filter.strftime('%d %b')} - {end_filter.strftime('%d %b')})")
            
            unique_days = sorted(daily_avg['TANGGAL'].unique())
            cols_per_row = 7
            
            for i in range(0, len(unique_days), cols_per_row):
                day_group = unique_days[i:i+cols_per_row]
                cols = st.columns(len(day_group))
                for j, day in enumerate(day_group):
                    day_data = daily_avg[daily_avg['TANGGAL'] == day]
                    if not day_data.empty:
                        avg_risk = day_data['Probabilitas_Risiko'].values[0]
                        level = day_data['Level'].values[0]
                        rain = day_data['Curah_Hujan_mm'].values[0]
                        
                        risk_pct = avg_risk * 100
                        if level == 'Aman':
                            bg = "linear-gradient(135deg, #6FCF97, #4AB88A)"; icon = "☀️"; label = "AMAN"
                        elif level == 'Waspada':
                            bg = "linear-gradient(135deg, #F2C94D, #F2994A)"; icon = "⛅"; label = "WASPADA"
                        elif level == 'Siaga':
                            bg = "linear-gradient(135deg, #FF6B6B, #FF4B4B)"; icon = "🌧️"; label = "SIAGA"
                        else:
                            bg = "linear-gradient(135deg, #8B0000, #CC0000)"; icon = "⛈️"; label = "BAHAYA"
                        
                        is_selected = day.date() == selected_date
                        border = "3px solid #FFD700" if is_selected else "none"
                        
                        with cols[j]:
                            st.markdown(f"""
                            <div style="background: {bg}; border-radius: 14px; padding: 12px 4px; text-align: center; color: white; border: {border};">
                                <div style="font-size:0.7rem; font-weight:600;">{day.strftime('%d %b')}</div>
                                <div style="font-size:1.2rem; margin:2px 0;">{icon}</div>
                                <div style="font-size:0.9rem; font-weight:700;">{risk_pct:.0f}%</div>
                                <div style="font-size:0.6rem; opacity:0.8;">{rain:.0f}mm</div>
                                <div style="font-size:0.7rem; font-weight:bold;">{label}</div>
                                {"<div style='font-size:0.6rem; background:gold; color:black; border-radius:4px; padding:1px 4px; margin-top:2px;'>📌</div>" if is_selected else ""}
                            </div>
                            """, unsafe_allow_html=True)
            
            st.divider()
            
            # ============================================================
            # TABEL LENGKAP
            # ============================================================
            with st.expander("📋 Tabel Lengkap (Semua Hari & Stasiun)"):
                full_table = df_filtered[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
                full_table['Probabilitas'] = (full_table['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
                full_table['Kategori BMKG'] = full_table['Curah_Hujan_mm'].apply(bmkg_category)
                full_table = full_table[['TANGGAL', 'STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
                full_table.columns = ['Tanggal', 'Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']
                
                st.dataframe(full_table.sort_values('Tanggal'), use_container_width=True, height=400)
                
                csv_download = full_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Data Range (CSV)", csv_download, f"prediksi_{selected_date.strftime('%Y%m%d')}.csv", "text/csv")
        
        else:
            st.warning("Tidak ada data dalam range yang dipilih.")
        
        # ============================================================
        # INFORMASI METODOLOGI
        # ============================================================
        with st.expander("📖 Tentang Prediksi Tahunan (Hybrid SARIMA-XGBoost)"):
            st.markdown("""
            ### Metodologi Hybrid
            
            **Lapis 1: SARIMA (Seasonal ARIMA)**
            - Agregasi data historis (2017-2025) ke level bulanan.
            - Memodelkan pola musiman (seasonality) 12 bulan.
            - Memprediksi total curah hujan bulanan untuk 2026-2028.
            
            **Lapis 2: XGBoost (Klasifikasi Risiko)**
            - Menggunakan model XGBoost yang sudah dilatih dengan data historis harian.
            - Mensimulasikan data harian berdasarkan proyeksi SARIMA.
            - Menghitung probabilitas risiko ekstrem untuk setiap hari.
            
            **Output untuk BMKG:**
            - Level peringatan (Aman, Waspada, Siaga, Bahaya) per hari.
            - Estimasi curah hujan harian.
            - Rekomendasi bulanan untuk mitigasi bencana.
            
            ⚠️ **Catatan:** Prediksi ini adalah **proyeksi tren/outlook musiman**. Semakin jauh ke depan, semakin besar ketidakpastiannya. Gunakan sebagai alat bantu perencanaan.
            """)
        
        st.caption(f"🎓 Prototipe Dashboard — Prediksi Hujan Ekstrem Kota Padang | Data: BMKG 2017-2028 | Periode: {min_date} s.d. {max_date}")

if __name__ == "__main__":
    main()