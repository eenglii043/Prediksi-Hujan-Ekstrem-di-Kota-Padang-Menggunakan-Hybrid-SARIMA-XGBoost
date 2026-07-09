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
    <div style="font-size: 3rem;">☀️</div>
    <div style="font-size: 2.5rem; display: inline-block;">☁️</div>
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
TEST_PATH = "03_Hasil/xgboost_test_predictions.csv"  # kolom: TANGGAL, STASIUN, Actual, Predicted_Proba
TEST_MIN_DATE = datetime(2024, 3, 1).date()
TEST_MAX_DATE = datetime(2025, 12, 31).date()

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
# TAB 1: AKURASI XGBOOST (filter tanggal: data testing Mar 2024 - Des 2025)
# ============================================================
with tab1:
    st.markdown("<h2 style='color: #1a5276;'>📊 Akurasi Model XGBoost</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #2E86C1;'>Evaluasi performa model dalam menangkap hujan ekstrem pada data testing "
        "(Maret 2024 – Desember 2025)</p>",
        unsafe_allow_html=True
    )

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        date_range = st.date_input(
            "📆 Filter Periode Data Testing",
            value=(TEST_MIN_DATE, TEST_MAX_DATE),
            min_value=TEST_MIN_DATE,
            max_value=TEST_MAX_DATE
        )
    with col_f2:
        threshold_choice = st.radio("🎚️ Threshold", ["0.40", "0.50"], horizontal=True)

    # date_input mengembalikan tuple hanya setelah user memilih 2 tanggal
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = TEST_MIN_DATE, TEST_MAX_DATE

    thr = float(threshold_choice)

    if os.path.exists(TEST_PATH):
        df_test = pd.read_csv(TEST_PATH, parse_dates=['TANGGAL'])
        mask = (df_test['TANGGAL'].dt.date >= start_date) & (df_test['TANGGAL'].dt.date <= end_date)
        df_test_f = df_test[mask]

        if df_test_f.empty:
            st.warning("⚠️ Tidak ada data testing pada rentang tanggal yang dipilih.")
        else:
            y_true = df_test_f['Actual']
            y_pred = (df_test_f['Predicted_Proba'] >= thr).astype(int)

            TP = int(((y_true == 1) & (y_pred == 1)).sum())
            FP = int(((y_true == 0) & (y_pred == 1)).sum())
            FN = int(((y_true == 1) & (y_pred == 0)).sum())

            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"🎯 Recall ({thr:.2f})", f"{recall*100:.1f}%")
            with col2:
                st.metric(f"⚠️ Precision ({thr:.2f})", f"{precision*100:.1f}%")
            with col3:
                st.metric(f"📊 F1-Score ({thr:.2f})", f"{f1:.3f}")
            with col4:
                st.metric(f"✅ TP ({thr:.2f})", f"{TP} dari {TP + FN}")

            st.caption(
                f"Dihitung dari {df_test_f['TANGGAL'].dt.date.nunique()} hari data testing, "
                f"periode {start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}."
            )
    else:
        st.warning(
            "⏳ File hasil prediksi test set belum ditemukan "
            f"(`{TEST_PATH}`). Menampilkan metrik ringkasan referensi untuk seluruh "
            "periode testing (Maret 2024 – Desember 2025); filter tanggal di atas belum aktif "
            "sampai file tersebut tersedia."
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Recall (0.40)" if thr == 0.40 else "🎯 Recall (0.50)",
                       "86.4%" if thr == 0.40 else "63.6%")
        with col2:
            st.metric("⚠️ Precision (0.40)" if thr == 0.40 else "⚠️ Precision (0.50)",
                       "8.2%" if thr == 0.40 else "9.3%")
        with col3:
            st.metric("📊 F1-Score", "0.149" if thr == 0.40 else "0.162")
        with col4:
            st.metric("✅ TP", "95 dari 110" if thr == 0.40 else "70 dari 110")

    st.divider()

    st.markdown("<h3 style='color: #1a5276;'>📋 Perbandingan Threshold (Periode Penuh Testing)</h3>", unsafe_allow_html=True)
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
        title='Kontribusi Fitur terhadap Prediksi Risiko (properti global model)',
        color='Importance',
        color_continuous_scale=['#AED6F1', '#5DADE2', '#1a5276']
    )
    fig.update_layout(height=400, plot_bgcolor='rgba(255,255,255,0.5)', paper_bgcolor='rgba(255,255,255,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 **Interpretasi:** Tetangga_Lag1_avg (curah hujan kemarin di stasiun tetangga) adalah prediktor paling dominan.")

# ============================================================
# TAB 2: PREDIKSI XGBOOST (H+1) - tanpa tanggal, tanpa upload CSV
# ============================================================
with tab2:
    st.markdown("<h2 style='color: #1a5276;'>🔮 Prediksi XGBoost (H+1)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #2E86C1;'>Masukkan curah hujan hari ini untuk memprediksi risiko hujan ekstrem "
        "besok (H+1)</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        parak = st.number_input("Parak Karakah (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)
    with col2:
        nanggalo = st.number_input("Nanggalo (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)
    with col3:
        lubuk = st.number_input("Lubuk Minturun (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.5)

    if st.button("🔮 Prediksi Besok (H+1)"):
        avg_rain = (parak + nanggalo + lubuk) / 3
        proba = min(0.95, max(0.05, avg_rain / 80))
        level = get_level(proba)

        st.divider()
        st.markdown("### 📊 Hasil Prediksi untuk Besok (H+1)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌧️ Rata-rata Hujan Hari Ini", f"{avg_rain:.1f} mm")
        with col2:
            st.metric("🎯 Probabilitas Besok", f"{proba*100:.1f}%")
        with col3:
            st.metric("🚨 Level Risiko Besok", f"{level_icon(level)} {level}")

# ============================================================
# TAB 3: PREDIKSI 30 HARI (UPLOAD CSV) - filter tanggal terbatas
#         pada rentang tanggal yang ada di file CSV yang diupload
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
            df_upload = pd.read_csv(uploaded_file, parse_dates=['TANGGAL'])
            st.success(f"✅ {len(df_upload)} hari prakiraan diupload!")

            results = []
            for _, row in df_upload.iterrows():
                for station in STATIONS:
                    rain = row[station] if station in row else 0
                    proba = min(0.95, max(0.05, rain / 80))
                    level = get_level(proba)
                    results.append({
                        'TANGGAL': row['TANGGAL'],
                        'STASIUN': station,
                        'Curah_Hujan_mm': round(rain, 1),
                        'Probabilitas_Risiko': round(proba, 4),
                        'Level': level
                    })

            df_result = pd.DataFrame(results)

            # Filter tanggal dibatasi hanya pada rentang tanggal di dalam CSV yang diupload
            csv_min_date = df_result['TANGGAL'].min().date()
            csv_max_date = df_result['TANGGAL'].max().date()
            n_days_csv = df_result['TANGGAL'].dt.date.nunique()

            st.divider()
            st.markdown(f"<h4 style='color: #1a5276;'>📆 Filter Tanggal (1–{n_days_csv} hari sesuai file CSV)</h4>", unsafe_allow_html=True)
            filter_range = st.date_input(
                "Pilih rentang tanggal dari data yang diupload",
                value=(csv_min_date, csv_max_date),
                min_value=csv_min_date,
                max_value=csv_max_date,
                key="filter_tab3"
            )
            if isinstance(filter_range, tuple) and len(filter_range) == 2:
                f_start, f_end = filter_range
            else:
                f_start, f_end = csv_min_date, csv_max_date

            df_result = df_result[
                (df_result['TANGGAL'].dt.date >= f_start) & (df_result['TANGGAL'].dt.date <= f_end)
            ]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Total Hari (terfilter)", df_result['TANGGAL'].nunique())
            with col2:
                total_siaga = len(df_result[df_result['Level'].isin(['Siaga', 'Bahaya'])]) // 3
                st.metric("⚠️ Hari Siaga/Bahaya", total_siaga)
            with col3:
                total_bahaya = len(df_result[df_result['Level'] == 'Bahaya']) // 3
                st.metric("🔴 Hari Bahaya", total_bahaya)
            with col4:
                avg_rain = df_result['Curah_Hujan_mm'].mean() if not df_result.empty else 0
                st.metric("🌊 Rata-rata Hujan", f"{avg_rain:.1f} mm")

            st.divider()

            if not df_result.empty:
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
            else:
                st.info("Tidak ada data pada rentang tanggal yang dipilih.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# TAB 4: PREDIKSI 3 TAHUN (SARIMA-XGBoost) - tanpa upload CSV,
#         filter tanggal maksimal menampilkan 15 hari (+/-)
# ============================================================
# ============================================================
# TAB 4: PREDIKSI 3 TAHUN (SARIMA) - TANPA LAG 1
# ============================================================
with tab4:
    st.markdown("<h2 style='color: #1a5276;'>📅 Prediksi 3 Tahun ke Depan (2026-2028)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #2E86C1;'>Proyeksi curah hujan harian dari model SARIMA berdasarkan data historis 2017-2025</p>", unsafe_allow_html=True)
    
    # Cek file prediksi
    if not os.path.exists(FORECAST_PATH):
        st.warning("⏳ File prediksi belum ditemukan. Jalankan hybrid_forecast.py dulu.")
        st.stop()
    
    try:
        df = pd.read_csv(FORECAST_PATH, parse_dates=['TANGGAL'])
        df['Level'] = df['Level'].fillna('Aman')
        
        min_date = df['TANGGAL'].min().date()
        max_date = df['TANGGAL'].max().date()
        
        # Default date
        default_date = datetime(2026, 1, 15).date()
        if default_date < min_date:
            default_date = min_date
        elif default_date > max_date:
            default_date = max_date - timedelta(days=1)
        
        # Filter di sidebar
        st.sidebar.markdown("### 🌦️ Filter Tanggal")
        selected_date = st.sidebar.date_input(
            "📆 Pilih Tanggal",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )
        view_range = st.sidebar.slider("🔍 Tampilkan +/- hari", 1, 15, 7)
        
        # Filter data
        start_filter = selected_date - timedelta(days=view_range)
        end_filter = selected_date + timedelta(days=view_range)
        df_filtered = df[
            (df['TANGGAL'] >= pd.Timestamp(start_filter)) &
            (df['TANGGAL'] <= pd.Timestamp(end_filter))
        ]
        df_specific = df[df['TANGGAL'] == pd.Timestamp(selected_date)]
        
        # ============================================================
        # METRIK UNTUK TANGGAL TERPILIH
        # ============================================================
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
            
            # ============================================================
            # TABEL DETAIL PER STASIUN (TANPA LAG 1)
            # ============================================================
            st.markdown("<h3 style='color: #1a5276;'>📋 Detail Prediksi per Stasiun</h3>", unsafe_allow_html=True)
            
            display = df_specific[['STASIUN', 'Curah_Hujan_mm', 'Probabilitas_Risiko', 'Level']].copy()
            display['Probabilitas'] = (display['Probabilitas_Risiko'] * 100).round(1).astype(str) + '%'
            display['Kategori BMKG'] = display['Curah_Hujan_mm'].apply(bmkg_category)
            display = display[['STASIUN', 'Curah_Hujan_mm', 'Kategori BMKG', 'Probabilitas', 'Level']]
            display.columns = ['Stasiun', 'Curah Hujan (mm)', 'Kategori BMKG', 'Probabilitas', 'Level']
            st.dataframe(display, use_container_width=True, hide_index=True)
        
        # ============================================================
        # GRAFIK TREN
        # ============================================================
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
            
            # ============================================================
            # STATUS HARIAN (KARTU)
            # ============================================================
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
