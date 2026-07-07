import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

# ============================================================
# KONFIGURASI
# ============================================================
DATA_PATH = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Padang_Filtered.csv"
MODEL_PATH = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\model_xgboost_final.pkl"
SCALER_PATH = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\scaler_final.pkl"
FEATURES_PATH = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\feature_cols_final.pkl"

OUTPUT_DIR = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "prediksi_tahunan_2026_2028.csv")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "step25_hybrid_forecast_report.txt")

print("="*60)
print("HYBRID SARIMA + XGBOOST - PREDIKSI TAHUNAN 2026-2028")
print("="*60)

try:
    # ============================================================
    # 1. LOAD DATA HISTORIS
    # ============================================================
    print("\n[1] Loading data historis...")
    df_raw = pd.read_csv(DATA_PATH, parse_dates=['DATA TIMESTAMP'])
    df_raw['TANGGAL'] = pd.to_datetime(df_raw['DATA TIMESTAMP'].dt.date)

    target_ids = ['13710202a', '13711001a', '13711102a']
    target_names = {
        '13710202a': 'Parak_Karakah',
        '13711001a': 'Nanggalo',
        '13711102a': 'Lubuk_Minturun'
    }
    
    df_target = df_raw[df_raw['POS HUJAN ID'].isin(target_ids)].copy()
    df_target['RAINFALL DAY MM'] = df_target['RAINFALL DAY MM'].replace([9999, 9999.0], np.nan)
    df_target['RAINFALL DAY MM'] = df_target['RAINFALL DAY MM'].replace([8888, 8888.0], 0)
    df_target['STASIUN'] = df_target['POS HUJAN ID'].map(target_names)
    
    df_wide = df_target.pivot_table(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM', aggfunc='mean')
    full_range = pd.date_range('2017-01-01', '2025-12-31', freq='D')
    df_wide = df_wide.reindex(full_range)
    df_wide.index.name = 'TANGGAL'
    
    knn = KNNImputer(n_neighbors=5, weights='distance')
    df_imp = pd.DataFrame(knn.fit_transform(df_wide), index=df_wide.index, columns=df_wide.columns)
    df_imp = df_imp.clip(lower=0)
    
    print(f"   Data historis: {len(df_imp)} hari, 3 stasiun")
    
    # ============================================================
    # 2. AGREGASI KE BULANAN
    # ============================================================
    print("\n[2] Agregasi ke bulanan...")
    df_monthly = df_imp.resample('M').sum()
    df_monthly.index = df_monthly.index.to_period('M')
    df_monthly['Rata_Rata'] = df_monthly.mean(axis=1)
    monthly_series = df_monthly['Rata_Rata']
    
    print(f"   Data bulanan: {len(monthly_series)} bulan (2017-2025)")
    
    # ============================================================
    # 3. FIT SARIMA
    # ============================================================
    print("\n[3] Fitting SARIMA...")
    
    # Coba import statsmodels, jika gagal beri pesan error
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
        
        # Uji stasioneritas
        result = adfuller(monthly_series.dropna())
        d = 1 if result[1] > 0.05 else 0
        print(f"   ADF p-value: {result[1]:.4f}, differencing: {d}")
        
        # Coba beberapa order sederhana
        best_aic = float('inf')
        best_order = (1, d, 1)
        best_seasonal = (1, 1, 1, 12)
        
        for p in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for Q in range(0, 2):
                        try:
                            model = SARIMAX(monthly_series, 
                                            order=(p, d, q),
                                            seasonal_order=(P, 1, Q, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                            results = model.fit(disp=False)
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_seasonal = (P, 1, Q, 12)
                        except:
                            continue
        
        print(f"   Best order: {best_order}, Seasonal: {best_seasonal}, AIC: {best_aic:.2f}")
        
        final_model = SARIMAX(monthly_series, 
                              order=best_order,
                              seasonal_order=best_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
        final_results = final_model.fit(disp=False)
        
        # Prediksi 36 bulan
        forecast_steps = 36
        forecast = final_results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Buat index tanggal
        last_date = monthly_series.index[-1].start_time
        future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        
        df_forecast = pd.DataFrame({
            'TANGGAL': future_months,
            'Prediksi_Total_Bulanan': forecast_mean.values,
            'CI_Lower': forecast_ci.iloc[:, 0].values,
            'CI_Upper': forecast_ci.iloc[:, 1].values
        })
        df_forecast['TANGGAL'] = df_forecast['TANGGAL'].dt.to_period('M')
        
        print(f"   Prediksi SARIMA: {forecast_steps} bulan (2026-2028)")
        
    except ImportError:
        print("   ⚠️ Library statsmodels tidak terinstall. Menggunakan pendekatan sederhana (rata-rata musiman).")
        # Fallback: gunakan rata-rata musiman
        monthly_avg = monthly_series.groupby(monthly_series.index.month).mean()
        future_months = pd.date_range(start='2026-01-01', periods=36, freq='M')
        pred_values = np.array([monthly_avg[m] for m in future_months.month])
        df_forecast = pd.DataFrame({
            'TANGGAL': future_months,
            'Prediksi_Total_Bulanan': pred_values,
            'CI_Lower': pred_values * 0.7,
            'CI_Upper': pred_values * 1.3
        })
        df_forecast['TANGGAL'] = df_forecast['TANGGAL'].dt.to_period('M')
    
    # ============================================================
    # 4. SIMULASI HARIAN DARI PREDIKSI BULANAN + XGBOOST
    # ============================================================
    print("\n[4] Simulasi harian & estimasi risiko XGBoost...")
    
    # Load model XGBoost
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        print("   ✅ Model XGBoost berhasil dimuat.")
    else:
        print("   ⚠️ Model XGBoost tidak ditemukan. Membuat model baru...")
        # Buat model sederhana jika tidak ada
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        scaler = None
        feature_cols = None
    
    # Ambil rata-rata harian historis per bulan
    df_imp['Bulan'] = df_imp.index.month
    monthly_avg = df_imp.groupby('Bulan').mean()
    
    results_daily = []
    
    for idx, row in df_forecast.iterrows():
        month_start = row['TANGGAL'].start_time
        month_end = row['TANGGAL'].end_time
        days_in_month = (month_end - month_start).days + 1
        
        daily_avg_target = row['Prediksi_Total_Bulanan'] / days_in_month
        
        month_num = row['TANGGAL'].month
        hist_pattern = monthly_avg.loc[month_num, ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']].values
        
        if hist_pattern.sum() > 0:
            scale_factor = daily_avg_target / hist_pattern.mean()
        else:
            scale_factor = daily_avg_target
        
        for day in range(days_in_month):
            current_date = month_start + pd.Timedelta(days=day)
            noise = np.random.normal(1, 0.3, 3)
            daily_rain = hist_pattern * scale_factor * noise
            daily_rain = np.maximum(daily_rain, 0)
            
            results_daily.append({
                'TANGGAL': current_date,
                'Parak_Karakah': daily_rain[0],
                'Nanggalo': daily_rain[1],
                'Lubuk_Minturun': daily_rain[2]
            })
    
    df_simulated = pd.DataFrame(results_daily)
    print(f"   Data simulasi harian: {len(df_simulated)} hari")
    
    # ============================================================
    # 5. PROSES FITUR & PREDIKSI XGBOOST
    # ============================================================
    print("\n[5] Menghitung fitur & prediksi risiko...")
    
    # Gunakan seed dari data historis
    historical_seed = df_imp.tail(30).copy()
    combined = pd.concat([historical_seed, df_simulated.set_index('TANGGAL')], axis=0)
    
    target_cols = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
    
    # Hitung fitur
    for col in target_cols:
        for lag in [1,2,3]:
            combined[f'{col}_lag{lag}'] = combined[col].shift(lag)
        combined[f'{col}_roll7'] = combined[col].shift(1).rolling(7, min_periods=1).mean()
        combined[f'{col}_roll30'] = combined[col].shift(1).rolling(30, min_periods=1).mean()
        combined[f'{col}_roll365'] = combined[col].shift(1).rolling(365, min_periods=1).mean()
    
    for st in target_cols:
        others = [c for c in target_cols if c != st]
        combined[f'{st}_Tetangga_Lag1_avg'] = combined[[f'{o}_lag1' for o in others]].mean(axis=1)
        combined[f'{st}_Tetangga_Roll7_avg'] = combined[[f'{o}_roll7' for o in others]].mean(axis=1)
    
    # Ambil bagian simulasi
    combined = combined.iloc[30:].copy()
    combined.reset_index(inplace=True)
    combined.rename(columns={'index': 'TANGGAL'}, inplace=True)
    
    # Reshape
    rows = []
    for st in target_cols:
        temp = pd.DataFrame()
        temp['TANGGAL'] = combined['TANGGAL']
        temp['STASIUN'] = st
        temp['Bulan'] = combined['TANGGAL'].dt.month
        temp['Hari_Tahun'] = combined['TANGGAL'].dt.dayofyear
        temp['Hari_Minggu'] = combined['TANGGAL'].dt.dayofweek
        for lag in [1,2,3]:
            temp[f'Lag{lag}'] = combined[f'{st}_lag{lag}']
        temp['Roll7'] = combined[f'{st}_roll7']
        temp['Roll30'] = combined[f'{st}_roll30']
        temp['Roll365'] = combined[f'{st}_roll365']
        temp['Tetangga_Lag1_avg'] = combined[f'{st}_Tetangga_Lag1_avg']
        temp['Tetangga_Roll7_avg'] = combined[f'{st}_Tetangga_Roll7_avg']
        temp['Curah_Hujan_mm'] = combined[st]
        rows.append(temp)
    
    df_long = pd.concat(rows, axis=0).reset_index(drop=True)
    df_long = df_long.dropna()
    print(f"   Shape data fitur: {df_long.shape}")
    
    # Prediksi dengan XGBoost
    if model is not None and scaler is not None and feature_cols is not None:
        df_encoded = pd.get_dummies(df_long, columns=['STASIUN'], drop_first=False)
        feature_vec = df_encoded[[c for c in df_encoded.columns if c in feature_cols]]
        X_scaled = scaler.transform(feature_vec)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        df_long['Probabilitas_Risiko'] = y_proba
    else:
        # Fallback: probabilitas dummy berdasarkan curah hujan
        df_long['Probabilitas_Risiko'] = np.clip(df_long['Curah_Hujan_mm'] / 100, 0, 1)
    
    # Level peringatan
    def get_level(prob):
        if prob < 0.25: return 'Aman'
        elif prob < 0.40: return 'Waspada'
        elif prob < 0.60: return 'Siaga'
        else: return 'Bahaya'
    
    df_long['Level'] = df_long['Probabilitas_Risiko'].apply(get_level)
    
    # ============================================================
    # 6. SIMPAN HASIL
    # ============================================================
    print("\n[6] Menyimpan hasil...")
    df_long.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✅ Data prediksi harian tersimpan di: {OUTPUT_CSV}")
    
    # Buat laporan
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("HYBRID SARIMA + XGBOOST - PREDIKSI TAHUNAN 2026-2028\n")
        f.write("="*60 + "\n")
        f.write(f"Total hari prediksi: {len(df_long)}\n")
        f.write(f"Rentang: {df_long['TANGGAL'].min()} s.d. {df_long['TANGGAL'].max()}\n\n")
        
        # Rekapitulasi per bulan
        df_long['Bulan_Tahun'] = df_long['TANGGAL'].dt.to_period('M')
        monthly_summary = df_long.groupby('Bulan_Tahun').agg({
            'Probabilitas_Risiko': 'mean',
            'Level': lambda x: x.value_counts().index[0],
            'Curah_Hujan_mm': 'mean'
        }).reset_index()
        monthly_summary.columns = ['Bulan', 'Rata_Risiko', 'Level_Dominan', 'Rata_Curah_Hujan']
        
        f.write("\nREKAPITULASI PER BULAN (2026-2028):\n")
        f.write(monthly_summary.to_string(index=False))
    
    print(f"   ✅ Laporan tersimpan di: {OUTPUT_REPORT}")
    
    # Visualisasi
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1 = axes[0]
        ax1.plot(monthly_series.index.astype(str), monthly_series.values, label='Historis', color='blue')
        ax1.plot(df_forecast['TANGGAL'].astype(str), df_forecast['Prediksi_Total_Bulanan'], 
                 label='Prediksi SARIMA', color='red', linestyle='--')
        ax1.fill_between(df_forecast['TANGGAL'].astype(str), 
                         df_forecast['CI_Lower'], df_forecast['CI_Upper'], 
                         alpha=0.2, color='red')
        ax1.set_title('Prediksi Curah Hujan Bulanan (SARIMA) 2026-2028')
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Total Curah Hujan (mm/bulan)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        monthly_risk = df_long.groupby(df_long['TANGGAL'].dt.to_period('M'))['Probabilitas_Risiko'].mean()
        monthly_risk.index = monthly_risk.index.astype(str)
        ax2.bar(monthly_risk.index, monthly_risk.values, color='orange', alpha=0.7)
        ax2.axhline(y=0.40, color='red', linestyle='--', label='Threshold Siaga (0.40)')
        ax2.axhline(y=0.60, color='darkred', linestyle='--', label='Threshold Bahaya (0.60)')
        ax2.set_title('Proyeksi Risiko Ekstrem Bulanan (XGBoost) 2026-2028')
        ax2.set_xlabel('Bulan')
        ax2.set_ylabel('Probabilitas Risiko Rata-rata')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, 'step25_hybrid_forecast_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"   ✅ Grafik tersimpan di: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Gagal membuat grafik: {e}")
    
    print("\n" + "="*60)
    print("✅ PROSES SELESAI! File prediksi siap digunakan di dashboard.")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()