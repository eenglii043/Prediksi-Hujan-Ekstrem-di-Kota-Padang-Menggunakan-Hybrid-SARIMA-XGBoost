import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
DATA_PATH = "03_Hasil/Data_Padang_Filtered.csv"
OUTPUT_CSV = "03_Hasil/prediksi_tahunan_2026_2028.csv"
STATIONS = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']

# ============================================================
# FUNGSI
# ============================================================
def get_level_from_rain(rain):
    """Level berdasarkan curah hujan (threshold lokal Padang)"""
    if rain < 10:
        return 'Aman'
    elif rain < 20:
        return 'Waspada'
    elif rain < 50:
        return 'Siaga'
    else:
        return 'Bahaya'

def get_probability_from_rain(rain):
    """Probabilitas risiko dari curah hujan (0-1)"""
    # Skala: 0mm = 0, 50mm = 0.8 (Siaga), 100mm = 1.0 (Bahaya)
    proba = min(1.0, rain / 50 * 0.8)
    return round(proba, 4)

def load_historical_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['DATA TIMESTAMP'])
    df['TANGGAL'] = pd.to_datetime(df['DATA TIMESTAMP'].dt.date)
    target_ids = ['13710202a', '13711001a', '13711102a']
    target_names = {
        '13710202a': 'Parak_Karakah',
        '13711001a': 'Nanggalo',
        '13711102a': 'Lubuk_Minturun'
    }
    df_target = df[df['POS HUJAN ID'].isin(target_ids)].copy()
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
    return df_imp

def fit_sarima(df_imp):
    df_monthly = df_imp.resample('M').sum()
    df_monthly.index = df_monthly.index.to_period('M')
    df_monthly['Total'] = df_monthly.sum(axis=1)  # Total dari 3 stasiun
    series = df_monthly['Total']
    model = SARIMAX(series, order=(0, 0, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print(f"SARIMA AIC: {results.aic:.2f}")
    return results, series

def generate_forecast(results, df_imp):
    # 1. Prediksi 36 bulan (total 3 stasiun)
    forecast_steps = 36
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    future_months = pd.date_range(start='2026-01-01', periods=forecast_steps, freq='M')
    
    print("\n=== PREDIKSI BULANAN SARIMA (2026-2028) ===")
    for i, (date, val) in enumerate(zip(future_months, forecast_mean.values)):
        print(f"  {date.strftime('%Y-%m')}: {val:.1f} mm (total 3 stasiun)")
    
    # 2. Ambil rata-rata pola harian per stasiun untuk setiap bulan dari data historis
    monthly_avg = {}
    for month in range(1, 13):
        month_data = df_imp[df_imp.index.month == month]
        if len(month_data) > 0:
            monthly_avg[month] = month_data[STATIONS].mean().values
        else:
            monthly_avg[month] = np.array([1/3, 1/3, 1/3]) * 10  # fallback
    
    # 3. Buat data harian
    rows = []
    for i, month_start in enumerate(future_months):
        month_num = month_start.month
        days_in_month = (month_start + pd.offsets.MonthEnd(0)).day
        monthly_total = forecast_mean.values[i]  # total dari 3 stasiun
        daily_avg_total = monthly_total / days_in_month  # rata-rata harian total
        
        # Ambil pola per stasiun untuk bulan ini (proporsi)
        pattern = monthly_avg[month_num]
        # Normalisasi proporsi
        if pattern.sum() > 0:
            proportion = pattern / pattern.sum()
        else:
            proportion = np.array([1/3, 1/3, 1/3])
        
        # Buat data per hari
        for day in range(days_in_month):
            current_date = month_start + pd.Timedelta(days=day)
            # Variasi acak 0.7-1.3 agar tiap hari beda
            noise = np.random.uniform(0.7, 1.3)
            daily_rain_total = daily_avg_total * noise
            # Bagi ke 3 stasiun sesuai proporsi
            daily_rain = daily_rain_total * proportion
            daily_rain = np.maximum(daily_rain, 0)
            
            for j, st in enumerate(STATIONS):
                rain = daily_rain[j]
                proba = get_probability_from_rain(rain)
                level = get_level_from_rain(rain)
                rows.append({
                    'TANGGAL': current_date,
                    'STASIUN': st,
                    'Curah_Hujan_mm': round(rain, 1),
                    'Probabilitas_Risiko': proba,
                    'Level': level
                })
    
    df_daily = pd.DataFrame(rows)
    print(f"\nTotal hari: {df_daily['TANGGAL'].nunique()} hari")
    print(f"Total baris: {len(df_daily)}")
    return df_daily

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("HYBRID SARIMA-XGBOOST PREDICTION 2026-2028 (V2)")
    print("="*60)
    
    print("\n[1] Loading historical data...")
    df_imp = load_historical_data()
    print(f"   Data shape: {df_imp.shape}")
    
    print("\n[2] Fitting SARIMA model...")
    sarima_model, series = fit_sarima(df_imp)
    
    print("\n[3] Generating daily forecast 2026-2028...")
    df_forecast = generate_forecast(sarima_model, df_imp)
    
    print(f"\n[4] Saving to {OUTPUT_CSV}...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_forecast.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ File saved successfully!")
    
    # Ringkasan bulanan
    print("\n[5] Monthly summary (2026-2028):")
    monthly_summary = df_forecast.groupby(df_forecast['TANGGAL'].dt.to_period('M')).agg({
        'Curah_Hujan_mm': 'mean',
        'Probabilitas_Risiko': 'mean'
    }).round(2)
    monthly_summary['Level'] = df_forecast.groupby(df_forecast['TANGGAL'].dt.to_period('M'))['Level'].agg(
        lambda x: x.value_counts().index[0]
    )
    print(monthly_summary.to_string())
    
    print("\n" + "="*60)
    print("✅ PROSES SELESAI!")
    print(f"   Output: {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()