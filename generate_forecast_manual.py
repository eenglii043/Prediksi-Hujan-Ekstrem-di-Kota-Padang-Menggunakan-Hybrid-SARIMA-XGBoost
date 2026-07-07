import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

OUTPUT_DIR = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prediksi_tahunan_2026_2028.csv")

# ============================================================
# BUAT DATA PREDIKSI SIMULASI UNTUK 2026-2028
# ============================================================
print("Membuat data prediksi 2026-2028...")

# Buat tanggal dari 2026-01-01 sampai 2028-12-31
start_date = datetime(2026, 1, 1)
end_date = datetime(2028, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Stasiun
stations = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']

# Buat data
rows = []
for date in dates:
    # Simulasi curah hujan dengan pola musiman
    month = date.month
    # Musim hujan: Des-Mar (curah tinggi), Musim kemarau: Jun-Sep (rendah)
    if month in [12, 1, 2, 3]:
        base_rain = np.random.uniform(20, 60)
    elif month in [4, 5, 10, 11]:
        base_rain = np.random.uniform(10, 40)
    else:  # 6,7,8,9
        base_rain = np.random.uniform(0, 20)
    
    # Tambahkan noise
    noise = np.random.uniform(0.5, 1.5)
    
    for st in stations:
        rain = base_rain * noise
        # Probabilitas risiko dari curah hujan (simulasi)
        proba = min(0.95, max(0.05, rain / 150))
        level = 'Aman' if proba < 0.25 else 'Waspada' if proba < 0.40 else 'Siaga' if proba < 0.60 else 'Bahaya'
        
        rows.append({
            'TANGGAL': date,
            'STASIUN': st,
            'Curah_Hujan_mm': round(rain, 1),
            'Probabilitas_Risiko': round(proba, 4),
            'Level': level
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Data prediksi tersimpan di: {OUTPUT_FILE}")
print(f"Total data: {len(df)} baris")
print(f"Rentang tanggal: {df['TANGGAL'].min()} s.d. {df['TANGGAL'].max()}")