import pandas as pd
import numpy as np

# ============================================
# STEP 12: RESHAPE KE LONG FORMAT UNTUK MODEL GABUNGAN (Opsi 2)
# 1 baris = 1 (tanggal, stasiun) -> 3x lipat jumlah baris
# PENTING: nilai curah hujan HARI INI dikeluarkan dari fitur (mencegah data leakage)
# karena itu adalah sumber langsung dari target Extreme_Event
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_FINAL_ReadyToModel.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_LongFormat_ReadyToModel.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step12_reshape_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']

lines = []
lines.append("="*60)
lines.append("LAPORAN RESHAPE KE LONG FORMAT - STEP 12")
lines.append("="*60)
lines.append(f"\nShape data wide (sebelum reshape): {df.shape}")

rows = []
for station in stations:
    others = [s for s in stations if s != station]
    
    temp = pd.DataFrame(index=df.index)
    temp['TANGGAL'] = df.index
    temp['STASIUN'] = station
    
    # Fitur waktu (sama untuk semua stasiun di tanggal yang sama)
    temp['Bulan'] = df['Bulan']
    temp['Hari_Tahun'] = df['Hari_Tahun']
    temp['Hari_Minggu'] = df['Hari_Minggu']
    temp['Musim'] = df['Musim']
    
    # Fitur historis milik stasiun itu sendiri (generic naming, tanpa nama stasiun)
    temp['Lag1'] = df[f'{station}_lag1']
    temp['Lag2'] = df[f'{station}_lag2']
    temp['Lag3'] = df[f'{station}_lag3']
    temp['Roll7'] = df[f'{station}_roll7']
    temp['Roll30'] = df[f'{station}_roll30']
    
    # Fitur spasial: rata-rata histori 2 stasiun tetangga (menangkap pola hujan regional)
    temp['Tetangga_Lag1_avg'] = df[[f'{o}_lag1' for o in others]].mean(axis=1)
    temp['Tetangga_Roll7_avg'] = df[[f'{o}_roll7' for o in others]].mean(axis=1)
    
    # Kolom referensi (BUKAN fitur, hanya untuk audit/verifikasi - tidak boleh dipakai saat training)
    temp['_Curah_Hujan_Hari_Ini_REF_ONLY'] = df[station]
    
    # Target
    temp['Extreme_Event'] = df[f'{station}_Extreme_Event']
    
    rows.append(temp)

df_long = pd.concat(rows, axis=0).reset_index(drop=True)
df_long = df_long.sort_values(['TANGGAL', 'STASIUN']).reset_index(drop=True)

lines.append(f"Shape data long format (sesudah reshape): {df_long.shape}")
lines.append(f"Ekspektasi baris: {df.shape[0]} hari x 3 stasiun = {df.shape[0]*3}")

lines.append(f"\n--- DAFTAR KOLOM DATASET LONG FORMAT ---")
for i, col in enumerate(df_long.columns, 1):
    tag = " [REFERENSI - JANGAN DIPAKAI SEBAGAI FITUR]" if 'REF_ONLY' in col else (" [TARGET]" if col == 'Extreme_Event' else "")
    lines.append(f"{i}. {col}{tag}")

# Cek NaN
lines.append(f"\n--- CEK NaN SETELAH RESHAPE (harus 0) ---")
lines.append(str(df_long.isna().sum()))

# Distribusi target keseluruhan (gabungan 3 stasiun)
lines.append(f"\n--- DISTRIBUSI TARGET GABUNGAN (Extreme_Event) ---")
target_counts = df_long['Extreme_Event'].value_counts()
pct_extreme = target_counts.get(1, 0) / len(df_long) * 100
lines.append(f"Normal (0): {target_counts.get(0, 0)}")
lines.append(f"Ekstrem (1): {target_counts.get(1, 0)} ({pct_extreme:.2f}%)")
lines.append(f"Rasio class imbalance: 1:{(100-pct_extreme)/pct_extreme:.1f}")

# Distribusi target per stasiun (validasi konsisten dengan Step 11)
lines.append(f"\n--- DISTRIBUSI TARGET PER STASIUN (VALIDASI ULANG, harus sama dengan Step 11) ---")
lines.append(df_long.groupby('STASIUN')['Extreme_Event'].value_counts().to_string())

# Definisikan kolom fitur final (eksplisit, exclude kolom referensi & target)
feature_cols = [c for c in df_long.columns if c not in 
                ['TANGGAL', 'STASIUN', '_Curah_Hujan_Hari_Ini_REF_ONLY', 'Extreme_Event', 'Musim']]
lines.append(f"\n--- KOLOM FITUR NUMERIK SIAP PAKAI (belum termasuk encoding STASIUN & Musim) ---")
lines.append(str(feature_cols))
lines.append(f"\nCatatan: kolom 'STASIUN' dan 'Musim' berupa kategori teks, perlu di-encode")
lines.append("(one-hot encoding atau label encoding) sebelum masuk ke model - akan dilakukan di Step 13.")

lines.append("\n--- PREVIEW 6 BARIS PERTAMA (2 hari x 3 stasiun) ---")
lines.append(df_long.head(6).to_string())

# Simpan
df_long.to_csv(output_csv, index=False)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Dataset long format tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape final: {df_long.shape}")
print(f"Distribusi target: Normal={target_counts.get(0,0)}, Ekstrem={target_counts.get(1,0)} ({pct_extreme:.2f}%)")