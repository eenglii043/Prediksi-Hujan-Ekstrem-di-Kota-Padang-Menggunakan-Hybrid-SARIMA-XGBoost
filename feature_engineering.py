import pandas as pd
import numpy as np

# ============================================
# STEP 9: FEATURE ENGINEERING
# Fitur lag, rolling, dan waktu dari data curah hujan yang sudah bersih & imputasi
# PENTING: pakai data BELUM di-scale (Data_3Pos_Imputed.csv), bukan hasil RobustScaler
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Imputed.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Features.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step9_feature_engineering_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)

lines = []
lines.append("="*60)
lines.append("LAPORAN FEATURE ENGINEERING - STEP 9")
lines.append("="*60)
lines.append(f"\nShape data awal: {df.shape}")
lines.append(f"Kolom awal: {df.columns.tolist()}")

stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']

# ============================================
# 1. FITUR WAKTU (Time Features)
# ============================================
df['Bulan'] = df.index.month
df['Hari_Tahun'] = df.index.dayofyear
df['Hari_Minggu'] = df.index.dayofweek  # 0=Senin, 6=Minggu

# Musim di Indonesia: umumnya Desember-Maret = musim hujan intensif (bisa disesuaikan referensi BMKG)
def get_musim(bulan):
    if bulan in [12, 1, 2, 3]:
        return 'Hujan_Intensif'
    elif bulan in [4, 5]:
        return 'Peralihan_1'
    elif bulan in [6, 7, 8, 9]:
        return 'Kemarau'
    else:  # 10, 11
        return 'Peralihan_2'

df['Musim'] = df.index.month.map(get_musim)

lines.append("\n--- FITUR WAKTU DIBUAT ---")
lines.append("Bulan, Hari_Tahun, Hari_Minggu, Musim")
lines.append("\nDistribusi Musim:")
lines.append(str(df['Musim'].value_counts()))

# ============================================
# 2. FITUR LAG (riwayat 1, 2, 3 hari sebelumnya) - per stasiun
# ============================================
for station in stations:
    for lag in [1, 2, 3]:
        df[f'{station}_lag{lag}'] = df[station].shift(lag)

lines.append("\n--- FITUR LAG DIBUAT ---")
lag_cols = [c for c in df.columns if '_lag' in c]
lines.append(f"Total kolom lag: {len(lag_cols)}")
lines.append(str(lag_cols))

# ============================================
# 3. FITUR ROLLING (rata-rata 7 hari & 30 hari terakhir) - per stasiun
# ============================================
for station in stations:
    df[f'{station}_roll7'] = df[station].shift(1).rolling(window=7, min_periods=1).mean()
    df[f'{station}_roll30'] = df[station].shift(1).rolling(window=30, min_periods=1).mean()

lines.append("\n--- FITUR ROLLING DIBUAT ---")
roll_cols = [c for c in df.columns if '_roll' in c]
lines.append(f"Total kolom rolling: {len(roll_cols)}")
lines.append(str(roll_cols))
lines.append("\nCatatan: rolling dihitung dari shift(1) dulu agar tidak bocor data hari ini ke fiturnya sendiri (menghindari data leakage)")

# ============================================
# CEK NaN yang muncul akibat lag/rolling di awal periode (wajar terjadi)
# ============================================
lines.append("\n--- JUMLAH NaN SETELAH FEATURE ENGINEERING (wajar untuk hari-hari awal karena belum ada histori) ---")
nan_summary = df.isna().sum()
lines.append(str(nan_summary[nan_summary > 0]))

n_rows_with_nan = df.isna().any(axis=1).sum()
lines.append(f"\nTotal baris yang mengandung NaN (karena histori lag/rolling belum cukup): {n_rows_with_nan}")
lines.append(f"Baris ini akan berada di {df.index.min()} sampai sekitar {df.index[29]} (30 hari pertama, karena roll30 butuh histori terpanjang)")

lines.append(f"\nShape data akhir (sebelum drop NaN): {df.shape}")
lines.append(f"Total kolom fitur: {len(df.columns)}")
lines.append(f"\nDaftar seluruh kolom:\n{df.columns.tolist()}")

# Preview
lines.append("\n--- PREVIEW 5 BARIS SETELAH FEATURE ENGINEERING LENGKAP (baris ke-35 dst, sudah ada histori) ---")
lines.append(df.iloc[35:40].to_string())

# Simpan (masih menyertakan NaN di awal periode - akan di-drop di step berikutnya setelah target dibuat)
df.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data dengan fitur tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape akhir: {df.shape}, Total kolom: {len(df.columns)}")