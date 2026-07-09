import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# ============================================
# STEP 16: EKSPANSI SPASIAL - VERSI BERSIH (8 STASIUN)
# Exclude Rs Bunga (28 hari data, 99% NaN) dan Stamar_Teluk_Bayur (88.5% NaN)
# karena keduanya tidak punya cukup data asli untuk diimputasi secara valid
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Padang_Filtered.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_8Pos_Imputed_Clean.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step16_expand_8stations_report.txt"

df = pd.read_csv(input_path)

lines = []
lines.append("="*60)
lines.append("LAPORAN EKSPANSI SPASIAL BERSIH - STEP 16 (8 STASIUN)")
lines.append("="*60)

# Exclude 2 stasiun bermasalah (bukti: laporan Step 2 & Step 15 sebelumnya)
exclude_stations = ['Rs Bunga', 'Stamar Teluk Bayur']
df = df[~df['NAME'].isin(exclude_stations)].copy()
lines.append(f"\nStasiun dikecualikan: {exclude_stations}")
lines.append("Alasan: Rs Bunga hanya 28 hari data unik (2017 saja, ~99% akan jadi NaN)")
lines.append("        Stamar Teluk Bayur 88.53% NaN pada rentang 2017-2025 (baru mulai 2018)")

station_map = {
    'Muara Palam - Parak Karakah': 'Parak_Karakah',
    'Nanggalo': 'Nanggalo',
    'Lubuk Minturun': 'Lubuk_Minturun',
    'Teluk Bayur': 'Teluk_Bayur',
    'Tambang Semen Padang': 'Tambang_Semen_Padang',
    'Water Plan Semen Padang': 'Water_Plan_Semen_Padang',
    'Bandar Buat': 'Bandar_Buat',
    'Limau Manih - UNAND': 'Limau_Manih_UNAND'
}
df['STASIUN'] = df['NAME'].map(station_map)
lines.append(f"\nTotal stasiun dipakai: {df['STASIUN'].nunique()}")
lines.append(str(sorted(df['STASIUN'].dropna().unique())))

df['TANGGAL'] = pd.to_datetime(df['DATA TIMESTAMP'].str.split(' ').str[0])

lines.append("\n--- CLEANING SENTINEL (8 STASIUN) ---")
n_9999 = (df['RAINFALL DAY MM'] == 9999).sum()
n_8888 = (df['RAINFALL DAY MM'] == 8888).sum()
lines.append(f"Jumlah 9999: {n_9999} -> diubah ke NaN")
lines.append(f"Jumlah 8888: {n_8888} -> diubah ke 0")
df['RAINFALL DAY MM'] = df['RAINFALL DAY MM'].replace([9999, 9999.0], np.nan)
df['RAINFALL DAY MM'] = df['RAINFALL DAY MM'].replace([8888, 8888.0], 0.0)

dup_check = df.groupby(['STASIUN', 'TANGGAL']).size()
duplicates = dup_check[dup_check > 1]
if len(duplicates) > 0:
    lines.append(f"\nPERINGATAN: {len(duplicates)} duplikat tanggal-stasiun, diambil rata-rata")
    df_piv = df.groupby(['STASIUN', 'TANGGAL'])['RAINFALL DAY MM'].mean().reset_index()
else:
    df_piv = df[['STASIUN', 'TANGGAL', 'RAINFALL DAY MM']]

df_wide = df_piv.pivot(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM')

full_range = pd.date_range(start='2017-01-01', end='2025-12-31', freq='D')
df_wide = df_wide.reindex(full_range)
df_wide.index.name = 'TANGGAL'

lines.append(f"\nShape setelah resampling 2017-2025: {df_wide.shape}")
lines.append("\n--- PERSENTASE NaN PER STASIUN (2017-2025) - VALIDASI SEMUA DI BAWAH 30% ---")
max_pct = 0
for col in df_wide.columns:
    pct = df_wide[col].isna().sum() / len(df_wide) * 100
    max_pct = max(max_pct, pct)
    flag = " OK" if pct < 30 else " MASIH TERLALU TINGGI!"
    lines.append(f"{col}: {pct:.2f}%{flag}")

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed_array = knn_imputer.fit_transform(df_wide)
df_imputed = pd.DataFrame(df_imputed_array, index=df_wide.index, columns=df_wide.columns)
df_imputed = df_imputed.clip(lower=0)

lines.append(f"\n--- SETELAH IMPUTASI KNN (8 stasiun bersih sebagai basis korelasi) ---")
lines.append(f"Total NaN tersisa: {df_imputed.isna().sum().sum()} (harus 0)")
lines.append(df_imputed.describe().to_string())

df_imputed.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data 8 stasiun bersih tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape final: {df_imputed.shape}")
print(f"Max NaN% di antara 8 stasiun: {max_pct:.2f}%")