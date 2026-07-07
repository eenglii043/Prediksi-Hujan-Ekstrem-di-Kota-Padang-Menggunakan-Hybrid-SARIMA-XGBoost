import pandas as pd
import numpy as np

# ============================================
# STEP 4: RESAMPLING KE FORMAT WIDE
# 1 baris = 1 tanggal, 3 kolom = 3 stasiun
# Rentang lengkap: 2015-01-01 s.d. 2025-12-31
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Cleaned.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Wide.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step4_resampling_report.txt"

df = pd.read_csv(input_path)

# Parse tanggal - ambil hanya bagian tanggal dari DATA TIMESTAMP
# Format asli: "2015-01-01 00:00:00.0 +0:00"
df['TANGGAL'] = pd.to_datetime(df['DATA TIMESTAMP'].str.split(' ').str[0])

lines = []
lines.append("="*60)
lines.append("LAPORAN RESAMPLING KE FORMAT WIDE - STEP 4")
lines.append("="*60)

# Mapping nama stasiun ke nama kolom yang lebih ringkas
station_map = {
    'Muara Palam - Parak Karakah': 'Parak_Karakah',
    'Nanggalo': 'Nanggalo',
    'Lubuk Minturun': 'Lubuk_Minturun'
}
df['STASIUN'] = df['NAME'].map(station_map)

# Cek duplikat tanggal per stasiun sebelum pivot (jaga-jaga)
dup_check = df.groupby(['STASIUN', 'TANGGAL']).size()
duplicates = dup_check[dup_check > 1]
lines.append(f"\nJumlah duplikat tanggal-stasiun (harus 0): {len(duplicates)}")
if len(duplicates) > 0:
    lines.append("PERINGATAN: ada duplikat, cek data berikut:")
    lines.append(str(duplicates))
    # Ambil rata-rata jika ada duplikat
    df = df.groupby(['STASIUN', 'TANGGAL'])['RAINFALL DAY MM'].mean().reset_index()
else:
    df = df[['STASIUN', 'TANGGAL', 'RAINFALL DAY MM']]

# Pivot ke wide format
df_wide = df.pivot(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM')

lines.append(f"\nBentuk data SEBELUM resampling tanggal lengkap: {df_wide.shape}")
lines.append(f"Rentang tanggal SEBELUM resampling: {df_wide.index.min()} s.d. {df_wide.index.max()}")

# Resampling ke rentang tanggal lengkap 2015-01-01 s.d. 2025-12-31
full_date_range = pd.date_range(start='2015-01-01', end='2025-12-31', freq='D')
df_wide = df_wide.reindex(full_date_range)
df_wide.index.name = 'TANGGAL'

lines.append(f"\nBentuk data SETELAH resampling tanggal lengkap: {df_wide.shape}")
lines.append(f"Total hari seharusnya (2015-01-01 s.d. 2025-12-31): {len(full_date_range)}")

# Cek NaN per kolom setelah resampling penuh
lines.append("\n--- JUMLAH NaN PER STASIUN SETELAH RESAMPLING PENUH ---")
lines.append(str(df_wide.isna().sum()))

lines.append("\n--- PERSENTASE NaN PER STASIUN ---")
for col in df_wide.columns:
    pct = df_wide[col].isna().sum() / len(df_wide) * 100
    lines.append(f"{col}: {pct:.2f}%")

# Cek NaN per tahun (buat lihat apakah ada tahun tertentu yang bolong parah)
lines.append("\n--- JUMLAH NaN PER TAHUN PER STASIUN ---")
df_wide_yearly = df_wide.copy()
df_wide_yearly['TAHUN'] = df_wide_yearly.index.year
nan_per_year = df_wide_yearly.groupby('TAHUN').apply(lambda x: x.drop(columns='TAHUN').isna().sum())
lines.append(nan_per_year.to_string())

# Preview 10 baris pertama dan terakhir
lines.append("\n--- PREVIEW 10 BARIS PERTAMA ---")
lines.append(df_wide.head(10).to_string())
lines.append("\n--- PREVIEW 10 BARIS TERAKHIR ---")
lines.append(df_wide.tail(10).to_string())

# Simpan
df_wide.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data wide format tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape akhir: {df_wide.shape}")