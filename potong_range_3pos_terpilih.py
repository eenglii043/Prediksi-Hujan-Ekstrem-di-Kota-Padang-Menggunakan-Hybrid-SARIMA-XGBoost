import pandas as pd

# ============================================
# STEP 5: POTONG RENTANG PENELITIAN KE 2017-2025
# Alasan: 2015 (mayoritas kosong) dan 2016 (100% kosong di 3 stasiun)
# tidak punya dasar statistik untuk diimputasi
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Wide.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Wide_2017_2025.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step5_cut_range_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)

lines = []
lines.append("="*60)
lines.append("LAPORAN PEMOTONGAN RENTANG PENELITIAN - STEP 5")
lines.append("="*60)

lines.append(f"\nShape SEBELUM potong (2015-2025): {df.shape}")

# Potong ke 2017-01-01 s.d. 2025-12-31
df_cut = df.loc['2017-01-01':'2025-12-31'].copy()

lines.append(f"Shape SETELAH potong (2017-2025): {df_cut.shape}")
lines.append(f"Rentang tanggal final: {df_cut.index.min()} s.d. {df_cut.index.max()}")
lines.append(f"Total hari: {len(df_cut)} (ekspektasi: 3287 hari untuk 9 tahun termasuk 3 tahun kabisat)")

# Validasi ulang NaN setelah potong
lines.append("\n--- JUMLAH NaN PER STASIUN SETELAH POTONG RENTANG ---")
lines.append(str(df_cut.isna().sum()))

lines.append("\n--- PERSENTASE NaN PER STASIUN (dari total hari 2017-2025) ---")
for col in df_cut.columns:
    pct = df_cut[col].isna().sum() / len(df_cut) * 100
    lines.append(f"{col}: {pct:.2f}%")

# Validasi ulang per tahun - pastikan tidak ada tahun yang bolong parah lagi
lines.append("\n--- JUMLAH NaN PER TAHUN PER STASIUN (VALIDASI ULANG) ---")
df_yearly = df_cut.copy()
df_yearly['TAHUN'] = df_yearly.index.year
nan_per_year = df_yearly.groupby('TAHUN').apply(lambda x: x.drop(columns='TAHUN').isna().sum())
lines.append(nan_per_year.to_string())

# Cek pola gap - apakah gap tersebar acak atau ada gap beruntun panjang
lines.append("\n--- ANALISIS PANJANG GAP BERUNTUN (consecutive NaN) PER STASIUN ---")
for col in df_cut.columns:
    is_na = df_cut[col].isna()
    # identifikasi grup gap beruntun
    gap_groups = (is_na != is_na.shift()).cumsum()[is_na]
    if len(gap_groups) > 0:
        gap_lengths = gap_groups.value_counts().sort_values(ascending=False)
        lines.append(f"\n{col}:")
        lines.append(f"  Jumlah episode gap terpisah: {len(gap_lengths)}")
        lines.append(f"  Gap terpanjang: {gap_lengths.max()} hari berturut-turut")
        lines.append(f"  5 gap terpanjang: {gap_lengths.head(5).tolist()}")
    else:
        lines.append(f"\n{col}: tidak ada gap")

# Simpan hasil potong
df_cut.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data terpotong tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape final: {df_cut.shape}")