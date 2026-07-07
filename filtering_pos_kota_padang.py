import pandas as pd

# ============================================
# STEP 2: FILTER KOTA PADANG (kode 1371) + CEK HUBUNGAN 8888 vs RAINFALL TRACE
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Hujan_Gabungan_2015_2025.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Padang_Filtered.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step2_filter_padang_report.txt"

df = pd.read_csv(input_path)

# Filter hanya stasiun dengan kode 1371 (Kota Padang)
df_padang = df[df['POS HUJAN ID'].str.startswith('1371')].copy()

# Simpan hasil filter ke csv baru (biar step selanjutnya lebih ringan diproses)
df_padang.to_csv(output_csv, index=False)

lines = []
lines.append("="*60)
lines.append("LAPORAN FILTER KOTA PADANG (kode 1371) - STEP 2")
lines.append("="*60)

lines.append(f"\nTotal baris sebelum filter: {df.shape[0]}")
lines.append(f"Total baris setelah filter (Kota Padang): {df_padang.shape[0]}")

# Daftar stasiun yang lolos filter beserta jumlah data & rentang tanggal
lines.append("\n--- DAFTAR STASIUN KOTA PADANG & KELENGKAPAN DATA ---")
station_summary = df_padang.groupby(['POS HUJAN ID', 'NAME']).agg(
    jumlah_baris=('RAINFALL DAY MM', 'count'),
    tanggal_min=('DATA TIMESTAMP', 'min'),
    tanggal_max=('DATA TIMESTAMP', 'max'),
    jumlah_nan=('RAINFALL DAY MM', lambda x: x.isna().sum())
).reset_index().sort_values('jumlah_baris', ascending=False)
lines.append(station_summary.to_string())

# Cek sentinel value khusus di data Padang saja
lines.append("\n--- SENTINEL VALUE DI DATA KOTA PADANG SAJA ---")
count_9999 = (df_padang['RAINFALL DAY MM'] == 9999).sum()
count_8888 = (df_padang['RAINFALL DAY MM'] == 8888).sum()
lines.append(f"Jumlah 9999 di data Padang: {count_9999}")
lines.append(f"Jumlah 8888 di data Padang: {count_8888}")

# Breakdown sentinel per stasiun
lines.append("\n--- SENTINEL VALUE PER STASIUN (Kota Padang) ---")
sentinel_per_station = df_padang.groupby(['POS HUJAN ID', 'NAME']).apply(
    lambda x: pd.Series({
        'count_9999': (x['RAINFALL DAY MM'] == 9999).sum(),
        'count_8888': (x['RAINFALL DAY MM'] == 8888).sum(),
        'total_baris': len(x)
    })
).reset_index()
lines.append(sentinel_per_station.to_string())

# INI YANG PALING PENTING: cek apakah 8888 berkorelasi dengan RAINFALL TRACE = 'Y'
lines.append("\n--- HUBUNGAN NILAI 8888 DENGAN KOLOM 'RAINFALL TRACE' ---")
trace_y_rows = df_padang[df_padang['RAINFALL TRACE'] == 'Y']
lines.append(f"Total baris dengan TRACE='Y' di data Padang: {len(trace_y_rows)}")
if len(trace_y_rows) > 0:
    lines.append("\nDistribusi nilai RAINFALL DAY MM saat TRACE='Y':")
    lines.append(str(trace_y_rows['RAINFALL DAY MM'].value_counts()))

lines_8888 = df_padang[df_padang['RAINFALL DAY MM'] == 8888]
if len(lines_8888) > 0:
    lines.append(f"\nDistribusi nilai RAINFALL TRACE saat RAINFALL DAY MM=8888 (n={len(lines_8888)}):")
    lines.append(str(lines_8888['RAINFALL TRACE'].value_counts()))
else:
    lines.append("\nTidak ada nilai 8888 di data Kota Padang.")

# Simpan laporan
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data terfilter tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Total baris Kota Padang: {df_padang.shape[0]}")