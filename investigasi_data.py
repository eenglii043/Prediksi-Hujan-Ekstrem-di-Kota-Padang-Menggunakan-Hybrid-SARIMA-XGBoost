import pandas as pd

# ============================================
# STEP 1: INVESTIGASI SENTINEL VALUE & STRUKTUR DATA
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Hujan_Gabungan_2015_2025.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step1_investigasi_report.txt"

df = pd.read_csv(input_path)

lines = []
lines.append("="*60)
lines.append("LAPORAN INVESTIGASI DATA - STEP 1")
lines.append("="*60)
lines.append(f"\nTotal baris: {df.shape[0]}")
lines.append(f"Total kolom: {df.shape[1]}")

# 1. Cek nilai sentinel di RAINFALL DAY MM
lines.append("\n--- CEK SENTINEL VALUE DI 'RAINFALL DAY MM' ---")
for val in [8888, 9999, 8888.0, 9999.0]:
    count = (df['RAINFALL DAY MM'] == val).sum()
    lines.append(f"Jumlah baris dengan nilai {val}: {count}")

# 2. Statistik deskriptif nilai curah hujan (buat lihat apakah ada outlier ganjil)
lines.append("\n--- STATISTIK DESKRIPTIF 'RAINFALL DAY MM' ---")
lines.append(str(df['RAINFALL DAY MM'].describe()))

# 3. Nilai tertinggi (top 20) - buat screening manual apakah ada sentinel tersembunyi
lines.append("\n--- TOP 20 NILAI TERTINGGI 'RAINFALL DAY MM' ---")
top20 = df.nlargest(20, 'RAINFALL DAY MM')[['POS HUJAN ID', 'NAME', 'DATA TIMESTAMP', 'RAINFALL DAY MM', 'RAINFALL TRACE']]
lines.append(top20.to_string())

# 4. Unique values di kolom RAINFALL TRACE
lines.append("\n--- UNIQUE VALUES DI 'RAINFALL TRACE' ---")
lines.append(str(df['RAINFALL TRACE'].value_counts()))

# 5. Daftar semua nama stasiun unik (buat identifikasi mana yang di Kota Padang)
lines.append("\n--- DAFTAR SEMUA NAMA STASIUN UNIK (untuk cari yang di Kota Padang) ---")
unique_stations = df[['POS HUJAN ID', 'NAME']].drop_duplicates().sort_values('NAME')
lines.append(unique_stations.to_string())

# 6. Cari stasiun yang namanya mengandung kata kunci Padang yang kita incar
lines.append("\n--- SEARCH: STASIUN DENGAN NAMA MIRIP TARGET KITA ---")
keywords = ['Parak', 'Karakah', 'Nanggalo', 'Lubuk Minturun', 'Teluk Bayur', 'Semen Padang', 'Padang']
for kw in keywords:
    matches = df[df['NAME'].str.contains(kw, case=False, na=False)][['POS HUJAN ID', 'NAME']].drop_duplicates()
    lines.append(f"\nKata kunci '{kw}':")
    lines.append(matches.to_string() if not matches.empty else "  (tidak ditemukan)")

# Simpan ke file txt
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan tersimpan di: {output_report}")
print(f"Total baris: {df.shape[0]}, Total kolom: {df.shape[1]}")