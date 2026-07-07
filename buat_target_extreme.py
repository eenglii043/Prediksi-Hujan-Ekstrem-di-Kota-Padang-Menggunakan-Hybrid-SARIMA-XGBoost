import pandas as pd
import numpy as np

# ============================================
# STEP 11: BUAT TARGET Extreme_Event (Persentil ke-95 per stasiun)
# Threshold final (dari Step 10):
#   Lubuk_Minturun : 64.00 mm
#   Nanggalo       : 62.70 mm
#   Parak_Karakah  : 55.07 mm
# Sekaligus finalisasi dataset: drop baris NaN awal (dari lag/rolling)
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Features.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_FINAL_ReadyToModel.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step11_target_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']

lines = []
lines.append("="*60)
lines.append("LAPORAN PEMBUATAN TARGET Extreme_Event - STEP 11")
lines.append("="*60)
lines.append(f"\nShape data sebelum: {df.shape}")

# ============================================
# Hitung ulang persentil-95 dari dataset ini (konsisten dengan Step 10)
# ============================================
p95_thresholds = {}
for station in stations:
    p95_thresholds[station] = df[station].quantile(0.95)

lines.append("\n--- THRESHOLD PERSENTIL-95 PER STASIUN (dipakai sebagai label) ---")
for station in stations:
    lines.append(f"{station}: {p95_thresholds[station]:.2f}mm")

# ============================================
# BUAT KOLOM TARGET per stasiun (masing-masing stasiun punya target sendiri)
# ============================================
for station in stations:
    df[f'{station}_Extreme_Event'] = (df[station] > p95_thresholds[station]).astype(int)

lines.append("\n--- DISTRIBUSI TARGET PER STASIUN ---")
for station in stations:
    col = f'{station}_Extreme_Event'
    counts = df[col].value_counts()
    pct_extreme = counts.get(1, 0) / len(df) * 100
    lines.append(f"\n{station}:")
    lines.append(f"  Normal (0): {counts.get(0, 0)}")
    lines.append(f"  Ekstrem (1): {counts.get(1, 0)} ({pct_extreme:.2f}%)")

# ============================================
# DROP baris dengan NaN (dari lag/rolling di awal periode - hanya 3 baris)
# ============================================
n_before_drop = len(df)
df_final = df.dropna()
n_after_drop = len(df_final)

lines.append(f"\n--- FINALISASI DATASET ---")
lines.append(f"Baris sebelum drop NaN: {n_before_drop}")
lines.append(f"Baris setelah drop NaN: {n_after_drop}")
lines.append(f"Baris yang dibuang: {n_before_drop - n_after_drop} (hari-hari awal tanpa histori lag/rolling cukup)")
lines.append(f"Rentang tanggal final: {df_final.index.min()} s.d. {df_final.index.max()}")

# Re-cek distribusi target setelah drop (pastikan tidak berubah signifikan)
lines.append("\n--- DISTRIBUSI TARGET SETELAH DROP NaN (VALIDASI ULANG) ---")
for station in stations:
    col = f'{station}_Extreme_Event'
    counts = df_final[col].value_counts()
    pct_extreme = counts.get(1, 0) / len(df_final) * 100
    lines.append(f"{station}: Ekstrem={counts.get(1,0)} ({pct_extreme:.2f}%), Normal={counts.get(0,0)}")

# ============================================
# RINGKASAN KOLOM FINAL
# ============================================
lines.append(f"\n--- DAFTAR SELURUH KOLOM DATASET FINAL ({len(df_final.columns)} kolom) ---")
for i, col in enumerate(df_final.columns, 1):
    lines.append(f"{i}. {col}")

lines.append(f"\nShape dataset final siap modeling: {df_final.shape}")

# Preview
lines.append("\n--- PREVIEW 5 BARIS PERTAMA DATASET FINAL ---")
lines.append(df_final.head().to_string())

# Simpan dataset final
df_final.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Dataset final siap modeling: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Shape final: {df_final.shape}")