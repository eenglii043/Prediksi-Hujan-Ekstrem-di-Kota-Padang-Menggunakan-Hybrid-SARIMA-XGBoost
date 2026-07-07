import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# ============================================
# STEP 8: NORMALISASI DATA
# RobustScaler dipilih sebagai metode utama karena tahan terhadap outlier
# ekstrem (curah hujan bisa melonjak tajam saat hujan deras).
# MinMax dan Standard disimpan sebagai pembanding untuk pembahasan skripsi.
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Imputed.csv"
output_robust = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Scaled_Robust.csv"
output_minmax = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Scaled_MinMax.csv"
output_standard = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Scaled_Standard.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step8_normalization_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)

lines = []
lines.append("="*60)
lines.append("LAPORAN NORMALISASI DATA - STEP 8")
lines.append("="*60)

lines.append(f"\nShape data: {df.shape}")
lines.append("\n--- STATISTIK SEBELUM NORMALISASI ---")
lines.append(df.describe().to_string())

# ============================================
# METODE UTAMA: RobustScaler (median & IQR based)
# ============================================
robust_scaler = RobustScaler()
df_robust = pd.DataFrame(
    robust_scaler.fit_transform(df),
    index=df.index, columns=df.columns
)
lines.append("\n--- HASIL ROBUSTSCALER (metode utama) ---")
lines.append(df_robust.describe().to_string())
lines.append("\nParameter RobustScaler per kolom (median & IQR yang dipakai):")
for i, col in enumerate(df.columns):
    lines.append(f"  {col}: median={robust_scaler.center_[i]:.4f}, IQR={robust_scaler.scale_[i]:.4f}")

# ============================================
# PEMBANDING 1: MinMaxScaler
# ============================================
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    index=df.index, columns=df.columns
)
lines.append("\n--- HASIL MINMAXSCALER (pembanding) ---")
lines.append(df_minmax.describe().to_string())

# ============================================
# PEMBANDING 2: StandardScaler
# ============================================
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(
    standard_scaler.fit_transform(df),
    index=df.index, columns=df.columns
)
lines.append("\n--- HASIL STANDARDSCALER (pembanding) ---")
lines.append(df_standard.describe().to_string())

# ============================================
# ANALISIS: kenapa RobustScaler lebih cocok - tunjukkan sensitivitas terhadap outlier
# ============================================
lines.append("\n--- ANALISIS SENSITIVITAS TERHADAP OUTLIER (nilai max) ---")
lines.append("Perbandingan skala nilai MAKSIMUM (hari hujan paling ekstrem) di tiap metode:")
for col in df.columns:
    lines.append(f"\n{col} (nilai asli max={df[col].max():.1f}mm):")
    lines.append(f"  RobustScaler  -> {df_robust[col].max():.4f}")
    lines.append(f"  MinMaxScaler  -> {df_minmax[col].max():.4f}")
    lines.append(f"  StandardScaler-> {df_standard[col].max():.4f}")

# Simpan ketiga versi
df_robust.to_csv(output_robust)
df_minmax.to_csv(output_minmax)
df_standard.to_csv(output_standard)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Selesai! 3 versi data ternormalisasi tersimpan:")
print(f"  - RobustScaler (utama): {output_robust}")
print(f"  - MinMaxScaler (pembanding): {output_minmax}")
print(f"  - StandardScaler (pembanding): {output_standard}")
print(f"Laporan tersimpan di: {output_report}")