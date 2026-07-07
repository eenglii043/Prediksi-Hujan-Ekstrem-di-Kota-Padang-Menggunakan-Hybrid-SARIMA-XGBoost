import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# ============================================
# STEP 7: TERAPKAN KNN IMPUTER KE DATA ASLI
# Berdasarkan hasil uji RMSE Step 6, KNN Imputer terbukti lebih akurat
# (RMSE rata-rata 19.32 vs Interpolasi Linear 28.14)
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Wide_2017_2025.csv"
output_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Imputed.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step7_imputation_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)

lines = []
lines.append("="*60)
lines.append("LAPORAN PENERAPAN KNN IMPUTER - STEP 7")
lines.append("="*60)

lines.append(f"\nShape data sebelum imputasi: {df.shape}")
lines.append("\n--- JUMLAH NaN SEBELUM IMPUTASI ---")
lines.append(str(df.isna().sum()))
lines.append(f"Total NaN: {df.isna().sum().sum()}")

# Simpan index dan nilai baris yang akan diimputasi (buat dicek manual nanti)
nan_rows_before = df[df.isna().any(axis=1)].copy()
lines.append(f"\n--- PREVIEW BARIS YANG AKAN DIIMPUTASI (sebelum) ---")
lines.append(nan_rows_before.to_string())

# ============================================
# TERAPKAN KNN IMPUTER (n_neighbors=5, weights='distance' — konsisten dengan Step 6)
# ============================================
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed_array = knn_imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed_array, index=df.index, columns=df.columns)

# Curah hujan tidak boleh negatif - clip ke 0 jika ada hasil imputasi yang minus
negative_check = (df_imputed < 0).sum()
lines.append(f"\n--- CEK NILAI NEGATIF HASIL IMPUTASI (harus 0) ---")
lines.append(str(negative_check))
df_imputed = df_imputed.clip(lower=0)

lines.append(f"\nShape data setelah imputasi: {df_imputed.shape}")
lines.append("\n--- JUMLAH NaN SETELAH IMPUTASI (harus 0 semua) ---")
lines.append(str(df_imputed.isna().sum()))

# Bandingkan statistik sebelum vs sesudah imputasi (pastikan distribusi tidak berubah drastis)
lines.append("\n--- STATISTIK SEBELUM IMPUTASI (data asli, tanpa NaN) ---")
lines.append(df.describe().to_string())

lines.append("\n--- STATISTIK SETELAH IMPUTASI (semua data) ---")
lines.append(df_imputed.describe().to_string())

# Preview nilai-nilai yang diimputasi (buat lampiran/verifikasi manual)
lines.append("\n--- NILAI HASIL IMPUTASI PADA BARIS YANG SEBELUMNYA NaN ---")
imputed_values = df_imputed.loc[nan_rows_before.index]
lines.append(imputed_values.to_string())

# Simpan hasil
df_imputed.to_csv(output_csv)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Data terimputasi tersimpan di: {output_csv}")
print(f"Laporan tersimpan di: {output_report}")
print(f"Total NaN tersisa: {df_imputed.isna().sum().sum()}")