import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error

# ============================================
# STEP 6: UJI AKURASI IMPUTASI (RMSE)
# Bandingkan Interpolasi Linear vs KNN Imputer
# Metode: sembunyikan sebagian data lengkap secara acak, lalu tebak & bandingkan
# ============================================

np.random.seed(42)  # biar hasilnya reproducible

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Wide_2017_2025.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step6_rmse_comparison_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)

lines = []
lines.append("="*60)
lines.append("LAPORAN UJI AKURASI IMPUTASI (RMSE) - STEP 6")
lines.append("="*60)

# Ambil hanya baris yang lengkap (tidak ada NaN sama sekali) sebagai basis uji
df_complete = df.dropna().copy()
lines.append(f"\nTotal baris lengkap (tanpa NaN sama sekali) yang dipakai untuk uji: {len(df_complete)}")

# Simulasikan missing: sembunyikan 10% data secara acak dari baris yang lengkap
mask_fraction = 0.10
df_test = df_complete.copy()
true_values = {}  # simpan nilai asli yang disembunyikan, per kolom

for col in df_test.columns:
    n_hide = int(len(df_test) * mask_fraction)
    hide_idx = np.random.choice(df_test.index, size=n_hide, replace=False)
    true_values[col] = df_test.loc[hide_idx, col].copy()
    df_test.loc[hide_idx, col] = np.nan

lines.append(f"Jumlah data yang disembunyikan per kolom: {int(len(df_test) * mask_fraction)} ({mask_fraction*100:.0f}%)")

# ============================================
# METODE 1: Interpolasi Linear
# ============================================
df_interp = df_test.interpolate(method='linear', limit_direction='both')

# ============================================
# METODE 2: KNN Imputer (memanfaatkan korelasi antar 3 stasiun)
# ============================================
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_knn_array = knn_imputer.fit_transform(df_test)
df_knn = pd.DataFrame(df_knn_array, index=df_test.index, columns=df_test.columns)

# ============================================
# HITUNG RMSE untuk masing-masing metode, per kolom
# ============================================
lines.append("\n--- HASIL RMSE PER STASIUN ---")
lines.append(f"{'Stasiun':<20}{'RMSE Interpolasi':<20}{'RMSE KNN':<20}{'Metode Terbaik'}")

rmse_summary = {}
for col in df_test.columns:
    idx = true_values[col].index
    y_true = true_values[col].values
    
    y_pred_interp = df_interp.loc[idx, col].values
    y_pred_knn = df_knn.loc[idx, col].values
    
    rmse_interp = np.sqrt(mean_squared_error(y_true, y_pred_interp))
    rmse_knn = np.sqrt(mean_squared_error(y_true, y_pred_knn))
    
    best_method = "Interpolasi" if rmse_interp < rmse_knn else "KNN"
    rmse_summary[col] = {'interp': rmse_interp, 'knn': rmse_knn, 'best': best_method}
    
    lines.append(f"{col:<20}{rmse_interp:<20.4f}{rmse_knn:<20.4f}{best_method}")

# Rata-rata RMSE keseluruhan
avg_rmse_interp = np.mean([v['interp'] for v in rmse_summary.values()])
avg_rmse_knn = np.mean([v['knn'] for v in rmse_summary.values()])
overall_best = "Interpolasi Linear" if avg_rmse_interp < avg_rmse_knn else "KNN Imputer"

lines.append(f"\n--- RATA-RATA RMSE KESELURUHAN ---")
lines.append(f"Interpolasi Linear: {avg_rmse_interp:.4f}")
lines.append(f"KNN Imputer: {avg_rmse_knn:.4f}")
lines.append(f"\n>>> METODE TERBAIK SECARA KESELURUHAN: {overall_best} <<<")

# Simpan laporan
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan tersimpan di: {output_report}")
print(f"Metode terbaik: {overall_best}")