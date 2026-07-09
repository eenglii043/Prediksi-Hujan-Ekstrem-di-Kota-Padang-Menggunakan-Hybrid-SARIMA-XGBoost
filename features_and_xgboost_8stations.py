import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report,
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 17: FEATURE ENGINEERING + XGBOOST - VERSI BERSIH (8 STASIUN)
# Fitur spasial sekarang dari rata-rata 7 tetangga bersih (bukan 8, karena
# 2 stasiun bermasalah sudah dibuang di Step 16)
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_8Pos_Imputed_Clean.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step17_xgboost_8stations_report.txt"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
target_stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']
all_stations = df.columns.tolist()  # 8 stasiun

lines = []
lines.append("="*60)
lines.append("LAPORAN FEATURE ENGINEERING + XGBOOST - STEP 17 (8 STASIUN BERSIH)")
lines.append("="*60)
lines.append(f"\nTotal stasiun tersedia sbg tetangga: {len(all_stations)}")
lines.append(f"Stasiun target (jadi baris prediksi): {target_stations}")

# ============================================
# FEATURE ENGINEERING per stasiun target
# ============================================
df['Bulan'] = df.index.month
df['Hari_Tahun'] = df.index.dayofyear
df['Hari_Minggu'] = df.index.dayofweek

for station in all_stations:
    df[f'{station}_lag1'] = df[station].shift(1)
    df[f'{station}_roll7'] = df[station].shift(1).rolling(7, min_periods=1).mean()

# Hitung threshold P95 per stasiun TARGET (bukan semua 8, cuma yang jadi label)
p95 = {s: df[s].quantile(0.95) for s in target_stations}
lines.append(f"\n--- THRESHOLD P95 (VALIDASI, harus dekat dengan Step 11: 64/62.7/55.07) ---")
for s in target_stations:
    lines.append(f"{s}: {p95[s]:.2f}mm")

# ============================================
# RESHAPE LONG FORMAT: 1 baris = 1 (tanggal, stasiun target)
# Fitur spasial = rata-rata dari SEMUA stasiun LAIN (7 tetangga, bukan diri sendiri)
# ============================================
rows = []
for station in target_stations:
    neighbors = [s for s in all_stations if s != station]  # 7 tetangga bersih
    
    temp = pd.DataFrame(index=df.index)
    temp.index.name = None  # <-- TAMBAHAN INI: hilangkan nama index biar tidak bentrok dengan kolom TANGGAL
    temp['TANGGAL'] = df.index
    temp['STASIUN'] = station
    temp['Bulan'] = df['Bulan']
    temp['Hari_Tahun'] = df['Hari_Tahun']
    temp['Hari_Minggu'] = df['Hari_Minggu']
    temp['Lag1'] = df[f'{station}_lag1']
    temp['Roll7'] = df[f'{station}_roll7']
    temp['Tetangga_Lag1_avg'] = df[[f'{n}_lag1' for n in neighbors]].mean(axis=1)
    temp['Tetangga_Roll7_avg'] = df[[f'{n}_roll7' for n in neighbors]].mean(axis=1)
    temp['_REF'] = df[station]
    temp['Extreme_Event'] = (df[station] > p95[station]).astype(int)
    rows.append(temp)

df_long = pd.concat(rows, axis=0).sort_values(['TANGGAL', 'STASIUN']).reset_index(drop=True)
df_long = df_long.dropna()

lines.append(f"\nShape long format: {df_long.shape}")
lines.append(f"Jumlah tetangga per stasiun target: 7 (dari 8 stasiun bersih, minus diri sendiri)")

# Distribusi target
target_dist = df_long['Extreme_Event'].value_counts()
lines.append(f"\nDistribusi target: Normal={target_dist.get(0,0)}, Ekstrem={target_dist.get(1,0)} ({target_dist.get(1,0)/len(df_long)*100:.2f}%)")

# ============================================
# SPLIT TIME-BASED 80/20
# ============================================
unique_dates = sorted(df_long['TANGGAL'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

df_train = df_long[df_long['TANGGAL'] < split_date].copy()
df_test = df_long[df_long['TANGGAL'] >= split_date].copy()

lines.append(f"\nSplit date: {pd.Timestamp(split_date).date()}")
lines.append(f"Train: {len(df_train)} baris | Test: {len(df_test)} baris")

feature_cols = ['Bulan', 'Hari_Tahun', 'Hari_Minggu', 'Lag1', 'Roll7', 'Tetangga_Lag1_avg', 'Tetangga_Roll7_avg']
X_train, y_train = df_train[feature_cols], df_train['Extreme_Event']
X_test, y_test = df_test[feature_cols], df_test['Extreme_Event']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
lines.append(f"scale_pos_weight: {scale_pos_weight:.4f}")

# ============================================
# XGBOOST + TUNING (grid sama seperti Step 15 biar perbandingan adil)
# ============================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 10],
    'scale_pos_weight': [1, scale_pos_weight]
}
tscv = TimeSeriesSplit(n_splits=5)
xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
grid_search = GridSearchCV(xgb_base, param_grid, cv=tscv, scoring='recall', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

lines.append(f"\n--- HASIL TUNING ---")
lines.append(f"Best params: {grid_search.best_params_}")
lines.append(f"Best CV recall: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
lines.append(f"\n--- CONFUSION MATRIX (8 STASIUN BERSIH, threshold default 0.5) ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")

rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

lines.append(f"\n--- METRIK UTAMA ---")
lines.append(f"Accuracy  : {acc:.4f}")
lines.append(f"Precision : {prec:.4f}")
lines.append(f"Recall    : {rec:.4f}")
lines.append(f"F1-Score  : {f1:.4f}")

lines.append(f"\n--- PERBANDINGAN 3 VERSI ---")
lines.append(f"{'Versi':<35}{'Recall':<12}{'Precision':<12}{'F1':<10}")
lines.append(f"{'3 stasiun (2 tetangga)':<35}{'0.5364':<12}{'0.0846':<12}{'0.1462':<10}")
lines.append(f"{'9/10 stasiun (bermasalah, 2 buruk)':<35}{'0.6364':<12}{'0.0906':<12}{'0.1586':<10}")
lines.append(f"{'8 stasiun BERSIH (final, defensible)':<35}{rec:<12.4f}{prec:<12.4f}{f1:<10.4f}")

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan: {output_report}")
print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")