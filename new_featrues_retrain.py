import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report,
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 17: FITUR BARU (tren, hari beruntun hujan, max 3 hari) + RETRAIN XGBOOST
# Fitur tambahan:
#   - Trend1_2   : Lag1 - Lag2 (naik/turun antar hari)
#   - Consec_Rain: jumlah hari beruntun hujan (>0mm) sebelum hari ini
#   - Max3       : curah hujan tertinggi dalam 3 hari terakhir
#   - Spasial_x_Roll7: interaksi Tetangga_Lag1_avg * Roll7
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Imputed.csv"
output_features_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_3Pos_Features_V2.csv"
output_long_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_LongFormat_V2.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step17_new_features_report.txt"
output_metrics_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step17_metrics_comparison.csv"
output_importance_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step17_feature_importance.csv"
output_predictions_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step17_predictions.csv"

df = pd.read_csv(input_path, index_col='TANGGAL', parse_dates=True)
stations = ['Lubuk_Minturun', 'Nanggalo', 'Parak_Karakah']

lines = []
lines.append("="*60)
lines.append("LAPORAN FITUR BARU & RETRAIN XGBOOST - STEP 17")
lines.append("="*60)
lines.append(f"\nShape data awal: {df.shape}")

# ============================================
# 1. FITUR WAKTU (sama seperti sebelumnya)
# ============================================
df['Bulan'] = df.index.month
df['Hari_Tahun'] = df.index.dayofyear
df['Hari_Minggu'] = df.index.dayofweek

def get_musim(bulan):
    if bulan in [12, 1, 2, 3]:
        return 'Hujan_Intensif'
    elif bulan in [4, 5]:
        return 'Peralihan_1'
    elif bulan in [6, 7, 8, 9]:
        return 'Kemarau'
    else:
        return 'Peralihan_2'

df['Musim'] = df.index.month.map(get_musim)

# ============================================
# 2. FITUR LAG & ROLLING (sama seperti sebelumnya)
# ============================================
for station in stations:
    for lag in [1, 2, 3]:
        df[f'{station}_lag{lag}'] = df[station].shift(lag)
    df[f'{station}_roll7'] = df[station].shift(1).rolling(window=7, min_periods=1).mean()
    df[f'{station}_roll30'] = df[station].shift(1).rolling(window=30, min_periods=1).mean()

# ============================================
# 3. FITUR BARU
# ============================================
for station in stations:
    # Trend: perubahan antara lag1 dan lag2 (positif = curah hujan meningkat)
    df[f'{station}_trend'] = df[f'{station}_lag1'] - df[f'{station}_lag2']

    # Consecutive rain days: jumlah hari beruntun hujan (>0mm) SEBELUM hari ini
    is_rain = (df[station].shift(1) > 0).astype(int)
    # hitung run-length dari hari hujan beruntun
    grp = (is_rain != is_rain.shift()).cumsum()
    df[f'{station}_consec_rain'] = is_rain.groupby(grp).cumsum() * is_rain

    # Max curah hujan dalam 3 hari terakhir (menangkap puncak, bukan cuma rata-rata)
    df[f'{station}_max3'] = df[station].shift(1).rolling(window=3, min_periods=1).max()

lines.append("\n--- FITUR BARU DITAMBAHKAN ---")
lines.append("Per stasiun: _trend, _consec_rain, _max3")

# Cek NaN yang muncul dari fitur baru
lines.append(f"\n--- CEK NaN SETELAH FEATURE ENGINEERING V2 ---")
nan_summary = df.isna().sum()
lines.append(str(nan_summary[nan_summary > 0]))

df.to_csv(output_features_csv)

# ============================================
# 4. RESHAPE KE LONG FORMAT (dengan fitur baru + interaksi spasial)
# ============================================
rows = []
for station in stations:
    others = [s for s in stations if s != station]

    temp = pd.DataFrame(index=df.index)
    temp['TANGGAL'] = df.index
    temp['STASIUN'] = station
    temp['Bulan'] = df['Bulan']
    temp['Hari_Tahun'] = df['Hari_Tahun']
    temp['Hari_Minggu'] = df['Hari_Minggu']
    temp['Musim'] = df['Musim']

    temp['Lag1'] = df[f'{station}_lag1']
    temp['Lag2'] = df[f'{station}_lag2']
    temp['Lag3'] = df[f'{station}_lag3']
    temp['Roll7'] = df[f'{station}_roll7']
    temp['Roll30'] = df[f'{station}_roll30']
    temp['Trend'] = df[f'{station}_trend']
    temp['Consec_Rain'] = df[f'{station}_consec_rain']
    temp['Max3'] = df[f'{station}_max3']

    temp['Tetangga_Lag1_avg'] = df[[f'{o}_lag1' for o in others]].mean(axis=1)
    temp['Tetangga_Roll7_avg'] = df[[f'{o}_roll7' for o in others]].mean(axis=1)
    temp['Tetangga_Max3_avg'] = df[[f'{o}_max3' for o in others]].mean(axis=1)

    # Fitur interaksi: sinyal spasial dikali tren lokal
    temp['Spasial_x_Roll7'] = temp['Tetangga_Lag1_avg'] * temp['Roll7']

    temp['_Curah_Hujan_Hari_Ini_REF_ONLY'] = df[station]

    # Target: pakai threshold persentil-95 yang sama seperti sebelumnya (dihitung ulang dari data ini)
    p95 = df[station].quantile(0.95)
    temp['Extreme_Event'] = (df[station] > p95).astype(int)

    rows.append(temp)

df_long = pd.concat(rows, axis=0).reset_index(drop=True)
df_long = df_long.sort_values(['TANGGAL', 'STASIUN']).reset_index(drop=True)

lines.append(f"\nShape long format V2: {df_long.shape}")

# Drop baris dengan NaN (dari histori lag/rolling/consec_rain di awal periode)
n_before = len(df_long)
df_long = df_long.dropna()
n_after = len(df_long)
lines.append(f"Baris dibuang karena NaN (histori awal belum cukup): {n_before - n_after}")

df_long.to_csv(output_long_csv, index=False)

lines.append(f"\n--- DISTRIBUSI TARGET ---")
target_counts = df_long['Extreme_Event'].value_counts()
pct = target_counts.get(1,0) / len(df_long) * 100
lines.append(f"Normal: {target_counts.get(0,0)}, Ekstrem: {target_counts.get(1,0)} ({pct:.2f}%)")

# ============================================
# 5. ENCODING + TIME-BASED SPLIT (sama seperti step 13)
# ============================================
df_encoded = pd.get_dummies(df_long, columns=['STASIUN', 'Musim'], drop_first=False)

unique_dates = sorted(df_encoded['TANGGAL'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

df_train = df_encoded[df_encoded['TANGGAL'] < split_date].copy()
df_test = df_encoded[df_encoded['TANGGAL'] >= split_date].copy()

lines.append(f"\n--- SPLIT ---")
lines.append(f"Tanggal split: {pd.Timestamp(split_date).date()}")
lines.append(f"Train: {len(df_train)} baris | Test: {len(df_test)} baris")

exclude_cols = ['TANGGAL', '_Curah_Hujan_Hari_Ini_REF_ONLY', 'Extreme_Event']
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

lines.append(f"\nTotal fitur (V2, termasuk fitur baru): {len(feature_cols)}")
lines.append(f"Daftar fitur: {feature_cols}")

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

# Scaling numerik (fit di train saja)
numeric_cols = ['Bulan', 'Hari_Tahun', 'Hari_Minggu', 'Lag1', 'Lag2', 'Lag3',
                 'Roll7', 'Roll30', 'Trend', 'Consec_Rain', 'Max3',
                 'Tetangga_Lag1_avg', 'Tetangga_Roll7_avg', 'Tetangga_Max3_avg', 'Spasial_x_Roll7']

scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ============================================
# 6. RETRAIN XGBOOST (tuning ulang karena fitur berubah)
# ============================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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
grid_search.fit(X_train_scaled, y_train)

lines.append(f"\n--- HASIL TUNING XGBOOST V2 ---")
lines.append(f"Best params: {grid_search.best_params_}")
lines.append(f"Best CV recall score: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)
y_pred_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

lines.append(f"\n" + "="*60)
lines.append("HASIL EVALUASI XGBOOST V2 (dengan fitur baru) DI TEST SET")
lines.append("="*60)
lines.append(f"\n--- CONFUSION MATRIX ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")
lines.append(f"\nAccuracy  : {acc:.4f}")
lines.append(f"Precision : {prec:.4f}")
lines.append(f"Recall    : {rec:.4f}")
lines.append(f"F1-Score  : {f1:.4f}")
lines.append(f"\n--- CLASSIFICATION REPORT ---")
lines.append(classification_report(y_test, y_pred, target_names=['Normal', 'Ekstrem']))

importance_df = pd.DataFrame({
    'fitur': feature_cols,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)
lines.append(f"\n--- FEATURE IMPORTANCE V2 ---")
lines.append(importance_df.to_string(index=False))

# ============================================
# 7. PERBANDINGAN DENGAN XGBOOST V1 (Step 15)
# ============================================
comparison_df = pd.DataFrame([
    {'model': 'XGBoost V1 (fitur asli)', 'accuracy': 0.6504, 'precision': 0.0846, 'recall': 0.5364, 'f1_score': 0.1462},
    {'model': 'XGBoost V2 (fitur baru)', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1},
])
lines.append(f"\n" + "="*60)
lines.append("PERBANDINGAN XGBOOST V1 vs V2")
lines.append("="*60)
lines.append(comparison_df.to_string(index=False))

improvement = f1 - 0.1462
lines.append(f"\nPerubahan F1-Score: {improvement:+.4f} ({'membaik' if improvement > 0 else 'memburuk/stagnan'})")

# Simpan semua output
with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
comparison_df.to_csv(output_metrics_csv, index=False)
importance_df.to_csv(output_importance_csv, index=False)

predictions_df = df_test[['TANGGAL']].copy()
predictions_df['y_true'] = y_test.values
predictions_df['y_pred'] = y_pred
predictions_df['y_pred_proba'] = y_pred_proba
predictions_df.to_csv(output_predictions_csv, index=False)

print(f"Selesai! Laporan: {output_report}")
print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
print(f"Perubahan F1 vs V1: {improvement:+.4f}")