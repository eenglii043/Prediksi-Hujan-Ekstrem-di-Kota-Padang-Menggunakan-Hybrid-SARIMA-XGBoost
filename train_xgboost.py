import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (recall_score, precision_score, f1_score, 
                             accuracy_score, confusion_matrix, classification_report)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. KONFIGURASI
# ============================================================
DATA_PATH = "03_Hasil/Data_Padang_Filtered.csv"
OUTPUT_DIR = "03_Hasil"

MODEL_PATH = os.path.join(OUTPUT_DIR, "model_xgboost_final.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_final.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_cols_final.pkl")

THRESHOLD_SIAGA = 0.40  # threshold untuk recall tertinggi

# ============================================================
# 2. LOAD & CLEAN DATA (3 STASIUN TARGET + 6 STASIUN TETANGGA)
# ============================================================
print("="*60)
print("TRAINING XGBOOST MODEL - PREDIKSI HUJAN EKSTREM")
print("="*60)

print("\n[1] Loading data...")
df_raw = pd.read_csv(DATA_PATH, parse_dates=['DATA TIMESTAMP'])
df_raw['TANGGAL'] = pd.to_datetime(df_raw['DATA TIMESTAMP'].dt.date)

# Filter 3 stasiun target
target_ids = ['13710202a', '13711001a', '13711102a']
target_names = {
    '13710202a': 'Parak_Karakah',
    '13711001a': 'Nanggalo',
    '13711102a': 'Lubuk_Minturun'
}

# Ambil semua stasiun Kota Padang (9 stasiun untuk fitur spasial)
all_stations = df_raw['POS HUJAN ID'].unique()
other_ids = [s for s in all_stations if s.startswith('1371') and s not in target_ids]
selected_ids = target_ids + other_ids

df_selected = df_raw[df_raw['POS HUJAN ID'].isin(selected_ids)].copy()

# Cleaning sentinel values
df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([9999, 9999.0], np.nan)
df_selected['RAINFALL DAY MM'] = df_selected['RAINFALL DAY MM'].replace([8888, 8888.0], 0)

# Pivot ke wide (9 stasiun)
df_selected['STASIUN'] = df_selected['POS HUJAN ID'].map(lambda x: target_names.get(x, x))
df_wide = df_selected.pivot_table(index='TANGGAL', columns='STASIUN', values='RAINFALL DAY MM', aggfunc='mean')

full_range = pd.date_range('2017-01-01', '2025-12-31', freq='D')
df_wide = df_wide.reindex(full_range)
df_wide.index.name = 'TANGGAL'

print(f"   Data shape: {df_wide.shape} ({len(df_wide)} hari x {len(df_wide.columns)} stasiun)")

# Imputasi KNN
print("\n[2] Imputasi missing values dengan KNN...")
knn = KNNImputer(n_neighbors=5, weights='distance')
df_imp = pd.DataFrame(knn.fit_transform(df_wide), index=df_wide.index, columns=df_wide.columns)
df_imp = df_imp.clip(lower=0)
print(f"   NaN tersisa: {df_imp.isna().sum().sum()}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3] Feature Engineering...")

target_cols = ['Parak_Karakah', 'Nanggalo', 'Lubuk_Minturun']
all_cols = df_imp.columns.tolist()
other_cols = [c for c in all_cols if c not in target_cols]

df_feat = df_imp.copy()

# Fitur waktu
df_feat['Bulan'] = df_feat.index.month
df_feat['Hari_Tahun'] = df_feat.index.dayofyear
df_feat['Hari_Minggu'] = df_feat.index.dayofweek

# Fitur lag & rolling untuk SEMUA stasiun
for col in all_cols:
    for lag in [1, 2, 3]:
        df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
    df_feat[f'{col}_roll7'] = df_feat[col].shift(1).rolling(7, min_periods=1).mean()
    df_feat[f'{col}_roll30'] = df_feat[col].shift(1).rolling(30, min_periods=1).mean()
    df_feat[f'{col}_roll365'] = df_feat[col].shift(1).rolling(365, min_periods=1).mean()

# Fitur spasial (rata-rata tetangga)
for st in target_cols:
    others = [o for o in other_cols if o != st]
    df_feat[f'{st}_Tetangga_Lag1_avg'] = df_feat[[f'{o}_lag1' for o in others]].mean(axis=1)
    df_feat[f'{st}_Tetangga_Roll7_avg'] = df_feat[[f'{o}_roll7' for o in others]].mean(axis=1)

print(f"   Total fitur: {len(df_feat.columns)}")

# ============================================================
# 4. RESHAPE KE LONG FORMAT
# ============================================================
print("\n[4] Reshape ke long format...")

rows = []
for st in target_cols:
    temp = pd.DataFrame(index=df_feat.index)
    temp['TANGGAL'] = df_feat.index
    temp['STASIUN'] = st
    temp['Bulan'] = df_feat['Bulan']
    temp['Hari_Tahun'] = df_feat['Hari_Tahun']
    temp['Hari_Minggu'] = df_feat['Hari_Minggu']
    for lag in [1, 2, 3]:
        temp[f'Lag{lag}'] = df_feat[f'{st}_lag{lag}']
    temp['Roll7'] = df_feat[f'{st}_roll7']
    temp['Roll30'] = df_feat[f'{st}_roll30']
    temp['Roll365'] = df_feat[f'{st}_roll365']
    temp['Tetangga_Lag1_avg'] = df_feat[f'{st}_Tetangga_Lag1_avg']
    temp['Tetangga_Roll7_avg'] = df_feat[f'{st}_Tetangga_Roll7_avg']
    temp['_REF'] = df_feat[st]  # hujan hari ini (tidak dipakai sebagai fitur)
    
    # Target: P95 per stasiun
    p95 = df_feat[st].quantile(0.95)
    temp['Extreme_Event'] = (df_feat[st] > p95).astype(int)
    rows.append(temp)

df_long = pd.concat(rows, axis=0).reset_index(drop=True)
df_long = df_long.dropna()  # hapus baris dengan NaN (akibat lag/rolling)
print(f"   Shape long: {df_long.shape}")

# ============================================================
# 5. ENCODING & SPLIT (TIME-BASED)
# ============================================================
print("\n[5] Split data (time-based 80/20)...")

df_encoded = pd.get_dummies(df_long, columns=['STASIUN'], drop_first=False)
unique_dates = sorted(df_encoded['TANGGAL'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

df_train = df_encoded[df_encoded['TANGGAL'] < split_date].copy()
df_test = df_encoded[df_encoded['TANGGAL'] >= split_date].copy()

exclude_cols = ['TANGGAL', '_REF', 'Extreme_Event']
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

print(f"   Train: {X_train.shape[0]} baris")
print(f"   Test: {X_test.shape[0]} baris")
print(f"   Fitur: {len(feature_cols)}")

# ============================================================
# 6. SCALING
# ============================================================
print("\n[6] Scaling data (RobustScaler)...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. TRAIN XGBOOST
# ============================================================
print("\n[7] Training XGBoost model...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.01,
    'min_child_weight': 5,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

model = XGBClassifier(**params)
model.fit(X_train_scaled, y_train)

# ============================================================
# 8. EVALUASI
# ============================================================
print("\n[8] Evaluasi model...")

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  Prediksi Normal  Prediksi Ekstrem")
print(f"Aktual Normal   {cm[0][0]:<10}     {cm[0][1]}")
print(f"Aktual Ekstrem  {cm[1][0]:<10}     {cm[1][1]}")

# Metrik
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\nMetrik pada threshold 0.50 (default):")
print(f"  Recall    : {rec:.4f} ({rec*100:.1f}%)")
print(f"  Precision : {prec:.4f} ({prec*100:.1f}%)")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")

# Recall pada threshold 0.40 (untuk prioritas)
y_pred_040 = (y_proba >= 0.40).astype(int)
rec_040 = recall_score(y_test, y_pred_040)
prec_040 = precision_score(y_test, y_pred_040)
f1_040 = f1_score(y_test, y_pred_040)
cm_040 = confusion_matrix(y_test, y_pred_040)

print(f"\nMetrik pada threshold 0.40 (prioritas Recall):")
print(f"  Recall    : {rec_040:.4f} ({rec_040*100:.1f}%)")
print(f"  Precision : {prec_040:.4f} ({prec_040*100:.1f}%)")
print(f"  F1-Score  : {f1_040:.4f}")
print(f"  TP: {cm_040[1,1]}, FP: {cm_040[0,1]}, FN: {cm_040[1,0]}")

# ============================================================
# 9. SIMPAN MODEL, SCALER, FEATURE COLS
# ============================================================
print("\n[9] Menyimpan model dan scaler...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_cols, FEATURES_PATH)

print(f"   ✅ Model: {MODEL_PATH}")
print(f"   ✅ Scaler: {SCALER_PATH}")
print(f"   ✅ Feature columns: {FEATURES_PATH}")

# ============================================================
# 10. SIMPAN METRIK KE CSV (untuk dashboard)
# ============================================================
print("\n[10] Menyimpan metrik...")

metrics_df = pd.DataFrame([
    {'Threshold': 0.50, 'Recall': rec, 'Precision': prec, 'F1': f1, 
     'TP': cm[1,1], 'FP': cm[0,1], 'FN': cm[1,0]},
    {'Threshold': 0.40, 'Recall': rec_040, 'Precision': prec_040, 'F1': f1_040,
     'TP': cm_040[1,1], 'FP': cm_040[0,1], 'FN': cm_040[1,0]}
])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "xgboost_metrics.csv"), index=False)
print(f"   ✅ Metrik: {os.path.join(OUTPUT_DIR, 'xgboost_metrics.csv')}")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
print("\n[11] Menyimpan feature importance...")

importance_df = pd.DataFrame({
    'Fitur': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "xgboost_feature_importance.csv"), index=False)
print(f"   ✅ Feature importance: {os.path.join(OUTPUT_DIR, 'xgboost_feature_importance.csv')}")
print("\nTop 10 fitur:")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# 12. RINGKASAN AKHIR
# ============================================================
print("\n" + "="*60)
print("✅ TRAINING SELESAI!")
print("="*60)
print(f"   Model terbaik: XGBoost dengan {len(feature_cols)} fitur")
print(f"   Recall @ 0.40: {rec_040*100:.1f}%")
print(f"   Precision @ 0.40: {prec_040*100:.1f}%")
print(f"   F1 @ 0.40: {f1_040:.4f}")
print("="*60)