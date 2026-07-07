import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 13: ENCODING + TIME-BASED SPLIT + BASELINE LOGISTIC REGRESSION
# Split berdasarkan TANGGAL (bukan urutan baris) agar 3 stasiun di tanggal
# yang sama tidak terpisah antara train dan test (mencegah leakage)
# ============================================

input_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_LongFormat_ReadyToModel.csv"
output_train_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Train.csv"
output_test_csv = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Test.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step13_baseline_report.txt"

df = pd.read_csv(input_path, parse_dates=['TANGGAL'])

lines = []
lines.append("="*60)
lines.append("LAPORAN SPLIT DATA & BASELINE MODEL - STEP 13")
lines.append("="*60)
lines.append(f"\nShape data awal: {df.shape}")

# ============================================
# 1. ENCODING KATEGORIKAL
# ============================================
df_encoded = pd.get_dummies(df, columns=['STASIUN', 'Musim'], drop_first=False)
lines.append(f"\nShape setelah one-hot encoding: {df_encoded.shape}")

encoded_cols = [c for c in df_encoded.columns if c.startswith('STASIUN_') or c.startswith('Musim_')]
lines.append(f"Kolom hasil encoding: {encoded_cols}")

# ============================================
# 2. TIME-BASED SPLIT (80% tanggal awal = train, 20% tanggal akhir = test)
# ============================================
unique_dates = sorted(df_encoded['TANGGAL'].unique())
n_dates = len(unique_dates)
split_idx = int(n_dates * 0.8)
split_date = unique_dates[split_idx]

lines.append(f"\n--- TIME-BASED SPLIT ---")
lines.append(f"Total hari unik: {n_dates}")
lines.append(f"Tanggal split: {pd.Timestamp(split_date).date()}")
lines.append(f"Train: {pd.Timestamp(unique_dates[0]).date()} s.d. {pd.Timestamp(unique_dates[split_idx-1]).date()}")
lines.append(f"Test : {pd.Timestamp(split_date).date()} s.d. {pd.Timestamp(unique_dates[-1]).date()}")

df_train = df_encoded[df_encoded['TANGGAL'] < split_date].copy()
df_test = df_encoded[df_encoded['TANGGAL'] >= split_date].copy()

lines.append(f"\nJumlah baris train: {len(df_train)} ({len(df_train)/len(df_encoded)*100:.1f}%)")
lines.append(f"Jumlah baris test : {len(df_test)} ({len(df_test)/len(df_encoded)*100:.1f}%)")

# Cek distribusi target di train vs test (pastikan tidak terlalu timpang)
lines.append(f"\n--- DISTRIBUSI TARGET DI TRAIN VS TEST ---")
train_dist = df_train['Extreme_Event'].value_counts()
test_dist = df_test['Extreme_Event'].value_counts()
lines.append(f"Train -> Normal: {train_dist.get(0,0)}, Ekstrem: {train_dist.get(1,0)} ({train_dist.get(1,0)/len(df_train)*100:.2f}%)")
lines.append(f"Test  -> Normal: {test_dist.get(0,0)}, Ekstrem: {test_dist.get(1,0)} ({test_dist.get(1,0)/len(df_test)*100:.2f}%)")

# ============================================
# 3. DEFINISIKAN FITUR (X) & TARGET (y) - EXCLUDE kolom referensi/leakage
# ============================================
exclude_cols = ['TANGGAL', '_Curah_Hujan_Hari_Ini_REF_ONLY', 'Extreme_Event']
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

lines.append(f"\n--- DAFTAR FITUR FINAL YANG DIPAKAI MODEL ({len(feature_cols)} fitur) ---")
lines.append(str(feature_cols))

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

# ============================================
# 4. SCALING (fit HANYA di train, transform ke train & test - mencegah data leakage)
# ============================================
numeric_cols = ['Bulan', 'Hari_Tahun', 'Hari_Minggu', 'Lag1', 'Lag2', 'Lag3', 
                 'Roll7', 'Roll30', 'Tetangga_Lag1_avg', 'Tetangga_Roll7_avg']

scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

lines.append(f"\nScaling: RobustScaler di-fit HANYA pada data train, lalu diterapkan ke train & test")
lines.append(f"(one-hot encoded columns tidak di-scale karena sudah biner 0/1)")

# ============================================
# 5. BASELINE MODEL: LOGISTIC REGRESSION
# class_weight='balanced' untuk menangani class imbalance (1:18.9)
# ============================================
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ============================================
# 6. EVALUASI
# ============================================
lines.append("\n" + "="*60)
lines.append("HASIL EVALUASI BASELINE LOGISTIC REGRESSION")
lines.append("="*60)

cm = confusion_matrix(y_test, y_pred)
lines.append(f"\n--- CONFUSION MATRIX ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")

lines.append(f"\n--- METRIK UTAMA (Recall & F1 jadi acuan utama, BUKAN akurasi) ---")
lines.append(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}  (kurang relevan karena imbalance, hanya info tambahan)")
lines.append(f"Precision : {precision_score(y_test, y_pred):.4f}")
lines.append(f"Recall    : {recall_score(y_test, y_pred):.4f}  <- metrik paling penting untuk mitigasi bencana")
lines.append(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")

lines.append(f"\n--- CLASSIFICATION REPORT LENGKAP ---")
lines.append(classification_report(y_test, y_pred, target_names=['Normal', 'Ekstrem']))

# Feature importance sederhana (koefisien logistic regression)
lines.append(f"\n--- KOEFISIEN MODEL (semakin besar/positif, semakin kuat pengaruhnya ke prediksi 'Ekstrem') ---")
coef_df = pd.DataFrame({'fitur': feature_cols, 'koefisien': model.coef_[0]}).sort_values('koefisien', ascending=False)
lines.append(coef_df.to_string(index=False))

# Simpan train/test set (dengan scaling) untuk dipakai model Random Forest & XGBoost besok
X_train_scaled.assign(Extreme_Event=y_train.values, TANGGAL=df_train['TANGGAL'].values).to_csv(output_train_csv, index=False)
X_test_scaled.assign(Extreme_Event=y_test.values, TANGGAL=df_test['TANGGAL'].values).to_csv(output_test_csv, index=False)

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Train/test set tersimpan.")
print(f"Laporan tersimpan di: {output_report}")
print(f"\nRecall: {recall_score(y_test, y_pred):.4f} | F1: {f1_score(y_test, y_pred):.4f} | Precision: {precision_score(y_test, y_pred):.4f}")