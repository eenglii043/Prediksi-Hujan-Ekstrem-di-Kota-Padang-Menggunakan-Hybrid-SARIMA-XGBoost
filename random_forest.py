import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report, 
                               recall_score, precision_score, f1_score, accuracy_score)

# ============================================
# STEP 14: RANDOM FOREST + HYPERPARAMETER TUNING
# Menggunakan TimeSeriesSplit untuk cross-validation (menjaga urutan waktu,
# tidak random shuffle, agar tidak ada data leakage saat tuning)
# ============================================

train_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Train.csv"
test_path = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\Data_Test.csv"
output_report = r"C:\Users\w11\Documents\awal tugas akhir\03_Hasil\step14_random_forest_report.txt"

df_train = pd.read_csv(train_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')
df_test = pd.read_csv(test_path, parse_dates=['TANGGAL']).sort_values('TANGGAL')

feature_cols = [c for c in df_train.columns if c not in ['TANGGAL', 'Extreme_Event']]

X_train = df_train[feature_cols]
y_train = df_train['Extreme_Event']
X_test = df_test[feature_cols]
y_test = df_test['Extreme_Event']

lines = []
lines.append("="*60)
lines.append("LAPORAN RANDOM FOREST - STEP 14")
lines.append("="*60)
lines.append(f"\nJumlah fitur: {len(feature_cols)}")
lines.append(f"Train: {len(X_train)} baris | Test: {len(X_test)} baris")

# ============================================
# HYPERPARAMETER TUNING dengan GridSearchCV + TimeSeriesSplit
# Scoring pakai 'recall' karena itu metrik prioritas untuk mitigasi bencana
# ============================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

tscv = TimeSeriesSplit(n_splits=5)

lines.append(f"\n--- KONFIGURASI TUNING ---")
lines.append(f"Param grid: {param_grid}")
lines.append(f"Cross-validation: TimeSeriesSplit (5 fold, menjaga urutan waktu)")
lines.append(f"Scoring: recall (prioritas utama untuk mitigasi bencana)")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf_base, param_grid, cv=tscv, scoring='recall', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

lines.append(f"\n--- HASIL TUNING ---")
lines.append(f"Best params: {grid_search.best_params_}")
lines.append(f"Best CV recall score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# ============================================
# EVALUASI DI TEST SET
# ============================================
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
lines.append(f"\n" + "="*60)
lines.append("HASIL EVALUASI RANDOM FOREST (TUNED) DI TEST SET")
lines.append("="*60)
lines.append(f"\n--- CONFUSION MATRIX ---")
lines.append(f"                 Prediksi Normal   Prediksi Ekstrem")
lines.append(f"Aktual Normal    {cm[0][0]:<17} {cm[0][1]}")
lines.append(f"Aktual Ekstrem   {cm[1][0]:<17} {cm[1][1]}")

lines.append(f"\n--- METRIK UTAMA ---")
lines.append(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
lines.append(f"Precision : {precision_score(y_test, y_pred):.4f}")
lines.append(f"Recall    : {recall_score(y_test, y_pred):.4f}")
lines.append(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")

lines.append(f"\n--- CLASSIFICATION REPORT LENGKAP ---")
lines.append(classification_report(y_test, y_pred, target_names=['Normal', 'Ekstrem']))

# Feature importance
lines.append(f"\n--- FEATURE IMPORTANCE (Random Forest) ---")
importance_df = pd.DataFrame({
    'fitur': feature_cols, 
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
lines.append(importance_df.to_string(index=False))

# ============================================
# PERBANDINGAN LANGSUNG DENGAN BASELINE (Logistic Regression)
# ============================================
lines.append(f"\n" + "="*60)
lines.append("PERBANDINGAN DENGAN BASELINE (Logistic Regression, dari Step 13)")
lines.append("="*60)
lines.append(f"{'Metrik':<15}{'Logistic Reg.':<18}{'Random Forest':<18}")
lines.append(f"{'Recall':<15}{'0.4455':<18}{recall_score(y_test, y_pred):<18.4f}")
lines.append(f"{'Precision':<15}{'0.0918':<18}{precision_score(y_test, y_pred):<18.4f}")
lines.append(f"{'F1-Score':<15}{'0.1522':<18}{f1_score(y_test, y_pred):<18.4f}")

with open(output_report, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Selesai! Laporan tersimpan di: {output_report}")
print(f"\nBest params: {grid_search.best_params_}")
print(f"Recall: {recall_score(y_test, y_pred):.4f} | Precision: {precision_score(y_test, y_pred):.4f} | F1: {f1_score(y_test, y_pred):.4f}")